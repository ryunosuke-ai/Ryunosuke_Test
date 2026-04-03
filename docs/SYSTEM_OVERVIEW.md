# システム全体仕様書

> 高齢者との自然な対話を目指す、参加姿勢推定ベースのマルチモーダル会話エージェント

---

## 1. システム概要

### 1.1 目的

本システムは、ユーザーの発話から会話参加姿勢を推定し、会話の深さと負荷を動的に調整するマルチモーダル会話エージェントである。  
単に長く話すかどうかではなく、ユーザーがどの程度会話を続けたいかを `p_want_talk` として逐次推定し、その値と現在フェーズに応じて質問の強さ、話題の進め方、終了判断を切り替える。

### 1.2 主要技術スタック

| 技術 | 用途 |
|---|---|
| Azure OpenAI | 発話分類、回想判定、会話メモ更新、応答生成 |
| Azure Speech SDK | 音声認識（STT）、音声合成（TTS） |
| Streamlit | 実行中の状態表示 UI |
| ベイズ推定 | `p_want_talk` の逐次更新 |
| OpenCV | 実験用画像の読み込みと JPEG エンコード |

### 1.3 システム構成図

```text
┌──────────────────────────────────────────────────────────────┐
│                     apps/bayes_v3.py                         │
│                 MultimodalAgent（本体）                      │
│                                                              │
│  listen() ─→ classify_action() ─→ update_posterior()         │
│      │               │                      │                │
│      │               └─ judge_memory_signal()                │
│      │                                      │                │
│      └────────────→ update_conv_memory() ─→ transition_policy() 
│                                                     │        │
│                                                     └─ think_and_reply() ─→ speak()
│
│  途中経過は log_*.txt / analysis_*.csv / agent_*.log に保存  │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
                 apps/ui_display.py が logs/ を監視
```

---

## 2. 実行モード

### 2.1 音声版

- エントリーポイント: `apps/bayes_v3.py`
- 入力: マイク音声
- 出力: スピーカー音声
- UI: `apps/ui_display.py` と連携

### 2.2 テキスト版

- エントリーポイント: `apps/text_chat.py`
- 入力: コンソール入力
- 出力: コンソール表示
- 用途: 会話制御ロジックの検証、`p_want_talk` の推移確認

テキスト版は音声 I/O と UI 待機を除き、会話制御の中核ロジックを音声版とほぼ共有する。

---

## 3. モジュール構成

```text
core/models.py  ← 共通データ定義
  ↑  ↑  ↑
  │  │  └── core/conv_memory.py
  │  └───── core/phase_manager.py
  └──────── core/bayes_engine.py
                     ↑
apps/bayes_v3.py ----+--→ core/phase_manager.py
apps/text_chat.py ---+    core/conv_memory.py

apps/ui_display.py  ← 独立。logs/ 配下のみを読む
```

### 3.1 `core/models.py`

共通データ型を定義する。

| 型 | 説明 |
|---|---|
| `ActionType` | ユーザー発話の主ラベル |
| `Phase` | 会話フェーズ |
| `PhaseConfig` | フェーズごとの指示、画像利用有無、最大ターン数 |
| `Observation` | 1ターン分の観測情報 |
| `ClassificationResult` | 発話分類結果と理由 |
| `MemoryUpdate` | 会話要約、繰り返し禁止項目、終了意思 |

#### `ActionType`

現行実装では、発話は次の4値で分類される。

| ラベル | 意味 |
|---|---|
| `ACTIVE` | 自発的に話を広げる姿勢が強い |
| `RESPONSIVE` | 質問や話題に意味ある返答をしている |
| `MINIMAL` | 最小限の反応はあるが広がりにくい |
| `DISENGAGE` | 会話から離れたい、拒否・終了方向が明確 |

### 3.2 `core/bayes_engine.py`

発話分類とベイズ更新を担当する。

#### 主な関数

| 関数 | 説明 |
|---|---|
| `update_posterior()` | `ActionType` を観測として `p_want_talk` を更新する純関数 |
| `classify_action()` | LLM で発話を 4 主ラベルに分類し、理由を返す |
| `judge_memory_signal()` | 発話に回想・具体的エピソード記憶が含まれるかを判定する |

#### 発話分類の考え方

分類軸は「自己開示かどうか」ではなく、「会話参加姿勢の強さ」である。  
LLM には `ACTIVE / RESPONSIVE / MINIMAL / DISENGAGE` のうち必ず 1 つを返させる。

#### 尤度テーブル

| 仮説 | ACTIVE | RESPONSIVE | MINIMAL | DISENGAGE |
|---|---|---|---|---|
| H1: 話したい | 0.45 | 0.30 | 0.17 | 0.08 |
| H0: 話したくない | 0.10 | 0.25 | 0.35 | 0.30 |

`update_posterior()` はこの尤度を使ってベイズ更新を行い、結果を `0.001 ～ 0.999` にクランプする。

#### 回想判定

`judge_memory_signal()` は `BRIDGE` / `DEEP_DIVE` で使う補助観測であり、次を主な基準に `memory_flag` を返す。

- 過去の出来事を時間的に位置づけて語っている
- 特定の場面、場所、人物が登場する体験を述べている
- 個人的な出来事を過去形で語っている

### 3.3 `core/phase_manager.py`

フェーズ遷移と質問強度の制御を担う。

#### 現在のフェーズ

| フェーズ | 概要 |
|---|---|
| `SETUP` | 目的共有と低負荷な開始 |
| `INTRO` | 挨拶とアイスブレイク |
| `SURROUNDINGS` | 画像を使った共同注意 |
| `BRIDGE` | 画像から体験・好みへの連想 |
| `DEEP_DIVE` | 回想やエピソードの深掘り |
| `ENDING` | 締めと終了 |

#### 内部状態

- `phase`: 現在フェーズ
- `turn_in_phase`: 現フェーズ内ターン数
- `bridge_fail_count`: `BRIDGE` で深まりにくかった回数
- `deep_drop_count`: `DEEP_DIVE` で反応が落ちた回数
- `consecutive_empathy_only`: 連続で質問なし応答を返した回数

#### フェーズ設定の要点

| フェーズ | `require_image` | `max_turns` | 要点 |
|---|---|---|---|
| `SETUP` | No | 1 | 目的共有、質問しない |
| `INTRO` | No | 2 | 挨拶、質問は最大1つ |
| `SURROUNDINGS` | Yes | 3 | 画像について負荷の小さい話題提供 |
| `BRIDGE` | Yes | 4 | 画像から過去や好みへ自然に接続 |
| `DEEP_DIVE` | No | 6 | 共感・要約を優先し、必要なら質問1つ |
| `ENDING` | No | 2 | 質問禁止、感謝と終了 |

#### フェーズ遷移の核

- `SETUP → INTRO`: 常に遷移
- `INTRO → SURROUNDINGS`: 常に遷移
- `SURROUNDINGS → BRIDGE`: フェーズ内 1 ターン以上経過で遷移
- `BRIDGE → DEEP_DIVE`: `memory_flag=True` または `ACTIVE`
- `BRIDGE → SURROUNDINGS`: `MINIMAL` または `p_want_talk < 0.35` が 2 回続く
- `BRIDGE → ENDING`: `DISENGAGE` かつ `p_want_talk < 0.30`
- `DEEP_DIVE → SURROUNDINGS`: 反応低下や拒否が出たが、まだ終了には早い場合
- `DEEP_DIVE → ENDING`: `DISENGAGE` または反応低下があり、`p_want_talk < 0.30`
- `ENDING`: 維持

#### インタラクションモード

`get_interaction_mode_instruction()` は `ActionType` と `p_want_talk` に応じて LLM への指示を切り替える。

| 条件 | モード |
|---|---|
| `ENDING` | 質問禁止で締める |
| `DISENGAGE` | 負荷を下げ、深掘りしない |
| `MINIMAL` | 短いコメント + 負担の小さい質問1つ |
| `ACTIVE` または `p_want_talk >= 0.70` | 共感重視。必要に応じて質問1つ |
| `RESPONSIVE` または `p_want_talk >= 0.40` | 軽い誘導 |
| それ以外 | 再点火。二択や yes/no を優先 |

`notify_reply()` は AI の返答に質問が含まれたかを見て `consecutive_empathy_only` を更新し、共感だけが続きすぎないようにする。

### 3.4 `core/conv_memory.py`

会話メモ管理と終了意思判定を担当する。

#### 主な関数

| 関数 | 説明 |
|---|---|
| `detect_stop_intent()` | 正規表現ベースで終了意思を検出する |
| `extract_recent_assistant_questions()` | 直近の AI 発話から質問文を抽出する |
| `update_conv_memory()` | LLM で `summary` と `do_not_ask` を更新する |

#### `MemoryUpdate`

- `summary`: 会話要約
- `do_not_ask`: 既に答えた内容で再質問してはいけない項目
- `stop_intent`: ユーザーが終えたい意思

`update_conv_memory()` は LLM 失敗時にルールベースの簡易フォールバックを持つ。

### 3.5 `apps/bayes_v3.py`

音声版エージェント本体。  
環境変数読み込み、音声 I/O、ログ出力、フェーズ制御、応答生成を一元管理する。

#### 主な責務

- `.env` の検証
- Azure OpenAI / Speech SDK 初期化
- 実験用画像のロード
- 実行ログの保存
- UI 起動待機
- ターンごとの会話制御

#### 主要設定値

| 設定 | 値 |
|---|---|
| `max_total_turns` | 15 |
| `p_want_talk` 初期値 | 0.5 |
| `initial_silence_timeout_ms` | 10000 |
| `segmentation_silence_timeout_ms` | 1100 |
| 応答生成 `max_tokens` | 250 |

#### 応答生成

`think_and_reply()` は次の情報をシステムプロンプトへ組み込んで応答を生成する。

1. エージェントの基本スタイル
2. 長さ制約
3. 禁止事項
4. 会話要約
5. `do_not_ask`
6. 現在フェーズの指示
7. インタラクションモード
8. 必要時のみ画像
9. 待機時は専用の待機指示

### 3.6 `apps/text_chat.py`

テキスト版エージェント本体。  
`apps/bayes_v3.py` の会話制御を再利用しつつ、音声 I/O の代わりに標準入出力を使う。

追加要素:

- `display_status()` で `ActionType`、`p_want_talk`、フェーズを可視化
- 空 Enter を「考え中」として扱い、待機応答を返す
- UI 起動待機なしで即開始

### 3.7 `apps/ui_display.py`

実行ログを読むだけの監視 UI。

#### 主な機能

- `logs/run_*` から最新セッションを特定
- `analysis_*.csv` の最終行から `p_want_talk`、フェーズ、ターンを表示
- `log_*.txt` をチャットバブル風に表示
- `experiment_image.jpg` を左カラムに表示
- 2 秒ごとの自動更新

#### UI 連携

1. `apps/ui_display.py` 起動時に `ui_ready.flag` を作成
2. `apps/bayes_v3.py` がそのフラグを検知して会話開始
3. UI はログをポーリングして追従表示

---

## 4. 1ターンの処理フロー

### 4.1 音声版

```text
listen()
  ↓
update_conv_memory()
  ↓
classify_action()
  ↓
update_posterior()
  ↓
judge_memory_signal()   ※ BRIDGE / DEEP_DIVE のみ
  ↓
transition_policy()
  ↓
think_and_reply()
  ↓
speak()
  ↓
log_interaction()
```

実際の `run()` では、次の補助処理も入る。

- ユーザーの終了意思があれば `ENDING` に強制遷移
- 最大ターン数到達時は `ENDING` に強制遷移
- 入力が空なら待機用の短い応答を返す

### 4.2 テキスト版

流れは音声版と同じだが、`listen()` / `speak()` の代わりに `get_input()` / `output()` を使う。  
加えて `display_status()` が更新結果をターミナルに表示する。

---

## 5. ベイズ更新の仕組み

### 5.1 数式

```text
P(H1|E) = P(E|H1) × P(H1)
          ─────────────────────────────────────────────
          P(E|H1) × P(H1) + P(E|H0) × (1 - P(H1))
```

- `H1`: ユーザーが話したい
- `H0`: ユーザーが話したくない
- `E`: 観測された `ActionType`

### 5.2 初期値

- `p_want_talk = 0.5`

### 5.3 更新の傾向

- `ACTIVE`: 大きく上がりやすい
- `RESPONSIVE`: 緩やかに上がる、または維持
- `MINIMAL`: 下がりやすい
- `DISENGAGE`: 大きく下がりやすい

---

## 6. 会話制御の設計意図

### 6.1 低負荷な開始

`SETUP` と `INTRO` では質問数を抑え、入りやすさを優先する。

### 6.2 画像を足場にした連想

`SURROUNDINGS` と `BRIDGE` では画像を共同注意の対象として使い、いきなり深い話に入らず、風景・季節・行事・食べ物・人などの足場を経由して過去や好みへつなぐ。

### 6.3 深掘りは条件付き

深掘りに入る条件は、単なる長文ではなく次のいずれかである。

- 具体的回想が出た
- 発話姿勢が `ACTIVE`

### 6.4 繰り返し質問の抑制

会話メモの `do_not_ask` をプロンプトに入れ、既に判明した好みや予定を再質問しないようにする。

### 6.5 離脱兆候への対応

`DISENGAGE` や `MINIMAL` の連続を検知した場合、深掘りを止める、周囲共有へ戻す、終了へ向かう、のいずれかで負荷を下げる。

---

## 7. ログ出力

各実行は `logs/run_YYYYMMDD_HHMMSS/` に保存される。

```text
logs/run_YYYYMMDD_HHMMSS/
├── log_*.txt
├── analysis_*.csv
└── agent_*.log
```

### 7.1 `log_*.txt`

- 会話本文の履歴
- 形式: `[HH:MM:SS] AI: ...` / `[HH:MM:SS] User: ...`

### 7.2 `analysis_*.csv`

現在の実装では次の列を出力する。

| 列名 | 内容 |
|---|---|
| `Timestamp` | 時刻 |
| `Turn` | 総ターン数 |
| `Phase` | フェーズ名 |
| `Speaker` | `AI` / `User` |
| `PrimaryLabel` | 分類ラベル |
| `LabelReason` | 分類理由 |
| `P_WantTalk` | 更新後の値 |
| `Text` | 発話本文 |

### 7.3 `agent_*.log`

- 内部ログ
- フェーズ遷移理由
- LLM 失敗や STT/TTS 失敗の警告

---

## 8. 起動方法

### 8.1 UI + 音声版

```bash
# ターミナル1
python3 -m streamlit run apps/ui_display.py

# ターミナル2
python3 -m apps.bayes_v3
```

### 8.2 テキスト版

```bash
python3 -m apps.text_chat
```

---

## 9. 環境変数

`.env` に以下を設定する。

```env
AZURE_SPEECH_KEY=
AZURE_SPEECH_REGION=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_DEPLOYMENT_NAME=
AZURE_OPENAI_API_VERSION=
```

テキスト版は Azure Speech を使わないが、音声版では必須。

---

## 10. 実験用画像

- ファイル名: `experiment_image.jpg`
- 配置場所: プロジェクトルート
- 用途: `SURROUNDINGS` と `BRIDGE` で会話生成時に参照

音声版では OpenCV で読み込み、Base64 JPEG にして LLM へ渡す。  
テキスト版ではファイルを直接 Base64 化して使う。

---

## 11. LLM 呼び出し一覧

1ターンあたり最大 4 回の LLM 呼び出しが発生する。

| 呼び出し元 | 用途 | max_tokens | temperature |
|---|---|---|---|
| `classify_action()` | 4 主ラベル分類 | 120 | 0.0 |
| `judge_memory_signal()` | 回想・具体的記憶の判定 | 80 | 0.0 |
| `update_conv_memory()` | 会話要約と `do_not_ask` 更新 | 512 | 0.0 |
| `think_and_reply()` | ユーザー向け応答生成 | 250 | 0.7 |

`judge_memory_signal()` は `BRIDGE` / `DEEP_DIVE` のときのみ呼ばれる。

---

## 12. 現行仕様での注意点

- 設計の中心は `SILENCE / NORMAL / DISCLOSURE` ではなく、`ACTIVE / RESPONSIVE / MINIMAL / DISENGAGE` の 4 ラベルである
- `consecutive_silence` は現状の遷移ロジックでは使われていない
- `SETUP` と `INTRO` は現在の `transition_policy()` では条件分岐なしに次フェーズへ進む
- 応答は「共感だけが続きすぎない」よう `consecutive_empathy_only` で微調整される

この文書は、現行実装のコードを正として更新している。
