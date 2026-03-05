# システム全体仕様書

> ベイズ推定を用いた対話型マルチモーダルエージェントシステム

---

## 1. システム概要

### 1.1 目的

本システムは、**高齢者との自然な対話**を目指すマルチモーダル会話エージェントである。ユーザーの発話をベイズ推定で分析し、「話したい確率（p_want_talk）」をリアルタイムに推定することで、ユーザーの負担に応じた適応的な会話を実現する。

### 1.2 主要技術スタック

| 技術 | 用途 |
|---|---|
| Azure OpenAI (GPT) | 発話分類・返信生成・会話メモ更新 |
| Azure Speech SDK | 音声認識（STT）・音声合成（TTS） |
| Streamlit | リアルタイムUI表示 |
| ベイズ推定 | ユーザーの「話したい確率」の逐次更新 |

### 1.3 システム構成図

```
┌─────────────────────────────────────────────────────────┐
│                     bayes_v3.py                         │
│                 （オーケストレータ）                       │
│                                                         │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ listen() │→│classify_action│→│ update_posterior  │   │
│  │  (STT)   │  │  (LLM分類)   │  │  (ベイズ更新)     │   │
│  └──────────┘  └──────────────┘  └──────────────────┘   │
│       │                                    │            │
│       │         ┌──────────────┐           │            │
│       │         │judge_memory  │           │            │
│       │         │_and_disclosure│          │            │
│       │         │  (LLM判定)   │           │            │
│       │         └──────────────┘           │            │
│       │                                    ↓            │
│       │         ┌──────────────┐  ┌────────────────┐    │
│       │         │update_conv   │  │transition      │    │
│       │         │_memory       │  │_policy         │    │
│       │         │ (会話メモ)    │  │(フェーズ遷移)   │    │
│       │         └──────────────┘  └────────────────┘    │
│       │                                    │            │
│       │         ┌──────────────┐           │            │
│       └────────→│think_and     │←──────────┘            │
│                 │_reply (LLM)  │                        │
│                 └──────┬───────┘                        │
│                        │                                │
│                 ┌──────┴───────┐                        │
│                 │  speak()     │                        │
│                 │  (TTS)       │                        │
│                 └──────────────┘                        │
└─────────────────────────────────────────────────────────┘
         │ CSV・ログ出力
         ↓
┌─────────────────────┐
│   ui_display.py     │
│  (Streamlit UI)     │
│  2秒ごとにポーリング  │
└─────────────────────┘
```

---

## 2. モジュール依存関係

```
models.py  ← 全モジュールが依存（データ構造定義、一方向のみ）
  ↑  ↑  ↑
  │  │  └── conv_memory.py     (MemoryUpdate)
  │  └───── phase_manager.py   (Phase, PhaseConfig, Observation, ActionType)
  └──────── bayes_engine.py    (ActionType)
                ↑
bayes_v3.py ----+--→ phase_manager.py
                +--→ conv_memory.py
                └--→ models.py

ui_display.py  ← 独立（bayes_v3 の出力ファイルを消費するのみ）
```

循環参照は存在しない。`models.py` が最下層のデータ定義層として機能し、他のモジュールはすべてここに依存する。

---

## 3. 各ファイルの役割と主要関数

### 3.1 models.py（45行）— データ構造定義

全モジュールが参照する共通のデータ型を定義する。

| クラス/Enum | 種別 | 説明 |
|---|---|---|
| `ActionType` | Enum | ユーザー発話の分類: `SILENCE`, `NORMAL`, `DISCLOSURE` |
| `Phase` | Enum | 会話フェーズ: `SETUP`, `INTRO`, `SURROUNDINGS`, `BRIDGE`, `DEEP_DIVE`, `ENDING` |
| `PhaseConfig` | dataclass | フェーズの設定（指示文、画像要否、最大ターン数） |
| `Observation` | dataclass | ユーザー観測情報（発話テキスト、分類、生返事、回想、自己開示） |
| `MemoryUpdate` | dataclass | 会話メモ更新（要約、繰り返し禁止リスト、終了意思） |

#### Phase（会話フェーズ）の値

| Phase | 表示名 |
|---|---|
| `SETUP` | 初期設定 |
| `INTRO` | 挨拶とアイスブレイク |
| `SURROUNDINGS` | 画像についての話題 |
| `BRIDGE` | 連想（体験や好みへの接続） |
| `DEEP_DIVE` | エピソードの深掘り |
| `ENDING` | エンディング |

---

### 3.2 bayes_engine.py（238行）— ベイズ更新 + 観測分類

ユーザーの発話を分類し、ベイズの定理で「話したい確率」を更新する。

#### 主要関数

| 関数 | 引数 | 戻り値 | 説明 |
|---|---|---|---|
| `update_posterior()` | `p_want_talk, action_type, minimal_reply` | `float` | ベイズ更新を実行し事後確率を返す（純関数） |
| `is_minimal_reply()` | `user_text` | `bool` | 「はい」「そう」など生返事かどうかを判定 |
| `classify_action()` | `client, deployment, user_text, logger` | `ActionType` | LLMで発話を DISCLOSURE / NORMAL に分類 |
| `judge_memory_and_disclosure()` | `client, deployment, user_text, logger` | `Tuple[bool, bool, Optional[str]]` | LLMで回想フラグ・自己開示フラグ・メモを判定 |

#### ベイズ更新の数式

```
P(H1|E) = L(E|H1) × P(H1) / [L(E|H1) × P(H1) + L(E|H0) × P(H0)]

H1 = 「ユーザーが話したい」
H0 = 「ユーザーが話したくない」
E  = 観測された発話タイプ（ActionType）
```

#### 尤度テーブル

| 仮説 | SILENCE | NORMAL | DISCLOSURE |
|---|---|---|---|
| H1（話したい） | 0.10 | 0.25 | 0.65 |
| H0（話したくない） | 0.45 | 0.50 | 0.05 |

- 生返事時（`is_minimal_reply=True`）はNORMALの尤度を上書き: H1=0.15, H0=0.70
- 事後確率は `0.001 ～ 0.999` にクランプ

#### 発話分類の基準（社会的浸透理論ベース）

LLMプロンプトで以下の基準により DISCLOSURE / NORMAL を判定:
- 感情・内面の表出（楽しい、寂しいなど）
- 個人的体験・記憶の言及（〜したことがある、昔は〜）
- 個人属性の共有（好き嫌い、習慣、家族構成）
- 願望・意図の表明（〜したい、〜してみたい）

---

### 3.3 phase_manager.py（226行）— フェーズ遷移ポリシー

会話のフェーズを管理し、観測と確率に応じた遷移を決定する。

#### PhaseManager クラス

| メソッド | 説明 |
|---|---|
| `__init__()` | フェーズ状態を初期化（SETUP から開始） |
| `transition_policy(obs, p_want_talk)` | フェーズ遷移ポリシーを適用（核ロジック） |
| `get_interaction_mode_instruction(obs, p_want_talk)` | p_want_talk に応じた質問強度の指示を返す |

#### 内部状態

```python
self.phase: Phase              # 現在のフェーズ
self.turn_in_phase: int        # フェーズ内ターン数
self.consecutive_silence: int  # 連続沈黙カウント
self.bridge_fail_count: int    # BRIDGE で回想が出なかった回数
self.deep_drop_count: int      # DEEP_DIVE で反応が落ちた回数
```

#### フェーズ設定テーブル

| フェーズ | max_turns | require_image | 指示の要点 |
|---|---|---|---|
| SETUP | 1 | No | 目的共有（超短く）、質問なし |
| INTRO | 2 | No | 挨拶・アイスブレイク、質問は最大1つ |
| SURROUNDINGS | 3 | Yes | 画像についての話、リアクション重視 |
| BRIDGE | 4 | Yes | 連想: 周囲→季節→行事→食べ物→人 |
| DEEP_DIVE | 6 | No | 深掘り: 感情・意味・関連記憶、質問より理解優先 |
| ENDING | 2 | No | エンディング: 質問禁止、感謝で閉じる |

---

### 3.4 conv_memory.py（137行）— 会話メモ更新 + 終了意思検出

会話の要約を管理し、繰り返し質問を防ぐ。

#### 主要関数

| 関数 | 引数 | 戻り値 | 説明 |
|---|---|---|---|
| `detect_stop_intent()` | `user_text` | `bool` | 正規表現で終了意思を検出 |
| `extract_recent_assistant_questions()` | `messages, max_questions=8` | `List[str]` | 直近AI発話から質問文を抽出 |
| `update_conv_memory()` | `client, deployment, conv_memory, user_text, history, recent_questions, logger` | `MemoryUpdate` | LLMで会話メモを更新 |

#### 終了意思の検出パターン

```python
_STOP_PATTERNS = [
    r"(そろそろ|もう|一旦).{0,6}(終わり|終える|やめ|切り|終了)",
    r"(会話|話).{0,6}(やめ|終わり|終了)",
    r"(終わりに|終わりで|ここまで|ストップ|停止)",
    r"(また今度|またね|ばいばい|バイバイ)",
]
```

#### 会話メモの構造

- `summary`: 会話要約（500文字以内）
- `do_not_ask`: ユーザーが既に答えた項目リスト（最大8件）
- `stop_intent`: ユーザーの終了意思（bool）

---

### 3.5 bayes_v3.py（512行）— エントリーポイント兼オーケストレータ

システム全体を統合する `MultimodalAgent` クラスを提供。音声I/O、LLM呼び出し、ベイズ更新、フェーズ遷移、ログ出力を一手に管理する。

#### MultimodalAgent クラス — 主要メソッド

| メソッド | 説明 |
|---|---|
| `__init__()` | Azure Speech SDK/OpenAI クライアント初期化、PhaseManager 生成 |
| `speak(text)` | テキストを Azure TTS で音声出力 |
| `listen()` | Azure STT でマイク入力を取得（沈黙なら None を返す） |
| `classify_action(user_text)` | bayes_engine に委譲して発話分類 |
| `update_posterior(action_type, minimal_reply)` | bayes_engine に委譲してベイズ更新 |
| `judge_memory_and_disclosure(user_text)` | bayes_engine に委譲して回想・自己開示判定 |
| `transition_policy(obs)` | phase_manager に委譲してフェーズ遷移 |
| `think_and_reply(obs, base64_image)` | LLMで返信を生成（システムプロンプト構築含む） |
| `log_interaction(speaker, text, action_type)` | CSV に分析データを記録 |
| `load_history_as_messages(max_messages)` | 会話ログをメッセージ形式で読み込み |
| `run()` | **メインループ**: UI待機 → 挨拶 → ターンループ → 終了 |

#### 主要設定値

| 設定 | 値 | 説明 |
|---|---|---|
| `max_total_turns` | 15 | 全フェーズの最大ターン数 |
| `p_want_talk` 初期値 | 0.5 | 話したい確率の初期値 |
| `initial_silence_timeout_ms` | 5000 | マイク待機時間 |
| `segmentation_silence_timeout_ms` | 1100 | セグメント間沈黙タイムアウト |
| `think_and_reply max_tokens` | 250 | 返信生成の最大トークン数 |

#### ログ出力構造

```
logs/run_YYYYMMDD_HHMMSS/
├── log_YYYYMMDD_HHMMSS.txt       # 会話ログ（テキスト）
├── analysis_YYYYMMDD_HHMMSS.csv  # 分析データ（CSV）
└── agent_YYYYMMDD_HHMMSS.log     # エージェントログ
```

#### think_and_reply のシステムプロンプト構成

```
1. エージェント説明（親しみやすい会話ロボット）
2. 返答スタイル（60～120文字、短いコメント→質問1つ）
3. 禁止事項（連続質問、聞き返し、説教）
4. 会話メモ（要約 + do_not_ask リスト）
5. 現在フェーズと指示文
6. インタラクションモード指示（p_want_talk に応じた質問強度）
7. 初期画像質問（SURROUNDINGS フェーズのみ）
```

---

### 3.6 ui_display.py（421行）— Streamlit リアルタイムUI

`bayes_v3.py` が出力するCSV・ログファイルを2秒ごとにポーリングし、リアルタイムに表示する。

#### 主要関数

| 関数 | 説明 |
|---|---|
| `find_latest_run()` | 最新の `run_*` ディレクトリを返す |
| `find_file_in_dir(directory, prefix, ext)` | 指定パターンでファイルを検索 |
| `read_analysis_csv(csv_path)` | CSVから最新状態を読み取る（p_want_talk, phase, turn） |
| `read_conversation_log(log_path)` | テキストログをパースして会話メッセージを返す |
| `render_status_bar(p, phase, turn)` | ステータスをコンパクトバーで表示 |
| `render_chat_log(messages)` | チャットバブル風HTMLで会話ログを表示 |
| `main()` | メイン実行関数 |

#### UIレイアウト

```
┌──────────────────────────────────────────┐
│  左カラム(2)    │  右カラム(3)             │
│                │                         │
│  実験用画像     │  ステータスバー           │
│  (experiment   │  ┌───────────────────┐  │
│   _image.jpg)  │  │P(話したい): 0.65  │  │
│                │  │フェーズ: 深掘り    │  │
│                │  │ターン: 5/15       │  │
│                │  └───────────────────┘  │
│                │                         │
│                │  チャットログ             │
│                │  ┌───────────────────┐  │
│                │  │🤖 こんにちは！     │  │
│                │  │👤 こんにちは       │  │
│                │  │🤖 この写真...     │  │
│                │  └───────────────────┘  │
└──────────────────────────────────────────┘
  フッター: 自動更新コントロール（停止/再開）
```

#### UI連携メカニズム

1. `ui_display.py` 起動時に `ui_ready.flag` を作成
2. `bayes_v3.py` はこのフラグを検出してから会話を開始
3. UIは CSV・ログを2秒ごとにポーリングして表示を更新

---

## 4. 会話フローの詳細

### 4.1 メインループ（1ターンの処理フロー）

```
while total_turns < max_total_turns:

  ① listen()
     → ユーザーの音声をテキストに変換（沈黙なら None）

  ② classify_action(user_text)
     → LLM で発話を DISCLOSURE / NORMAL に分類
     → 沈黙の場合は SILENCE

  ③ is_minimal_reply(user_text)
     → 「はい」「そう」等の生返事を検出

  ④ update_posterior(action_type, minimal_reply)
     → ベイズ更新で p_want_talk を更新

  ⑤ judge_memory_and_disclosure(user_text)
     → LLM で回想フラグ・自己開示フラグを判定

  ⑥ Observation オブジェクト生成
     → user_text, action_type, minimal_reply, memory_flag, self_disclosure_flag

  ⑦ update_conv_memory(user_text)
     → LLM で会話メモを更新（summary, do_not_ask, stop_intent）

  ⑧ 終了意思チェック
     → detect_stop_intent() または conv_memory.stop_intent が True → ENDING へ

  ⑨ transition_policy(obs, p_want_talk)
     → フェーズ遷移ポリシーを適用

  ⑩ think_and_reply(obs, image_b64)
     → LLM でシステムプロンプト + 会話履歴から返信を生成

  ⑪ speak(reply)
     → Azure TTS で返信を音声出力

  ⑫ log_interaction()
     → CSV に分析データを記録

  ⑬ phase == ENDING なら break
```

### 4.2 インタラクションモード（p_want_talk による質問強度の調整）

| p_want_talk | モード | 質問強度 |
|---|---|---|
| >= 0.70 | 共感重視 | 共感/要約/感想が中心、質問は最大1つ（できればしない） |
| >= 0.40 | 軽い誘導 | コメント → 小さな質問1つ（はい/いいえ、二択） |
| < 0.40 | 再点火 | 周囲要素を1つ拾うコメント、二択か「はい/いいえ」のみ |
| SILENCE 時 | 低負荷 | 短い気遣い + 沈黙の余白、質問なし |
| 生返事時 | 軽い誘導 | 短いコメント → 小さな質問1つ |

---

## 5. フェーズ遷移ルール

### 5.1 遷移フロー図

```
SETUP ──(常に遷移)──→ INTRO
                        │
                   (沈黙以外の応答)
                        ↓
                    SURROUNDINGS ←──────────────────┐
                        │                           │
                   (1ターン経過 + 沈黙以外)           │
                        ↓                           │
                      BRIDGE                        │
                     ╱     ╲                        │
          (回想 or              (沈黙×2 or           │
           DISCLOSURE)           p<0.35 が2回)       │
                ↓                    ↓              │
            DEEP_DIVE          SURROUNDINGSに戻る ──┘
             ╱     ╲
   (NORMAL/SILENCE×2    (DISCLOSURE継続)
    + p<0.30)                ↓
        ↓              （留まる）
      ENDING
        │
   (p>=0.30 の場合)
        ↓
   SURROUNDINGSに戻る ──────────────────────────────┘
```

### 5.2 各フェーズの遷移条件

| 現在フェーズ | 条件 | 遷移先 | 理由 |
|---|---|---|---|
| SETUP | 常に | INTRO | セットアップ完了 |
| INTRO | 沈黙以外の応答 | SURROUNDINGS | 導入完了 |
| SURROUNDINGS | 1ターン経過 + 沈黙以外 | BRIDGE | 共同注意が成立 |
| BRIDGE | memory_flag=True OR DISCLOSURE | DEEP_DIVE | 回想・自己開示が出た |
| BRIDGE | 失敗2回（沈黙 OR p<0.35） | SURROUNDINGS | 回想が出にくい、戻って仕切り直し |
| DEEP_DIVE | DISCLOSURE 継続 | （留まる） | エンゲージメント維持 |
| DEEP_DIVE | 反応低下2回 + p<0.30 | ENDING | 負担が高い、終了 |
| DEEP_DIVE | 反応低下2回 + p>=0.30 | SURROUNDINGS | 負担を下げて仕切り直し |
| 全フェーズ | 連続沈黙2回 + p<0.20 | ENDING | 安全な終了（負担が高い） |
| 全フェーズ | 連続沈黙2回 + p>=0.20 | SURROUNDINGS | 負担を下げる |
| 全フェーズ | max_turns 到達 | 次フェーズ（線形） | ターン上限 |

### 5.3 沈黙処理

```
連続沈黙カウント（consecutive_silence）:
  - SILENCE → +1
  - SILENCE 以外 → 0 にリセット

連続沈黙 >= 2 の場合:
  p_want_talk < 0.20 → ENDING（安全な終了）
  p_want_talk >= 0.20 → SURROUNDINGS へ戻す（負担を下げる）
```

---

## 6. ベイズ更新の仕組み（詳細）

### 6.1 ベイズの定理

```
          P(E|H1) × P(H1)
P(H1|E) = ─────────────────────────────────
          P(E|H1) × P(H1) + P(E|H0) × P(H0)
```

- **H1**: ユーザーが話したい（p_want_talk）
- **H0**: ユーザーが話したくない（1 - p_want_talk）
- **E**: 観測された行動（ActionType）

### 6.2 尤度テーブル

```
                 SILENCE    NORMAL    DISCLOSURE
H1（話したい）    0.10       0.25       0.65
H0（話したくない） 0.45       0.50       0.05
```

**設計意図**:
- DISCLOSUREは「話したい」を強く示唆（H1: 0.65 vs H0: 0.05）
- SILENCEは「話したくない」を示唆（H1: 0.10 vs H0: 0.45）
- NORMALは弱いシグナル（H1: 0.25 vs H0: 0.50）

### 6.3 生返事の補正

NORMALかつ生返事（「はい」「そう」「うん」等）の場合、尤度を上書き:

```
H1: 0.15（話したい人でも生返事はあるが信号性は低い）
H0: 0.70（話したくない人が生返事する確率は高い）
```

### 6.4 更新例

```
初期: p_want_talk = 0.50

観測: DISCLOSURE の場合
  P(H1|E) = (0.65 × 0.50) / (0.65 × 0.50 + 0.05 × 0.50)
           = 0.325 / 0.35
           ≈ 0.929   → 話したい確率が大幅に上昇

観測: SILENCE の場合
  P(H1|E) = (0.10 × 0.50) / (0.10 × 0.50 + 0.45 × 0.50)
           = 0.05 / 0.275
           ≈ 0.182   → 話したい確率が大幅に低下
```

---

## 7. コード規模サマリ

| ファイル | 行数 | 役割 |
|---|---|---|
| `models.py` | 45 | データ構造定義 |
| `bayes_engine.py` | 238 | ベイズ更新 + 観測分類 |
| `phase_manager.py` | 226 | フェーズ遷移ポリシー |
| `conv_memory.py` | 137 | 会話メモ更新 + 終了意思検出 |
| `bayes_v3.py` | 512 | エントリーポイント兼オーケストレータ |
| `ui_display.py` | 421 | Streamlit リアルタイムUI |
| **合計** | **1,579** | |

---

## 8. 起動方法

```bash
# ターミナル1: UIを先に起動
streamlit run ui_display.py

# ターミナル2: エージェントを起動（UIの準備完了を待機してから開始）
python bayes_v3.py
```

`bayes_v3.py` は `ui_ready.flag` ファイルが作成されるまで待機する。このフラグは `ui_display.py` が起動時に作成する。

---

## 9. 環境変数

`.env` ファイルに以下が必要:

```
AZURE_SPEECH_KEY=
AZURE_SPEECH_REGION=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_DEPLOYMENT_NAME=
AZURE_OPENAI_API_VERSION=
```

---

## 10. LLM呼び出し一覧

本システムでは1ターンあたり最大4回のLLM呼び出しが発生する。

| 呼び出し元 | 用途 | max_tokens | temperature |
|---|---|---|---|
| `classify_action()` | 発話を DISCLOSURE / NORMAL に分類 | 6 | 0.0 |
| `judge_memory_and_disclosure()` | 回想・自己開示の2軸判定 | 80 | 0.0 |
| `update_conv_memory()` | 会話メモ更新（要約 + do_not_ask） | 512 | 0.3 |
| `think_and_reply()` | ユーザーへの返信生成 | 250 | 0.7 |
