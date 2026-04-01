# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 言語設定
- 常に日本語で会話する
- コメントも日本語で記述する
- エラーメッセージの説明も日本語で行う
- ドキュメントも日本語で生成する

## 作業ルール
- 作業を開始する前に、gitが初期化されていない場合は `git init` を実行すること
- その後、必ず以下を順番に実行すること
  1. 変更内容を表すコミットメッセージで `git add -A && git commit` を実行すること
  2. `git push origin master`
- コミットメッセージは以下の形式に従うこと：
  - `feat: ○○機能を追加` — 新機能
  - `fix: ○○のバグを修正` — バグ修正
  - `refactor: ○○をリファクタリング` — 動作変更なしのコード整理
  - `docs: ○○のドキュメントを更新` — ドキュメントのみの変更
  - `chore: ○○` — ビルド・設定ファイル等の雑務
- リモート: https://github.com/ryunosuke-ai/Ryunosuke_Test.git

## 起動方法

メインシステム（`bayes_v3.py` + `ui_display.py`）の起動手順：

```bash
# ターミナル1: UIを先に起動
python3 -m streamlit run ui_display.py

# ターミナル2: エージェントを起動（UIの準備完了を待機してから開始）
python3 bayes_v3.py
```

`bayes_v3.py` は `ui_ready.flag` ファイルが作成されるまで待機する。このフラグは `ui_display.py` が起動時に作成する。

テキストベース（音声なし）で動作確認する場合：

```bash
python3 text_chat.py
```

## テスト

```bash
# 全テスト実行
python3 -m pytest test_bayes_v3.py test_ui_display.py -v

# 単一テストクラス
python3 -m pytest test_bayes_v3.py::TestUpdatePosterior -v

# 単一テストメソッド
python3 -m pytest test_bayes_v3.py::TestUpdatePosterior::test_disclosure_raises_probability -v
```

テストは Azure Speech SDK / cv2 / OpenAI API をすべてモック化しており、APIキーやネットワーク接続なしで実行可能。テンポラリディレクトリを使用するためクリーンアップ不要。

## 環境変数（.env）

`.env` ファイルに以下が必要：

```
AZURE_SPEECH_KEY=
AZURE_SPEECH_REGION=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_DEPLOYMENT_NAME=
AZURE_OPENAI_API_VERSION=
```

## 実験用画像

プロジェクトルートに `experiment_image.jpg` を配置すること。エージェントはこの静止画を会話に使用する（SURROUNDINGS・BRIDGEフェーズでLLMに送信）。画像の内容が会話の話題に直接影響する。

---

## コードアーキテクチャ

### 主要ファイルの役割

| ファイル | 役割 |
|---|---|
| `bayes_v3.py` | **エントリーポイント兼オーケストレータ**。MultimodalAgent: 音声I/O, ベイズ更新, フェーズ遷移, LLM返信生成, ログ出力 |
| `text_chat.py` | **テキスト版エージェント**。TextChatAgent: 音声I/O・UI同期なし、コンソールでステータス表示。調査・デバッグ用 |
| `models.py` | データ構造定義（ActionType, Phase, PhaseConfig, Observation, MemoryUpdate） |
| `bayes_engine.py` | ベイズ更新 + 観測分類（update_posterior, is_minimal_reply, classify_action, judge_memory_and_disclosure） |
| `phase_manager.py` | フェーズ遷移ポリシー + 設定テーブル + インタラクションモード指示 |
| `conv_memory.py` | 会話メモ更新 + 終了意思検出 |
| `ui_display.py` | **メインUI**。Streamlitでログをリアルタイム表示（bayes_v3.pyの出力ファイルを消費するのみ） |
| `test_bayes_v3.py` | bayes_v3 / bayes_engine / phase_manager / conv_memory のユニットテスト |
| `test_ui_display.py` | ui_display のユニットテスト |

### モジュール依存関係（循環参照なし）

```
models.py  ← 全モジュールが依存（一方向のみ）
  ↑  ↑  ↑
  |  |  └── conv_memory.py
  |  └───── phase_manager.py
  └──────── bayes_engine.py
                ↑
bayes_v3.py ----+--→ phase_manager.py
text_chat.py ---+    conv_memory.py
```

`ui_display.py` は他モジュールに依存せず、CSV・ログファイルのみを消費する。

### 1ターンの処理フロー

```
listen() → classify_action() → update_posterior() → judge_memory_and_disclosure()
               ↓                      ↓                        ↓
        ActionType判定           p_want_talk更新          回想/自己開示フラグ
        (SILENCE/NORMAL/                                       ↓
         DISCLOSURE)                              update_conv_memory() → メモ更新
                                                               ↓
                                                  transition_policy() → フェーズ遷移
                                                               ↓
                                                  think_and_reply() → speak()
```

1ターンあたり最大4回のLLM呼び出しが発生：

| 関数 | 用途 | max_tokens | temperature |
|---|---|---|---|
| `classify_action()` | 発話分類 DISCLOSURE/NORMAL | 6 | 0.0 |
| `judge_memory_and_disclosure()` | 回想・自己開示の2軸判定 | 80 | 0.0 |
| `update_conv_memory()` | 会話メモ更新 | 512 | 0.3 |
| `think_and_reply()` | 返信生成 | 250 | 0.7 |

### ベイズ更新の仕組み

`p_want_talk`（ユーザーが話したい確率）をベイズの定理で逐次更新する。初期値は0.5。

尤度テーブル（`bayes_engine.py` の `DEFAULT_LIKELIHOODS`）：

| 仮説 | SILENCE | NORMAL | DISCLOSURE |
|---|---|---|---|
| H1（話したい） | 0.10 | 0.25 | 0.65 |
| H0（話したくない） | 0.45 | 0.50 | 0.05 |

生返事（「はい」「そう」等）時は NORMAL の尤度を H1=0.15, H0=0.70 に上書きする。

### フェーズ遷移

6段階のフェーズを線形に進行しつつ、条件に応じて後退・スキップする：

```
SETUP → INTRO → SURROUNDINGS → BRIDGE → DEEP_DIVE → ENDING
                     ↑            |           |
                     └────────────+───────────┘
                      (反応低下時に戻す)
```

- **BRIDGE → DEEP_DIVE**: 回想 or 自己開示が検出されたとき
- **BRIDGE → SURROUNDINGS**: 2回失敗（沈黙 or p<0.35）で仕切り直し
- **DEEP_DIVE → ENDING**: 反応低下2回 + p<0.30
- **全フェーズ → ENDING**: 連続沈黙2回 + p<0.20（安全終了）

### インタラクションモード（質問強度の動的調整）

`phase_manager.get_interaction_mode_instruction()` が p_want_talk に応じて指示を切り替える：

| p_want_talk | モード | 質問強度 |
|---|---|---|
| >= 0.70 | 共感重視 | 質問は最大1つ、できればしない |
| 0.40〜0.70 | 軽い誘導 | コメント → 小さな質問1つ |
| < 0.40 | 再点火 | 二択 or はい/いいえのみ |
| SILENCE時 | 低負荷 | 短い気遣い、質問なし |

### ログ出力

実行ごとに `logs/run_YYYYMMDD_HHMMSS/` ディレクトリへ保存：

```
logs/run_YYYYMMDD_HHMMSS/
├── log_*.txt          # 会話ログ（[HH:MM:SS] AI/User: テキスト）
├── analysis_*.csv     # 分析データ（Timestamp,Turn,Phase,Speaker,ActionType,P_WantTalk,Text）
└── agent_*.log        # エージェントログ
```

UI連携: `ui_ready.flag` で起動同期、UIは CSV/ログを2秒ごとにポーリング表示
