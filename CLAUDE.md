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
  1. `git add -A && git commit -m "before claude edit"`
  2. `git push origin master`
- リモート: https://github.com/ryunosuke-ai/Ryunosuke_Test.git

## 起動方法

メインシステム（`bayes_v3.py` + `ui_display.py`）の起動手順：

```bash
# ターミナル1: UIを先に起動
streamlit run ui_display.py

# ターミナル2: エージェントを起動（UIの準備完了を待機してから開始）
python bayes_v3.py
```

`bayes_v3.py` は `ui_ready.flag` ファイルが作成されるまで待機する。このフラグは `ui_display.py` が起動時に作成する。

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

プロジェクトルートに `experiment_image.jpg` を配置すること。エージェントはこの静止画を会話に使用する。

---

## コードアーキテクチャ

### 主要ファイルの役割

| ファイル | 役割 |
|---|---|
| `bayes_v3.py` | **現行メインエージェント**。ベイズ更新・フェーズ管理・会話メモを統合 |
| `ui_display.py` | **メインUI**。Streamlitでログをリアルタイム表示 |
| `omomi_score.py` | `bayes_v3.py` の前身（重み係数ベースの更新、カメラ使用） |
| `enshutu_main.py` | フェーズ管理の初期実装（カメラ使用） |
| `main.py` | 原型の単純なマルチモーダルエージェント（カメラ使用） |
| `bayes_v3_4class.py` / `bayes_v3_4class_llm.py` | ActionTypeを4クラスに拡張した実験バリアント |
| `app.py` | バックグラウンドスレッドでエージェントを動かす別UI |
| `STT.py`, `LLM.py`, `Vision.py` | 各機能の単体テスト用スクリプト |

### `bayes_v3.py` の内部構造

#### 会話フェーズ（Phase）
`SETUP → INTRO → SURROUNDINGS → BRIDGE → DEEP_DIVE → ENDING` の順で進む。`SURROUNDINGS` と `BRIDGE` のみ画像を使用。BRIDGEは回想が引き出せなければ `SURROUNDINGS` に戻る。

#### ベイズ更新（`update_posterior`）
- `p_want_talk`（0〜1）でユーザーの「話したい度」を管理
- `ActionType`: SILENCE / NORMAL / DISCLOSURE の3クラス
- H1（話したい）とH0（話したくない）の尤度テーブルを使い正規化更新
- 生返事（`minimal_reply`）は専用の尤度で強めに下げる

#### 会話メモ（`conv_memory: MemoryUpdate`）
- `summary`: これまでの会話の要約（LLMで更新）
- `do_not_ask`: 繰り返し質問禁止リスト（最大8件）
- `stop_intent`: ユーザーの終了意思フラグ

#### ログ出力
実行ごとに `logs/run_YYYYMMDD_HHMMSS/` を作成：
- `log_*.txt` — 会話ログ（`[HH:MM:SS] AI:` / `[HH:MM:SS] User:` 形式）
- `analysis_*.csv` — ターンごとの分析データ（P_WantTalk, Phase, Turn 等）
- `agent_*.log` — Pythonロギング出力

#### `ui_display.py` との連携
`bayes_v3.py` は起動時に `ui_ready.flag` の存在を0.5秒ごとにポーリングして待機。`ui_display.py` は起動時にこのフラグを作成し、エージェントに起動を通知する。UIは `analysis_*.csv` と `log_*.txt` を2秒ごとに再読み込みしてリアルタイム表示する。
