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
streamlit run ui_display.py

# ターミナル2: エージェントを起動（UIの準備完了を待機してから開始）
python bayes_v3.py
```

`bayes_v3.py` は `ui_ready.flag` ファイルが作成されるまで待機する。このフラグは `ui_display.py` が起動時に作成する。

## テスト

```bash
# 全テスト実行（60件）
python -m pytest test_bayes_v3.py test_ui_display.py -v

# 単一テストクラス
python -m pytest test_bayes_v3.py::TestUpdatePosterior -v

# 単一テストメソッド
python -m pytest test_bayes_v3.py::TestUpdatePosterior::test_disclosure_raises_probability -v
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

プロジェクトルートに `experiment_image.jpg` を配置すること。エージェントはこの静止画を会話に使用する。

---

## コードアーキテクチャ

### 主要ファイルの役割

| ファイル | 役割 |
|---|---|
| `bayes_v3.py` | **エントリーポイント兼オーケストレータ**。MultimodalAgent: init, speak, listen, ログ, think_and_reply, run |
| `models.py` | データ構造定義（ActionType, Phase, PhaseConfig, Observation, MemoryUpdate） |
| `bayes_engine.py` | ベイズ更新 + 観測分類（update_posterior, is_minimal_reply, classify_action, judge_memory_and_disclosure） |
| `phase_manager.py` | フェーズ遷移ポリシー + 設定テーブル + インタラクションモード指示 |
| `conv_memory.py` | 会話メモ更新 + 終了意思検出 |
| `ui_display.py` | **メインUI**。Streamlitでログをリアルタイム表示 |
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
                └--→ conv_memory.py
```

### 会話フローの概要

```
listen() → classify_action() → update_posterior() → think_and_reply() → speak()
               ↓                      ↓
        ActionType判定           p_want_talk更新
        (SILENCE/NORMAL/          ↓
         DISCLOSURE)         phase_manager が
                             フェーズ遷移を判定
```

- フェーズ: `SETUP → INTRO → SURROUNDINGS → BRIDGE → DEEP_DIVE → ENDING`
- ログ出力: 実行ごとに `logs/run_YYYYMMDD_HHMMSS/` へ会話ログ・分析CSV・agentログを保存
- UI連携: `ui_ready.flag` で起動同期、UIは CSV/ログを2秒ごとにポーリング表示
