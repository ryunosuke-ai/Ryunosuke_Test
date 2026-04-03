# Repository Guidelines

## このシステムについて
このリポジトリは、高齢者との自然な対話を目指すマルチモーダル会話エージェントです。ユーザー発話を `SILENCE`、`NORMAL`、`DISCLOSURE` に分類し、ベイズ推定で `p_want_talk` を更新しながら、会話の負荷と深さを調整します。メイン実装は `apps/bayes_v3.py`、テキスト検証版は `apps/text_chat.py`、監視 UI は `apps/ui_display.py` です。

## まず把握したい全体像
会話は 1 ターンごとに、入力取得、発話分類、確率更新、会話メモ更新、フェーズ遷移、応答生成の順で進みます。UI は会話ロジックに直接依存せず、実行中に出力されるログと CSV を読み取って表示します。

```text
入力取得 → classify_action() → update_posterior()
        → judge_memory_and_disclosure() → update_conv_memory()
        → transition_policy() → think_and_reply()
```

## 起動方法
通常は 2 つのターミナルを使います。先に UI を起動し、その後にエージェント本体を起動してください。

```bash
# ターミナル1: UI
python3 -m streamlit run apps/ui_display.py

# ターミナル2: エージェント本体
python3 -m apps.bayes_v3
```

`apps/ui_display.py` は起動時に `ui_ready.flag` を作成し、`apps/bayes_v3.py` はそのフラグを確認してから開始します。音声 I/O を使わずに会話ロジックだけ確認したい場合は、`python3 -m apps.text_chat` を使います。

## 使い方
通常運用では、マイク入力と音声出力を使って会話します。会話中はユーザー発話の種類と `p_want_talk` に応じて、質問の強さや話題の深さが変わります。テキスト版では、コンソール上で会話しながらフェーズ、ターン数、観測タイプ、`p_want_talk` の推移を確認できます。

実行ログは `logs/run_YYYYMMDD_HHMMSS/` に保存されます。
- `log_*.txt`: 会話ログ
- `analysis_*.csv`: 分析用データ
- `agent_*.log`: エージェント内部ログ

## 会話フェーズ
会話は以下のフェーズを段階的に進みます。

```text
SETUP → INTRO → SURROUNDINGS → BRIDGE → DEEP_DIVE → ENDING
                 ↑               |           |
                 └───────────────+───────────┘
```

主な遷移条件:
- `BRIDGE → DEEP_DIVE`: 回想または自己開示が出たとき
- `BRIDGE → SURROUNDINGS`: 低反応や沈黙が続いたとき
- `DEEP_DIVE → ENDING`: 深掘り継続が不自然になったとき
- `全フェーズ → ENDING`: 連続沈黙時の安全終了

## ファイル構成
主要ファイルは用途別ディレクトリに整理されています。

- `apps/bayes_v3.py`: 音声 I/O を含むメインエージェント
- `apps/text_chat.py`: テキスト入力ベースの検証用エージェント
- `apps/ui_display.py`: Streamlit によるリアルタイム表示 UI
- `apps/simple_text_chat.py`: Azure OpenAI で返信生成する簡易テキスト会話
- `core/bayes_engine.py`: 発話分類とベイズ更新
- `core/phase_manager.py`: フェーズ遷移と応答方針の切り替え
- `core/conv_memory.py`: 会話要約、繰り返し防止、終了意思判定
- `core/models.py`: 共通データ型定義
- `llm/gpt_oss/simple_text_chat_gpt_oss.py`: gpt-oss 版簡易テキスト会話
- `llm/qwen/simple_text_chat_qwen.py`: Qwen 版簡易テキスト会話
- `docs/SYSTEM_OVERVIEW.md`: システム仕様の詳細
- `CLAUDE.md`: 作業ルールと補足資料

依存関係は次のとおりです。

```text
core/models.py  ← 共通データ定義
  ↑  ↑  ↑
  |  |  └── core/conv_memory.py
  |  └───── core/phase_manager.py
  └──────── core/bayes_engine.py
                     ↑
apps/bayes_v3.py ----+--→ core/phase_manager.py
apps/text_chat.py ---+    core/conv_memory.py

apps/ui_display.py  ← 独立。logs 配下のみを読む
```

## セットアップ
依存関係の導入:

```bash
python3 -m venv test_env
source test_env/bin/activate
python3 -m pip install -r requirements.txt
```

`.env` には以下を設定します。

```env
AZURE_SPEECH_KEY=
AZURE_SPEECH_REGION=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_DEPLOYMENT_NAME=
AZURE_OPENAI_API_VERSION=
```

また、プロジェクトルートに `experiment_image.jpg` を配置してください。この画像は `SURROUNDINGS` と `BRIDGE` フェーズで話題生成に使われます。

## テスト
既存ドキュメントでは以下のコマンドを前提にしています。

```bash
python3 -m pytest tests/test_label4_system.py tests/test_local_llm_utils.py -v
python3 -m pytest tests/test_label4_system.py -v
python3 -m pytest tests/test_local_llm_utils.py -v
```

テストでは Azure Speech SDK、`cv2`、OpenAI API をモック化し、ネットワーク不要で実行できる状態を維持してください。

## コーディング方針
Python は PEP 8 準拠、インデントは 4 スペースです。関数、変数、ファイルは `snake_case`、クラスは `PascalCase`、定数は `UPPER_SNAKE_CASE` を使います。コメント、エラーメッセージ説明、ドキュメントは日本語で統一してください。新しいロジックは `apps/bayes_v3.py` に集中させすぎず、判定処理を各専門モジュールへ分ける構成を維持します。

## 言語設定
- 常に日本語で会話する
- コメントも日本語で記述する
- エラーメッセージの説明も日本語で行う
- ドキュメントも日本語で生成する

## 作業ルール
- 作業開始前に git が初期化されていない場合は `git init` を実行する
- 作業ごとに必ず `git add -A && git commit` を行い、その後 `git push origin master` を実行する
- コミットメッセージは以下の形式に従う
  - `feat: ○○機能を追加`
  - `fix: ○○のバグを修正`
  - `refactor: ○○をリファクタリング`
  - `docs: ○○のドキュメントを更新`
  - `chore: ○○`
- リモートは `https://github.com/ryunosuke-ai/Ryunosuke_Test.git` を使用する

## 注意事項
`.env`、API キー、個人情報を含むログはコミット禁止です。`experiment_image.jpg` を差し替える場合は、会話内容への影響を明記してください。表示変更がある場合は、PR にスクリーンショットを添付してください。
