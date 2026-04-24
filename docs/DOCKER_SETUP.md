# Docker 運用ガイド

## 概要

このプロジェクトは、Docker 上で `apps/text_chat.py` と `apps/ui_display.py` を安定して動かせるように構成する。  
初期段階では、音声 I/O を含む `apps/bayes_v3.py` は Docker 対象外とし、テキスト会話と UI の検証を優先する。

将来的なローカル LLM の追加学習や強化学習を見据え、モデル資産はホスト側ディレクトリとして次の 3 系統に分離して保持する。

- `models/base`: 学習前のベースモデル
- `hf_cache`: Hugging Face のキャッシュ
- `artifacts`: 学習結果、LoRA、評価結果

この分離により、実験が失敗した場合でもベースモデルを汚さずに再学習できる。

## 前提条件

- Docker Engine と Docker Compose Plugin が導入済みであること
- GPU を使う場合は、ホスト側に NVIDIA ドライバが導入済みであること
- GPU を Docker から使う場合は `nvidia-container-toolkit` が設定済みであること
- プロジェクトルートに `.env` と `experiment_image.jpg` があること

## ビルド

```bash
docker compose build
```

## 起動方法

### UI を起動する

```bash
docker compose up ui
```

ブラウザから `http://localhost:8501` を開く。

### テキスト会話を起動する

別ターミナルで次を実行する。

```bash
docker compose run --rm app python3 -m apps.text_chat
```

### テストを実行する

```bash
docker compose run --rm app python3 -m pytest tests/test_label4_system.py -v
docker compose run --rm app python3 -m pytest tests/test_local_llm_utils.py -v
```

## GPU 確認

コンテナ内で GPU を確認する場合は、次を実行する。

```bash
docker compose run --rm app python3 -c "import torch; print(torch.cuda.is_available())"
docker compose run --rm app nvidia-smi
```

`torch.cuda.is_available()` が `True` にならない場合は、ホスト側の NVIDIA ドライバと `nvidia-container-toolkit` を見直す。

## モデル資産の運用方針

- 学習前のベースモデルは `models/base` に保存し、上書きしない
- 実験結果や学習済み重みは `artifacts` に保存する
- Hugging Face 由来のキャッシュは `hf_cache` に集約する
- 再学習したい場合は `artifacts` 側だけを切り替え、`models/base` はそのまま使う

この方針にすると、試行錯誤が増えても「どのモデルを起点に学習したか」を保ちやすい。

## 今回の対象外

- `apps/bayes_v3.py` のマイク・スピーカー連携
- 学習専用コンテナの追加
- 分散学習や複数 GPU 学習のジョブ設計

将来、学習ジョブを追加する場合は、`docker-compose.yml` に `train` サービスを追加し、`models/base` と `artifacts` を共有する構成を推奨する。
