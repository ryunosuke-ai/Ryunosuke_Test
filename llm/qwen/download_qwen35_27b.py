"""Qwen3.5-27B をローカルに事前取得するスクリプト。"""

import argparse
import sys

try:
    import torch
    from transformers import AutoProcessor, AutoTokenizer, Qwen3_5ForConditionalGeneration
except ImportError:
    print("エラー: `transformers` / `torch` が見つかりません。")
    print("Qwen3.5 に対応した依存関係をインストールしてから再実行してください。")
    print("例: python3 -m pip install -r requirements.txt")
    sys.exit(1)

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None


DEFAULT_MODEL_ID = "Qwen/Qwen3.5-27B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3.5-27B をローカルにダウンロードします。")
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"取得するモデルID（既定: {DEFAULT_MODEL_ID}）",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="取得するリビジョン（任意）",
    )
    parser.add_argument(
        "--enable-4bit",
        action="store_true",
        help="4bit量子化を有効化する（既定は無効）",
    )
    return parser.parse_args()


def _load_tokenizer(args: argparse.Namespace):
    """トークナイザを取得する。"""
    try:
        processor = AutoProcessor.from_pretrained(
            args.model_id,
            revision=args.revision,
            trust_remote_code=True,
        )
        return getattr(processor, "tokenizer", None) or processor
    except Exception as processor_error:
        print(f"警告: AutoProcessor の読み込みに失敗したため AutoTokenizer に切り替えます: {processor_error}")
        return AutoTokenizer.from_pretrained(
            args.model_id,
            revision=args.revision,
            trust_remote_code=True,
        )


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        print("警告: CUDA対応GPUが見つかりません。")
        print("ダウンロード自体は続行しますが、ローカル推論の実行時にはGPUが必要です。")

    print("モデル取得を開始します...")
    print(f"  model_id : {args.model_id}")
    if args.revision:
        print(f"  revision : {args.revision}")

    tokenizer = _load_tokenizer(args)
    loaded_with_4bit = False

    if BitsAndBytesConfig is not None and args.enable_4bit:
        try:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            _ = Qwen3_5ForConditionalGeneration.from_pretrained(
                args.model_id,
                revision=args.revision,
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True,
            )
            loaded_with_4bit = True
            print("4bit量子化で取得しました。")
        except Exception as e:
            print(f"警告: 4bit量子化での取得に失敗しました。FP16/BF16へ切り替えます: {e}")
    elif args.enable_4bit and BitsAndBytesConfig is None:
        print("警告: BitsAndBytesConfig が利用できないため、4bit量子化を使わずFP16/BF16で取得します。")

    if not loaded_with_4bit:
        use_bf16 = torch.cuda.is_bf16_supported()
        load_dtype = torch.bfloat16 if use_bf16 else torch.float16
        _ = Qwen3_5ForConditionalGeneration.from_pretrained(
            args.model_id,
            revision=args.revision,
            torch_dtype=load_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        print(f"{'BF16' if use_bf16 else 'FP16'}で取得しました。")

    print("モデル取得が完了しました。")
    print("Hugging Face のローカルキャッシュへ保存されています。")
    print(f"tokenizer vocab size: {len(tokenizer)}")


if __name__ == "__main__":
    main()
