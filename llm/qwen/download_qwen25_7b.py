"""Qwen 2.5 7B Instruct をローカルに事前取得するスクリプト。"""

import argparse
import sys

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
except ModuleNotFoundError:
    print("エラー: `transformers` / `torch` が見つかりません。")
    print("依存関係をインストールしてから再実行してください。")
    print("例: python3 -m pip install transformers torch accelerate sentencepiece")
    sys.exit(1)

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen 2.5 7B をローカルにダウンロードします。")
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
        "--disable-4bit",
        action="store_true",
        help="4bit量子化を使わず、FP16/BF16で取得する",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        print("警告: CUDA対応GPUが見つかりません。")
        print("ダウンロード自体は続行しますが、4bit推論の実行時にはGPUが必要です。")

    print("モデル取得を開始します...")
    print(f"  model_id : {args.model_id}")
    if args.revision:
        print(f"  revision : {args.revision}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.revision,
        trust_remote_code=True,
    )
    loaded_with_4bit = False
    if BitsAndBytesConfig is not None and not args.disable_4bit:
        try:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            _ = AutoModelForCausalLM.from_pretrained(
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

    if not loaded_with_4bit:
        use_bf16 = torch.cuda.is_bf16_supported()
        load_dtype = torch.bfloat16 if use_bf16 else torch.float16
        _ = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            revision=args.revision,
            dtype=load_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        print(f"{'BF16' if use_bf16 else 'FP16'}で取得しました。")

    print("モデル取得が完了しました。")
    print("Hugging Face のローカルキャッシュへ保存されています。")
    print(f"tokenizer vocab size: {len(tokenizer)}")


if __name__ == "__main__":
    main()
