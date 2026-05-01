"""Qwen3.5 を LoRA/QLoRA で DPO 学習するスクリプト。"""

from __future__ import annotations

import argparse
import inspect
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_DATASET_PATH = "artifacts/noxij_dpo_preferences_ai_user.jsonl"
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-27B"
DEFAULT_OUTPUT_DIR = "artifacts/qwen35_dpo_lora"
TRUE_VALUES = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class PreferenceDatasetSplit:
    """DPO 学習用データの train/eval 分割。"""

    train: list[dict[str, str]]
    eval: list[dict[str, str]]


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""
    parser = argparse.ArgumentParser(description="Qwen3.5-27B を NoXi+J DPO データで LoRA 学習します。")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_PATH, help=f"DPO JSONL（既定: {DEFAULT_DATASET_PATH}）。")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help=f"ベースモデルIDまたはパス（既定: {DEFAULT_MODEL_ID}）。")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help=f"LoRA出力先（既定: {DEFAULT_OUTPUT_DIR}）。")
    parser.add_argument("--num-train-epochs", type=float, default=3.0, help="学習エポック数。")
    parser.add_argument("--learning-rate", type=float, default=5e-6, help="学習率。")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta。")
    parser.add_argument("--max-length", type=int, default=1024, help="prompt + 応答の最大トークン長。")
    parser.add_argument("--max-prompt-length", type=int, default=768, help="prompt の最大トークン長。")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1, help="デバイスごとの学習バッチサイズ。")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8, help="勾配蓄積ステップ数。")
    parser.add_argument("--eval-ratio", type=float, default=0.1, help="評価用に分割する割合。0なら評価なし。")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード。")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank。")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha。")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout。")
    parser.add_argument("--no-4bit", action="store_true", help="4bit量子化を無効化し、FP16/BF16で読み込みます。")
    parser.add_argument("--dry-run", action="store_true", help="学習せず、データと設定だけ確認します。")
    parser.add_argument("--logging-steps", type=int, default=1, help="ログ出力間隔。")
    parser.add_argument("--save-steps", type=int, default=25, help="チェックポイント保存間隔。")
    parser.add_argument("--max-grad-norm", type=float, default=0.3, help="勾配クリッピング。")
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="ウォームアップ割合。")
    return parser.parse_args()


def _require_nonempty_string(record: dict[str, Any], key: str, *, line_number: int) -> str:
    """必須文字列列を検証して返す。"""
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{line_number}行目の `{key}` が空、または文字列ではありません。")
    return value


def read_preference_records(path: Path) -> list[dict[str, str]]:
    """DPO JSONL から prompt/chosen/rejected を読み込む。"""
    records: list[dict[str, str]] = []
    try:
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{line_number}行目をJSONとして読めません: {exc}") from exc
                records.append(
                    {
                        "prompt": _require_nonempty_string(payload, "prompt", line_number=line_number),
                        "chosen": _require_nonempty_string(payload, "chosen", line_number=line_number),
                        "rejected": _require_nonempty_string(payload, "rejected", line_number=line_number),
                    }
                )
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"DPO JSONL が見つかりません: {path}") from exc
    if not records:
        raise ValueError("DPO JSONL に有効なレコードがありません。")
    return records


def split_records(records: list[dict[str, str]], *, eval_ratio: float, seed: int) -> PreferenceDatasetSplit:
    """DPOレコードを train/eval に分割する。"""
    if not 0.0 <= eval_ratio < 1.0:
        raise ValueError("`eval_ratio` は 0.0 以上 1.0 未満を指定してください。")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    if eval_ratio == 0.0 or len(shuffled) < 2:
        return PreferenceDatasetSplit(train=shuffled, eval=[])
    eval_size = max(1, int(round(len(shuffled) * eval_ratio)))
    eval_size = min(eval_size, len(shuffled) - 1)
    return PreferenceDatasetSplit(train=shuffled[eval_size:], eval=shuffled[:eval_size])


def summarize_records(records: list[dict[str, str]]) -> dict[str, int]:
    """dry-run 表示用の簡易統計を返す。"""
    return {
        "count": len(records),
        "max_prompt_chars": max(len(record["prompt"]) for record in records),
        "max_chosen_chars": max(len(record["chosen"]) for record in records),
        "max_rejected_chars": max(len(record["rejected"]) for record in records),
    }


def print_dry_run_summary(args: argparse.Namespace, split: PreferenceDatasetSplit) -> None:
    """dry-run 用にデータ概要を表示する。"""
    all_records = [*split.train, *split.eval]
    stats = summarize_records(all_records)
    print("DPO LoRA dry-run")
    print(f"  dataset: {args.dataset}")
    print(f"  model_id: {args.model_id}")
    print(f"  output_dir: {args.output_dir}")
    print(f"  records: {stats['count']} 件")
    print(f"  train/eval: {len(split.train)} / {len(split.eval)}")
    print(f"  max_prompt_chars: {stats['max_prompt_chars']}")
    print(f"  max_chosen_chars: {stats['max_chosen_chars']}")
    print(f"  max_rejected_chars: {stats['max_rejected_chars']}")
    print(f"  4bit: {'無効' if args.no_4bit else '有効'}")
    sample = split.train[0]
    print("  sample:")
    print(f"    prompt: {sample['prompt'][:120]!r}")
    print(f"    chosen: {sample['chosen'][:80]!r}")
    print(f"    rejected: {sample['rejected'][:80]!r}")


def load_training_dependencies() -> dict[str, Any]:
    """DPO学習に必要な依存関係を遅延読み込みする。"""
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
        from transformers import AutoProcessor, AutoTokenizer, BitsAndBytesConfig
        try:
            from transformers import Qwen3_5ForConditionalGeneration as ModelClass
        except ImportError:
            from transformers import AutoModelForCausalLM as ModelClass
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        raise RuntimeError(
            "DPO学習に必要な依存関係が不足しています。"
            "`python3 -m pip install -r requirements.txt` を実行してください。"
        ) from exc
    return {
        "torch": torch,
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "TaskType": TaskType,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoProcessor": AutoProcessor,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "ModelClass": ModelClass,
        "DPOConfig": DPOConfig,
        "DPOTrainer": DPOTrainer,
    }


def load_tokenizer(model_id: str, deps: dict[str, Any]):
    """Qwen3.5 の tokenizer を読み込む。"""
    try:
        processor = deps["AutoProcessor"].from_pretrained(model_id, trust_remote_code=True)
        tokenizer = getattr(processor, "tokenizer", None) or processor
    except Exception:
        tokenizer = deps["AutoTokenizer"].from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(args: argparse.Namespace, deps: dict[str, Any]):
    """DPO学習用のQwen3.5モデルを読み込む。"""
    torch = deps["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA対応GPUが見つかりません。Qwen3.5-27B のDPO学習にはGPU環境が必要です。")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if args.no_4bit:
        model_kwargs["torch_dtype"] = dtype
    else:
        model_kwargs["quantization_config"] = deps["BitsAndBytesConfig"](
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
    model = deps["ModelClass"].from_pretrained(args.model_id, **model_kwargs)
    if hasattr(model, "config"):
        model.config.use_cache = False
    if not args.no_4bit:
        model = deps["prepare_model_for_kbit_training"](model)
    return model, dtype


def build_lora_config(args: argparse.Namespace, deps: dict[str, Any]):
    """LoRA設定を作成する。"""
    return deps["LoraConfig"](
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=deps["TaskType"].CAUSAL_LM,
        target_modules="all-linear",
    )


def build_training_args(args: argparse.Namespace, deps: dict[str, Any], *, dtype: Any, has_eval: bool):
    """TRL DPOConfig を作成する。"""
    torch = deps["torch"]
    use_bf16 = dtype == torch.bfloat16
    kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "beta": args.beta,
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": True,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_strategy": "steps",
        "eval_strategy": "steps" if has_eval else "no",
        "eval_steps": args.save_steps if has_eval else None,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "lr_scheduler_type": "cosine",
        "optim": "paged_adamw_8bit" if not args.no_4bit else "adamw_torch",
        "bf16": use_bf16,
        "fp16": not use_bf16,
        "report_to": [],
        "remove_unused_columns": False,
        "seed": args.seed,
    }
    parameters = inspect.signature(deps["DPOConfig"].__init__).parameters
    if "eval_strategy" in kwargs and "eval_strategy" not in parameters and "evaluation_strategy" in parameters:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    if not any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        kwargs = {key: value for key, value in kwargs.items() if key in parameters}
    return deps["DPOConfig"](**kwargs)


def train(args: argparse.Namespace, split: PreferenceDatasetSplit) -> None:
    """DPO LoRA学習を実行する。"""
    deps = load_training_dependencies()
    tokenizer = load_tokenizer(args.model_id, deps)
    model, dtype = load_model(args, deps)
    lora_config = build_lora_config(args, deps)
    train_dataset = deps["Dataset"].from_list(split.train)
    eval_dataset = deps["Dataset"].from_list(split.eval) if split.eval else None
    training_args = build_training_args(args, deps, dtype=dtype, has_eval=eval_dataset is not None)

    trainer_kwargs = {
        "model": model,
        "ref_model": None,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "peft_config": lora_config,
    }
    try:
        trainer = deps["DPOTrainer"](**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        trainer = deps["DPOTrainer"](**trainer_kwargs, tokenizer=tokenizer)

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA adapter を保存しました: {args.output_dir}")


def main() -> int:
    """CLIエントリポイント。"""
    args = parse_args()
    try:
        records = read_preference_records(Path(args.dataset))
        split = split_records(records, eval_ratio=args.eval_ratio, seed=args.seed)
    except (FileNotFoundError, ValueError) as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print_dry_run_summary(args, split)
        return 0

    try:
        train(args, split)
    except RuntimeError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
