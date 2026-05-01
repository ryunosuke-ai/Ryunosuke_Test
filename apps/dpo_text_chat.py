"""DPO学習後のQwen3.5だけをターミナルで対話表示するCLI。"""

from __future__ import annotations

import argparse
import os
import sys
import logging
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    print("エラー: `python-dotenv` パッケージが見つかりません。")
    print("依存関係をインストールしてから再実行してください。")
    print("例: python3 -m pip install python-dotenv")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.local_llm_utils import extract_qwen_final_text  # noqa: E402


load_dotenv()

DEFAULT_BASE_MODEL_ID = "Qwen/Qwen3.5-27B"
DEFAULT_LORA_PATH = "artifacts/qwen35_dpo_lora"
DEFAULT_MAX_NEW_TOKENS = 160
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.8
DEFAULT_REPETITION_PENALTY = 1.0


@dataclass(frozen=True)
class ChatBundle:
    """生成に必要なモデル一式。"""

    tokenizer: object
    model: object
    torch: object

    @property
    def input_device(self):
        """入力テンソルを載せる先のデバイス。"""
        try:
            return next(self.model.parameters()).device
        except Exception:
            return self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")


def read_env_value(name: str, default: str) -> str:
    """環境変数から設定値を読む。"""
    return os.getenv(name, default).strip() or default


def create_run_dir() -> tuple[str, str]:
    """会話ログ保存用のrunディレクトリと履歴ファイルを作る。"""
    os.makedirs("logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"logs/run_{ts}"
    os.makedirs(run_dir, exist_ok=True)
    history_file = f"{run_dir}/log_{ts}.txt"
    return run_dir, history_file


def append_history_line(history_file: str, role: str, text: str) -> None:
    """履歴ファイルへ1行追記する。"""
    safe_text = str(text).replace("\n", " ").strip()
    if not safe_text:
        return
    with open(history_file, "a", encoding="utf-8") as file:
        file.write(f"[{datetime.now().strftime('%H:%M:%S')}] {role}: {safe_text}\n")


def write_session_header(history_file: str, *, base_model_id: str, lora_path: str, use_4bit: bool, args: argparse.Namespace) -> None:
    """履歴ファイルの先頭にセッション情報を残す。"""
    with open(history_file, "a", encoding="utf-8") as file:
        file.write(f"# session_start: {datetime.now().isoformat(timespec='seconds')}\n")
        file.write(f"# base_model_id: {base_model_id}\n")
        file.write(f"# lora_path: {lora_path}\n")
        file.write(f"# use_4bit: {use_4bit}\n")
        file.write(f"# max_new_tokens: {args.max_new_tokens}\n")
        file.write(f"# temperature: {args.temperature}\n")
        file.write(f"# top_p: {args.top_p}\n")
        file.write(f"# repetition_penalty: {args.repetition_penalty}\n")
        file.write(f"# seed: {args.seed}\n")
        file.write("\n")


def setup_logger(run_dir: str, timestamp: str) -> logging.Logger:
    """会話ログをファイルへ残すロガーを作る。"""
    logger = logging.getLogger("dpo_text_chat")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S")

    file_handler = logging.FileHandler(f"{run_dir}/agent_{timestamp}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""
    parser = argparse.ArgumentParser(description="DPO学習後のQwen3.5をターミナルで対話表示します。")
    parser.add_argument(
        "--base-model-id",
        default=read_env_value("DPO_TEXT_CHAT_BASE_MODEL", DEFAULT_BASE_MODEL_ID),
        help=f"ベースモデルID（既定: {DEFAULT_BASE_MODEL_ID}）。",
    )
    parser.add_argument(
        "--lora-path",
        default=read_env_value("DPO_TEXT_CHAT_LORA_PATH", DEFAULT_LORA_PATH),
        help=f"LoRA adapterの保存先（既定: {DEFAULT_LORA_PATH}）。",
    )
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="最大生成トークン数。")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="生成temperature。")
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P, help="生成top_p。")
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=DEFAULT_REPETITION_PENALTY,
        help="repetition penalty。",
    )
    parser.add_argument("--seed", type=int, default=42, help="乱数シード。")
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="4bit量子化で読み込みます。環境が整っていない場合は使わないでください。",
    )
    return parser.parse_args()


def build_dpo_prompt(user_text: str) -> str:
    """学習データと同じ形式のpromptを作る。"""
    return (
        "以下の会話の次のAI返答を生成してください。\n\n"
        "これまでの会話:\n"
        f"User: {user_text.strip()}\n\n"
        "AI:"
    )


def strip_prompt_prefix(decoded_text: str, prompt_text: str) -> str:
    """decode結果にpromptが含まれる場合、生成部分だけを取り出す。"""
    if decoded_text.startswith(prompt_text):
        return decoded_text[len(prompt_text):].strip()
    marker_index = decoded_text.rfind("AI:")
    if marker_index >= 0:
        return decoded_text[marker_index + len("AI:"):].strip()
    return decoded_text.strip()


def cleanup_generated_text(decoded_text: str, prompt_text: str) -> str:
    """生成結果を表示用の返答本文へ整える。"""
    generated_text = strip_prompt_prefix(decoded_text, prompt_text)
    reply = extract_qwen_final_text(generated_text)
    if reply:
        return reply
    fallback = extract_qwen_final_text(generated_text, show_thinking=True)
    return fallback or generated_text


def load_training_modules() -> dict[str, object]:
    """重い依存を遅延読み込みする。"""
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
        try:
            from transformers import Qwen3_5ForConditionalGeneration as ModelClass
        except ImportError:
            ModelClass = AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "DPOチャットの実行に必要な依存関係が不足しています。"
            "`python3 -m pip install -r requirements.txt` を実行してください。"
        ) from exc
    return {
        "torch": torch,
        "PeftModel": PeftModel,
        "AutoProcessor": AutoProcessor,
        "AutoTokenizer": AutoTokenizer,
        "ModelClass": ModelClass,
    }


def disable_peft_bitsandbytes_dispatch() -> None:
    """PEFTのLoRA挿入時にbitsandbytes backendを使わせない。"""
    try:
        import peft.import_utils as peft_import_utils
        import peft.tuners.lora.model as peft_lora_model
    except ImportError:
        return

    def _always_false() -> bool:
        return False

    for module in (peft_import_utils, peft_lora_model):
        for name in ("is_bnb_available", "is_bnb_4bit_available"):
            detector = getattr(module, name, None)
            if hasattr(detector, "cache_clear"):
                detector.cache_clear()
            setattr(module, name, _always_false)


def load_tokenizer(model_id: str, deps: dict[str, object]):
    """Qwen tokenizerを読み込む。"""
    try:
        processor = deps["AutoProcessor"].from_pretrained(model_id, trust_remote_code=True)
        tokenizer = getattr(processor, "tokenizer", None) or processor
    except Exception:
        tokenizer = deps["AutoTokenizer"].from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_chat_bundle(base_model_id: str, lora_path: str, *, no_4bit: bool) -> ChatBundle:
    """ベースモデルにDPO LoRA adapterを載せて読み込む。"""
    deps = load_training_modules()
    disable_peft_bitsandbytes_dispatch()
    torch = deps["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA対応GPUが見つかりません。Qwen3.5-27B の実行にはGPU環境が必要です。")
    if not Path(lora_path).exists():
        raise RuntimeError(f"LoRA adapter が見つかりません: {lora_path}")

    tokenizer = load_tokenizer(base_model_id, deps)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model_kwargs: dict[str, object] = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if no_4bit:
        model_kwargs["torch_dtype"] = dtype
    else:
        try:
            from transformers import BitsAndBytesConfig
        except Exception as exc:
            raise RuntimeError(
                "4bit量子化に必要な bitsandbytes が見つかりません。"
                " `--no-4bit` を付けるか、依存関係を見直してください。"
            ) from exc
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )

    base_model = deps["ModelClass"].from_pretrained(base_model_id, **model_kwargs)
    if hasattr(base_model, "config"):
        base_model.config.use_cache = True
    model = deps["PeftModel"].from_pretrained(base_model, lora_path)
    model.eval()
    return ChatBundle(tokenizer=tokenizer, model=model, torch=torch)


def generate_reply(
    bundle: ChatBundle,
    user_text: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    seed: int,
) -> str:
    """DPO学習後モデルで返答を生成する。"""
    prompt_text = build_dpo_prompt(user_text)
    bundle.torch.manual_seed(seed)
    if bundle.torch.cuda.is_available():
        bundle.torch.cuda.manual_seed_all(seed)

    model_inputs = bundle.tokenizer(prompt_text, return_tensors="pt")
    model_inputs = {
        key: value.to(bundle.input_device) if hasattr(value, "to") else value
        for key, value in model_inputs.items()
    }
    input_ids = model_inputs["input_ids"]
    if "attention_mask" not in model_inputs:
        model_inputs["attention_mask"] = bundle.torch.ones_like(input_ids, device=bundle.input_device)

    with bundle.torch.no_grad():
        output_ids = bundle.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=bundle.tokenizer.eos_token_id,
            pad_token_id=bundle.tokenizer.eos_token_id,
        )

    generated = output_ids[0][input_ids.shape[1]:]
    decoded = bundle.tokenizer.decode(generated, skip_special_tokens=False).strip()
    return cleanup_generated_text(decoded, prompt_text)


def print_banner() -> None:
    """起動時の案内を表示する。"""
    line = "=" * 56
    print(f"\n{line}")
    print("DPO後モデルのターミナルチャット")
    print(line)
    print("  exit / quit / :q で終了")
    print("  返答は学習後モデルのみ表示します")


def run_repl() -> int:
    """ターミナル対話ループを実行する。"""
    args = parse_args()
    try:
        run_dir, history_file = create_run_dir()
        timestamp = Path(history_file).stem.replace("log_", "")
        logger = setup_logger(run_dir, timestamp)
        write_session_header(
            history_file,
            base_model_id=args.base_model_id,
            lora_path=args.lora_path,
            use_4bit=args.use_4bit,
            args=args,
        )
        bundle = load_chat_bundle(args.base_model_id, args.lora_path, no_4bit=not args.use_4bit)
    except RuntimeError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1

    print_banner()
    print(f"  ベースモデル: {args.base_model_id}")
    print(f"  LoRA adapter : {args.lora_path}")
    print(f"  4bit         : {'有効' if args.use_4bit else '無効'}")
    print(f"  ログ出力先   : {run_dir}/")
    print()
    logger.info("会話ログ: %s", history_file)
    logger.info("ベースモデル: %s", args.base_model_id)
    logger.info("LoRA adapter: %s", args.lora_path)
    logger.info("4bit: %s", "有効" if args.use_4bit else "無効")

    while True:
        try:
            user_text = input("あなた: ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            break

        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit", ":q"}:
            break

        append_history_line(history_file, "User", user_text)
        logger.info("User: %s", user_text)

        try:
            reply = generate_reply(
                bundle,
                user_text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                seed=args.seed,
            )
        except Exception as exc:
            print(f"AI: 生成に失敗しました: {exc}")
            logger.warning("生成失敗: %s", exc)
            continue

        final_reply = reply or "（空の返答）"
        print(f"AI: {final_reply}")
        append_history_line(history_file, "AI", final_reply)
        logger.info("AI: %s", final_reply)

    print(f"\n会話ログを保存しました: {run_dir}/")
    return 0


def main() -> int:
    """CLIエントリポイント。"""
    return run_repl()


if __name__ == "__main__":
    raise SystemExit(main())
