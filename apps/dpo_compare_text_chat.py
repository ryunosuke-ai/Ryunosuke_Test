"""DPO学習前後のQwen3.5返答をターミナルで比較するCLI。"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import warnings
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

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
from apps.dpo_text_chat import (  # noqa: E402
    ChatBundle,
    DEFAULT_BASE_MODEL_ID,
    DEFAULT_LORA_PATH,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    append_history_line,
    create_run_dir,
    disable_peft_bitsandbytes_dispatch,
    load_tokenizer,
    load_training_modules,
    read_env_value,
)


load_dotenv()

DEFAULT_ENV_BASE_MODEL = "DPO_COMPARE_BASE_MODEL"
DEFAULT_ENV_LORA_PATH = "DPO_COMPARE_LORA_PATH"

ROLE_MARKER_PATTERN = re.compile(
    r"(?<!\w)(?:User|AI|assistant|system|ユーザー|アシスタント)\s*[:：]",
    re.IGNORECASE,
)
LEADING_ROLE_MARKER_PATTERN = re.compile(
    r"^\s*(?:User|AI|assistant|system|ユーザー|アシスタント)\s*[:：]\s*",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CompareResult:
    """比較生成の結果。"""

    base_reply: str
    dpo_reply: str


def read_compare_env_value(name: str, default: str) -> str:
    """比較CLI向けに環境変数から設定値を読む。"""
    return os.getenv(name, default).strip() or default


def create_run_dir_with_compare_logs() -> tuple[str, str]:
    """比較用のrunディレクトリと履歴ファイルを作る。"""
    return create_run_dir()


def setup_logger(run_dir: str, timestamp: str) -> logging.Logger:
    """ファイル専用のロガーを作る。"""
    logger = logging.getLogger("dpo_compare_text_chat")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S")
    file_handler = logging.FileHandler(f"{run_dir}/agent_{timestamp}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def suppress_external_warnings() -> None:
    """外部ライブラリの警告をできるだけ抑える。"""
    warnings.filterwarnings(
        "ignore",
        message=r"You are sending unauthenticated requests to the HF Hub.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"The fast path is not available because one of the required library is not installed.*",
    )
    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
    except Exception:
        pass

    for name in ("huggingface_hub", "transformers", "peft", "urllib3"):
        logging.getLogger(name).setLevel(logging.ERROR)


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""
    parser = argparse.ArgumentParser(description="DPO学習前後のQwen3.5をターミナルで比較表示します。")
    parser.add_argument(
        "--base-model-id",
        default=read_compare_env_value(DEFAULT_ENV_BASE_MODEL, DEFAULT_BASE_MODEL_ID),
        help=f"ベースモデルID（既定: {DEFAULT_BASE_MODEL_ID}）。",
    )
    parser.add_argument(
        "--lora-path",
        default=read_compare_env_value(DEFAULT_ENV_LORA_PATH, DEFAULT_LORA_PATH),
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


def build_dpo_compare_prompt(user_text: str) -> str:
    """学習データと同じ形式の比較用promptを作る。"""
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


def _strip_leading_role_markers(text: str) -> str:
    """先頭に混入したUser/AIラベルを落とす。"""
    return LEADING_ROLE_MARKER_PATTERN.sub("", text, count=1).strip()


def _truncate_at_role_marker(text: str) -> str:
    """途中に混入したUser/AIラベル以降を切り落とす。"""
    match = ROLE_MARKER_PATTERN.search(text)
    if match:
        return text[:match.start()].strip()
    return text.strip()


def cleanup_generated_text(decoded_text: str, prompt_text: str) -> str:
    """生成結果を表示用の返答本文へ整える。"""
    generated_text = strip_prompt_prefix(decoded_text, prompt_text)
    reply = extract_qwen_final_text(generated_text)
    if not reply:
        reply = extract_qwen_final_text(generated_text, show_thinking=True) or generated_text
    reply = _strip_leading_role_markers(reply)
    reply = _truncate_at_role_marker(reply)
    return reply.strip() or generated_text.strip()


@contextmanager
def adapter_disabled(model: object) -> Iterator[None]:
    """PEFT adapterを一時的に無効化する。"""
    disable_adapter = getattr(model, "disable_adapter", None)
    if disable_adapter is None:
        yield
        return
    with disable_adapter():
        yield


def write_session_header(
    history_file: str,
    *,
    base_model_id: str,
    lora_path: str,
    use_4bit: bool,
    args: argparse.Namespace,
) -> None:
    """履歴ファイルの先頭にセッション情報を残す。"""
    with open(history_file, "a", encoding="utf-8") as file:
        file.write(f"# session_start: {datetime.now().isoformat(timespec='seconds')}\n")
        file.write("# mode: compare\n")
        file.write(f"# base_model_id: {base_model_id}\n")
        file.write(f"# lora_path: {lora_path}\n")
        file.write(f"# use_4bit: {use_4bit}\n")
        file.write(f"# max_new_tokens: {args.max_new_tokens}\n")
        file.write(f"# temperature: {args.temperature}\n")
        file.write(f"# top_p: {args.top_p}\n")
        file.write(f"# repetition_penalty: {args.repetition_penalty}\n")
        file.write(f"# seed: {args.seed}\n")
        file.write("# outputs: base,dpo\n")
        file.write("\n")


def load_compare_bundle(base_model_id: str, lora_path: str, *, use_4bit: bool) -> ChatBundle:
    """ベースモデルにDPO LoRA adapterを載せて読み込む。"""
    suppress_external_warnings()
    deps = load_training_modules()
    disable_peft_bitsandbytes_dispatch()
    torch = deps["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA対応GPUが見つかりません。Qwen3.5-27B の比較にはGPU環境が必要です。")
    if not Path(lora_path).exists():
        raise RuntimeError(f"LoRA adapter が見つかりません: {lora_path}")

    tokenizer = load_tokenizer(base_model_id, deps)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model_kwargs: dict[str, object] = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except Exception as exc:
            raise RuntimeError(
                "4bit量子化に必要な bitsandbytes が見つかりません。"
                " `--use-4bit` を外すか、依存関係を見直してください。"
            ) from exc
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
    else:
        model_kwargs["torch_dtype"] = dtype

    base_model = deps["ModelClass"].from_pretrained(base_model_id, **model_kwargs)
    if hasattr(base_model, "config"):
        base_model.config.use_cache = True
    model = deps["PeftModel"].from_pretrained(base_model, lora_path)
    model.eval()
    return ChatBundle(tokenizer=tokenizer, model=model, torch=torch)


def generate_reply(
    bundle: ChatBundle,
    prompt_text: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    seed: int,
    use_adapter: bool,
) -> str:
    """ベースまたはDPO後モデルで返答を生成する。"""
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

    def _run_generate():
        with bundle.torch.no_grad():
            return bundle.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=bundle.tokenizer.eos_token_id,
                pad_token_id=bundle.tokenizer.eos_token_id,
            )

    if use_adapter:
        output_ids = _run_generate()
    else:
        with adapter_disabled(bundle.model):
            output_ids = _run_generate()

    generated = output_ids[0][input_ids.shape[1]:]
    decoded = bundle.tokenizer.decode(generated, skip_special_tokens=False).strip()
    return cleanup_generated_text(decoded, prompt_text)


def run_repl() -> int:
    """ターミナル対話ループを実行する。"""
    args = parse_args()
    try:
        run_dir, history_file = create_run_dir_with_compare_logs()
        timestamp = Path(history_file).stem.replace("log_", "")
        logger = setup_logger(run_dir, timestamp)
        write_session_header(
            history_file,
            base_model_id=args.base_model_id,
            lora_path=args.lora_path,
            use_4bit=args.use_4bit,
            args=args,
        )
        bundle = load_compare_bundle(args.base_model_id, args.lora_path, use_4bit=args.use_4bit)
    except RuntimeError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1

    logger.info("会話ログ: %s", history_file)
    logger.info("ベースモデル: %s", args.base_model_id)
    logger.info("LoRA adapter: %s", args.lora_path)
    logger.info("4bit: %s", "有効" if args.use_4bit else "無効")

    while True:
        try:
            user_text = input("User: ").strip()
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

        prompt_text = build_dpo_compare_prompt(user_text)
        try:
            base_reply = generate_reply(
                bundle,
                prompt_text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                seed=args.seed,
                use_adapter=False,
            )
            dpo_reply = generate_reply(
                bundle,
                prompt_text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                seed=args.seed,
                use_adapter=True,
            )
        except Exception as exc:
            print(f"AI(base): 生成に失敗しました: {exc}", file=sys.stderr)
            logger.warning("生成失敗: %s", exc)
            continue

        base_final = base_reply or "（空の返答）"
        dpo_final = dpo_reply or "（空の返答）"
        print(f"AI(base): {base_final}")
        print(f"AI(dpo): {dpo_final}")
        append_history_line(history_file, "AI(base)", base_final)
        append_history_line(history_file, "AI(dpo)", dpo_final)
        logger.info("User: %s", user_text)
        logger.info("AI(base): %s", base_final)
        logger.info("AI(dpo): %s", dpo_final)

    return 0


def main() -> int:
    """CLIエントリポイント。"""
    return run_repl()


if __name__ == "__main__":
    raise SystemExit(main())
