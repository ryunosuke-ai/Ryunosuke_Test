"""DPO学習前後のQwen3.5返答を比較するStreamlitアプリ。"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.local_llm_utils import extract_qwen_final_text  # noqa: E402


DEFAULT_BASE_MODEL_ID = "Qwen/Qwen3.5-27B"
DEFAULT_LORA_PATH = "artifacts/qwen35_dpo_lora"
DEFAULT_MAX_NEW_TOKENS = 160
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.8
DEFAULT_REPETITION_PENALTY = 1.0


@dataclass
class CompareModels:
    """比較UIで使うモデル一式。"""

    tokenizer: object
    model: object
    device: str


def read_env_value(name: str, default: str) -> str:
    """環境変数から設定値を読む。"""
    return os.getenv(name, default).strip() or default


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
    marker = "AI:"
    marker_index = decoded_text.rfind(marker)
    if marker_index >= 0:
        return decoded_text[marker_index + len(marker):].strip()
    return decoded_text.strip()


def cleanup_generated_text(decoded_text: str, prompt_text: str) -> str:
    """生成結果を表示用の返答本文へ整える。"""
    generated_text = strip_prompt_prefix(decoded_text, prompt_text)
    reply = extract_qwen_final_text(generated_text)
    if reply:
        return reply
    fallback = extract_qwen_final_text(generated_text, show_thinking=True)
    return fallback or generated_text


@contextmanager
def adapter_disabled(model: object) -> Iterator[None]:
    """PEFT adapterを一時的に無効化する。"""
    disable_adapter = getattr(model, "disable_adapter", None)
    if disable_adapter is None:
        yield
        return
    with disable_adapter():
        yield


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
            "比較UIの実行に必要な依存関係が不足しています。"
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


@st.cache_resource(show_spinner=False)
def load_compare_models(base_model_id: str, lora_path: str) -> CompareModels:
    """ベースモデルにDPO LoRA adapterを載せて読み込む。"""
    deps = load_training_modules()
    disable_peft_bitsandbytes_dispatch()
    torch = deps["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA対応GPUが見つかりません。Qwen3.5-27B の比較にはGPU環境が必要です。")
    if not Path(lora_path).exists():
        raise RuntimeError(f"LoRA adapter が見つかりません: {lora_path}")

    tokenizer = load_tokenizer(base_model_id, deps)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base_model = deps["ModelClass"].from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    if hasattr(base_model, "config"):
        base_model.config.use_cache = True
    model = deps["PeftModel"].from_pretrained(base_model, lora_path)
    model.eval()
    return CompareModels(tokenizer=tokenizer, model=model, device="cuda")


def generate_reply(
    compare_models: CompareModels,
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
    torch = load_training_modules()["torch"]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tokenizer = compare_models.tokenizer
    model_inputs = tokenizer(prompt_text, return_tensors="pt")
    model_inputs = {
        key: value.to(compare_models.device) if hasattr(value, "to") else value
        for key, value in model_inputs.items()
    }
    input_ids = model_inputs["input_ids"]
    if "attention_mask" not in model_inputs:
        model_inputs["attention_mask"] = torch.ones_like(input_ids, device=compare_models.device)

    def _run_generate():
        with torch.no_grad():
            return compare_models.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

    if use_adapter:
        output_ids = _run_generate()
    else:
        with adapter_disabled(compare_models.model):
            output_ids = _run_generate()

    generated = output_ids[0][input_ids.shape[1]:]
    decoded = tokenizer.decode(generated, skip_special_tokens=False).strip()
    return cleanup_generated_text(decoded, prompt_text)


def render_app() -> None:
    """Streamlit UIを描画する。"""
    st.set_page_config(page_title="DPO前後比較チャット", layout="wide")
    st.title("DPO前後比較チャット")

    with st.sidebar:
        st.header("モデル設定")
        base_model_id = st.text_input(
            "ベースモデル",
            value=read_env_value("DPO_COMPARE_BASE_MODEL", DEFAULT_BASE_MODEL_ID),
        )
        lora_path = st.text_input(
            "DPO LoRA adapter",
            value=read_env_value("DPO_COMPARE_LORA_PATH", DEFAULT_LORA_PATH),
        )
        st.header("生成設定")
        max_new_tokens = st.slider("max_new_tokens", 32, 512, DEFAULT_MAX_NEW_TOKENS, step=16)
        temperature = st.slider("temperature", 0.0, 1.5, DEFAULT_TEMPERATURE, step=0.05)
        top_p = st.slider("top_p", 0.1, 1.0, DEFAULT_TOP_P, step=0.05)
        repetition_penalty = st.slider(
            "repetition_penalty",
            0.8,
            1.5,
            DEFAULT_REPETITION_PENALTY,
            step=0.05,
        )
        seed = st.number_input("seed", value=42, min_value=0, max_value=2_147_483_647)

    user_text = st.text_area(
        "User入力",
        height=160,
        placeholder="例: 最近、昔やっていたギターの話を思い出して、また少し弾いてみたくなってきました。",
    )
    prompt_text = build_dpo_compare_prompt(user_text) if user_text.strip() else ""
    with st.expander("生成に使うprompt", expanded=False):
        st.code(prompt_text or "User入力後に表示されます。", language="text")

    if st.button("ベースモデルとDPO後モデルで生成", type="primary", disabled=not user_text.strip()):
        try:
            with st.spinner("モデルを読み込んでいます。初回は時間がかかります。"):
                compare_models = load_compare_models(base_model_id, lora_path)
            with st.spinner("返答を生成しています。"):
                base_reply = generate_reply(
                    compare_models,
                    prompt_text,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    seed=int(seed),
                    use_adapter=False,
                )
                dpo_reply = generate_reply(
                    compare_models,
                    prompt_text,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    seed=int(seed),
                    use_adapter=True,
                )
        except Exception as exc:
            st.error(f"生成に失敗しました: {exc}")
            return

        left, right = st.columns(2)
        with left:
            st.subheader("学習前: Qwen3.5")
            st.markdown(base_reply or "（空の返答）")
        with right:
            st.subheader("学習後: DPO LoRA")
            st.markdown(dpo_reply or "（空の返答）")


if __name__ == "__main__":
    render_app()
