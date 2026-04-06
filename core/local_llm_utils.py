"""ローカルLLMの生成結果を整形する補助関数。"""

import re
from typing import Any


GPT_OSS_FINAL_PREFIX = "<|channel|>final<|message|>"
GPT_OSS_FINAL_MARKER = "<|start|>assistant<|channel|>final<|message|>"
GPT_OSS_END_MARKERS = ("<|return|>", "<|end|>")
GPT_OSS_INLINE_END_MARKERS = ("<|return|>", "<|start|>", "<|end|>")
QWEN_THINK_BLOCK_PATTERN = re.compile(r"<think>\s*.*?\s*</think>\s*", re.DOTALL)


def _normalize_reply_text(text: str) -> str:
    """表示用に余分な空白を整える。"""
    return re.sub(r"\s+", " ", text).strip()


def extract_gpt_oss_final_text(raw_text: str) -> str:
    """gpt-oss の生出力から final チャンネル本文だけを取り出す。"""
    if not raw_text:
        return ""

    start = raw_text.rfind(GPT_OSS_FINAL_MARKER)
    if start == -1:
        return ""

    reply = raw_text[start + len(GPT_OSS_FINAL_MARKER):]
    end_positions = [reply.find(marker) for marker in GPT_OSS_END_MARKERS if marker in reply]
    if end_positions:
        reply = reply[:min(end_positions)]
    return _normalize_reply_text(reply)


def _cleanup_fallback_text(text: str) -> str:
    """特殊トークンが消えた後の漏れテキストをできるだけ除去する。"""
    cleaned = text.strip()

    if "assistantfinal" in cleaned:
        cleaned = cleaned.split("assistantfinal", 1)[1]
    elif "final" in cleaned and cleaned.startswith("analysis"):
        cleaned = cleaned.rsplit("final", 1)[-1]

    cleaned = re.sub(r"^(analysis|assistantfinal|final)\s*", "", cleaned).strip()
    return _normalize_reply_text(cleaned)


def decode_local_llm_reply(tokenizer: Any, generated_ids: Any) -> str:
    """ローカルLLMの生成結果からユーザー向け返答だけを抽出する。"""
    raw_text = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
    final_text = extract_gpt_oss_final_text(raw_text)
    if final_text:
        return final_text

    inline_text = raw_text
    inline_end_positions = [inline_text.find(marker) for marker in GPT_OSS_INLINE_END_MARKERS if marker in inline_text]
    if inline_end_positions:
        inline_text = inline_text[:min(inline_end_positions)]
    inline_text = _normalize_reply_text(inline_text)
    if inline_text:
        return inline_text

    fallback_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return _cleanup_fallback_text(fallback_text)


def build_gpt_oss_final_prompt(tokenizer: Any, messages: list[dict]) -> str:
    """gpt-oss の生成開始位置を final チャンネルへ固定したプロンプトを作る。"""
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt + GPT_OSS_FINAL_PREFIX


def clean_qwen_thinking_output(text: str, show_thinking: bool = False) -> str:
    """Qwen3.5 の think ブロックを必要に応じて除去する。"""
    if not text:
        return ""
    if show_thinking:
        return _normalize_reply_text(text)
    return _normalize_reply_text(QWEN_THINK_BLOCK_PATTERN.sub("", text))
