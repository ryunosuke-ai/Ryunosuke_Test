"""ローカルLLMの生成結果を整形する補助関数。"""

import re
from typing import Any


GPT_OSS_FINAL_PREFIX = "<|channel|>final<|message|>"
GPT_OSS_FINAL_MARKER = "<|start|>assistant<|channel|>final<|message|>"
GPT_OSS_END_MARKERS = ("<|return|>", "<|end|>")
GPT_OSS_INLINE_END_MARKERS = ("<|return|>", "<|start|>", "<|end|>")
QWEN_THINK_BLOCK_PATTERN = re.compile(r"<think>\s*.*?\s*</think>\s*", re.DOTALL)
QWEN_SPECIAL_TOKEN_PATTERN = re.compile(r"<\|[^>]+?\|>")
QWEN_REASONING_HEAD_PATTERN = re.compile(
    r"^\s*(Thinking Process:|The user is asking\b|Analyze the Request:|Wait, looking at\b)",
    re.IGNORECASE,
)
QWEN_ANSWER_MARKER_PATTERN = re.compile(
    r"(Final Answer:|Final Response:|Response:|Assistant:|回答:|返答:)",
    re.IGNORECASE,
)
JAPANESE_CHAR_PATTERN = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]")


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


def _strip_qwen_special_tokens(text: str) -> str:
    """Qwen 系で漏れやすい特殊トークンを除去する。"""
    if not text:
        return ""
    return QWEN_SPECIAL_TOKEN_PATTERN.sub(" ", text)


def _japanese_ratio(text: str) -> float:
    """文字列中の日本語比率をざっくり求める。"""
    if not text:
        return 0.0
    japanese_chars = len(JAPANESE_CHAR_PATTERN.findall(text))
    visible_chars = len(re.sub(r"\s+", "", text))
    if visible_chars == 0:
        return 0.0
    return japanese_chars / visible_chars


def extract_qwen_final_text(raw_text: str, show_thinking: bool = False) -> str:
    """Qwen3.5 の生出力からユーザー向け最終回答だけをできるだけ抽出する。"""
    if not raw_text:
        return ""

    cleaned = _strip_qwen_special_tokens(raw_text)
    if show_thinking:
        return _normalize_reply_text(cleaned)

    cleaned = QWEN_THINK_BLOCK_PATTERN.sub(" ", cleaned).strip()
    if not cleaned:
        return ""

    answer_markers = list(QWEN_ANSWER_MARKER_PATTERN.finditer(cleaned))
    if answer_markers:
        cleaned = cleaned[answer_markers[-1].end():].strip()

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n+", cleaned)
        if paragraph.strip()
    ]

    if paragraphs and QWEN_REASONING_HEAD_PATTERN.match(paragraphs[0]):
        if len(paragraphs) == 1 and not answer_markers:
            return ""
        candidate_paragraphs = [
            paragraph
            for paragraph in paragraphs
            if not QWEN_REASONING_HEAD_PATTERN.match(paragraph)
            and _japanese_ratio(paragraph) >= 0.2
        ]
        if candidate_paragraphs:
            cleaned = candidate_paragraphs[-1]

    if QWEN_REASONING_HEAD_PATTERN.match(cleaned) and _japanese_ratio(cleaned) < 0.2:
        return ""

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


def build_qwen_generation_prompt(tokenizer: Any, messages: list[dict], enable_thinking: bool = False) -> str:
    """Qwen3.5 の生成用プロンプトを組み立てる。"""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def clean_qwen_thinking_output(text: str, show_thinking: bool = False) -> str:
    """Qwen3.5 の think ブロックを必要に応じて除去する。"""
    if not text:
        return ""
    return extract_qwen_final_text(text, show_thinking=show_thinking)


def decode_qwen_local_llm_reply(tokenizer: Any, generated_ids: Any, show_thinking: bool = False) -> str:
    """Qwen3.5 の生成結果から特殊トークンや推論を除いた返答本文を抽出する。"""
    raw_text = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
    reply = extract_qwen_final_text(raw_text, show_thinking=show_thinking)
    if reply:
        return reply

    fallback_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return extract_qwen_final_text(fallback_text, show_thinking=show_thinking)
