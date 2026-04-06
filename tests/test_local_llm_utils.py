"""ローカルLLMの返答抽出ロジックのテスト。"""

from core.local_llm_utils import (
    build_gpt_oss_final_prompt,
    build_qwen_display_fallback_text,
    build_qwen_generation_prompt,
    clean_qwen_thinking_output,
    decode_local_llm_reply,
    decode_qwen_local_llm_reply,
    extract_gpt_oss_final_text,
    extract_qwen_final_text,
)


class DummyTokenizer:
    """特殊トークンあり・なしの復号結果を返す簡易スタブ。"""

    def __init__(self, with_special_tokens: str, without_special_tokens: str):
        self.with_special_tokens = with_special_tokens
        self.without_special_tokens = without_special_tokens
        self.last_enable_thinking = None

    def decode(self, generated_ids, skip_special_tokens=False):
        return self.without_special_tokens if skip_special_tokens else self.with_special_tokens

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=False):
        assert tokenize is False
        assert add_generation_prompt is True
        self.last_enable_thinking = enable_thinking
        return "<|start|>assistant"


def test_extract_gpt_oss_final_text_returns_only_final_message():
    raw = (
        "<|start|>assistant<|channel|>analysis<|message|>内部推論<|end|>"
        "<|start|>assistant<|channel|>final<|message|>"
        "こんにちは！今日はどこかへお出かけされましたか？<|return|>"
    )

    result = extract_gpt_oss_final_text(raw)

    assert result == "こんにちは！今日はどこかへお出かけされましたか？"


def test_decode_local_llm_reply_prefers_gpt_oss_final_channel():
    tokenizer = DummyTokenizer(
        with_special_tokens=(
            "<|start|>assistant<|channel|>analysis<|message|>内部推論<|end|>"
            "<|start|>assistant<|channel|>final<|message|>"
            "それはゆっくりできそうですね。\n今日は好きな本を読みましたか？<|return|>"
        ),
        without_special_tokens=(
            "analysis内部推論assistantfinal"
            "それはゆっくりできそうですね。\n今日は好きな本を読みましたか？"
        ),
    )

    result = decode_local_llm_reply(tokenizer, [1, 2, 3])

    assert result == "それはゆっくりできそうですね。 今日は好きな本を読みましたか？"


def test_decode_local_llm_reply_cleans_leaked_analysis_prefix_without_special_tokens():
    tokenizer = DummyTokenizer(
        with_special_tokens="",
        without_special_tokens=(
            "analysisWe need to respond following the rules."
            "assistantfinalこんにちは！今日はどこかへお出かけされましたか？"
        ),
    )

    result = decode_local_llm_reply(tokenizer, [1, 2, 3])

    assert result == "こんにちは！今日はどこかへお出かけされましたか？"


def test_decode_local_llm_reply_keeps_normal_model_output():
    tokenizer = DummyTokenizer(
        with_special_tokens="今日はのんびりできてよかったですね。お茶は飲まれましたか？",
        without_special_tokens="今日はのんびりできてよかったですね。お茶は飲まれましたか？",
    )

    result = decode_local_llm_reply(tokenizer, [1, 2, 3])

    assert result == "今日はのんびりできてよかったですね。お茶は飲まれましたか？"


def test_decode_local_llm_reply_trims_trailing_next_message_tokens():
    tokenizer = DummyTokenizer(
        with_special_tokens=(
            "わかったよ。気になることがあればいつでも話せるからね。"
            "<|start|>assistant<|channel|>analysis<|message|>internal reasoning"
        ),
        without_special_tokens=(
            "わかったよ。気になることがあればいつでも話せるからね。assistantanalysisinternal reasoning"
        ),
    )

    result = decode_local_llm_reply(tokenizer, [1, 2, 3])

    assert result == "わかったよ。気になることがあればいつでも話せるからね。"


def test_build_gpt_oss_final_prompt_appends_final_channel_marker():
    tokenizer = DummyTokenizer(with_special_tokens="", without_special_tokens="")

    result = build_gpt_oss_final_prompt(tokenizer, [{"role": "user", "content": "こんにちは"}])

    assert result == "<|start|>assistant<|channel|>final<|message|>"


def test_build_qwen_generation_prompt_passes_enable_thinking():
    tokenizer = DummyTokenizer(with_special_tokens="", without_special_tokens="")

    result = build_qwen_generation_prompt(
        tokenizer,
        [{"role": "user", "content": "こんにちは"}],
        enable_thinking=True,
    )

    assert result == "<|start|>assistant"
    assert tokenizer.last_enable_thinking is True


def test_clean_qwen_thinking_output_hides_thinking_by_default():
    raw = "<think>\n内部で考えています。\n</think>\n\n今日はのんびりできましたか？"

    result = clean_qwen_thinking_output(raw)

    assert result == "今日はのんびりできましたか？"


def test_clean_qwen_thinking_output_can_keep_thinking_for_debug():
    raw = "<think>考え中</think>\n今日はのんびりできましたか？"

    result = clean_qwen_thinking_output(raw, show_thinking=True)

    assert result == "<think>考え中</think> 今日はのんびりできましたか？"


def test_extract_qwen_final_text_removes_im_end_token():
    raw = "ただいま、お元気ですか？<|im_end|>"

    result = extract_qwen_final_text(raw)

    assert result == "ただいま、お元気ですか？"


def test_extract_qwen_final_text_prefers_final_answer_after_think_block():
    raw = "<think>内部で考えています。</think>\n\n今日はのんびりできましたか？<|im_end|>"

    result = extract_qwen_final_text(raw)

    assert result == "今日はのんびりできましたか？"


def test_extract_qwen_final_text_returns_empty_when_only_plain_reasoning_leaks():
    raw = (
        "Thinking Process: 1. Analyze the Request. "
        "The user said \"なんで英語で出力されるのでしょうか？日本語で話せますか？\" "
        "Wait, looking at the actual conversation flow."
    )

    result = extract_qwen_final_text(raw)

    assert result == ""


def test_extract_qwen_final_text_keeps_japanese_answer_after_reasoning_prefix():
    raw = (
        "Thinking Process: 1. Analyze the Request.\n\n"
        "日本語でお話しできますよ。さっきは表示が崩れてしまいました。"
    )

    result = extract_qwen_final_text(raw)

    assert result == "日本語でお話しできますよ。さっきは表示が崩れてしまいました。"


def test_decode_qwen_local_llm_reply_uses_fallback_decode_when_needed():
    tokenizer = DummyTokenizer(
        with_special_tokens="",
        without_special_tokens="元気そうで何よりです。最近、楽しかったことはありましたか？<|im_end|>",
    )

    result = decode_qwen_local_llm_reply(tokenizer, [1, 2, 3])

    assert result == "元気そうで何よりです。最近、楽しかったことはありましたか？"


def test_build_qwen_display_fallback_text_keeps_reasoning_when_no_final_answer_exists():
    raw = "Thinking Process:\n\n1. Analyze the Request.\n2. Continue carefully."

    result = build_qwen_display_fallback_text(raw)

    assert result == "Thinking Process: 1. Analyze the Request. 2. Continue carefully."


def test_build_qwen_display_fallback_text_removes_special_tokens_and_answer_prefix():
    raw = "Assistant: こんにちは。<|im_end|>"

    result = build_qwen_display_fallback_text(raw)

    assert result == "こんにちは。"


class DummyBrokenProcessorTokenizer:
    """Qwen3.5 の不安定経路を模したスタブ。"""

    eos_token_id = 99

    def __init__(self):
        self.last_prompt = None
        self.last_enable_thinking = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=False):
        if tokenize:
            raise TypeError("string indices must be integers")
        assert add_generation_prompt is True
        self.last_enable_thinking = enable_thinking
        self.last_prompt = "\n".join(f"{m['role']}:{m['content']}" for m in messages)
        return self.last_prompt

    def __call__(self, prompt, return_tensors="pt"):
        assert return_tensors == "pt"
        assert prompt == self.last_prompt
        return {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}


def test_qwen_text_path_can_tokenize_after_chat_template_without_processor_fast_path():
    tokenizer = DummyBrokenProcessorTokenizer()
    messages = [
        {"role": "system", "content": "会話してください"},
        {"role": "user", "content": "こんにちは"},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    model_inputs = tokenizer(prompt, return_tensors="pt")

    assert prompt == "system:会話してください\nuser:こんにちは"
    assert model_inputs["input_ids"] == [[1, 2, 3]]
    assert model_inputs["attention_mask"] == [[1, 1, 1]]
    assert tokenizer.last_enable_thinking is True
