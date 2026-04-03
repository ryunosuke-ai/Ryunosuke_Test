"""ローカルLLMの返答抽出ロジックのテスト。"""

from core.local_llm_utils import build_gpt_oss_final_prompt, decode_local_llm_reply, extract_gpt_oss_final_text


class DummyTokenizer:
    """特殊トークンあり・なしの復号結果を返す簡易スタブ。"""

    def __init__(self, with_special_tokens: str, without_special_tokens: str):
        self.with_special_tokens = with_special_tokens
        self.without_special_tokens = without_special_tokens

    def decode(self, generated_ids, skip_special_tokens=False):
        return self.without_special_tokens if skip_special_tokens else self.with_special_tokens

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        assert add_generation_prompt is True
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
