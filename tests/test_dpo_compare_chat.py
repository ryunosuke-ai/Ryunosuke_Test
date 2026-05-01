"""DPO前後比較UIの軽量テスト。"""

from apps.dpo_compare_chat import (
    DEFAULT_BASE_MODEL_ID,
    DEFAULT_LORA_PATH,
    build_dpo_compare_prompt,
    cleanup_generated_text,
    disable_peft_bitsandbytes_dispatch,
    strip_prompt_prefix,
)


def test_build_dpo_compare_prompt_matches_training_template():
    prompt = build_dpo_compare_prompt("最近、またギターを弾きたくなってきました。")

    assert prompt == (
        "以下の会話の次のAI返答を生成してください。\n\n"
        "これまでの会話:\n"
        "User: 最近、またギターを弾きたくなってきました。\n\n"
        "AI:"
    )


def test_strip_prompt_prefix_removes_prompt_when_present():
    prompt = build_dpo_compare_prompt("旅行の話をしたいです。")
    decoded = prompt + "いいですね。どんな場所が気になっていますか？"

    assert strip_prompt_prefix(decoded, prompt) == "いいですね。どんな場所が気になっていますか？"


def test_strip_prompt_prefix_uses_last_ai_marker_as_fallback():
    decoded = "User: 旅行の話をしたいです。\n\nAI: いいですね。どんな場所が気になっていますか？"

    assert strip_prompt_prefix(decoded, "missing prompt") == "いいですね。どんな場所が気になっていますか？"


def test_cleanup_generated_text_removes_qwen_special_tokens():
    prompt = build_dpo_compare_prompt("お酒の話をしたいです。")
    decoded = "バーボン、お好きなんですね。<|im_end|>"

    assert cleanup_generated_text(decoded, prompt) == "バーボン、お好きなんですね。"


def test_default_paths():
    assert DEFAULT_BASE_MODEL_ID == "Qwen/Qwen3.5-27B"
    assert DEFAULT_LORA_PATH == "artifacts/qwen35_dpo_lora"


def test_disable_peft_bitsandbytes_dispatch_forces_detectors_false():
    try:
        import peft.import_utils as peft_import_utils
        import peft.tuners.lora.model as peft_lora_model
    except ImportError:
        return

    disable_peft_bitsandbytes_dispatch()

    assert peft_import_utils.is_bnb_available() is False
    assert peft_import_utils.is_bnb_4bit_available() is False
    assert peft_lora_model.is_bnb_available() is False
    assert peft_lora_model.is_bnb_4bit_available() is False
