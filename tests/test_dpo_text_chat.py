"""DPO後モデルのターミナルチャットの軽量テスト。"""

import sys

from apps.dpo_text_chat import (
    DEFAULT_BASE_MODEL_ID,
    DEFAULT_LORA_PATH,
    build_dpo_prompt,
    cleanup_generated_text,
    parse_args,
    strip_prompt_prefix,
)


def test_build_dpo_prompt_matches_dataset_style():
    prompt = build_dpo_prompt("最近、またギターを弾きたくなってきました。")

    assert prompt == (
        "以下の会話の次のAI返答を生成してください。\n\n"
        "これまでの会話:\n"
        "User: 最近、またギターを弾きたくなってきました。\n\n"
        "AI:"
    )


def test_strip_prompt_prefix_removes_prompt_when_present():
    prompt = build_dpo_prompt("旅行の話をしたいです。")
    decoded = prompt + "いいですね。どんな場所が気になっていますか？"

    assert strip_prompt_prefix(decoded, prompt) == "いいですね。どんな場所が気になっていますか？"


def test_cleanup_generated_text_removes_qwen_special_tokens():
    prompt = build_dpo_prompt("お酒の話をしたいです。")
    decoded = "バーボン、お好きなんですね。<|im_end|>"

    assert cleanup_generated_text(decoded, prompt) == "バーボン、お好きなんですね。"


def test_default_paths():
    assert DEFAULT_BASE_MODEL_ID == "Qwen/Qwen3.5-27B"
    assert DEFAULT_LORA_PATH == "artifacts/qwen35_dpo_lora"


def test_parse_args_defaults_to_non_4bit(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["apps.dpo_text_chat"])

    args = parse_args()

    assert args.use_4bit is False
