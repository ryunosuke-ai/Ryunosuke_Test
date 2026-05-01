"""DPO前後比較ターミナルCLIの軽量テスト。"""

import logging
import sys
from pathlib import Path

from apps.dpo_compare_text_chat import (
    DEFAULT_BASE_MODEL_ID,
    DEFAULT_LORA_PATH,
    append_history_line,
    build_dpo_compare_prompt,
    cleanup_generated_text,
    create_run_dir_with_compare_logs,
    parse_args,
    setup_logger,
    strip_prompt_prefix,
    write_session_header,
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


def test_cleanup_generated_text_truncates_role_markers():
    prompt = build_dpo_compare_prompt("読書をしていました。")
    decoded = "今日は読書の気分ですね。User: その本、面白いですか？"

    assert cleanup_generated_text(decoded, prompt) == "今日は読書の気分ですね。"


def test_cleanup_generated_text_removes_qwen_special_tokens():
    prompt = build_dpo_compare_prompt("お酒の話をしたいです。")
    decoded = "バーボン、お好きなんですね。<|im_end|>"

    assert cleanup_generated_text(decoded, prompt) == "バーボン、お好きなんですね。"


def test_default_paths():
    assert DEFAULT_BASE_MODEL_ID == "Qwen/Qwen3.5-27B"
    assert DEFAULT_LORA_PATH == "artifacts/qwen35_dpo_lora"


def test_parse_args_defaults_to_non_4bit(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["apps.dpo_compare_text_chat"])

    args = parse_args()

    assert args.use_4bit is False


def test_append_history_line_writes_timestamped_line(tmp_path):
    history_file = tmp_path / "log.txt"

    append_history_line(str(history_file), "User", "こんにちは\n元気ですか?")

    content = history_file.read_text(encoding="utf-8").strip()
    assert content.startswith("[")
    assert "] User: こんにちは 元気ですか?" in content


def test_write_session_header_writes_compare_metadata(tmp_path):
    history_file = tmp_path / "log.txt"
    args = type(
        "Args",
        (),
        {
            "max_new_tokens": 160,
            "temperature": 0.7,
            "top_p": 0.8,
            "repetition_penalty": 1.0,
            "seed": 42,
        },
    )()

    write_session_header(
        str(history_file),
        base_model_id=DEFAULT_BASE_MODEL_ID,
        lora_path=DEFAULT_LORA_PATH,
        use_4bit=False,
        args=args,
    )

    content = history_file.read_text(encoding="utf-8")
    assert "# mode: compare" in content
    assert "# base_model_id: Qwen/Qwen3.5-27B" in content
    assert "# lora_path: artifacts/qwen35_dpo_lora" in content
    assert "# use_4bit: False" in content
    assert "# outputs: base,dpo" in content


def test_create_run_dir_creates_log_dir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    run_dir, history_file = create_run_dir_with_compare_logs()

    assert Path(run_dir).exists()
    assert Path(history_file).parent == Path(run_dir)


def test_setup_logger_uses_file_handler_only(tmp_path):
    logger = setup_logger(str(tmp_path), "20260501_170756")

    assert any(isinstance(handler, logging.FileHandler) for handler in logger.handlers)
    assert not any(type(handler) is logging.StreamHandler for handler in logger.handlers)
