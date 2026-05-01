"""Qwen3.5 DPO LoRA 学習スクリプトの軽量テスト。"""

import json
from argparse import Namespace
from pathlib import Path

import pytest

from tools.train_qwen35_dpo_lora import (
    PreferenceDatasetSplit,
    disable_peft_bitsandbytes_dispatch,
    print_dry_run_summary,
    read_preference_records,
    split_records,
    summarize_records,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """テスト用JSONLを書き込む。"""
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def make_row(index: int = 1) -> dict[str, str]:
    """DPOレコードを作る。"""
    return {
        "prompt": f"以下の会話の次のAI返答を生成してください。\n\nこれまでの会話:\nUser: 話題{index}\n\nAI:",
        "chosen": f"いいですね、話題{index}についてもう少し聞かせてください。",
        "rejected": f"話題{index}は一般的に重要ですね。",
        "metadata": {"source_rank": index},
    }


def test_read_preference_records_keeps_only_required_columns(tmp_path: Path):
    path = tmp_path / "dpo.jsonl"
    write_jsonl(path, [make_row(1)])

    records = read_preference_records(path)

    assert records == [
        {
            "prompt": make_row(1)["prompt"],
            "chosen": make_row(1)["chosen"],
            "rejected": make_row(1)["rejected"],
        }
    ]


def test_read_preference_records_rejects_missing_required_column(tmp_path: Path):
    path = tmp_path / "dpo.jsonl"
    row = make_row(1)
    row.pop("rejected")
    write_jsonl(path, [row])

    with pytest.raises(ValueError, match="rejected"):
        read_preference_records(path)


def test_read_preference_records_rejects_empty_dataset(tmp_path: Path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n", encoding="utf-8")

    with pytest.raises(ValueError, match="有効なレコード"):
        read_preference_records(path)


def test_split_records_uses_eval_ratio_and_seed():
    records = [make_row(index) for index in range(10)]

    first = split_records(records, eval_ratio=0.2, seed=123)
    second = split_records(records, eval_ratio=0.2, seed=123)

    assert len(first.train) == 8
    assert len(first.eval) == 2
    assert first == second
    assert {row["prompt"] for row in first.train}.isdisjoint({row["prompt"] for row in first.eval})


def test_split_records_can_disable_eval():
    records = [make_row(index) for index in range(3)]

    split = split_records(records, eval_ratio=0.0, seed=123)

    assert len(split.train) == 3
    assert split.eval == []


def test_summarize_records_returns_max_lengths():
    records = [make_row(1), make_row(2)]

    summary = summarize_records(records)

    assert summary["count"] == 2
    assert summary["max_prompt_chars"] == max(len(row["prompt"]) for row in records)
    assert summary["max_chosen_chars"] == max(len(row["chosen"]) for row in records)
    assert summary["max_rejected_chars"] == max(len(row["rejected"]) for row in records)


def test_print_dry_run_summary_outputs_core_settings(capsys):
    args = Namespace(
        dataset="artifacts/noxij_dpo_preferences_ai_user.jsonl",
        model_id="Qwen/Qwen3.5-27B",
        output_dir="artifacts/qwen35_dpo_lora",
        no_4bit=False,
    )
    split = PreferenceDatasetSplit(train=[make_row(1)], eval=[])

    print_dry_run_summary(args, split)

    output = capsys.readouterr().out
    assert "DPO LoRA dry-run" in output
    assert "Qwen/Qwen3.5-27B" in output
    assert "train/eval: 1 / 0" in output


def test_disable_peft_bitsandbytes_dispatch_forces_detectors_false():
    try:
        import peft.import_utils as peft_import_utils
        import peft.tuners.lora.model as peft_lora_model
    except ImportError:
        pytest.skip("peft がインストールされていません")

    disable_peft_bitsandbytes_dispatch()

    assert peft_import_utils.is_bnb_available() is False
    assert peft_import_utils.is_bnb_4bit_available() is False
    assert peft_lora_model.is_bnb_available() is False
    assert peft_lora_model.is_bnb_4bit_available() is False
