"""DPO preference データセット作成スクリプトのテスト。"""

import csv
import json
import random
from pathlib import Path

from tools.build_dpo_preference_dataset import (
    PreferenceExample,
    SourceExample,
    build_generation_messages,
    classify_prompt_type,
    extract_rejected_text,
    generate_preference_for_example,
    read_source_examples,
    select_source_examples,
    validate_rejected,
    write_csv_files,
    write_jsonl,
)


class StubGenerator:
    """テスト用 rejected 生成器。"""

    model_id = "stub-qwen"

    def __init__(self, outputs: list[str]):
        self.outputs = outputs
        self.calls = 0

    def generate(self, messages):
        self.calls += 1
        return self.outputs.pop(0)


def make_source(
    *,
    rank: int = 1,
    prompt: str = "神奈川に行ったのは何かの聖地なんですか?",
    chosen: str = "そうですね、好きなアニメの舞台で、一度行ってみたかった場所なんです。",
    final_score: float = 0.8,
    expert_start_sec: float = 20.0,
) -> SourceExample:
    return SourceExample(
        source_rank=rank,
        session_id="001",
        prompt=prompt,
        chosen=chosen,
        final_score=final_score,
        engagement_delta=0.1,
        expert_start_sec=expert_start_sec,
        expert_end_sec=expert_start_sec + 5.0,
        novice_start_sec=expert_start_sec - 3.0,
        novice_end_sec=expert_start_sec - 1.0,
        raw_row={},
    )


def write_score_csv(path: Path) -> None:
    fieldnames = [
        "session_id",
        "expert_start_sec",
        "expert_end_sec",
        "novice_start_sec",
        "novice_end_sec",
        "final_score",
        "engagement_delta",
        "novice_text",
        "expert_text",
    ]
    rows = [
        {
            "session_id": "001",
            "expert_start_sec": "5.0",
            "expert_end_sec": "8.0",
            "novice_start_sec": "1.0",
            "novice_end_sec": "4.0",
            "final_score": "0.9",
            "engagement_delta": "0.2",
            "novice_text": "聞こえてますか?",
            "expert_text": "はい、聞こえてます。初めまして。",
        },
        {
            "session_id": "001",
            "expert_start_sec": "20.0",
            "expert_end_sec": "25.0",
            "novice_start_sec": "16.0",
            "novice_end_sec": "19.0",
            "final_score": "0.8",
            "engagement_delta": "0.1",
            "novice_text": "旅行は好きなんですか?",
            "expert_text": "はい、移動時間も含めて楽しむタイプなんです。",
        },
        {
            "session_id": "002",
            "expert_start_sec": "30.0",
            "expert_end_sec": "35.0",
            "novice_start_sec": "27.0",
            "novice_end_sec": "29.0",
            "final_score": "0.7",
            "engagement_delta": "0.05",
            "novice_text": "ちょっと難しいですね。",
            "expert_text": "難しかったら身近な例で言い直しますね。",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_select_source_examples_skips_setup_rows(tmp_path: Path):
    path = tmp_path / "score.csv"
    write_score_csv(path)

    examples = select_source_examples(read_source_examples(path), top_n=2)

    assert [example.source_rank for example in examples] == [2, 3]


def test_classify_prompt_type():
    assert classify_prompt_type("それはいくらぐらいなんですか?") == "question"
    assert classify_prompt_type("ちょっと難しいですね。") == "difficulty"
    assert classify_prompt_type("はいはいはい") == "backchannel"
    assert classify_prompt_type("昔から旅行が好きで、普通電車で遠くまで行くのが楽しいんですよ。") == "self_disclosure"


def test_build_generation_messages_contains_constraints():
    source = make_source()

    messages = build_generation_messages(source, "question", "内容を拾わない一般論")
    joined = "\n".join(message["content"] for message in messages)

    assert "JSON" in joined
    assert "暴言" in joined
    assert "chosen よりも、相手が話し続けにくくなる" in joined


def test_extract_rejected_text_from_json():
    text = '{"rejected": "旅行は気分転換になりますよね。"}'

    assert extract_rejected_text(text) == "旅行は気分転換になりますよね。"


def test_validate_rejected_rejects_duplicates():
    source = make_source()

    valid, reason = validate_rejected(source, source.chosen)

    assert valid is False
    assert "chosen" in reason


def test_generate_preference_for_example_retries_until_valid():
    source = make_source()
    generator = StubGenerator([
        '{"rejected": "そうですね、好きなアニメの舞台で、一度行ってみたかった場所なんです。"}',
        '{"rejected": "旅行は気分転換になりますよね。"}',
    ])

    result = generate_preference_for_example(source, generator=generator, rng=random.Random(1))

    assert isinstance(result, PreferenceExample)
    assert result.rejected == "旅行は気分転換になりますよね。"
    assert generator.calls == 2


def test_write_jsonl_and_csv_files(tmp_path: Path):
    source = make_source()
    example = PreferenceExample(
        prompt=source.prompt,
        chosen=source.chosen,
        rejected="旅行は気分転換になりますよね。",
        prompt_type="question",
        rejected_strategy="内容を拾わない一般論",
        source=source,
        model_id="stub-qwen",
    )
    jsonl_path = tmp_path / "dpo.jsonl"
    csv_path = tmp_path / "dpo.csv"
    failed_path = tmp_path / "failed.csv"

    write_jsonl([example], jsonl_path)
    write_csv_files([example], [], output_csv=csv_path, failed_csv=failed_path)

    record = json.loads(jsonl_path.read_text(encoding="utf-8").strip())
    assert record["prompt"] == source.prompt
    assert record["metadata"]["source_rank"] == 1
    with csv_path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["rejected"] == "旅行は気分転換になりますよね。"
