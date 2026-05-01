"""DPO 候補用 engagement 増加ランキング作成スクリプトのテスト。"""

import csv
from pathlib import Path

from tools.build_dpo_engagement_ranking import (
    CSV_COLUMNS,
    build_rows,
    mean_in_window,
    write_rows,
)


def write_transcript(path: Path, rows: list[tuple[float, float, str]]) -> None:
    """テスト用 transcript を書き込む。"""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=";")
        for start_sec, end_sec, text in rows:
            writer.writerow([start_sec, end_sec, text, "0.00"])


def write_engagement(path: Path, values: list[float]) -> None:
    """テスト用 engagement を書き込む。"""
    path.write_text("".join(f"{value:.6f}\n" for value in values), encoding="utf-8")


def make_session(
    dataset_dir: Path,
    session_id: str,
    *,
    expert_rows: list[tuple[float, float, str]],
    novice_rows: list[tuple[float, float, str]],
    engagement_values: list[float],
) -> None:
    """テスト用セッションディレクトリを作成する。"""
    session_dir = dataset_dir / session_id
    session_dir.mkdir(parents=True)
    write_transcript(session_dir / "expert.audio.transcript.annotation.csv", expert_rows)
    write_transcript(session_dir / "novice.audio.transcript.annotation.csv", novice_rows)
    write_engagement(session_dir / "novice.engagement.annotation.csv", engagement_values)


def test_mean_in_window_uses_frame_timestamps_and_clips_bounds():
    values = [0.0, 1.0, 2.0, 3.0]

    mean, count = mean_in_window(values, start_sec=-1.0, end_sec=0.21, sample_rate=10.0)

    assert mean == 1.0
    assert count == 3


def test_build_rows_keeps_only_expert_response_after_novice_and_sorts_by_delta(tmp_path: Path):
    dataset_dir = tmp_path / "datasets"
    make_session(
        dataset_dir,
        "001",
        expert_rows=[
            (0.0, 1.0, "冒頭説明"),
            (3.0, 4.0, "小さく上がる返答"),
            (6.0, 7.0, "連続した expert 返答"),
            (8.0, 9.0, "大きく上がる返答"),
        ],
        novice_rows=[
            (1.5, 2.0, "novice の質問1"),
            (7.2, 7.5, "novice の質問2"),
        ],
        engagement_values=[
            0.1,
            0.1,
            0.1,
            0.2,
            0.2,
            0.2,
            0.2,
            0.3,
            0.8,
            0.8,
        ],
    )

    rows, warnings = build_rows(dataset_dir, window_sec=1.0, sample_rate=1.0)

    assert warnings == []
    assert [row.expert_text for row in rows] == ["大きく上がる返答", "小さく上がる返答"]
    assert rows[0].novice_text == "novice の質問2"
    assert rows[0].before_engagement_mean == 0.3
    assert rows[0].after_engagement_mean == 0.8
    assert rows[0].engagement_delta == 0.5
    assert rows[1].engagement_delta == 0.1


def test_build_rows_skips_empty_before_or_after_window(tmp_path: Path):
    dataset_dir = tmp_path / "datasets"
    make_session(
        dataset_dir,
        "001",
        expert_rows=[(1.0, 4.0, "後続窓が空になる返答")],
        novice_rows=[(0.2, 0.8, "直前 novice")],
        engagement_values=[0.2, 0.2, 0.2, 0.2],
    )

    rows, warnings = build_rows(dataset_dir, window_sec=1.0, sample_rate=1.0)

    assert rows == []
    assert len(warnings) == 1
    assert "前後窓が空" in warnings[0]


def test_write_rows_outputs_expected_columns_and_formatted_values(tmp_path: Path):
    dataset_dir = tmp_path / "datasets"
    output_path = tmp_path / "ranking.csv"
    make_session(
        dataset_dir,
        "001",
        expert_rows=[(2.0, 3.0, "返答")],
        novice_rows=[(1.0, 1.5, "質問")],
        engagement_values=[0.1, 0.1, 0.5, 0.5],
    )
    rows, _ = build_rows(dataset_dir, window_sec=1.0, sample_rate=1.0)

    write_rows(rows, output_path)

    with output_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        output_rows = list(reader)

    assert reader.fieldnames == CSV_COLUMNS
    assert output_rows[0]["session_id"] == "001"
    assert output_rows[0]["engagement_delta"] == "0.400000"
    assert output_rows[0]["expert_text"] == "返答"
    assert output_rows[0]["novice_text"] == "質問"
