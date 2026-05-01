"""DPO 候補用 engagement 増加ランキング作成スクリプトのテスト。"""

import csv
from pathlib import Path

import pytest

from tools.build_dpo_engagement_ranking import (
    CSV_COLUMNS,
    build_rows,
    compute_openface_features,
    compute_openpose_features,
    mean_in_window,
    read_dict_rows,
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


OPENPOSE_POINTS = [
    "nose",
    "neck",
    "r_eye",
    "l_eye",
    "r_ear",
    "l_ear",
    "r_shoulder",
    "l_shoulder",
    "r_elbow",
    "l_elbow",
    "r_wrist",
    "l_wrist",
]


def write_openface(path: Path, rows: list[tuple[float, float, float, float, float]]) -> None:
    """テスト用 OpenFace CSV を書き込む。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "timestamp", "AU06_r", "AU06_c", "AU12_r", "AU12_c"])
        for frame, timestamp, au06_r, au06_c, au12_r in rows:
            writer.writerow([frame, timestamp, au06_r, au06_c, au12_r, 1.0 if au12_r > 0 else 0.0])


def base_openpose_row(frame: int, timestamp: float, *, x_offset: float = 0.0) -> dict[str, float]:
    """テスト用 OpenPose 行を作る。"""
    coordinates = {
        "nose": (50.0 + x_offset, 10.0),
        "neck": (50.0 + x_offset, 30.0),
        "r_eye": (45.0 + x_offset, 8.0),
        "l_eye": (55.0 + x_offset, 8.0),
        "r_ear": (40.0 + x_offset, 10.0),
        "l_ear": (60.0 + x_offset, 10.0),
        "r_shoulder": (30.0, 40.0),
        "l_shoulder": (70.0, 40.0),
        "r_elbow": (25.0 + x_offset, 55.0),
        "l_elbow": (75.0 + x_offset, 55.0),
        "r_wrist": (20.0 + x_offset, 70.0),
        "l_wrist": (80.0 + x_offset, 70.0),
    }
    row: dict[str, float] = {"frame": float(frame), "timestamp": timestamp}
    for point_name, (x, y) in coordinates.items():
        row[f"{point_name}_x"] = x
        row[f"{point_name}_y"] = y
        row[f"{point_name}_conf"] = 1.0
    return row


def write_openpose(path: Path, rows: list[dict[str, float]]) -> None:
    """テスト用 OpenPose CSV を書き込む。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["frame", "timestamp"]
    for point_name in OPENPOSE_POINTS:
        fieldnames.extend([f"{point_name}_x", f"{point_name}_y", f"{point_name}_conf"])
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_multimodal_session(multimodal_dir: Path, session_id: str) -> None:
    """テスト用 multimodal CSV を作成する。"""
    session_dir = multimodal_dir / session_id
    write_openface(
        session_dir / "novice.openface2.smile_au.csv",
        [
            (1.0, 4.0, 0.1, 1.0, 0.2),
            (2.0, 4.5, 0.1, 1.0, 0.2),
            (5.0, 3.0, 0.1, 1.0, 0.2),
            (6.0, 3.5, 0.1, 1.0, 0.2),
            (3.0, 9.0, 0.2, 1.0, 0.4),
            (4.0, 9.5, 0.2, 1.0, 0.4),
        ],
    )
    write_openpose(
        session_dir / "novice.openpose.csv",
        [
            base_openpose_row(5, 3.0, x_offset=0.0),
            base_openpose_row(6, 3.5, x_offset=1.0),
            base_openpose_row(1, 4.0, x_offset=0.0),
            base_openpose_row(2, 4.5, x_offset=1.0),
            base_openpose_row(3, 9.0, x_offset=0.0),
            base_openpose_row(4, 9.5, x_offset=2.0),
        ],
    )


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
    multimodal_dir = tmp_path / "multimodal_csv"
    make_multimodal_session(multimodal_dir, "001")

    rows, warnings = build_rows(dataset_dir, multimodal_dir=multimodal_dir, window_sec=1.0, sample_rate=1.0)

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
    assert any("前後窓が空" in warning for warning in warnings)


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


def test_compute_openface_features_uses_after_window_smile_values(tmp_path: Path):
    path = tmp_path / "openface.csv"
    write_openface(
        path,
        [
            (1.0, 0.0, 0.0, 0.0, 0.0),
            (2.0, 1.0, 0.2, 1.0, 0.5),
            (3.0, 1.5, 0.4, 0.0, 0.7),
        ],
    )

    features = compute_openface_features(read_dict_rows(path), start_sec=1.0, end_sec=2.0)

    assert features.frame_count == 2
    assert features.au06_after_mean == pytest.approx(0.3)
    assert features.au12_after_mean == pytest.approx(0.6)
    assert features.au06_after_ratio == pytest.approx(0.5)
    assert features.au12_after_ratio == pytest.approx(1.0)
    assert features.smile_score_raw == pytest.approx(0.7 * 0.6 + 0.3 * 0.3)


def test_compute_openpose_features_normalizes_motion_by_shoulder_width(tmp_path: Path):
    path = tmp_path / "openpose.csv"
    write_openpose(
        path,
        [
            base_openpose_row(1, 1.0, x_offset=0.0),
            base_openpose_row(2, 1.5, x_offset=4.0),
        ],
    )

    features = compute_openpose_features(
        read_dict_rows(path),
        start_sec=1.0,
        end_sec=2.0,
        confidence_threshold=0.3,
    )

    assert features.frame_pair_count == 1
    assert features.head_motion_score_raw == pytest.approx(0.1)
    assert features.upper_body_activation_score_raw == pytest.approx((5 * 4.0 / 40.0) / 7)
    assert features.hand_energy_score_raw == pytest.approx(0.1)


def test_build_rows_outputs_multimodal_score_columns(tmp_path: Path):
    dataset_dir = tmp_path / "datasets"
    multimodal_dir = tmp_path / "multimodal_csv"
    make_session(
        dataset_dir,
        "001",
        expert_rows=[(2.0, 3.0, "返答")],
        novice_rows=[(1.0, 1.5, "質問")],
        engagement_values=[0.1, 0.1, 0.5, 0.5],
    )
    make_multimodal_session(multimodal_dir, "001")

    rows, warnings = build_rows(dataset_dir, multimodal_dir=multimodal_dir, window_sec=1.0, sample_rate=1.0)

    assert warnings == []
    assert rows[0].smile_score_raw is not None
    assert rows[0].head_motion_score_raw is not None
    assert rows[0].final_score == 0.5
