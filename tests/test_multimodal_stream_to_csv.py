"""OpenPose / eGeMAPS ストリーム CSV 変換スクリプトのテスト。"""

import csv
import struct
from pathlib import Path

import pytest

from tools.multimodal_stream_to_csv import (
    MODALITY_EGEMAPS,
    MODALITY_OPENPOSE,
    build_egemaps_column_names,
    build_modality_specs,
    build_openpose_column_names,
    convert_subject_directory,
    read_and_write_csv,
)


def write_meta_file(
    path: Path,
    *,
    dim: int,
    frame_count: int = 2,
    root_name: str = "stream",
    meta_type: str = "",
) -> None:
    """テスト用 `.stream` を生成する。"""
    content = f"""<{root_name} ssi-v ="2">
    <info ftype="BINARY" sr="25.000" dim="{dim}" byte="4" type="FLOAT" delim=" " />
    <meta type="{meta_type}" />
    <chunk from="0.000" to="0.080" byte="0" num="{frame_count}" />
</{root_name}>
"""
    path.write_text(content, encoding="utf-8")


def write_data_file(path: Path, *, dim: int, frame_count: int) -> None:
    """テスト用 `.stream~` を生成する。"""
    values = [float(index) for index in range(frame_count * dim)]
    packed = struct.pack("<" + ("f" * len(values)), *values)
    path.write_bytes(packed)


def test_build_openpose_column_names_has_body_keypoints_and_extras():
    columns = build_openpose_column_names()

    assert len(columns) == 139
    assert columns[:9] == [
        "nose_x",
        "nose_y",
        "nose_conf",
        "neck_x",
        "neck_y",
        "neck_conf",
        "r_shoulder_x",
        "r_shoulder_y",
        "r_shoulder_conf",
    ]
    assert columns[-1] == "openpose_extra_084"


def test_build_egemaps_column_names_has_88_columns():
    columns = build_egemaps_column_names()

    assert len(columns) == 88
    assert columns[0] == "egemaps_000"
    assert columns[-1] == "egemaps_087"


def test_read_and_write_csv_converts_openpose_stream(tmp_path: Path):
    specs = build_modality_specs()
    spec = specs[MODALITY_OPENPOSE]
    meta_path = tmp_path / "novice.openpose.stream"
    data_path = tmp_path / "novice.openpose.stream~"
    output_path = tmp_path / "novice.openpose.csv"
    write_meta_file(meta_path, dim=139, frame_count=2)
    write_data_file(data_path, dim=139, frame_count=2)

    frame_count, dimension = read_and_write_csv(meta_path, data_path, output_path, spec)

    assert frame_count == 2
    assert dimension == 139
    with output_path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))
    assert rows[0][:5] == ["frame", "timestamp", "nose_x", "nose_y", "nose_conf"]
    assert len(rows[0]) == 141
    assert rows[1][0] == "1"
    assert rows[1][1] == "0.0"
    assert rows[2][1] == "0.04"


def test_read_and_write_csv_converts_egemaps_stream(tmp_path: Path):
    specs = build_modality_specs()
    spec = specs[MODALITY_EGEMAPS]
    meta_path = tmp_path / "novice.audio.egemapsv2.stream"
    data_path = tmp_path / "novice.audio.egemapsv2.stream~"
    output_path = tmp_path / "novice.audio.egemapsv2.csv"
    write_meta_file(
        meta_path,
        dim=88,
        frame_count=2,
        root_name="annotation",
        meta_type="eGeMAPSv02_Functionals",
    )
    write_data_file(data_path, dim=88, frame_count=2)

    frame_count, dimension = read_and_write_csv(meta_path, data_path, output_path, spec)

    assert frame_count == 2
    assert dimension == 88
    with output_path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))
    assert rows[0][:4] == ["frame", "timestamp", "egemaps_000", "egemaps_001"]
    assert len(rows[0]) == 90


def test_read_and_write_csv_fails_for_wrong_dimension(tmp_path: Path):
    spec = build_modality_specs()[MODALITY_OPENPOSE]
    meta_path = tmp_path / "novice.openpose.stream"
    data_path = tmp_path / "novice.openpose.stream~"
    output_path = tmp_path / "novice.openpose.csv"
    write_meta_file(meta_path, dim=138, frame_count=1)
    write_data_file(data_path, dim=138, frame_count=1)

    with pytest.raises(ValueError, match="次元数が想定と異なります"):
        read_and_write_csv(meta_path, data_path, output_path, spec)


def test_read_and_write_csv_fails_for_short_binary(tmp_path: Path):
    spec = build_modality_specs()[MODALITY_EGEMAPS]
    meta_path = tmp_path / "novice.audio.egemapsv2.stream"
    data_path = tmp_path / "novice.audio.egemapsv2.stream~"
    output_path = tmp_path / "novice.audio.egemapsv2.csv"
    write_meta_file(
        meta_path,
        dim=88,
        frame_count=2,
        root_name="annotation",
        meta_type="eGeMAPSv02_Functionals",
    )
    write_data_file(data_path, dim=88, frame_count=1)

    with pytest.raises(ValueError, match="サイズがメタ情報と一致しません"):
        read_and_write_csv(meta_path, data_path, output_path, spec)


def test_convert_subject_directory_writes_selected_modalities(tmp_path: Path):
    input_dir = tmp_path / "datasets"
    output_dir = tmp_path / "multimodal_csv"
    unrelated_dir = input_dir / "openface_csv"
    unrelated_dir.mkdir(parents=True)
    subject_dir = input_dir / "086"
    subject_dir.mkdir(parents=True)
    write_meta_file(subject_dir / "novice.openpose.stream", dim=139, frame_count=1)
    write_data_file(subject_dir / "novice.openpose.stream~", dim=139, frame_count=1)
    write_meta_file(
        subject_dir / "novice.audio.egemapsv2.stream",
        dim=88,
        frame_count=1,
        root_name="annotation",
        meta_type="eGeMAPSv02_Functionals",
    )
    write_data_file(subject_dir / "novice.audio.egemapsv2.stream~", dim=88, frame_count=1)

    results, failures = convert_subject_directory(
        input_dir,
        output_dir,
        [MODALITY_OPENPOSE, MODALITY_EGEMAPS],
    )

    assert len(results) == 2
    assert failures == []
    assert (output_dir / "086" / "novice.openpose.csv").exists()
    assert (output_dir / "086" / "novice.audio.egemapsv2.csv").exists()


def test_convert_subject_directory_skips_missing_modality_file(tmp_path: Path):
    input_dir = tmp_path / "datasets"
    output_dir = tmp_path / "multimodal_csv"
    subject_dir = input_dir / "086"
    subject_dir.mkdir(parents=True)
    write_meta_file(subject_dir / "novice.openpose.stream", dim=139, frame_count=1)
    write_data_file(subject_dir / "novice.openpose.stream~", dim=139, frame_count=1)

    results, failures = convert_subject_directory(
        input_dir,
        output_dir,
        [MODALITY_OPENPOSE, MODALITY_EGEMAPS],
    )

    assert len(results) == 1
    assert len(failures) == 1
    assert failures[0][0] == MODALITY_EGEMAPS
    assert "メタ情報ファイルが見つかりません" in failures[0][2]
