"""OpenFace / OpenPose ストリーム CSV 変換スクリプトのテスト。"""

import csv
import struct
from pathlib import Path

import pytest

from tools.multimodal_stream_to_csv import (
    MODALITY_OPENFACE,
    MODALITY_OPENPOSE,
    SMILE_RELATED_COLUMNS,
    build_openface_column_names,
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


def test_build_openface_column_names_contains_smile_related_columns():
    columns = build_openface_column_names()

    assert len(columns) == 714
    assert "AU06_r" in columns
    assert "AU12_r" in columns
    assert "AU06_c" in columns
    assert "AU12_c" in columns


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


def test_read_and_write_csv_extracts_openface_smile_columns_by_default(tmp_path: Path):
    specs = build_modality_specs()
    spec = specs[MODALITY_OPENFACE]
    meta_path = tmp_path / "novice.openface2.stream"
    data_path = tmp_path / "novice.openface2.stream~"
    output_path = tmp_path / "novice.openface2.smile_au.csv"
    write_meta_file(meta_path, dim=714, frame_count=1)
    write_data_file(data_path, dim=714, frame_count=1)

    frame_count, dimension = read_and_write_csv(meta_path, data_path, output_path, spec)

    assert frame_count == 1
    assert dimension == 714
    with output_path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))

    assert rows[0] == SMILE_RELATED_COLUMNS
    assert len(rows[1]) == len(SMILE_RELATED_COLUMNS)

    all_columns = build_openface_column_names()
    expected_values = {
        name: str(float(all_columns.index(name)))
        for name in SMILE_RELATED_COLUMNS
    }
    assert rows[1][0] == expected_values["frame"]
    assert rows[1][1] == expected_values["timestamp"]
    assert rows[1][2] == expected_values["AU06_r"]
    assert rows[1][3] == expected_values["AU06_c"]
    assert rows[1][4] == expected_values["AU12_r"]
    assert rows[1][5] == expected_values["AU12_c"]


def test_read_and_write_csv_can_write_all_openface_columns(tmp_path: Path):
    specs = build_modality_specs(openface_all_columns=True)
    spec = specs[MODALITY_OPENFACE]
    meta_path = tmp_path / "novice.openface2.stream"
    data_path = tmp_path / "novice.openface2.stream~"
    output_path = tmp_path / "novice.openface2.csv"
    write_meta_file(meta_path, dim=714, frame_count=1)
    write_data_file(data_path, dim=714, frame_count=1)

    read_and_write_csv(meta_path, data_path, output_path, spec)

    with output_path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))

    assert len(rows[0]) == 714
    assert rows[0][0] == "frame"
    assert rows[0][-1] == "AU45_c"


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


def test_read_and_write_csv_fails_for_wrong_dimension(tmp_path: Path):
    spec = build_modality_specs()[MODALITY_OPENPOSE]
    meta_path = tmp_path / "novice.openpose.stream"
    data_path = tmp_path / "novice.openpose.stream~"
    output_path = tmp_path / "novice.openpose.csv"
    write_meta_file(meta_path, dim=138, frame_count=1)
    write_data_file(data_path, dim=138, frame_count=1)

    with pytest.raises(ValueError, match="次元数が想定と異なります"):
        read_and_write_csv(meta_path, data_path, output_path, spec)


def test_read_and_write_csv_fails_when_data_file_is_missing(tmp_path: Path):
    spec = build_modality_specs()[MODALITY_OPENFACE]
    meta_path = tmp_path / "novice.openface2.stream"
    output_path = tmp_path / "novice.openface2.smile_au.csv"
    write_meta_file(meta_path, dim=714, frame_count=1)

    with pytest.raises(FileNotFoundError, match="実データファイルが見つかりません"):
        read_and_write_csv(meta_path, tmp_path / "missing.stream~", output_path, spec)


def test_read_and_write_csv_fails_for_short_binary(tmp_path: Path):
    spec = build_modality_specs()[MODALITY_OPENPOSE]
    meta_path = tmp_path / "novice.openpose.stream"
    data_path = tmp_path / "novice.openpose.stream~"
    output_path = tmp_path / "novice.openpose.csv"
    write_meta_file(meta_path, dim=139, frame_count=2)
    write_data_file(data_path, dim=139, frame_count=1)

    with pytest.raises(ValueError, match="サイズがメタ情報と一致しません"):
        read_and_write_csv(meta_path, data_path, output_path, spec)


def test_convert_subject_directory_writes_openface_and_openpose_csv(tmp_path: Path):
    input_dir = tmp_path / "datasets"
    output_dir = tmp_path / "multimodal_csv"
    unrelated_dir = input_dir / "openface_csv"
    unrelated_dir.mkdir(parents=True)
    subject_dir = input_dir / "086"
    subject_dir.mkdir(parents=True)
    write_meta_file(subject_dir / "novice.openface2.stream", dim=714, frame_count=1)
    write_data_file(subject_dir / "novice.openface2.stream~", dim=714, frame_count=1)
    write_meta_file(subject_dir / "novice.openpose.stream", dim=139, frame_count=1)
    write_data_file(subject_dir / "novice.openpose.stream~", dim=139, frame_count=1)

    results, failures = convert_subject_directory(
        input_dir,
        output_dir,
        [MODALITY_OPENFACE, MODALITY_OPENPOSE],
    )

    assert len(results) == 2
    assert failures == []
    assert (output_dir / "086" / "novice.openface2.smile_au.csv").exists()
    assert (output_dir / "086" / "novice.openpose.csv").exists()


def test_convert_subject_directory_can_write_only_openpose(tmp_path: Path):
    input_dir = tmp_path / "datasets"
    output_dir = tmp_path / "multimodal_csv"
    subject_dir = input_dir / "086"
    subject_dir.mkdir(parents=True)
    write_meta_file(subject_dir / "novice.openface2.stream", dim=714, frame_count=1)
    write_data_file(subject_dir / "novice.openface2.stream~", dim=714, frame_count=1)
    write_meta_file(subject_dir / "novice.openpose.stream", dim=139, frame_count=1)
    write_data_file(subject_dir / "novice.openpose.stream~", dim=139, frame_count=1)

    results, failures = convert_subject_directory(input_dir, output_dir, [MODALITY_OPENPOSE])

    assert len(results) == 1
    assert failures == []
    assert results[0].modality == MODALITY_OPENPOSE
    assert not (output_dir / "086" / "novice.openface2.smile_au.csv").exists()
    assert (output_dir / "086" / "novice.openpose.csv").exists()


def test_convert_subject_directory_can_write_all_openface_columns(tmp_path: Path):
    input_dir = tmp_path / "datasets"
    output_dir = tmp_path / "multimodal_csv"
    subject_dir = input_dir / "086"
    subject_dir.mkdir(parents=True)
    write_meta_file(subject_dir / "novice.openface2.stream", dim=714, frame_count=1)
    write_data_file(subject_dir / "novice.openface2.stream~", dim=714, frame_count=1)

    results, failures = convert_subject_directory(
        input_dir,
        output_dir,
        [MODALITY_OPENFACE],
        openface_all_columns=True,
    )

    assert len(results) == 1
    assert failures == []
    output_path = output_dir / "086" / "novice.openface2.csv"
    with output_path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))
    assert len(rows[0]) == 714


def test_convert_subject_directory_ignores_dirs_without_target_stream(tmp_path: Path):
    input_dir = tmp_path / "datasets"
    output_dir = tmp_path / "multimodal_csv"
    subject_dir = input_dir / "086"
    subject_dir.mkdir(parents=True)

    with pytest.raises(ValueError, match="対象 stream を含む被験者ディレクトリ"):
        convert_subject_directory(input_dir, output_dir, [MODALITY_OPENFACE, MODALITY_OPENPOSE])
