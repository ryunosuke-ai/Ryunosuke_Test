"""OpenFace ストリーム CSV 変換スクリプトのテスト。"""

import csv
import struct
from pathlib import Path

import pytest

from tools.openface_stream_to_csv import (
    EXPECTED_DIMENSION,
    SMILE_RELATED_COLUMNS,
    build_openface_column_names,
    convert_subject_directory,
    convert_stream_to_csv,
    find_subject_meta_files,
    resolve_output_path,
)


def write_meta_file(path: Path, *, dim: int = EXPECTED_DIMENSION, frame_count: int = 2) -> None:
    """テスト用 `.stream` を生成する。"""
    content = f"""<stream ssi-v ="2">
    <info ftype="BINARY" sr="25.000" dim="{dim}" byte="4" type="FLOAT" delim=" " />
    <meta type="" />
    <chunk from="0.000" to="0.080" byte="0" num="{frame_count}" />
</stream>
"""
    path.write_text(content, encoding="utf-8")


def write_data_file(path: Path, frame_count: int) -> None:
    """テスト用 `.stream~` を生成する。"""
    values = [float(index) for index in range(frame_count * EXPECTED_DIMENSION)]
    packed = struct.pack("<" + ("f" * len(values)), *values)
    path.write_bytes(packed)


def test_build_openface_column_names_contains_smile_related_columns():
    columns = build_openface_column_names()

    assert len(columns) == EXPECTED_DIMENSION
    assert "AU06_r" in columns
    assert "AU12_r" in columns
    assert "AU06_c" in columns
    assert "AU12_c" in columns


def test_convert_stream_to_csv_writes_expected_shape(tmp_path: Path):
    meta_path = tmp_path / "sample.openface2.stream"
    data_path = tmp_path / "sample.openface2.stream~"
    output_path = tmp_path / "sample.csv"
    write_meta_file(meta_path, frame_count=2)
    write_data_file(data_path, frame_count=2)

    frame_count, dimension = convert_stream_to_csv(meta_path, data_path, output_path)

    assert frame_count == 2
    assert dimension == EXPECTED_DIMENSION

    with output_path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))

    assert len(rows) == 3
    assert len(rows[0]) == EXPECTED_DIMENSION
    assert len(rows[1]) == EXPECTED_DIMENSION
    assert rows[0][0] == "frame"
    assert rows[0][-1] == "AU45_c"
    assert rows[1][0] == "0.0"
    assert rows[2][-1] == str(float((2 * EXPECTED_DIMENSION) - 1))


def test_convert_stream_to_csv_fails_for_unsupported_dimension(tmp_path: Path):
    meta_path = tmp_path / "sample.openface2.stream"
    data_path = tmp_path / "sample.openface2.stream~"
    output_path = tmp_path / "sample.csv"
    write_meta_file(meta_path, dim=713, frame_count=1)
    write_data_file(data_path, frame_count=1)

    with pytest.raises(ValueError, match="未対応の次元数"):
        convert_stream_to_csv(meta_path, data_path, output_path)


def test_convert_stream_to_csv_fails_for_short_binary(tmp_path: Path):
    meta_path = tmp_path / "sample.openface2.stream"
    data_path = tmp_path / "sample.openface2.stream~"
    output_path = tmp_path / "sample.csv"
    write_meta_file(meta_path, frame_count=2)
    write_data_file(data_path, frame_count=1)

    with pytest.raises(ValueError, match="サイズがメタ情報と一致しません"):
        convert_stream_to_csv(meta_path, data_path, output_path)


def test_convert_stream_to_csv_fails_when_data_file_is_missing(tmp_path: Path):
    meta_path = tmp_path / "sample.openface2.stream"
    output_path = tmp_path / "sample.csv"
    write_meta_file(meta_path, frame_count=1)

    with pytest.raises(FileNotFoundError, match="実データファイルが見つかりません"):
        convert_stream_to_csv(meta_path, tmp_path / "missing.stream~", output_path)


def test_convert_stream_to_csv_can_extract_smile_related_columns_only(tmp_path: Path):
    meta_path = tmp_path / "sample.openface2.stream"
    data_path = tmp_path / "sample.openface2.stream~"
    output_path = tmp_path / "smile_only.csv"
    write_meta_file(meta_path, frame_count=1)
    write_data_file(data_path, frame_count=1)

    convert_stream_to_csv(meta_path, data_path, output_path, smile_au_only=True)

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


def test_resolve_output_path_uses_default_suffixes(tmp_path: Path):
    meta_path = tmp_path / "novice.openface2.stream"

    assert resolve_output_path(meta_path, None, False) == tmp_path / "novice.openface2.csv"
    assert resolve_output_path(meta_path, None, True) == tmp_path / "novice.openface2.smile_au.csv"


def test_find_subject_meta_files_discovers_novice_openface2_only(tmp_path: Path):
    subject_dir = tmp_path / "086"
    subject_dir.mkdir()
    write_meta_file(subject_dir / "novice.openface2.stream", frame_count=1)
    write_meta_file(subject_dir / "expert.openface2.stream", frame_count=1)

    meta_files = find_subject_meta_files(tmp_path)

    assert meta_files == [subject_dir / "novice.openface2.stream"]


def test_convert_subject_directory_writes_per_subject_csv(tmp_path: Path):
    input_dir = tmp_path / "datasets"
    output_dir = tmp_path / "openface_csv"
    for subject_id in ("086", "101"):
        subject_dir = input_dir / subject_id
        subject_dir.mkdir(parents=True)
        write_meta_file(subject_dir / "novice.openface2.stream", frame_count=1)
        write_data_file(subject_dir / "novice.openface2.stream~", frame_count=1)

    results, failures = convert_subject_directory(input_dir, output_dir, smile_au_only=True)

    assert len(results) == 2
    assert failures == []
    for subject_id in ("086", "101"):
        output_path = output_dir / subject_id / "novice.openface2.smile_au.csv"
        assert output_path.exists()
        with output_path.open("r", newline="", encoding="utf-8") as file:
            rows = list(csv.reader(file))
        assert rows[0] == SMILE_RELATED_COLUMNS


def test_convert_subject_directory_skips_invalid_subject_files(tmp_path: Path):
    input_dir = tmp_path / "datasets"
    output_dir = tmp_path / "openface_csv"

    valid_dir = input_dir / "086"
    valid_dir.mkdir(parents=True)
    write_meta_file(valid_dir / "novice.openface2.stream", frame_count=1)
    write_data_file(valid_dir / "novice.openface2.stream~", frame_count=1)

    invalid_dir = input_dir / "102"
    invalid_dir.mkdir(parents=True)
    write_meta_file(invalid_dir / "novice.openface2.stream", frame_count=2)
    write_data_file(invalid_dir / "novice.openface2.stream~", frame_count=1)

    results, failures = convert_subject_directory(input_dir, output_dir, smile_au_only=True)

    assert len(results) == 1
    assert results[0][0] == valid_dir / "novice.openface2.stream"
    assert len(failures) == 1
    assert failures[0][0] == invalid_dir / "novice.openface2.stream"
    assert "サイズがメタ情報と一致しません" in failures[0][1]
