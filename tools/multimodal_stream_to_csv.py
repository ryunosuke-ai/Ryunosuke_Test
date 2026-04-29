"""OpenPose / eGeMAPS の SSI ストリームを CSV に変換する CLI。"""

from __future__ import annotations

import argparse
import csv
import struct
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_INPUT_DIR = "datasets"
DEFAULT_OUTPUT_DIR_NAME = "multimodal_csv"
FLOAT32_BYTE_SIZE = 4

MODALITY_OPENPOSE = "openpose"
MODALITY_EGEMAPS = "egemaps"
ALL_MODALITIES = (MODALITY_OPENPOSE, MODALITY_EGEMAPS)


@dataclass(frozen=True)
class StreamMetadata:
    """`.stream` に含まれるメタ情報。"""

    sample_rate: float
    dimension: int
    byte_size: int
    value_type: str
    frame_count: int
    start_time: float
    end_time: float
    meta_type: str


@dataclass(frozen=True)
class ModalitySpec:
    """変換対象モダリティの仕様。"""

    name: str
    stream_name: str
    expected_dimension: int
    output_filename: str
    columns: list[str]
    expected_meta_type: str | None = None


@dataclass(frozen=True)
class ConversionResult:
    """1 ファイル分の変換結果。"""

    modality: str
    meta_path: Path
    output_path: Path
    frame_count: int
    dimension: int


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""
    parser = argparse.ArgumentParser(
        description="OpenPose / eGeMAPS の `.stream` / `.stream~` を CSV に変換します。"
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help=f"被験者ごとのフォルダを含むディレクトリ（既定: {DEFAULT_INPUT_DIR}）。",
    )
    parser.add_argument(
        "--output-dir",
        help=f"出力先ルート。省略時は `--input-dir/{DEFAULT_OUTPUT_DIR_NAME}` を使います。",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=ALL_MODALITIES,
        default=list(ALL_MODALITIES),
        help="変換するモダリティ。複数指定できます（既定: openpose egemaps）。",
    )
    return parser.parse_args()


def build_openpose_column_names() -> list[str]:
    """OpenPose 139 次元の列名を生成する。"""
    body_parts = [
        "nose",
        "neck",
        "r_shoulder",
        "r_elbow",
        "r_wrist",
        "l_shoulder",
        "l_elbow",
        "l_wrist",
        "r_hip",
        "r_knee",
        "r_ankle",
        "l_hip",
        "l_knee",
        "l_ankle",
        "r_eye",
        "l_eye",
        "r_ear",
        "l_ear",
    ]
    columns: list[str] = []
    for body_part in body_parts:
        columns.extend([f"{body_part}_x", f"{body_part}_y", f"{body_part}_conf"])

    extra_count = 139 - len(columns)
    columns.extend(f"openpose_extra_{index:03d}" for index in range(extra_count))
    return columns


def build_egemaps_column_names() -> list[str]:
    """eGeMAPSv02 Functionals 88 次元の列名を生成する。"""
    return [f"egemaps_{index:03d}" for index in range(88)]


def build_modality_specs() -> dict[str, ModalitySpec]:
    """対応モダリティの仕様を返す。"""
    return {
        MODALITY_OPENPOSE: ModalitySpec(
            name=MODALITY_OPENPOSE,
            stream_name="novice.openpose.stream",
            expected_dimension=139,
            output_filename="novice.openpose.csv",
            columns=build_openpose_column_names(),
        ),
        MODALITY_EGEMAPS: ModalitySpec(
            name=MODALITY_EGEMAPS,
            stream_name="novice.audio.egemapsv2.stream",
            expected_dimension=88,
            output_filename="novice.audio.egemapsv2.csv",
            columns=build_egemaps_column_names(),
            expected_meta_type="eGeMAPSv02_Functionals",
        ),
    }


def load_stream_metadata(meta_path: Path) -> StreamMetadata:
    """`.stream` ファイルからメタ情報を読み取る。"""
    try:
        root = ET.fromstring(meta_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"メタ情報ファイルが見つかりません: {meta_path}") from exc
    except ET.ParseError as exc:
        raise ValueError(f"メタ情報ファイルの XML を解析できません: {meta_path}") from exc

    info = root.find("info")
    chunk = root.find("chunk")
    meta = root.find("meta")
    if info is None or chunk is None:
        raise ValueError("`.stream` に `<info>` または `<chunk>` 要素がありません。")

    try:
        return StreamMetadata(
            sample_rate=float(info.attrib["sr"]),
            dimension=int(info.attrib["dim"]),
            byte_size=int(info.attrib["byte"]),
            value_type=info.attrib["type"],
            frame_count=int(chunk.attrib["num"]),
            start_time=float(chunk.attrib["from"]),
            end_time=float(chunk.attrib["to"]),
            meta_type=meta.attrib.get("type", "") if meta is not None else "",
        )
    except KeyError as exc:
        raise ValueError(f"`.stream` の必須属性が不足しています: {exc}") from exc
    except ValueError as exc:
        raise ValueError("`.stream` の属性値を数値として解釈できません。") from exc


def resolve_data_path(meta_path: Path) -> Path:
    """実データファイルのパスを確定する。"""
    return Path(f"{meta_path}~")


def resolve_output_root(input_dir: Path, explicit_output_dir: str | None) -> Path:
    """一括変換時の出力ルートを確定する。"""
    if explicit_output_dir:
        return Path(explicit_output_dir)
    return input_dir / DEFAULT_OUTPUT_DIR_NAME


def validate_metadata(metadata: StreamMetadata, spec: ModalitySpec) -> None:
    """対応しているメタ情報か確認する。"""
    if metadata.dimension != spec.expected_dimension:
        raise ValueError(
            f"{spec.name} の次元数が想定と異なります: "
            f"{metadata.dimension}。想定は {spec.expected_dimension} です。"
        )
    if metadata.byte_size != FLOAT32_BYTE_SIZE:
        raise ValueError(
            f"未対応の byte サイズです: {metadata.byte_size}。"
            f"float32 ({FLOAT32_BYTE_SIZE} byte) を想定しています。"
        )
    if metadata.value_type.upper() != "FLOAT":
        raise ValueError(f"未対応の型です: {metadata.value_type}。`type=\"FLOAT\"` を想定しています。")
    if spec.expected_meta_type is not None and metadata.meta_type != spec.expected_meta_type:
        raise ValueError(
            f"{spec.name} の meta type が想定と異なります: "
            f"{metadata.meta_type}。想定は {spec.expected_meta_type} です。"
        )


def read_and_write_csv(meta_path: Path, data_path: Path, output_path: Path, spec: ModalitySpec) -> tuple[int, int]:
    """1 つの SSI ストリームを CSV に変換する。"""
    metadata = load_stream_metadata(meta_path)
    validate_metadata(metadata, spec)

    if len(spec.columns) != metadata.dimension:
        raise ValueError(
            f"{spec.name} の列名数が次元数と一致しません: {len(spec.columns)} != {metadata.dimension}"
        )
    if not data_path.exists():
        raise FileNotFoundError(f"実データファイルが見つかりません: {data_path}")

    expected_file_size = metadata.frame_count * metadata.dimension * metadata.byte_size
    actual_file_size = data_path.stat().st_size
    if actual_file_size != expected_file_size:
        raise ValueError(
            "実データのサイズがメタ情報と一致しません。"
            f" 期待値: {expected_file_size} byte, 実際: {actual_file_size} byte"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_format = "<" + ("f" * metadata.dimension)
    row_size = struct.calcsize(row_format)

    with data_path.open("rb") as data_file, output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["frame", "timestamp", *spec.columns])

        for frame_index in range(metadata.frame_count):
            row_bytes = data_file.read(row_size)
            if len(row_bytes) != row_size:
                raise ValueError(f"{frame_index} 行目の読み取り中にデータ長が不足しました。")
            timestamp = metadata.start_time + (frame_index / metadata.sample_rate)
            writer.writerow([frame_index + 1, timestamp, *struct.unpack(row_format, row_bytes)])

        extra_bytes = data_file.read(1)
        if extra_bytes:
            raise ValueError("実データ末尾に余分なバイトが含まれています。")

    return metadata.frame_count, metadata.dimension


def find_subject_dirs(input_dir: Path) -> list[Path]:
    """被験者ディレクトリを列挙する。"""
    if not input_dir.exists():
        raise FileNotFoundError(f"入力ディレクトリが見つかりません: {input_dir}")
    if not input_dir.is_dir():
        raise ValueError(f"入力パスがディレクトリではありません: {input_dir}")
    return sorted(path for path in input_dir.iterdir() if path.is_dir())


def convert_subject_directory(
    input_dir: Path,
    output_root: Path,
    modality_names: Iterable[str],
) -> tuple[list[ConversionResult], list[tuple[str, Path, str]]]:
    """被験者ごとに OpenPose / eGeMAPS を CSV に変換する。"""
    specs = build_modality_specs()
    modality_names = list(modality_names)
    unknown_modalities = [name for name in modality_names if name not in specs]
    if unknown_modalities:
        raise ValueError(f"未対応のモダリティです: {', '.join(unknown_modalities)}")

    subject_dirs = [
        subject_dir
        for subject_dir in find_subject_dirs(input_dir)
        if any((subject_dir / specs[modality_name].stream_name).exists() for modality_name in modality_names)
    ]
    if not subject_dirs:
        raise ValueError(f"`{input_dir}` 配下に対象 stream を含む被験者ディレクトリが見つかりませんでした。")

    results: list[ConversionResult] = []
    failures: list[tuple[str, Path, str]] = []
    for subject_dir in subject_dirs:
        for modality_name in modality_names:
            spec = specs[modality_name]
            meta_path = subject_dir / spec.stream_name
            data_path = resolve_data_path(meta_path)
            output_path = output_root / subject_dir.name / spec.output_filename

            try:
                frame_count, dimension = read_and_write_csv(meta_path, data_path, output_path, spec)
            except (FileNotFoundError, ValueError) as exc:
                failures.append((modality_name, meta_path, str(exc)))
                continue

            results.append(
                ConversionResult(
                    modality=modality_name,
                    meta_path=meta_path,
                    output_path=output_path,
                    frame_count=frame_count,
                    dimension=dimension,
                )
            )

    return results, failures


def print_summary(results: Iterable[ConversionResult], failures: Iterable[tuple[str, Path, str]]) -> None:
    """変換結果を表示する。"""
    results = list(results)
    failures = list(failures)
    print(f"変換件数: {len(results)}")
    print(f"スキップ件数: {len(failures)}")
    for result in results:
        print(
            f"{result.modality}: {result.meta_path} -> {result.output_path} "
            f"({result.frame_count} フレーム, {result.dimension} 次元)"
        )
    for modality, meta_path, reason in failures:
        print(f"スキップ: {modality}: {meta_path} ({reason})")


def main() -> int:
    """CLI のエントリーポイント。"""
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_root = resolve_output_root(input_dir, args.output_dir)

    try:
        results, failures = convert_subject_directory(input_dir, output_root, args.modalities)
    except (FileNotFoundError, ValueError) as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1

    print_summary(results, failures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
