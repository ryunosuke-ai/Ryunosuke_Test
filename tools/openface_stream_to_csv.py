"""OpenFace の SSI ストリームを CSV に変換する CLI。"""

from __future__ import annotations

import argparse
import csv
import struct
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


EXPECTED_DIMENSION = 714
FLOAT32_BYTE_SIZE = 4
SMILE_RELATED_COLUMNS = ["frame", "timestamp", "AU06_r", "AU06_c", "AU12_r", "AU12_c"]


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


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""
    parser = argparse.ArgumentParser(
        description="OpenFace の `.stream` / `.stream~` を列名付き CSV に変換します。"
    )
    parser.add_argument(
        "--meta",
        required=True,
        help="SSI メタ情報ファイル (`.stream`) のパス。",
    )
    parser.add_argument(
        "--data",
        help="実データファイル (`.stream~`) のパス。省略時は `--meta` に `~` を付けたパスを使います。",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="出力する CSV ファイルのパス。",
    )
    parser.add_argument(
        "--smile-au-only",
        action="store_true",
        help="`frame`, `timestamp`, `AU06_*`, `AU12_*` のみを抽出して出力します。",
    )
    return parser.parse_args()


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
        )
    except KeyError as exc:
        raise ValueError(f"`.stream` の必須属性が不足しています: {exc}") from exc
    except ValueError as exc:
        raise ValueError("`.stream` の属性値を数値として解釈できません。") from exc


def build_openface_column_names() -> list[str]:
    """714 次元の OpenFace 列名一覧を生成する。"""
    columns = [
        "frame",
        "face_id",
        "timestamp",
        "confidence",
        "success",
        "gaze_0_x",
        "gaze_0_y",
        "gaze_0_z",
        "gaze_1_x",
        "gaze_1_y",
        "gaze_1_z",
        "gaze_angle_x",
        "gaze_angle_y",
    ]

    for prefix in ("eye_lmk_x", "eye_lmk_y", "eye_lmk_X", "eye_lmk_Y", "eye_lmk_Z"):
        columns.extend(f"{prefix}_{index}" for index in range(56))

    columns.extend(
        [
            "pose_Tx",
            "pose_Ty",
            "pose_Tz",
            "pose_Rx",
            "pose_Ry",
            "pose_Rz",
        ]
    )

    for prefix in ("x", "y", "X", "Y", "Z"):
        columns.extend(f"{prefix}_{index}" for index in range(68))

    columns.extend(["p_scale", "p_rx", "p_ry", "p_rz", "p_tx", "p_ty"])
    columns.extend(f"p_{index}" for index in range(34))

    columns.extend(
        [
            "AU01_r",
            "AU02_r",
            "AU04_r",
            "AU05_r",
            "AU06_r",
            "AU07_r",
            "AU09_r",
            "AU10_r",
            "AU12_r",
            "AU14_r",
            "AU15_r",
            "AU17_r",
            "AU20_r",
            "AU23_r",
            "AU25_r",
            "AU26_r",
            "AU45_r",
        ]
    )
    columns.extend(
        [
            "AU01_c",
            "AU02_c",
            "AU04_c",
            "AU05_c",
            "AU06_c",
            "AU07_c",
            "AU09_c",
            "AU10_c",
            "AU12_c",
            "AU14_c",
            "AU15_c",
            "AU17_c",
            "AU20_c",
            "AU23_c",
            "AU25_c",
            "AU26_c",
            "AU28_c",
            "AU45_c",
        ]
    )

    if len(columns) != EXPECTED_DIMENSION:
        raise AssertionError(f"列数が {EXPECTED_DIMENSION} になっていません: {len(columns)}")
    return columns


def resolve_data_path(meta_path: Path, explicit_data_path: str | None) -> Path:
    """実データファイルのパスを確定する。"""
    if explicit_data_path:
        return Path(explicit_data_path)
    return Path(f"{meta_path}~")


def build_selected_column_indices(all_columns: list[str], smile_au_only: bool) -> list[int]:
    """出力対象列のインデックス一覧を返す。"""
    if not smile_au_only:
        return list(range(len(all_columns)))

    return [all_columns.index(column_name) for column_name in SMILE_RELATED_COLUMNS]


def validate_metadata(metadata: StreamMetadata) -> None:
    """対応しているメタ情報か確認する。"""
    if metadata.dimension != EXPECTED_DIMENSION:
        raise ValueError(
            f"未対応の次元数です: {metadata.dimension}。このスクリプトは {EXPECTED_DIMENSION} 次元専用です。"
        )
    if metadata.byte_size != FLOAT32_BYTE_SIZE:
        raise ValueError(
            f"未対応の byte サイズです: {metadata.byte_size}。float32 ({FLOAT32_BYTE_SIZE} byte) を想定しています。"
        )
    if metadata.value_type.upper() != "FLOAT":
        raise ValueError(
            f"未対応の型です: {metadata.value_type}。`type=\"FLOAT\"` を想定しています。"
        )


def convert_stream_to_csv(
    meta_path: Path,
    data_path: Path,
    output_path: Path,
    *,
    smile_au_only: bool = False,
) -> tuple[int, int]:
    """OpenFace のストリームを CSV に変換する。"""
    metadata = load_stream_metadata(meta_path)
    validate_metadata(metadata)
    columns = build_openface_column_names()
    selected_indices = build_selected_column_indices(columns, smile_au_only)
    selected_columns = [columns[index] for index in selected_indices]

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
        writer.writerow(selected_columns)

        for frame_index in range(metadata.frame_count):
            row_bytes = data_file.read(row_size)
            if len(row_bytes) != row_size:
                raise ValueError(
                    f"{frame_index} 行目の読み取り中にデータ長が不足しました。"
                )
            row_values = struct.unpack(row_format, row_bytes)
            writer.writerow([row_values[index] for index in selected_indices])

        extra_bytes = data_file.read(1)
        if extra_bytes:
            raise ValueError("実データ末尾に余分なバイトが含まれています。")

    return metadata.frame_count, metadata.dimension


def main() -> int:
    """CLI のエントリーポイント。"""
    args = parse_args()
    meta_path = Path(args.meta)
    data_path = resolve_data_path(meta_path, args.data)
    output_path = Path(args.output)

    try:
        frame_count, dimension = convert_stream_to_csv(
            meta_path,
            data_path,
            output_path,
            smile_au_only=args.smile_au_only,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1

    print(f"変換完了: {output_path}")
    print(f"フレーム数: {frame_count}")
    print(f"次元数: {dimension}")
    if args.smile_au_only:
        print(f"抽出列: {', '.join(SMILE_RELATED_COLUMNS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
