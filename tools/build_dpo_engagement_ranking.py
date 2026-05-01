"""NoXi+J から DPO 候補用の engagement 増加ランキング CSV を作成する。"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_DATASET_DIR = "datasets"
DEFAULT_MULTIMODAL_DIR = "datasets/multimodal_csv"
DEFAULT_OUTPUT_PATH = "artifacts/noxij_dpo_multimodal_score.csv"
DEFAULT_WINDOW_SEC = 5.0
DEFAULT_SAMPLE_RATE = 25.0
DEFAULT_CONFIDENCE_THRESHOLD = 0.3
ZSCORE_CLIP = 3.0

EXPERT_TRANSCRIPT_NAME = "expert.audio.transcript.annotation.csv"
NOVICE_TRANSCRIPT_NAME = "novice.audio.transcript.annotation.csv"
NOVICE_ENGAGEMENT_NAME = "novice.engagement.annotation.csv"
OPENFACE_NAME = "novice.openface2.smile_au.csv"
OPENPOSE_NAME = "novice.openpose.csv"

CSV_COLUMNS = [
    "session_id",
    "expert_start_sec",
    "expert_end_sec",
    "novice_start_sec",
    "novice_end_sec",
    "before_engagement_mean",
    "after_engagement_mean",
    "engagement_delta",
    "expert_text",
    "novice_text",
    "before_frame_count",
    "after_frame_count",
    "final_score",
    "engagement_delta_norm",
    "smile_score_raw",
    "smile_score_norm",
    "head_motion_score_raw",
    "head_motion_score_norm",
    "upper_body_activation_score_raw",
    "upper_body_activation_score_norm",
    "hand_energy_score_raw",
    "hand_energy_score_norm",
    "au12_after_mean",
    "au06_after_mean",
    "au12_after_ratio",
    "au06_after_ratio",
    "openface_frame_count",
    "openpose_frame_pair_count",
]


@dataclass(frozen=True)
class Utterance:
    """transcript 由来の 1 発話。"""

    session_id: str
    speaker: str
    start_sec: float
    end_sec: float
    text: str


@dataclass(frozen=True)
class OpenFaceFeatures:
    """OpenFace から計算した表情特徴。"""

    smile_score_raw: float | None
    au12_after_mean: float | None
    au06_after_mean: float | None
    au12_after_ratio: float | None
    au06_after_ratio: float | None
    frame_count: int


@dataclass(frozen=True)
class OpenPoseFeatures:
    """OpenPose から計算した身体動作特徴。"""

    head_motion_score_raw: float | None
    upper_body_activation_score_raw: float | None
    hand_energy_score_raw: float | None
    frame_pair_count: int


@dataclass(frozen=True)
class EngagementDeltaRow:
    """expert 返答前後の novice engagement 差分。"""

    session_id: str
    expert_start_sec: float
    expert_end_sec: float
    novice_start_sec: float
    novice_end_sec: float
    before_engagement_mean: float
    after_engagement_mean: float
    engagement_delta: float
    expert_text: str
    novice_text: str
    before_frame_count: int
    after_frame_count: int
    smile_score_raw: float | None = None
    head_motion_score_raw: float | None = None
    upper_body_activation_score_raw: float | None = None
    hand_energy_score_raw: float | None = None
    au12_after_mean: float | None = None
    au06_after_mean: float | None = None
    au12_after_ratio: float | None = None
    au06_after_ratio: float | None = None
    openface_frame_count: int = 0
    openpose_frame_pair_count: int = 0
    engagement_delta_norm: float = 0.5
    smile_score_norm: float = 0.5
    head_motion_score_norm: float = 0.5
    upper_body_activation_score_norm: float = 0.5
    hand_energy_score_norm: float = 0.5
    final_score: float = 0.5

    def to_csv_row(self) -> dict[str, str]:
        """CSV 出力用の文字列表現に変換する。"""
        return {
            "session_id": self.session_id,
            "expert_start_sec": format_float(self.expert_start_sec),
            "expert_end_sec": format_float(self.expert_end_sec),
            "novice_start_sec": format_float(self.novice_start_sec),
            "novice_end_sec": format_float(self.novice_end_sec),
            "before_engagement_mean": format_float(self.before_engagement_mean),
            "after_engagement_mean": format_float(self.after_engagement_mean),
            "engagement_delta": format_float(self.engagement_delta),
            "expert_text": self.expert_text,
            "novice_text": self.novice_text,
            "before_frame_count": str(self.before_frame_count),
            "after_frame_count": str(self.after_frame_count),
            "final_score": format_float(self.final_score),
            "engagement_delta_norm": format_float(self.engagement_delta_norm),
            "smile_score_raw": format_optional_float(self.smile_score_raw),
            "smile_score_norm": format_float(self.smile_score_norm),
            "head_motion_score_raw": format_optional_float(self.head_motion_score_raw),
            "head_motion_score_norm": format_float(self.head_motion_score_norm),
            "upper_body_activation_score_raw": format_optional_float(self.upper_body_activation_score_raw),
            "upper_body_activation_score_norm": format_float(self.upper_body_activation_score_norm),
            "hand_energy_score_raw": format_optional_float(self.hand_energy_score_raw),
            "hand_energy_score_norm": format_float(self.hand_energy_score_norm),
            "au12_after_mean": format_optional_float(self.au12_after_mean),
            "au06_after_mean": format_optional_float(self.au06_after_mean),
            "au12_after_ratio": format_optional_float(self.au12_after_ratio),
            "au06_after_ratio": format_optional_float(self.au06_after_ratio),
            "openface_frame_count": str(self.openface_frame_count),
            "openpose_frame_pair_count": str(self.openpose_frame_pair_count),
        }


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""
    parser = argparse.ArgumentParser(
        description="expert 返答前後の novice engagement 差分を計算し、降順 CSV を作成します。"
    )
    parser.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
        help=f"NoXi+J データセットのルートディレクトリ（既定: {DEFAULT_DATASET_DIR}）。",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help=f"ランキング CSV の出力先（既定: {DEFAULT_OUTPUT_PATH}）。",
    )
    parser.add_argument(
        "--multimodal-dir",
        default=DEFAULT_MULTIMODAL_DIR,
        help=f"変換済み OpenFace/OpenPose CSV のルートディレクトリ（既定: {DEFAULT_MULTIMODAL_DIR}）。",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=DEFAULT_WINDOW_SEC,
        help=f"返答前後の平均 engagement を計算する秒数（既定: {DEFAULT_WINDOW_SEC}）。",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=DEFAULT_SAMPLE_RATE,
        help=f"engagement annotation のフレームレート（既定: {DEFAULT_SAMPLE_RATE}）。",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help=f"OpenPose の関節点を有効とみなす confidence 閾値（既定: {DEFAULT_CONFIDENCE_THRESHOLD}）。",
    )
    return parser.parse_args()


def format_float(value: float) -> str:
    """CSV で読みやすい固定精度の数値文字列を返す。"""
    return f"{value:.6f}"


def format_optional_float(value: float | None) -> str:
    """欠損を空文字として出力する。"""
    if value is None:
        return ""
    return format_float(value)


def read_transcript(path: Path, *, session_id: str, speaker: str) -> list[Utterance]:
    """transcript annotation CSV を読み込む。"""
    utterances: list[Utterance] = []
    try:
        with path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.reader(file, delimiter=";")
            for line_number, row in enumerate(reader, start=1):
                if not row or all(not column.strip() for column in row):
                    continue
                if len(row) < 3:
                    raise ValueError(f"{path}:{line_number} の列数が不足しています。")
                try:
                    start_sec = float(row[0])
                    end_sec = float(row[1])
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_number} の時刻を数値として解釈できません。") from exc
                text = ";".join(row[2:-1] if len(row) > 3 else row[2:]).strip()
                utterances.append(
                    Utterance(
                        session_id=session_id,
                        speaker=speaker,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        text=text,
                    )
                )
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"transcript ファイルが見つかりません: {path}") from exc
    return utterances


def read_engagement(path: Path) -> list[float]:
    """novice engagement annotation を読み込む。"""
    values: list[float] = []
    try:
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                value = line.strip()
                if not value:
                    continue
                try:
                    values.append(float(value))
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_number} の engagement を数値として解釈できません。") from exc
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"engagement ファイルが見つかりません: {path}") from exc
    return values


def mean_in_window(
    values: list[float],
    *,
    start_sec: float,
    end_sec: float,
    sample_rate: float,
) -> tuple[float | None, int]:
    """指定時間窓に含まれる engagement 平均とフレーム数を返す。"""
    if end_sec <= start_sec:
        return None, 0

    start_index = max(0, math.ceil(start_sec * sample_rate - 1e-9))
    end_index = min(len(values), math.ceil(end_sec * sample_rate - 1e-9))
    if end_index <= start_index:
        return None, 0

    window_values = values[start_index:end_index]
    return sum(window_values) / len(window_values), len(window_values)


def read_dict_rows(path: Path) -> list[dict[str, str]]:
    """ヘッダー付き CSV を辞書のリストとして読み込む。"""
    try:
        with path.open("r", newline="", encoding="utf-8") as file:
            return list(csv.DictReader(file))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"CSV ファイルが見つかりません: {path}") from exc


def rows_in_time_window(rows: list[dict[str, str]], *, start_sec: float, end_sec: float) -> list[dict[str, str]]:
    """timestamp 列が指定時間窓に含まれる行を返す。"""
    selected_rows: list[dict[str, str]] = []
    for row in rows:
        try:
            timestamp = float(row["timestamp"])
        except (KeyError, ValueError) as exc:
            raise ValueError("CSV の timestamp 列を数値として解釈できません。") from exc
        if start_sec <= timestamp < end_sec:
            selected_rows.append(row)
    return selected_rows


def mean_numeric_column(rows: list[dict[str, str]], column: str) -> float | None:
    """指定列の平均を返す。"""
    values: list[float] = []
    for row in rows:
        try:
            values.append(float(row[column]))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"CSV の {column} 列を数値として解釈できません。") from exc
    if not values:
        return None
    return sum(values) / len(values)


def ratio_positive_column(rows: list[dict[str, str]], column: str) -> float | None:
    """指定列が 0 より大きいフレーム比率を返す。"""
    values: list[float] = []
    for row in rows:
        try:
            values.append(float(row[column]))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"CSV の {column} 列を数値として解釈できません。") from exc
    if not values:
        return None
    return sum(1 for value in values if value > 0.0) / len(values)


def compute_openface_features(
    rows: list[dict[str, str]],
    *,
    start_sec: float,
    end_sec: float,
) -> OpenFaceFeatures:
    """expert 返答後窓の OpenFace 笑顔特徴を計算する。"""
    window_rows = rows_in_time_window(rows, start_sec=start_sec, end_sec=end_sec)
    au12_after_mean = mean_numeric_column(window_rows, "AU12_r")
    au06_after_mean = mean_numeric_column(window_rows, "AU06_r")
    au12_after_ratio = ratio_positive_column(window_rows, "AU12_c")
    au06_after_ratio = ratio_positive_column(window_rows, "AU06_c")
    if au12_after_mean is None or au06_after_mean is None:
        smile_score_raw = None
    else:
        smile_score_raw = 0.7 * au12_after_mean + 0.3 * au06_after_mean
    return OpenFaceFeatures(
        smile_score_raw=smile_score_raw,
        au12_after_mean=au12_after_mean,
        au06_after_mean=au06_after_mean,
        au12_after_ratio=au12_after_ratio,
        au06_after_ratio=au06_after_ratio,
        frame_count=len(window_rows),
    )


def point_from_row(row: dict[str, str], point_name: str, *, confidence_threshold: float) -> tuple[float, float] | None:
    """OpenPose 行から有効な関節点座標を取り出す。"""
    try:
        confidence = float(row[f"{point_name}_conf"])
        x = float(row[f"{point_name}_x"])
        y = float(row[f"{point_name}_y"])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"OpenPose CSV の {point_name} 座標を数値として解釈できません。") from exc
    if confidence < confidence_threshold or x < 0.0 or y < 0.0:
        return None
    return x, y


def shoulder_scale(row: dict[str, str], *, confidence_threshold: float) -> float | None:
    """左右肩幅を正規化スケールとして返す。"""
    right = point_from_row(row, "r_shoulder", confidence_threshold=confidence_threshold)
    left = point_from_row(row, "l_shoulder", confidence_threshold=confidence_threshold)
    if right is None or left is None:
        return None
    distance = euclidean_distance(right, left)
    if distance <= 0.0:
        return None
    return distance


def euclidean_distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    """2点間距離を返す。"""
    return math.sqrt((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2)


def normalized_movement(
    previous_row: dict[str, str],
    current_row: dict[str, str],
    point_names: list[str],
    *,
    confidence_threshold: float,
) -> float | None:
    """指定関節群の平均フレーム間移動量を肩幅で正規化して返す。"""
    scale = shoulder_scale(current_row, confidence_threshold=confidence_threshold)
    if scale is None:
        return None

    movements: list[float] = []
    for point_name in point_names:
        previous_point = point_from_row(previous_row, point_name, confidence_threshold=confidence_threshold)
        current_point = point_from_row(current_row, point_name, confidence_threshold=confidence_threshold)
        if previous_point is None or current_point is None:
            continue
        movements.append(euclidean_distance(previous_point, current_point) / scale)
    if not movements:
        return None
    return sum(movements) / len(movements)


def mean_optional(values: list[float]) -> float | None:
    """空リストなら None、値があれば平均を返す。"""
    if not values:
        return None
    return sum(values) / len(values)


def compute_openpose_features(
    rows: list[dict[str, str]],
    *,
    start_sec: float,
    end_sec: float,
    confidence_threshold: float,
) -> OpenPoseFeatures:
    """expert 返答後窓の OpenPose 身体動作特徴を計算する。"""
    window_rows = rows_in_time_window(rows, start_sec=start_sec, end_sec=end_sec)
    if len(window_rows) < 2:
        return OpenPoseFeatures(None, None, None, 0)

    head_points = ["nose", "neck", "r_eye", "l_eye", "r_ear", "l_ear"]
    upper_body_points = ["neck", "r_shoulder", "l_shoulder", "r_elbow", "l_elbow", "r_wrist", "l_wrist"]
    hand_points = ["r_elbow", "l_elbow", "r_wrist", "l_wrist"]
    head_movements: list[float] = []
    upper_body_movements: list[float] = []
    hand_movements: list[float] = []
    valid_pair_count = 0

    for previous_row, current_row in zip(window_rows, window_rows[1:]):
        head_movement = normalized_movement(
            previous_row,
            current_row,
            head_points,
            confidence_threshold=confidence_threshold,
        )
        upper_body_movement = normalized_movement(
            previous_row,
            current_row,
            upper_body_points,
            confidence_threshold=confidence_threshold,
        )
        hand_movement = normalized_movement(
            previous_row,
            current_row,
            hand_points,
            confidence_threshold=confidence_threshold,
        )
        if head_movement is not None or upper_body_movement is not None or hand_movement is not None:
            valid_pair_count += 1
        if head_movement is not None:
            head_movements.append(head_movement)
        if upper_body_movement is not None:
            upper_body_movements.append(upper_body_movement)
        if hand_movement is not None:
            hand_movements.append(hand_movement)

    return OpenPoseFeatures(
        head_motion_score_raw=mean_optional(head_movements),
        upper_body_activation_score_raw=mean_optional(upper_body_movements),
        hand_energy_score_raw=mean_optional(hand_movements),
        frame_pair_count=valid_pair_count,
    )


def find_previous_utterance(expert: Utterance, utterances: Iterable[Utterance]) -> Utterance | None:
    """expert 発話の直前にある発話を返す。"""
    previous_candidates = [
        utterance
        for utterance in utterances
        if utterance.start_sec < expert.start_sec
    ]
    if not previous_candidates:
        return None
    return max(previous_candidates, key=lambda utterance: (utterance.start_sec, utterance.end_sec))


def build_session_rows(
    session_dir: Path,
    *,
    multimodal_dir: Path,
    window_sec: float,
    sample_rate: float,
    confidence_threshold: float,
) -> tuple[list[EngagementDeltaRow], list[str]]:
    """1 セッション分の engagement 差分行を作成する。"""
    session_id = session_dir.name
    warnings: list[str] = []
    expert_utterances = read_transcript(
        session_dir / EXPERT_TRANSCRIPT_NAME,
        session_id=session_id,
        speaker="expert",
    )
    novice_utterances = read_transcript(
        session_dir / NOVICE_TRANSCRIPT_NAME,
        session_id=session_id,
        speaker="novice",
    )
    engagement = read_engagement(session_dir / NOVICE_ENGAGEMENT_NAME)
    all_utterances = sorted(
        [*expert_utterances, *novice_utterances],
        key=lambda utterance: (utterance.start_sec, utterance.end_sec, utterance.speaker),
    )
    openface_rows: list[dict[str, str]] = []
    openpose_rows: list[dict[str, str]] = []
    try:
        openface_rows = read_dict_rows(multimodal_dir / session_id / OPENFACE_NAME)
    except FileNotFoundError as exc:
        warnings.append(str(exc))
    try:
        openpose_rows = read_dict_rows(multimodal_dir / session_id / OPENPOSE_NAME)
    except FileNotFoundError as exc:
        warnings.append(str(exc))

    rows: list[EngagementDeltaRow] = []
    for expert in expert_utterances:
        previous = find_previous_utterance(expert, all_utterances)
        if previous is None or previous.speaker != "novice":
            continue

        before_mean, before_count = mean_in_window(
            engagement,
            start_sec=max(0.0, expert.start_sec - window_sec),
            end_sec=expert.start_sec,
            sample_rate=sample_rate,
        )
        after_mean, after_count = mean_in_window(
            engagement,
            start_sec=expert.end_sec,
            end_sec=expert.end_sec + window_sec,
            sample_rate=sample_rate,
        )
        if before_mean is None or after_mean is None:
            warnings.append(
                f"{session_id}: {expert.start_sec:.2f}-{expert.end_sec:.2f} 秒の前後窓が空のためスキップしました。"
            )
            continue

        after_window_start = expert.end_sec
        after_window_end = expert.end_sec + window_sec
        openface_features = OpenFaceFeatures(None, None, None, None, None, 0)
        openpose_features = OpenPoseFeatures(None, None, None, 0)
        if openface_rows:
            openface_features = compute_openface_features(
                openface_rows,
                start_sec=after_window_start,
                end_sec=after_window_end,
            )
            if openface_features.smile_score_raw is None:
                warnings.append(
                    f"{session_id}: {expert.start_sec:.2f}-{expert.end_sec:.2f} 秒の OpenFace 有効フレームがありません。"
                )
        if openpose_rows:
            openpose_features = compute_openpose_features(
                openpose_rows,
                start_sec=after_window_start,
                end_sec=after_window_end,
                confidence_threshold=confidence_threshold,
            )
            if (
                openpose_features.head_motion_score_raw is None
                and openpose_features.upper_body_activation_score_raw is None
                and openpose_features.hand_energy_score_raw is None
            ):
                warnings.append(
                    f"{session_id}: {expert.start_sec:.2f}-{expert.end_sec:.2f} 秒の OpenPose 有効フレームペアがありません。"
                )

        rows.append(
            EngagementDeltaRow(
                session_id=session_id,
                expert_start_sec=expert.start_sec,
                expert_end_sec=expert.end_sec,
                novice_start_sec=previous.start_sec,
                novice_end_sec=previous.end_sec,
                before_engagement_mean=before_mean,
                after_engagement_mean=after_mean,
                engagement_delta=after_mean - before_mean,
                expert_text=expert.text,
                novice_text=previous.text,
                before_frame_count=before_count,
                after_frame_count=after_count,
                smile_score_raw=openface_features.smile_score_raw,
                head_motion_score_raw=openpose_features.head_motion_score_raw,
                upper_body_activation_score_raw=openpose_features.upper_body_activation_score_raw,
                hand_energy_score_raw=openpose_features.hand_energy_score_raw,
                au12_after_mean=openface_features.au12_after_mean,
                au06_after_mean=openface_features.au06_after_mean,
                au12_after_ratio=openface_features.au12_after_ratio,
                au06_after_ratio=openface_features.au06_after_ratio,
                openface_frame_count=openface_features.frame_count,
                openpose_frame_pair_count=openpose_features.frame_pair_count,
            )
        )

    return rows, warnings


def discover_session_dirs(dataset_dir: Path) -> list[Path]:
    """対象セッションディレクトリを列挙する。"""
    if not dataset_dir.exists():
        raise FileNotFoundError(f"データセットディレクトリが見つかりません: {dataset_dir}")
    session_dirs = [
        path
        for path in dataset_dir.iterdir()
        if path.is_dir() and (path / EXPERT_TRANSCRIPT_NAME).exists()
    ]
    if not session_dirs:
        raise ValueError(f"`{dataset_dir}` 配下に対象セッションが見つかりませんでした。")
    return sorted(session_dirs, key=lambda path: path.name)


def zscore_to_unit(value: float | None, values: list[float]) -> float:
    """値を z-score 化して 0〜1 に写像する。欠損や分散なしは中立値を返す。"""
    if value is None or len(values) < 2:
        return 0.5
    mean_value = sum(values) / len(values)
    variance = sum((item - mean_value) ** 2 for item in values) / len(values)
    stddev = math.sqrt(variance)
    if stddev == 0.0:
        return 0.5
    zscore = (value - mean_value) / stddev
    clipped = min(ZSCORE_CLIP, max(-ZSCORE_CLIP, zscore))
    return (clipped + ZSCORE_CLIP) / (ZSCORE_CLIP * 2.0)


def raw_values(rows: list[EngagementDeltaRow], attribute: str) -> list[float]:
    """行リストから欠損なしの raw 値だけを取り出す。"""
    values: list[float] = []
    for row in rows:
        value = getattr(row, attribute)
        if value is not None:
            values.append(value)
    return values


def rows_by_session(rows: list[EngagementDeltaRow]) -> dict[str, list[EngagementDeltaRow]]:
    """候補行を session_id ごとにまとめる。"""
    grouped_rows: dict[str, list[EngagementDeltaRow]] = {}
    for row in rows:
        grouped_rows.setdefault(row.session_id, []).append(row)
    return grouped_rows


def apply_normalization_and_scores(rows: list[EngagementDeltaRow]) -> list[EngagementDeltaRow]:
    """raw 特徴を正規化し、final_score を付与する。"""
    engagement_values = [row.engagement_delta for row in rows]
    session_rows = rows_by_session(rows)
    normalized_rows: list[EngagementDeltaRow] = []

    for row in rows:
        rows_in_session = session_rows[row.session_id]
        engagement_delta_norm = zscore_to_unit(row.engagement_delta, engagement_values)
        smile_score_norm = zscore_to_unit(row.smile_score_raw, raw_values(rows_in_session, "smile_score_raw"))
        head_motion_score_norm = zscore_to_unit(
            row.head_motion_score_raw,
            raw_values(rows_in_session, "head_motion_score_raw"),
        )
        upper_body_activation_score_norm = zscore_to_unit(
            row.upper_body_activation_score_raw,
            raw_values(rows_in_session, "upper_body_activation_score_raw"),
        )
        hand_energy_score_norm = zscore_to_unit(
            row.hand_energy_score_raw,
            raw_values(rows_in_session, "hand_energy_score_raw"),
        )
        final_score = (
            0.50 * engagement_delta_norm
            + 0.20 * smile_score_norm
            + 0.15 * head_motion_score_norm
            + 0.10 * upper_body_activation_score_norm
            + 0.05 * hand_energy_score_norm
        )
        normalized_rows.append(
            EngagementDeltaRow(
                session_id=row.session_id,
                expert_start_sec=row.expert_start_sec,
                expert_end_sec=row.expert_end_sec,
                novice_start_sec=row.novice_start_sec,
                novice_end_sec=row.novice_end_sec,
                before_engagement_mean=row.before_engagement_mean,
                after_engagement_mean=row.after_engagement_mean,
                engagement_delta=row.engagement_delta,
                expert_text=row.expert_text,
                novice_text=row.novice_text,
                before_frame_count=row.before_frame_count,
                after_frame_count=row.after_frame_count,
                smile_score_raw=row.smile_score_raw,
                head_motion_score_raw=row.head_motion_score_raw,
                upper_body_activation_score_raw=row.upper_body_activation_score_raw,
                hand_energy_score_raw=row.hand_energy_score_raw,
                au12_after_mean=row.au12_after_mean,
                au06_after_mean=row.au06_after_mean,
                au12_after_ratio=row.au12_after_ratio,
                au06_after_ratio=row.au06_after_ratio,
                openface_frame_count=row.openface_frame_count,
                openpose_frame_pair_count=row.openpose_frame_pair_count,
                engagement_delta_norm=engagement_delta_norm,
                smile_score_norm=smile_score_norm,
                head_motion_score_norm=head_motion_score_norm,
                upper_body_activation_score_norm=upper_body_activation_score_norm,
                hand_energy_score_norm=hand_energy_score_norm,
                final_score=final_score,
            )
        )
    return normalized_rows


def build_rows(
    dataset_dir: Path,
    *,
    multimodal_dir: Path = Path(DEFAULT_MULTIMODAL_DIR),
    window_sec: float = DEFAULT_WINDOW_SEC,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> tuple[list[EngagementDeltaRow], list[str]]:
    """全セッションの engagement 差分行を作成する。"""
    if window_sec <= 0:
        raise ValueError("`window_sec` は 0 より大きい値を指定してください。")
    if sample_rate <= 0:
        raise ValueError("`sample_rate` は 0 より大きい値を指定してください。")
    if confidence_threshold < 0:
        raise ValueError("`confidence_threshold` は 0 以上の値を指定してください。")

    rows: list[EngagementDeltaRow] = []
    warnings: list[str] = []
    for session_dir in discover_session_dirs(dataset_dir):
        session_rows, session_warnings = build_session_rows(
            session_dir,
            multimodal_dir=multimodal_dir,
            window_sec=window_sec,
            sample_rate=sample_rate,
            confidence_threshold=confidence_threshold,
        )
        rows.extend(session_rows)
        warnings.extend(session_warnings)

    rows = apply_normalization_and_scores(rows)
    rows.sort(
        key=lambda row: (
            row.final_score,
            row.engagement_delta_norm,
            row.after_engagement_mean,
            -row.expert_start_sec,
        ),
        reverse=True,
    )
    return rows, warnings


def write_rows(rows: list[EngagementDeltaRow], output_path: Path) -> None:
    """ランキング行を CSV に書き出す。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_row())


def main() -> int:
    """CLI エントリポイント。"""
    args = parse_args()
    try:
        rows, warnings = build_rows(
            Path(args.dataset_dir),
            multimodal_dir=Path(args.multimodal_dir),
            window_sec=args.window_sec,
            sample_rate=args.sample_rate,
            confidence_threshold=args.confidence_threshold,
        )
        write_rows(rows, Path(args.output))
    except (FileNotFoundError, ValueError) as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1

    for warning in warnings:
        print(f"警告: {warning}", file=sys.stderr)
    print(f"{len(rows)} 件の expert 返答を {args.output} に出力しました。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
