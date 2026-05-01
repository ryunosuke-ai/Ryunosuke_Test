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
DEFAULT_OUTPUT_PATH = "artifacts/noxij_expert_response_engagement_delta.csv"
DEFAULT_WINDOW_SEC = 5.0
DEFAULT_SAMPLE_RATE = 25.0

EXPERT_TRANSCRIPT_NAME = "expert.audio.transcript.annotation.csv"
NOVICE_TRANSCRIPT_NAME = "novice.audio.transcript.annotation.csv"
NOVICE_ENGAGEMENT_NAME = "novice.engagement.annotation.csv"

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
    return parser.parse_args()


def format_float(value: float) -> str:
    """CSV で読みやすい固定精度の数値文字列を返す。"""
    return f"{value:.6f}"


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
    window_sec: float,
    sample_rate: float,
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


def build_rows(
    dataset_dir: Path,
    *,
    window_sec: float = DEFAULT_WINDOW_SEC,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
) -> tuple[list[EngagementDeltaRow], list[str]]:
    """全セッションの engagement 差分行を作成する。"""
    if window_sec <= 0:
        raise ValueError("`window_sec` は 0 より大きい値を指定してください。")
    if sample_rate <= 0:
        raise ValueError("`sample_rate` は 0 より大きい値を指定してください。")

    rows: list[EngagementDeltaRow] = []
    warnings: list[str] = []
    for session_dir in discover_session_dirs(dataset_dir):
        session_rows, session_warnings = build_session_rows(
            session_dir,
            window_sec=window_sec,
            sample_rate=sample_rate,
        )
        rows.extend(session_rows)
        warnings.extend(session_warnings)

    rows.sort(
        key=lambda row: (
            row.engagement_delta,
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
            window_sec=args.window_sec,
            sample_rate=args.sample_rate,
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
