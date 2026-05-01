"""multimodal score CSV から rejected 付き DPO 学習データを作成する。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Protocol

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.local_llm_utils import build_qwen_generation_prompt, extract_qwen_final_text


DEFAULT_INPUT_PATH = "artifacts/noxij_dpo_multimodal_score.csv"
DEFAULT_OUTPUT_JSONL = "artifacts/noxij_dpo_preferences.jsonl"
DEFAULT_OUTPUT_CSV = "artifacts/noxij_dpo_preferences.csv"
DEFAULT_FAILED_CSV = "artifacts/noxij_dpo_preferences_failed.csv"
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-27B"
DEFAULT_TOP_N = 50
DEFAULT_MAX_NEW_TOKENS = 160
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.8
DEFAULT_SEED = 42
MAX_GENERATION_ATTEMPTS = 3

TRUE_VALUES = {"1", "true", "yes", "on"}

PROMPT_TYPE_QUESTION = "question"
PROMPT_TYPE_DIFFICULTY = "difficulty"
PROMPT_TYPE_BACKCHANNEL = "backchannel"
PROMPT_TYPE_SELF_DISCLOSURE = "self_disclosure"
PROMPT_TYPE_OTHER = "other"

STRATEGY_GENERIC = "内容を拾わない一般論"
STRATEGY_SELF_FOCUS = "話題を自分側に寄せる"
STRATEGY_SHALLOW = "浅い相槌または雑な質問"
STRATEGY_OVEREXPLAIN = "説明しすぎて話す余地を減らす"

DPO_COLUMNS = [
    "prompt",
    "chosen",
    "rejected",
    "prompt_type",
    "rejected_strategy",
    "source_rank",
    "session_id",
    "final_score",
    "engagement_delta",
    "model_id",
]

FAILED_COLUMNS = [
    "source_rank",
    "session_id",
    "prompt",
    "chosen",
    "prompt_type",
    "rejected_strategy",
    "final_score",
    "failure_reason",
]


@dataclass(frozen=True)
class SourceExample:
    """DPO候補元の1行。"""

    source_rank: int
    session_id: str
    prompt: str
    chosen: str
    final_score: float
    engagement_delta: float
    expert_start_sec: float
    expert_end_sec: float
    novice_start_sec: float
    novice_end_sec: float
    raw_row: dict[str, str]


@dataclass(frozen=True)
class PreferenceExample:
    """DPO 学習に使う preference 例。"""

    prompt: str
    chosen: str
    rejected: str
    prompt_type: str
    rejected_strategy: str
    source: SourceExample
    model_id: str

    def to_jsonl_record(self) -> dict:
        """JSONL 用レコードへ変換する。"""
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "metadata": {
                "session_id": self.source.session_id,
                "expert_start_sec": self.source.expert_start_sec,
                "expert_end_sec": self.source.expert_end_sec,
                "novice_start_sec": self.source.novice_start_sec,
                "novice_end_sec": self.source.novice_end_sec,
                "final_score": self.source.final_score,
                "engagement_delta": self.source.engagement_delta,
                "prompt_type": self.prompt_type,
                "rejected_strategy": self.rejected_strategy,
                "source_rank": self.source.source_rank,
                "model_id": self.model_id,
            },
        }

    def to_csv_row(self) -> dict[str, str]:
        """CSV 用レコードへ変換する。"""
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "prompt_type": self.prompt_type,
            "rejected_strategy": self.rejected_strategy,
            "source_rank": str(self.source.source_rank),
            "session_id": self.source.session_id,
            "final_score": f"{self.source.final_score:.6f}",
            "engagement_delta": f"{self.source.engagement_delta:.6f}",
            "model_id": self.model_id,
        }


@dataclass(frozen=True)
class FailedExample:
    """rejected 生成に失敗した候補。"""

    source: SourceExample
    prompt_type: str
    rejected_strategy: str
    failure_reason: str

    def to_csv_row(self) -> dict[str, str]:
        """失敗CSV用レコードへ変換する。"""
        return {
            "source_rank": str(self.source.source_rank),
            "session_id": self.source.session_id,
            "prompt": self.source.prompt,
            "chosen": self.source.chosen,
            "prompt_type": self.prompt_type,
            "rejected_strategy": self.rejected_strategy,
            "final_score": f"{self.source.final_score:.6f}",
            "failure_reason": self.failure_reason,
        }


class RejectedGenerator(Protocol):
    """rejected 生成器のインターフェース。"""

    model_id: str

    def generate(self, messages: list[dict[str, str]]) -> str:
        """messages から rejected JSON 文字列を生成する。"""


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""
    parser = argparse.ArgumentParser(description="NoXi+J DPO preference データセットを作成します。")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help=f"入力CSV（既定: {DEFAULT_INPUT_PATH}）。")
    parser.add_argument(
        "--output-jsonl",
        default=DEFAULT_OUTPUT_JSONL,
        help=f"DPO JSONL 出力先（既定: {DEFAULT_OUTPUT_JSONL}）。",
    )
    parser.add_argument(
        "--output-csv",
        default=DEFAULT_OUTPUT_CSV,
        help=f"目視確認用 CSV 出力先（既定: {DEFAULT_OUTPUT_CSV}）。",
    )
    parser.add_argument(
        "--failed-csv",
        default=DEFAULT_FAILED_CSV,
        help=f"生成失敗候補 CSV 出力先（既定: {DEFAULT_FAILED_CSV}）。",
    )
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help=f"使用する上位候補数（既定: {DEFAULT_TOP_N}）。")
    parser.add_argument("--min-final-score", type=float, default=None, help="最低 final_score（任意）。")
    parser.add_argument("--model-id", default=os.getenv("LOCAL_QWEN_MODEL_ID", DEFAULT_MODEL_ID), help="QwenモデルID。")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="rejected生成の最大トークン数。")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="生成temperature。")
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P, help="生成top_p。")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="乱数シード。")
    parser.add_argument("--dry-run", action="store_true", help="モデルを読まず、候補選定結果だけ表示します。")
    parser.add_argument("--resume", action="store_true", help="既存JSONLの source_rank をスキップして再開します。")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    """比較用に空白を正規化する。"""
    return re.sub(r"\s+", "", text or "").strip()


def is_setup_or_audio_check(example: SourceExample) -> bool:
    """聞こえ確認や冒頭セットアップに近い候補を除外する。"""
    prompt = example.prompt
    chosen = example.chosen
    if example.expert_start_sec <= 10.0 and ("聞こえ" in prompt or "聞こえ" in chosen or "初めまして" in chosen):
        return True
    return False


def read_source_examples(input_path: Path) -> list[SourceExample]:
    """multimodal score CSV を読み込む。"""
    try:
        with input_path.open("r", newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"入力CSVが見つかりません: {input_path}") from exc

    examples: list[SourceExample] = []
    for rank, row in enumerate(rows, start=1):
        try:
            examples.append(
                SourceExample(
                    source_rank=rank,
                    session_id=row["session_id"],
                    prompt=row["novice_text"].strip(),
                    chosen=row["expert_text"].strip(),
                    final_score=float(row["final_score"]),
                    engagement_delta=float(row["engagement_delta"]),
                    expert_start_sec=float(row["expert_start_sec"]),
                    expert_end_sec=float(row["expert_end_sec"]),
                    novice_start_sec=float(row["novice_start_sec"]),
                    novice_end_sec=float(row["novice_end_sec"]),
                    raw_row=row,
                )
            )
        except KeyError as exc:
            raise ValueError(f"入力CSVの必須列が不足しています: {exc}") from exc
        except ValueError as exc:
            raise ValueError("入力CSVの数値列を解釈できません。") from exc
    return examples


def select_source_examples(
    examples: list[SourceExample],
    *,
    top_n: int,
    min_final_score: float | None = None,
) -> list[SourceExample]:
    """DPO化する上位候補を選ぶ。"""
    if top_n <= 0:
        raise ValueError("`top_n` は 1 以上を指定してください。")
    selected: list[SourceExample] = []
    for example in examples:
        if min_final_score is not None and example.final_score < min_final_score:
            continue
        if is_setup_or_audio_check(example):
            continue
        selected.append(example)
        if len(selected) >= top_n:
            break
    return selected


def is_backchannel_text(text: str) -> bool:
    """相槌中心の短い発話かをざっくり判定する。"""
    normalized = normalize_text(text)
    if len(normalized) <= 2:
        return True
    backchannel_chars = re.sub(r"(はい|うん|ええ|なるほど|そうですね|そうなんですね|あー|んー|ほう|へー|まあ|、|。|!|！|\\?|？)", "", normalized)
    return len(backchannel_chars) <= 2 and len(normalized) <= 30


def classify_prompt_type(prompt: str) -> str:
    """prompt の会話状態を分類する。"""
    if any(word in prompt for word in ("難しい", "わから", "分から", "困", "迷って", "定まって")):
        return PROMPT_TYPE_DIFFICULTY
    if "?" in prompt or "？" in prompt or prompt.endswith("ですか") or prompt.endswith("ますか"):
        return PROMPT_TYPE_QUESTION
    if is_backchannel_text(prompt):
        return PROMPT_TYPE_BACKCHANNEL
    if len(normalize_text(prompt)) >= 35 or any(word in prompt for word in ("好き", "楽しい", "思って", "行って", "やって")):
        return PROMPT_TYPE_SELF_DISCLOSURE
    return PROMPT_TYPE_OTHER


def choose_rejected_strategy(prompt_type: str, rng: random.Random) -> str:
    """promptタイプに応じた rejected 方針を選ぶ。"""
    if prompt_type == PROMPT_TYPE_QUESTION:
        return rng.choice([STRATEGY_GENERIC, STRATEGY_SELF_FOCUS, STRATEGY_SHALLOW])
    if prompt_type == PROMPT_TYPE_DIFFICULTY:
        return rng.choice([STRATEGY_SHALLOW, STRATEGY_GENERIC])
    if prompt_type == PROMPT_TYPE_BACKCHANNEL:
        return rng.choice([STRATEGY_OVEREXPLAIN, STRATEGY_SELF_FOCUS])
    if prompt_type == PROMPT_TYPE_SELF_DISCLOSURE:
        return rng.choice([STRATEGY_GENERIC, STRATEGY_SELF_FOCUS, STRATEGY_SHALLOW])
    return rng.choice([STRATEGY_GENERIC, STRATEGY_SHALLOW, STRATEGY_OVEREXPLAIN])


def build_generation_messages(example: SourceExample, prompt_type: str, strategy: str) -> list[dict[str, str]]:
    """Qwen に渡す rejected 生成プロンプトを作る。"""
    system = (
        "あなたはDPO学習用データを作る日本語会話データ作成者です。"
        "目的は、自然だがchosenより会話意欲を少し下げるrejected返答を1つ作ることです。"
        "暴言、攻撃、差別、倫理的に問題のある内容、露骨に失礼な返答は禁止です。"
        "出力はJSONのみで、キーは rejected だけにしてください。"
    )
    user = f"""以下の会話候補から rejected を作成してください。

prompt（novice発話）:
{example.prompt}

chosen（実際に反応が良かったexpert返答）:
{example.chosen}

prompt_type: {prompt_type}
rejected_strategy: {strategy}

条件:
- 日本語の1文または2文にする。
- 実際の会話であり得る自然な返答にする。
- ただし chosen よりも、相手が話し続けにくくなる返答にする。
- prompt の具体語を拾いすぎない。
- 共感、深掘り、言い換え、促しのうち少なくとも1つを欠かす。
- chosen や prompt をコピーしない。
- chosen と同程度か、少し短めにする。
- 説明や分析を出さず、JSONだけを返す。

出力形式:
{{"rejected": "ここに返答"}}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def extract_rejected_text(model_output: str) -> str:
    """モデル出力から rejected 文字列を抽出する。"""
    text = extract_qwen_final_text(model_output) or model_output.strip()
    try:
        payload = json.loads(text)
        rejected = payload.get("rejected", "")
        return str(rejected).strip()
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return ""
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return ""
        return str(payload.get("rejected", "")).strip()


def is_too_similar(first: str, second: str) -> bool:
    """2つのテキストが近すぎるかを返す。"""
    first_norm = normalize_text(first)
    second_norm = normalize_text(second)
    if not first_norm or not second_norm:
        return False
    if first_norm == second_norm:
        return True
    return SequenceMatcher(None, first_norm, second_norm).ratio() >= 0.82


def validate_rejected(example: SourceExample, rejected: str) -> tuple[bool, str]:
    """生成された rejected の最低限の品質を検査する。"""
    if not rejected:
        return False, "rejectedが空です"
    if is_too_similar(example.chosen, rejected):
        return False, "chosenと近すぎます"
    if is_too_similar(example.prompt, rejected):
        return False, "promptと近すぎます"
    if len(rejected) > max(180, int(len(example.chosen) * 1.4)):
        return False, "rejectedが長すぎます"
    if any(marker in rejected for marker in ("```", "rejected", "prompt", "chosen")):
        return False, "JSONや説明文が混入しています"
    return True, ""


class QwenRejectedGenerator:
    """ローカル Qwen3.5 を使う rejected 生成器。"""

    def __init__(
        self,
        *,
        model_id: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> None:
        try:
            import torch
            from transformers import AutoProcessor, AutoTokenizer, Qwen3_5ForConditionalGeneration
            try:
                from transformers import BitsAndBytesConfig
            except Exception:
                BitsAndBytesConfig = None
        except ImportError as exc:
            raise RuntimeError("Qwen3.5 実行に必要な transformers / torch が見つかりません。") from exc

        if not torch.cuda.is_available():
            raise RuntimeError("Qwen3.5 の rejected 生成には CUDA 対応 GPU が必要です。")

        self.torch = torch
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.device = "cuda"

        try:
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            self.tokenizer = getattr(processor, "tokenizer", None) or processor
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        enable_4bit = os.getenv("LOCAL_QWEN_ENABLE_4BIT", "0").strip().lower() in TRUE_VALUES
        if enable_4bit and BitsAndBytesConfig is not None:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )
        self.model.eval()

    def generate(self, messages: list[dict[str, str]]) -> str:
        """Qwen3.5 で rejected JSON を生成する。"""
        prompt_text = build_qwen_generation_prompt(self.tokenizer, messages, enable_thinking=False)
        model_inputs = self.tokenizer(prompt_text, return_tensors="pt")
        model_inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in model_inputs.items()
        }
        input_ids = model_inputs["input_ids"]
        if "attention_mask" not in model_inputs:
            model_inputs["attention_mask"] = self.torch.ones_like(input_ids, device=self.device)
        with self.torch.no_grad():
            output_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output_ids[0][input_ids.shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=False).strip()


def generate_preference_for_example(
    example: SourceExample,
    *,
    generator: RejectedGenerator,
    rng: random.Random,
) -> PreferenceExample | FailedExample:
    """1候補から preference 例を生成する。"""
    prompt_type = classify_prompt_type(example.prompt)
    strategy = choose_rejected_strategy(prompt_type, rng)
    messages = build_generation_messages(example, prompt_type, strategy)
    last_reason = "生成されませんでした"
    for _ in range(MAX_GENERATION_ATTEMPTS):
        output = generator.generate(messages)
        rejected = extract_rejected_text(output)
        valid, reason = validate_rejected(example, rejected)
        if valid:
            return PreferenceExample(
                prompt=example.prompt,
                chosen=example.chosen,
                rejected=rejected,
                prompt_type=prompt_type,
                rejected_strategy=strategy,
                source=example,
                model_id=generator.model_id,
            )
        last_reason = reason
    return FailedExample(
        source=example,
        prompt_type=prompt_type,
        rejected_strategy=strategy,
        failure_reason=last_reason,
    )


def existing_source_ranks(jsonl_path: Path) -> set[int]:
    """既存JSONLから source_rank を集める。"""
    if not jsonl_path.exists():
        return set()
    ranks: set[int] = set()
    with jsonl_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                ranks.add(int(payload["metadata"]["source_rank"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return ranks


def write_jsonl(examples: list[PreferenceExample], output_path: Path, *, append: bool = False) -> None:
    """DPO JSONLを書き出す。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with output_path.open(mode, encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(example.to_jsonl_record(), ensure_ascii=False) + "\n")


def write_csv_files(
    examples: list[PreferenceExample],
    failures: list[FailedExample],
    *,
    output_csv: Path,
    failed_csv: Path,
) -> None:
    """目視確認CSVと失敗CSVを書き出す。"""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=DPO_COLUMNS)
        writer.writeheader()
        for example in examples:
            writer.writerow(example.to_csv_row())

    failed_csv.parent.mkdir(parents=True, exist_ok=True)
    with failed_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FAILED_COLUMNS)
        writer.writeheader()
        for failure in failures:
            writer.writerow(failure.to_csv_row())


def print_dry_run_summary(examples: list[SourceExample]) -> None:
    """dry-run 用に候補概要を表示する。"""
    print(f"{len(examples)} 件の候補を選択しました。")
    for example in examples[:10]:
        prompt_type = classify_prompt_type(example.prompt)
        print(
            f"rank={example.source_rank} session={example.session_id} "
            f"score={example.final_score:.6f} prompt_type={prompt_type} "
            f"prompt={example.prompt[:60]!r} chosen={example.chosen[:60]!r}"
        )


def main() -> int:
    """CLIエントリポイント。"""
    args = parse_args()
    rng = random.Random(args.seed)
    try:
        sources = read_source_examples(Path(args.input))
        selected_sources = select_source_examples(
            sources,
            top_n=args.top_n,
            min_final_score=args.min_final_score,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print_dry_run_summary(selected_sources)
        return 0

    skipped_ranks = existing_source_ranks(Path(args.output_jsonl)) if args.resume else set()
    selected_sources = [source for source in selected_sources if source.source_rank not in skipped_ranks]

    try:
        generator = QwenRejectedGenerator(
            model_id=args.model_id,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    except RuntimeError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1

    successes: list[PreferenceExample] = []
    failures: list[FailedExample] = []
    for index, source in enumerate(selected_sources, start=1):
        result = generate_preference_for_example(source, generator=generator, rng=rng)
        if isinstance(result, PreferenceExample):
            successes.append(result)
            print(f"[{index}/{len(selected_sources)}] rank={source.source_rank} 生成成功")
        else:
            failures.append(result)
            print(f"[{index}/{len(selected_sources)}] rank={source.source_rank} 生成失敗: {result.failure_reason}")

    write_jsonl(successes, Path(args.output_jsonl), append=args.resume)
    write_csv_files(successes, failures, output_csv=Path(args.output_csv), failed_csv=Path(args.failed_csv))
    print(f"DPO JSONL: {args.output_jsonl} ({len(successes)} 件)")
    print(f"確認CSV: {args.output_csv}")
    print(f"失敗CSV: {args.failed_csv} ({len(failures)} 件)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
