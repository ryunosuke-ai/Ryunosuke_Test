"""フェーズ遷移なしのテキスト会話ツール（話したい度推定あり）"""

import logging
import os
import sys
from collections import deque
from datetime import datetime
from typing import Deque

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    print("エラー: `python-dotenv` パッケージが見つかりません。")
    print("依存関係をインストールしてから再実行してください。")
    print("例: python3 -m pip install python-dotenv")
    sys.exit(1)

try:
    from openai import AzureOpenAI
except ModuleNotFoundError:
    print("エラー: `openai` パッケージが見つかりません。")
    print("依存関係をインストールしてから再実行してください。")
    print("例: python3 -m pip install openai")
    sys.exit(1)

try:
    import torch
    from transformers import AutoProcessor, AutoTokenizer, Qwen3_5ForConditionalGeneration
except ImportError:
    print("エラー: `transformers` / `torch` が見つかりません。")
    print("Qwen3.5 に対応した依存関係をインストールしてから再実行してください。")
    print("例: python3 -m pip install -r requirements.txt")
    sys.exit(1)

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv()

from core.models import ActionType, MemoryUpdate, Observation  # noqa: E402
from core.bayes_engine import DEFAULT_LIKELIHOODS  # noqa: E402
from core.local_llm_utils import (  # noqa: E402
    build_qwen_display_fallback_text,
    build_qwen_generation_prompt,
    decode_qwen_local_llm_reply,
)
from llm.gpt_oss.simple_text_chat_gpt_oss import SimpleTextChatAgent as GptOssSimpleTextChatAgent  # noqa: E402


TRUE_VALUES = {"1", "true", "yes", "on"}


def _read_env_value(primary_name: str, default: str | None = None, legacy_names: tuple[str, ...] = ()) -> str | None:
    """環境変数を優先順で取得する。"""
    for env_name in (primary_name, *legacy_names):
        value = os.getenv(env_name)
        if value not in (None, ""):
            return value
    return default


def _read_env_flag(primary_name: str, default: bool = False, legacy_names: tuple[str, ...] = ()) -> bool:
    """真偽値の環境変数を解釈する。"""
    fallback = "1" if default else "0"
    value = _read_env_value(primary_name, fallback, legacy_names=legacy_names)
    return str(value).strip().lower() in TRUE_VALUES


class SimpleTextChatAgent(GptOssSimpleTextChatAgent):
    """
    フェーズ遷移を使わないテキスト会話エージェント。
    返信生成のみ Qwen3.5-27B を使い、その他の会話ロジックは既存実装を流用する。
    """

    PHASE_LABEL = "SIMPLE_CHAT_QWEN35"
    DEFAULT_MODEL_ID = "Qwen/Qwen3.5-27B"
    BANNER_TITLE = "単純会話ツール Qwen3.5-27B版（フェーズ遷移なし）"
    MAX_GENERATION_ATTEMPTS = 2

    def __init__(self):
        # --- 1) 設定読み込み ---
        self.openai_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        self.local_model_id = _read_env_value(
            "LOCAL_QWEN_MODEL_ID",
            self.DEFAULT_MODEL_ID,
            legacy_names=("LOCAL_QWEN35_MODEL_ID",),
        )
        self.enable_thinking = _read_env_flag(
            "LOCAL_QWEN_ENABLE_THINKING",
            default=True,
            legacy_names=("LOCAL_QWEN35_ENABLE_THINKING",),
        )
        self.show_thinking = _read_env_flag(
            "LOCAL_QWEN_SHOW_THINKING",
            default=False,
            legacy_names=("LOCAL_QWEN35_SHOW_THINKING",),
        )

        missing = [k for k, v in [
            ("AZURE_OPENAI_API_KEY", self.openai_key),
            ("AZURE_OPENAI_ENDPOINT", self.openai_endpoint),
            ("AZURE_OPENAI_DEPLOYMENT_NAME", self.deployment_name),
        ] if not v]
        if missing:
            print(f"エラー: .env の設定が不足しています: {', '.join(missing)}")
            sys.exit(1)

        # --- 2) ログ準備 ---
        os.makedirs("logs", exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = f"logs/run_{ts}"
        os.makedirs(self.run_dir, exist_ok=True)

        self.history_file = f"{self.run_dir}/log_{ts}.txt"
        self.analysis_csv = f"{self.run_dir}/analysis_{ts}.csv"

        with open(self.analysis_csv, "w", encoding="utf-8") as f:
            f.write("Timestamp,Turn,Phase,Speaker,PrimaryLabel,LabelReason,P_WantTalk,Text\n")

        self._setup_logger(ts)

        self.logger.info("会話ログ: %s", self.history_file)
        self.logger.info("分析用CSV: %s", self.analysis_csv)

        # --- 3) Azure OpenAI 初期化（分類・会話メモ更新で使用） ---
        self.openai_client = AzureOpenAI(
            api_key=self.openai_key,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=self.openai_endpoint,
        )
        self.logger.info("分類・会話メモ更新は Azure OpenAI を使用します。")

        # --- 4) ローカルQwen3.5-27B 初期化（返信生成で使用） ---
        if not torch.cuda.is_available():
            print("エラー: CUDA対応GPUが見つかりません。")
            print("Qwen3.5-27B のローカル実行には GPU 環境が必要です。")
            print("`nvidia-smi` と `python3 -c 'import torch; print(torch.cuda.is_available())'` を確認してください。")
            sys.exit(1)
        self.local_device = "cuda"
        self.processor = None
        self.tokenizer = None
        self.local_model = None
        self._load_local_qwen()

        # --- 5) 状態 ---
        self.total_turns: int = 0
        self.p_want_talk: float = 0.5
        self.likelihoods = DEFAULT_LIKELIHOODS
        self.conv_memory: MemoryUpdate = MemoryUpdate(summary="", do_not_ask=[])
        self.force_end: bool = False
        self.max_total_turns: int = 15
        self.flow_stage: str = self.FLOW_WARMUP
        self.stage_turns: int = 0
        self.last_transition_turn: int = 0
        self.low_engagement_streak: int = 0
        self.p_history: Deque[float] = deque([self.p_want_talk], maxlen=6)

    def _setup_logger(self, ts: str) -> None:
        self.logger = logging.getLogger("simple_text_chat_qwen35_agent")
        self.logger.setLevel(logging.INFO)

        fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S")
        file_handler = logging.FileHandler(f"{self.run_dir}/agent_{ts}.log", encoding="utf-8")
        file_handler.setFormatter(fmt)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(fmt)

        self.logger.handlers.clear()
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

    def _load_local_qwen(self) -> None:
        """ローカルQwen3.5-27Bモデルを読み込む。"""
        try:
            try:
                self.processor = AutoProcessor.from_pretrained(
                    self.local_model_id,
                    trust_remote_code=True,
                )
                self.tokenizer = getattr(self.processor, "tokenizer", None) or self.processor
            except Exception as processor_error:
                self.logger.warning(
                    "AutoProcessor の読み込みに失敗したため AutoTokenizer に切り替えます: %s",
                    processor_error,
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.local_model_id,
                    trust_remote_code=True,
                )

            enable_4bit = _read_env_flag(
                "LOCAL_QWEN_ENABLE_4BIT",
                default=False,
                legacy_names=("LOCAL_QWEN35_ENABLE_4BIT",),
            )
            loaded_with_4bit = False

            if BitsAndBytesConfig is not None and enable_4bit:
                try:
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                    self.local_model = Qwen3_5ForConditionalGeneration.from_pretrained(
                        self.local_model_id,
                        quantization_config=quant_config,
                        device_map="auto",
                        trust_remote_code=True,
                    )
                    loaded_with_4bit = True
                    self.logger.info("Qwen3.5-27Bを4bit量子化で読み込みました。")
                except Exception as quant_error:
                    self.logger.warning(
                        "4bit読み込みに失敗したため、FP16/BF16へフォールバックします: %s",
                        quant_error,
                    )
            elif enable_4bit and BitsAndBytesConfig is None:
                self.logger.warning(
                    "BitsAndBytesConfig が利用できないため、4bit量子化をスキップしてFP16/BF16を使います。"
                )

            if not loaded_with_4bit:
                use_bf16 = torch.cuda.is_bf16_supported()
                load_dtype = torch.bfloat16 if use_bf16 else torch.float16
                self.local_model = Qwen3_5ForConditionalGeneration.from_pretrained(
                    self.local_model_id,
                    torch_dtype=load_dtype,
                    device_map="auto",
                    trust_remote_code=True,
                )
                self.logger.info(
                    "Qwen3.5-27Bを%sで読み込みました（4bit未使用）。",
                    "BF16" if use_bf16 else "FP16",
                )
            self.local_model.eval()
            self.logger.info("ローカルQwen3.5-27Bモデル読み込み完了: %s", self.local_model_id)
            self.logger.info("Qwen3.5 thinkingモード: %s", "有効" if self.enable_thinking else "無効")
        except Exception as e:
            self.logger.error("ローカルQwen3.5-27Bモデルの読み込みに失敗: %s", e)
            print("エラー: ローカルQwen3.5-27Bモデルの読み込みに失敗しました。")
            print("先に `python3 -m llm.qwen.download_qwen35_27b` を実行し、依存関係とGPU環境を確認してください。")
            print("4bitを使う場合は `LOCAL_QWEN_ENABLE_4BIT=1` を付けて実行してください。")
            sys.exit(1)

    def _build_qwen_prompt_text(self, messages: list[dict]) -> str:
        """Qwen 用のチャットプロンプトを組み立てる。"""
        return build_qwen_generation_prompt(
            self.tokenizer,
            messages,
            enable_thinking=self.enable_thinking,
        )

    def _build_model_inputs(self, prompt_text: str) -> dict:
        """プロンプトをモデル入力へ変換して GPU へ載せる。"""
        model_inputs = self.tokenizer(prompt_text, return_tensors="pt")
        return {
            key: value.to(self.local_device) if hasattr(value, "to") else value
            for key, value in model_inputs.items()
        }

    def _decode_qwen_reply(self, generated_ids) -> str:
        """生成結果からユーザー表示用の返答本文を抽出する。"""
        return decode_qwen_local_llm_reply(
            self.tokenizer,
            generated_ids,
            show_thinking=self.show_thinking,
        )

    def _build_output_control_instruction(self, strict_output: bool = False) -> str:
        """thinking ON 時でも最終回答だけを返すための制約を返す。"""
        base_instruction = (
            "【出力制御】内部では十分に考えてよいが、ユーザーに見せる出力は最終回答本文のみとする。\n"
            "【出力制御】Thinking Process、分析、推論、<think>、内部メモは出力しない。\n"
            "【出力制御】必ず自然な日本語の返答本文を1つ以上出す。\n"
        )
        if strict_output:
            return (
                base_instruction
                + "【出力制御】今回は前回の出力が思考文に偏ったため、説明や分析ではなく、ユーザー向け返答本文だけを直ちに出力する。\n"
            )
        return base_instruction

    def _generate_once(self, messages: list[dict], strict_output: bool = False) -> tuple[str, str]:
        """Qwen で1回生成し、抽出本文と生返答を返す。"""
        prompt_text = self._build_qwen_prompt_text(messages)
        model_inputs = self._build_model_inputs(prompt_text)
        input_ids = model_inputs["input_ids"]

        if "attention_mask" not in model_inputs:
            model_inputs["attention_mask"] = torch.ones_like(input_ids, device=self.local_device)
        with torch.no_grad():
            output_ids = self.local_model.generate(
                **model_inputs,
                max_new_tokens=self.LOCAL_MAX_NEW_TOKENS,
                do_sample=True,
                temperature=self.LOCAL_TEMPERATURE,
                top_p=self.LOCAL_TOP_P,
                repetition_penalty=1.05,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = output_ids[0][input_ids.shape[1]:]
        raw_text = self.tokenizer.decode(generated, skip_special_tokens=False).strip()
        reply = self._decode_qwen_reply(generated)

        if strict_output:
            self.logger.warning("Qwen3.5-27B生出力（再生成）: %r", raw_text)
        else:
            self.logger.warning("Qwen3.5-27B生出力: %r", raw_text)
        return reply, raw_text

    def think_and_reply(self, obs: Observation, waiting_mode: bool = False, ending_mode: bool = False) -> str:
        history = self.load_history_as_messages(max_messages=10)

        summary_text = self.conv_memory.summary if self.conv_memory.summary else "（まだ要約が少ないです）"
        if self.conv_memory.do_not_ask:
            do_not_ask_text = "- " + "\n- ".join(self.conv_memory.do_not_ask)
        else:
            do_not_ask_text = "（特になし）"

        if waiting_mode:
            interaction_mode = (
                "【現在状況】ユーザーは返答内容を考えている途中です。\n"
                "【モード】急かさず、履歴に沿った短いひと言だけ述べる。\n"
                "【禁止】深掘りの継続、連続質問、終了誘導。\n"
            )
            length_instruction = "【長さ】30〜70文字程度。\n"
        else:
            interaction_mode = self._build_interaction_mode_instruction(obs, ending_mode)
            length_instruction = "【長さ】60〜120文字程度。\n"
        stage_instruction = self._build_stage_instruction()
        stage_goal_instruction = self._build_stage_goal_instruction()
        reminiscence_examples = self._build_reminiscence_examples() if self.flow_stage == self.FLOW_REMINISCENCE else ""

        system_prompt = (
            "あなたは親しみやすい会話ロボットです。\n"
            "【目的】会話履歴を参照し、会話を円滑に進めながら自然に回想へつなぐ。\n"
            "【重要】返答は自然な日本語。できるだけカタカナ語を避ける。\n"
            f"{length_instruction}"
            "【形式】短いコメント→質問は最大1つ。\n"
            "【形式】質問をする場合は、コメントの後に1文だけ置く。\n"
            "【禁止】連続質問、同じ内容の聞き返し、説教、詰問。\n"
            "【禁止】直前のユーザー発話をそのまま繰り返す・引用する応答。\n"
            "【回想法ルール】記憶の正確さを試さない。答えやすい語りを優先する。\n"
            "【回想法ルール】正解確認・年号確認・矛盾指摘はしない。\n"
            "【回想法ルール】答えづらそうなら無理に深掘りせず、最近の話題へ戻す。\n"
            "【回想法ルール】回想質問は、情景・人物・気持ち・意味のいずれか1点に絞る。\n"
            "\n"
            f"【現在の話したい度】{self.p_want_talk:.2f}\n"
            f"【話したい度の最近の上昇量】{self._delta_window():.2f}\n"
            f"【直近の観測ラベル】{obs.action_type.value}\n"
            f"{stage_instruction}"
            f"{stage_goal_instruction}\n"
            "【会話メモ（要約）】\n"
            f"{summary_text}\n"
            "【繰り返し禁止（すでに確認済み）】\n"
            f"{do_not_ask_text}\n"
            "【重要】上の『繰り返し禁止』に含まれる内容は、言い換えても再質問しない。\n"
            f"{self._build_output_control_instruction(strict_output=False)}"
            f"{reminiscence_examples}"
            f"{interaction_mode}\n"
        )

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)

        if obs.user_text:
            user_content = obs.user_text
        else:
            user_content = "(ユーザーは返答を考えています。急かさず、短い待機のひと言だけ返してください。)"

        messages.append({"role": "user", "content": user_content})

        try:
            reply, raw_text = self._generate_once(messages, strict_output=False)
            if reply:
                return reply

            retry_messages = list(messages)
            retry_messages[0] = {
                "role": "system",
                "content": system_prompt.replace(
                    self._build_output_control_instruction(strict_output=False),
                    self._build_output_control_instruction(strict_output=True),
                ),
            }
            self.logger.warning("Qwen3.5-27Bの本文抽出に失敗したため、thinking ON のまま再生成します。")
            retry_reply, retry_raw_text = self._generate_once(retry_messages, strict_output=True)
            if retry_reply:
                return retry_reply

            fallback_text = build_qwen_display_fallback_text(retry_raw_text or raw_text)
            if fallback_text:
                self.logger.warning("Qwen3.5-27Bの本文抽出に再度失敗したため、生返答をそのまま表示します。")
                return fallback_text
            return build_qwen_display_fallback_text(raw_text)
        except Exception as e:
            self.logger.warning("ローカルQwen3.5-27B生成失敗: %s", e)
            return "ごめんなさい、少し調子が悪いみたいです。落ち着いたらまた話しかけてくださいね。"

    def run(self) -> None:
        self._banner(self.BANNER_TITLE)
        print("  空Enter = 考え中として待機 / Ctrl+C = 終了")
        print(f"  最大ターン数: {self.max_total_turns}")
        print(f"  ローカルモデル: {self.local_model_id}")
        print(f"  thinkingモード: {'ON' if self.enable_thinking else 'OFF'}")
        print(f"  ログ出力先: {self.run_dir}/")
        print()

        initial_greeting = "こんにちは。今日は気軽にお話ししましょう。"
        self.output(initial_greeting)
        self.log_interaction("AI", initial_greeting, "")

        try:
            while True:
                user_text = self.get_input()
                if not user_text:
                    waiting_obs = Observation(user_text=None, action_type=ActionType.MINIMAL)
                    wait_reply = self.think_and_reply(waiting_obs, waiting_mode=True, ending_mode=self.force_end)
                    self.output(wait_reply)
                    if self.force_end:
                        print("\n会話を終了します。ログは以下に保存されています:")
                        print(f"  {self.run_dir}/")
                        break
                    continue

                self.total_turns += 1
                self.update_conv_memory(user_text)

                if self.conv_memory.stop_intent:
                    self.force_end = True

                classification = self.classify_action(user_text)
                action_type = classification.action_type

                prior = float(self.p_want_talk)
                self.update_posterior(action_type)
                posterior = float(self.p_want_talk)
                self.p_history.append(self.p_want_talk)
                if action_type in (ActionType.MINIMAL, ActionType.DISENGAGE):
                    self.low_engagement_streak += 1
                else:
                    self.low_engagement_streak = 0

                self.display_status(action_type, prior, posterior, classification.reason)
                self.log_interaction("User", user_text, action_type.value, classification.reason or "")

                obs = Observation(
                    user_text=user_text,
                    action_type=action_type,
                    label_reason=classification.reason,
                )

                if self.total_turns >= self.max_total_turns:
                    self.force_end = True
                if not self.force_end:
                    self._update_flow_stage(obs)

                reply = self.think_and_reply(obs, ending_mode=self.force_end)
                self.output(reply)
                self.log_interaction("AI", reply, "")

                if self.force_end:
                    print("\n会話を終了します。ログは以下に保存されています:")
                    print(f"  {self.run_dir}/")
                    break

        except KeyboardInterrupt:
            print("\n\n中断しました。ログは以下に保存されています:")
            print(f"  {self.run_dir}/")


if __name__ == "__main__":
    agent = SimpleTextChatAgent()
    agent.run()
