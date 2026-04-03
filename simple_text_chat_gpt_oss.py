"""フェーズ遷移なしのテキスト会話ツール（話したい度推定あり）"""

import os
import sys
import io
import re
import logging
from collections import deque
from datetime import datetime
from typing import Optional, List, Deque

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
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ModuleNotFoundError:
    print("エラー: `transformers` / `torch` が見つかりません。")
    print("依存関係をインストールしてから再実行してください。")
    print("例: python3 -m pip install transformers torch accelerate sentencepiece")
    sys.exit(1)

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

# 文字化け対策（Windowsターミナル想定）
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

load_dotenv()

from models import ActionType, Observation, MemoryUpdate, ClassificationResult  # noqa: E402
from bayes_engine import (  # noqa: E402
    update_posterior as _update_posterior,
    classify_action as _classify_action,
    DEFAULT_LIKELIHOODS,
)
from conv_memory import (  # noqa: E402
    extract_recent_assistant_questions,
    update_conv_memory as _update_conv_memory,
)
from local_llm_utils import decode_local_llm_reply  # noqa: E402


class SimpleTextChatAgent:
    """
    フェーズ遷移を使わないテキスト会話エージェント。
    ベイズ推定による話したい度更新と会話メモ更新は維持し、
    会話履歴を参照して盛り上がる質問を生成する。
    """

    PHASE_LABEL = "SIMPLE_CHAT_GPT_OSS"
    FLOW_WARMUP = "WARMUP"
    FLOW_BRIDGE = "BRIDGE"
    FLOW_REMINISCENCE = "REMINISCENCE"

    TRANSITION_COOLDOWN_TURNS = 1

    WARMUP_TO_BRIDGE_ABS = 0.45
    WARMUP_TO_BRIDGE_FLOOR = 0.28
    WARMUP_TO_BRIDGE_DELTA = 0.15

    BRIDGE_TO_REMINISCENCE_ABS = 0.60
    BRIDGE_TO_REMINISCENCE_FLOOR = 0.38
    BRIDGE_TO_REMINISCENCE_DELTA = 0.18

    BACK_TO_WARMUP_ABS = 0.35
    DELTA_WINDOW_TURNS = 3
    DEFAULT_MODEL_ID = "openai/gpt-oss-20b"
    LOCAL_MAX_NEW_TOKENS = 180
    LOCAL_TEMPERATURE = 0.7
    LOCAL_TOP_P = 0.9

    def __init__(self):
        # --- 1) 設定読み込み ---
        self.openai_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        self.local_model_id = os.getenv("LOCAL_GPT_OSS_MODEL_ID", self.DEFAULT_MODEL_ID)

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

        # --- 4) ローカルgpt-oss 初期化（返信生成で使用） ---
        if not torch.cuda.is_available():
            print("エラー: CUDA対応GPUが見つかりません。")
            print("gpt-oss-20b のローカル実行には GPU 環境が必要です。")
            print("`nvidia-smi` と `python3 -c 'import torch; print(torch.cuda.is_available())'` を確認してください。")
            sys.exit(1)
        self.local_device = "cuda"
        self.tokenizer = None
        self.local_model = None
        self._load_local_gpt_oss()

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

    def _load_local_gpt_oss(self) -> None:
        """ローカルgpt-ossモデルを読み込む。"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_id, trust_remote_code=True)
            enable_4bit = os.getenv("LOCAL_GPT_OSS_ENABLE_4BIT", "0") == "1"
            loaded_with_4bit = False

            if BitsAndBytesConfig is not None and enable_4bit:
                try:
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                    self.local_model = AutoModelForCausalLM.from_pretrained(
                        self.local_model_id,
                        quantization_config=quant_config,
                        device_map="auto",
                        trust_remote_code=True,
                    )
                    loaded_with_4bit = True
                    self.logger.info("gpt-ossを4bit量子化で読み込みました。")
                except Exception as quant_error:
                    self.logger.warning("4bit読み込みに失敗したため、FP16/BF16へフォールバックします: %s", quant_error)

            if not loaded_with_4bit:
                use_bf16 = torch.cuda.is_bf16_supported()
                load_dtype = torch.bfloat16 if use_bf16 else torch.float16
                self.local_model = AutoModelForCausalLM.from_pretrained(
                    self.local_model_id,
                    dtype=load_dtype,
                    device_map="auto",
                    trust_remote_code=True,
                )
                self.logger.info(
                    "gpt-ossを%sで読み込みました（4bit未使用）。",
                    "BF16" if use_bf16 else "FP16",
                )
            self.local_model.eval()
            self.logger.info("ローカルgpt-ossモデル読み込み完了: %s", self.local_model_id)
        except Exception as e:
            self.logger.error("ローカルgpt-ossモデルの読み込みに失敗: %s", e)
            print("エラー: ローカルgpt-ossモデルの読み込みに失敗しました。")
            print("先に `python3 download_gpt_oss_20b.py` を実行し、依存関係とGPU環境を確認してください。")
            print("4bitを使う場合は `LOCAL_GPT_OSS_ENABLE_4BIT=1` を付けて実行してください。")
            sys.exit(1)

    # -----------------------------
    # ログ・表示
    # -----------------------------
    def _setup_logger(self, ts: str) -> None:
        self.logger = logging.getLogger("simple_text_chat_agent")
        self.logger.setLevel(logging.INFO)

        fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S")
        file_handler = logging.FileHandler(f"{self.run_dir}/agent_{ts}.log", encoding="utf-8")
        file_handler.setFormatter(fmt)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(fmt)

        self.logger.handlers.clear()
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

    def _banner(self, title: str) -> None:
        line = "=" * 56
        print(f"\n{line}\n{title}\n{line}")

    def append_to_history(self, role: str, text: str) -> None:
        try:
            with open(self.history_file, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {role}: {text}\n")
        except Exception:
            pass

    # -----------------------------
    # テキスト I/O
    # -----------------------------
    def get_input(self) -> Optional[str]:
        """テキスト入力を取得。空Enterは考え中（None）として扱う。"""
        try:
            text = input("\nあなた: ").strip()
        except EOFError:
            return None
        if text:
            self.append_to_history("User", text)
            return text
        print("  （考え中として待機します）")
        return None

    def output(self, text: str) -> None:
        """AI応答を表示し、履歴に記録する。"""
        if not text or not text.strip():
            return
        print(f"\nAI: {text}")
        self.append_to_history("AI", text)

    # -----------------------------
    # 調査用ステータス表示
    # -----------------------------
    def display_status(
        self,
        action_type: ActionType,
        prior: float,
        posterior: float,
        label_reason: Optional[str] = None,
    ) -> None:
        """ターン・観測・話したい度バーを表示する。"""
        tag = action_type.value

        # 話したい度バー（20文字幅）
        bar_len = 20
        filled = int(posterior * bar_len)
        bar = "\u2588" * filled + "\u2591" * (bar_len - filled)

        print(f"\n--- 状態 {'─' * 32}")
        print(f"  フェーズ  : {self.PHASE_LABEL}")
        print(f"  会話段階  : {self.flow_stage}")
        print(f"  ターン    : {self.total_turns}/{self.max_total_turns}")
        print(f"  観測      : {tag}")
        if label_reason:
            print(f"  判定理由  : {label_reason}")
        print(f"  話したい度: {prior:.2f} → {posterior:.2f}  [{bar}]")
        print(f"{'─' * 42}")

    # -----------------------------
    # 分析用データのロギング
    # -----------------------------
    def log_interaction(
        self,
        speaker: str,
        text: str,
        primary_label: str = "",
        label_reason: str = "",
    ) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        safe_text = str(text).replace('"', '""').replace("\n", " ")
        safe_reason = str(label_reason).replace('"', '""').replace("\n", " ")
        try:
            with open(self.analysis_csv, "a", encoding="utf-8") as f:
                f.write(
                    f'{ts},{self.total_turns},{self.PHASE_LABEL},{speaker},'
                    f'{primary_label},"{safe_reason}",{self.p_want_talk:.2f},"{safe_text}"\n'
                )
        except Exception as e:
            self.logger.warning("CSV書き込み失敗: %s", e)

    # -----------------------------
    # ベイズ更新（委譲）
    # -----------------------------
    def update_posterior(self, action_type: ActionType) -> float:
        prior = float(self.p_want_talk)
        self.p_want_talk = _update_posterior(
            prior,
            action_type,
            likelihoods=self.likelihoods,
        )
        return self.p_want_talk

    # -----------------------------
    # 観測分類（委譲）
    # -----------------------------
    def classify_action(self, user_text: Optional[str]) -> ClassificationResult:
        return _classify_action(self.openai_client, self.deployment_name, user_text, self.logger)

    # -----------------------------
    # 会話メモ更新（委譲）
    # -----------------------------
    def update_conv_memory(self, user_text: Optional[str]) -> None:
        if not user_text:
            return
        history = self.load_history_as_messages(max_messages=12)
        msgs = self.load_history_as_messages(max_messages=12)
        recent_questions = extract_recent_assistant_questions(msgs)
        self.conv_memory = _update_conv_memory(
            self.openai_client,
            self.deployment_name,
            self.conv_memory,
            user_text,
            history,
            recent_questions,
            self.logger,
        )

    def _build_interaction_mode_instruction(self, obs: Observation, ending_mode: bool) -> str:
        """話したい度と観測ラベルに応じた会話モード指示を返す。"""
        if ending_mode:
            return (
                "【モード】終了。\n"
                "・最後の発話を受け止め、感謝して締める\n"
                "・質問禁止。新しい話題を出さない\n"
            )

        if obs.action_type == ActionType.DISENGAGE or self.p_want_talk < 0.30:
            return (
                "【モード】負荷を下げる。\n"
                "・受け止め中心で短く返す\n"
                "・質問は原則しない。する場合も確認1つだけ\n"
                "・今は回想を深めず、現在や最近の話題に戻す\n"
            )

        if obs.action_type == ActionType.MINIMAL or self.p_want_talk < 0.45:
            return (
                "【モード】軽い再点火。\n"
                "・短い共感の後、質問するならはい/いいえか二択で1つまで\n"
                "・答えやすさを優先し、負担の大きい想起は避ける\n"
            )

        if obs.action_type == ActionType.RESPONSIVE or self.p_want_talk < 0.70:
            return (
                "【モード】自然な拡張。\n"
                "・共感の後、具体化しやすい質問を1つまで\n"
            )

        return (
            "【モード】盛り上げ。\n"
            "・ユーザーの話を広げる自由回答型の質問を1つまで\n"
            "・質問がなくても自然なら無理に聞かない\n"
        )

    def _build_stage_instruction(self) -> str:
        if self.flow_stage == self.FLOW_WARMUP:
            return (
                "【会話段階】WARMUP（最近の出来事・身の回り）。\n"
                "・今日/今週/最近の出来事など、いまに近い話題を優先\n"
                "・まだ回想は無理に求めない\n"
                "・質問は生活の事実確認に寄せ、感情の深掘りは控える\n"
            )
        if self.flow_stage == self.FLOW_BRIDGE:
            return (
                "【会話段階】BRIDGE（最近の話題から過去への橋渡し）。\n"
                "・最近の出来事と似た過去体験があるかを、やわらかく確認\n"
                "・橋渡し質問は1つまで。答えづらければ現在の話題へ戻す\n"
                "・接続語を使って自然に移る（例: そういえば、ちなみに、似た場面で）\n"
            )
        return (
            "【会話段階】REMINISCENCE（回想法モード）。\n"
            "・いまは回想法（Reminiscence）に基づいて会話する\n"
            "・目的は記憶の正確さ確認ではなく、体験の想起と語りの促進\n"
            "・質問は次の4タイプのうち1つだけ選ぶ: 情景 / 人物 / 気持ち / 意味\n"
            "・話したい度が低い時は情景か人物のみ。高い時のみ気持ちや意味へ進む\n"
            "・重くなりすぎたら現在や最近の話題へ戻す\n"
        )

    def _build_stage_goal_instruction(self) -> str:
        if self.flow_stage == self.FLOW_WARMUP:
            return "【段階目標】話しやすい空気を作り、最近の出来事を具体化する。"
        if self.flow_stage == self.FLOW_BRIDGE:
            return "【段階目標】最近の話題と過去体験の接点を1つ見つける。"
        return "【段階目標】回想法として、思い出の情景・人物・気持ち・意味のいずれかを自然に語ってもらう。"

    def _build_reminiscence_examples(self) -> str:
        return (
            "【良い質問例】\n"
            "・最近よく歩く道は、昔にも似た場所がありましたか？\n"
            "・その頃の景色で、今でも浮かびやすい場面はありますか？\n"
            "・その思い出を振り返ると、どんな気持ちが残っていますか？\n"
            "【悪い質問例】\n"
            "・それは何年何月何日ですか？正確に教えてください。\n"
            "・前にも聞きましたが、もう一度同じ内容を詳しく答えてください。\n"
        )

    def _detect_reminiscence_cue(self, text: Optional[str]) -> bool:
        if not text:
            return False
        return bool(re.search(r"(昔|当時|以前|子どもの頃|学生時代|若い頃|初めて|思い出)", text))

    def _delta_window(self) -> float:
        if not self.p_history:
            return 0.0
        recent = list(self.p_history)[-self.DELTA_WINDOW_TURNS:]
        current = recent[-1]
        return current - min(recent)

    def _transition_guard(self) -> bool:
        # 遷移直後の揺れを避けるため、1ターンのクールダウンを置く
        return (self.total_turns - self.last_transition_turn) > self.TRANSITION_COOLDOWN_TURNS

    def _recovery_condition(self, floor: float, delta_threshold: float, action_type: ActionType) -> bool:
        if action_type not in (ActionType.RESPONSIVE, ActionType.ACTIVE):
            return False
        if self.low_engagement_streak > 0:
            return False
        return (self.p_want_talk >= floor) and (self._delta_window() >= delta_threshold)

    def _set_flow_stage(self, new_stage: str, reason: str) -> None:
        if self.flow_stage == new_stage:
            self.stage_turns += 1
            return
        prev = self.flow_stage
        self.flow_stage = new_stage
        self.stage_turns = 1
        self.last_transition_turn = self.total_turns
        self.logger.info(
            "会話段階遷移: %s -> %s（%s, p=%.2f, delta=%.2f）",
            prev,
            new_stage,
            reason,
            self.p_want_talk,
            self._delta_window(),
        )

    def _update_flow_stage(self, obs: Observation) -> None:
        if not self._transition_guard():
            self.stage_turns += 1
            return

        if obs.action_type in (ActionType.MINIMAL, ActionType.DISENGAGE) or self.p_want_talk < self.BACK_TO_WARMUP_ABS:
            self._set_flow_stage(self.FLOW_WARMUP, "反応低下または話したい度低下")
            return

        if self.flow_stage == self.FLOW_WARMUP:
            abs_ready = self.p_want_talk >= self.WARMUP_TO_BRIDGE_ABS
            rec_ready = self._recovery_condition(
                self.WARMUP_TO_BRIDGE_FLOOR,
                self.WARMUP_TO_BRIDGE_DELTA,
                obs.action_type,
            )
            if self.stage_turns >= 1 and (abs_ready or rec_ready):
                reason = "絶対閾値到達" if abs_ready else "回復量条件到達"
                self._set_flow_stage(self.FLOW_BRIDGE, reason)
                return
            self.stage_turns += 1
            return

        if self.flow_stage == self.FLOW_BRIDGE:
            abs_ready = self.p_want_talk >= self.BRIDGE_TO_REMINISCENCE_ABS
            rec_ready = self._recovery_condition(
                self.BRIDGE_TO_REMINISCENCE_FLOOR,
                self.BRIDGE_TO_REMINISCENCE_DELTA,
                obs.action_type,
            )
            cue_ready = self._detect_reminiscence_cue(obs.user_text) or obs.action_type == ActionType.ACTIVE
            if cue_ready and (abs_ready or rec_ready):
                reason = "絶対閾値+橋渡し手がかり" if abs_ready else "回復量+橋渡し手がかり"
                self._set_flow_stage(self.FLOW_REMINISCENCE, reason)
                return
            self.stage_turns += 1
            return

        # REMINISCENCE維持
        self.stage_turns += 1

    # -----------------------------
    # 返信生成（LLM）
    # -----------------------------
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
            model_inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            model_inputs = {k: v.to(self.local_device) for k, v in model_inputs.items()}
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
            reply = decode_local_llm_reply(self.tokenizer, generated)
            if not reply:
                return "少し考えがまとまりませんでした。よければ、もう一度だけ聞かせてください。"
            return reply
        except Exception as e:
            self.logger.warning("ローカルgpt-oss生成失敗: %s", e)
            return "ごめんなさい、少し調子が悪いみたいです。落ち着いたらまた話しかけてくださいね。"

    def load_history_as_messages(self, max_messages: int = 10) -> List[dict]:
        if not os.path.exists(self.history_file):
            return []
        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                raw = f.read()
            parts = re.split(r"(\[\d{2}:\d{2}:\d{2}\] (?:User|AI):)", raw)
            messages = []
            role = None
            for p in parts:
                if not p.strip():
                    continue
                if "User:" in p:
                    role = "user"
                elif "AI:" in p:
                    role = "assistant"
                else:
                    if role:
                        messages.append({"role": role, "content": p.strip()})
            return messages[-max_messages:]
        except Exception:
            return []

    # -----------------------------
    # メインループ
    # -----------------------------
    def run(self) -> None:
        self._banner("単純会話ツール gpt-oss版（フェーズ遷移なし）")
        print("  空Enter = 考え中として待機 / Ctrl+C = 終了")
        print(f"  最大ターン数: {self.max_total_turns}")
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
