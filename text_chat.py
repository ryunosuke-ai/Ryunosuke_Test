"""テキストベース会話ツール: 話したい度（p_want_talk）の推定精度調査用"""

import os
import sys
import io
import re
import base64
import logging
from datetime import datetime
from typing import Optional, List

from dotenv import load_dotenv
from openai import AzureOpenAI

# 文字化け対策（Windowsターミナル想定）
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

load_dotenv()

from models import ActionType, Phase, PhaseConfig, Observation, MemoryUpdate  # noqa: E402
from bayes_engine import (  # noqa: E402
    update_posterior as _update_posterior,
    is_minimal_reply,
    classify_action as _classify_action,
    judge_memory_and_disclosure as _judge_memory_and_disclosure,
    DEFAULT_LIKELIHOODS,
    DEFAULT_MINIMAL_LIKELIHOOD,
)
from phase_manager import PhaseManager  # noqa: E402
from conv_memory import (  # noqa: E402
    detect_stop_intent,
    extract_recent_assistant_questions,
    update_conv_memory as _update_conv_memory,
)


class TextChatAgent:
    """
    テキスト入力ベースの会話エージェント。
    bayes_v3.py の MultimodalAgent から音声I/O・UI同期・cv2依存を除去し、
    話したい度の推移を調査しやすいステータス表示を追加したもの。
    """

    def __init__(self, image_path: str = "experiment_image.jpg"):
        self.static_image_path = image_path
        self.static_image_b64 = None

        # --- 1) 設定読み込み ---
        self.openai_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

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
            f.write("Timestamp,Turn,Phase,Speaker,ActionType,P_WantTalk,Text\n")

        self._setup_logger(ts)
        self._load_static_image()

        self.logger.info("会話ログ: %s", self.history_file)
        self.logger.info("分析用CSV: %s", self.analysis_csv)

        # --- 3) Azure OpenAI 初期化 ---
        self.openai_client = AzureOpenAI(
            api_key=self.openai_key,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=self.openai_endpoint,
        )

        # --- 4) フェーズマネージャ ---
        self.phase_mgr = PhaseManager(logger=self.logger)

        # --- 5) 状態 ---
        self.total_turns: int = 0
        self.p_want_talk: float = 0.5

        self.likelihoods = DEFAULT_LIKELIHOODS
        self.minimal_normal_likelihood = DEFAULT_MINIMAL_LIKELIHOOD

        self.asked_initial_image_question: bool = False

        # --- 6) 会話メモ ---
        self.conv_memory: MemoryUpdate = MemoryUpdate(summary="", do_not_ask=[])
        self.force_end: bool = False
        self.max_total_turns: int = 15

    # -----------------------------
    # ログ・表示
    # -----------------------------
    def _setup_logger(self, ts: str) -> None:
        self.logger = logging.getLogger("text_chat_agent")
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
    # 画像（cv2不使用）
    # -----------------------------
    def _load_static_image(self) -> None:
        if not os.path.exists(self.static_image_path):
            print(f"警告: 実験用画像が見つかりません ({self.static_image_path})")
            print("  → experiment_image.jpg をプロジェクトルートに配置してください")
            return

        try:
            with open(self.static_image_path, "rb") as f:
                self.static_image_b64 = base64.b64encode(f.read()).decode("utf-8")
            print(f"実験用画像を読み込みました: {self.static_image_path}")
        except Exception as e:
            print(f"警告: 画像の読み込みに失敗しました ({self.static_image_path}): {e}")

    # -----------------------------
    # テキスト I/O
    # -----------------------------
    def get_input(self) -> Optional[str]:
        """テキスト入力を取得。空Enterは沈黙（None）として扱う。"""
        try:
            text = input("\nあなた: ").strip()
        except EOFError:
            return None
        if text:
            self.append_to_history("User", text)
            return text
        print("  （沈黙として処理します）")
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
    def display_status(self, action_type: ActionType, minimal_reply: bool,
                       prior: float, posterior: float) -> None:
        """フェーズ・ターン・観測・話したい度バーを表示する。"""
        phase_name = self.phase_mgr.phase.value
        turn_total = self.total_turns
        turn_in_phase = self.phase_mgr.turn_in_phase

        tag = action_type.value
        if action_type == ActionType.NORMAL and minimal_reply:
            tag += "(MIN)"

        # 話したい度バー（20文字幅）
        bar_len = 20
        filled = int(posterior * bar_len)
        bar = "\u2588" * filled + "\u2591" * (bar_len - filled)

        print(f"\n--- 状態 {'─' * 32}")
        print(f"  フェーズ  : {phase_name}")
        print(f"  ターン    : {turn_total}/{self.max_total_turns} (フェーズ内: {turn_in_phase})")
        print(f"  観測      : {tag}")
        print(f"  話したい度: {prior:.2f} → {posterior:.2f}  [{bar}]")
        print(f"{'─' * 42}")

    # -----------------------------
    # 分析用データのロギング
    # -----------------------------
    def log_interaction(self, speaker: str, text: str, action_type: str = "") -> None:
        ts = datetime.now().strftime('%H:%M:%S')
        safe_text = str(text).replace('"', '""').replace('\n', ' ')
        try:
            with open(self.analysis_csv, "a", encoding="utf-8") as f:
                f.write(f"{ts},{self.total_turns},{self.phase_mgr.phase.name},{speaker},{action_type},{self.p_want_talk:.2f},\"{safe_text}\"\n")
        except Exception as e:
            self.logger.warning("CSV書き込み失敗: %s", e)

    # -----------------------------
    # ベイズ更新（委譲）
    # -----------------------------
    def update_posterior(self, action_type: ActionType, minimal_reply: bool = False) -> float:
        prior = float(self.p_want_talk)
        self.p_want_talk = _update_posterior(
            prior, action_type, minimal_reply,
            likelihoods=self.likelihoods,
            minimal_likelihood=self.minimal_normal_likelihood,
        )
        return self.p_want_talk

    # -----------------------------
    # 観測分類（委譲）
    # -----------------------------
    def classify_action(self, user_text: Optional[str]) -> ActionType:
        return _classify_action(self.openai_client, self.deployment_name, user_text, self.logger)

    def judge_memory_and_disclosure(self, user_text: Optional[str]):
        return _judge_memory_and_disclosure(self.openai_client, self.deployment_name, user_text, self.logger)

    # -----------------------------
    # フェーズ遷移（委譲）
    # -----------------------------
    def transition_policy(self, obs: Observation) -> None:
        self.phase_mgr.transition_policy(obs, self.p_want_talk)

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
            self.openai_client, self.deployment_name,
            self.conv_memory, user_text, history, recent_questions,
            self.logger,
        )

    # -----------------------------
    # 返信生成（LLM）
    # -----------------------------
    def think_and_reply(self, obs: Observation, base64_image: Optional[str]) -> str:
        cfg = self.phase_mgr.phase_configs[self.phase_mgr.phase]

        history = self.load_history_as_messages(max_messages=10)

        summary_text = self.conv_memory.summary if self.conv_memory.summary else "（まだ要約が少ないです）"
        if self.conv_memory.do_not_ask:
            do_not_ask_text = "- " + "\n- ".join(self.conv_memory.do_not_ask)
        else:
            do_not_ask_text = "（特になし）"

        initial_question_instruction = ""
        if self.phase_mgr.phase == Phase.SURROUNDINGS and not self.asked_initial_image_question:
            #initial_question_instruction = "【重要】このターンの発言の最初 または 最後に、必ず「なぜフード付きのシャツを着た男性とパイプを持っている男性が話していると思いますか？」という趣旨の質問を組み込んでください。\n"
            self.asked_initial_image_question = True

        interaction_mode = self.phase_mgr.get_interaction_mode_instruction(obs, self.p_want_talk)

        system_prompt = (
            "あなたは親しみやすい会話ロボットです。\n"
            "【重要】返答は自然な日本語。できるだけカタカナ語を避ける。\n"
            "【長さ】60〜120文字程度。\n"
            "【形式】短いコメント→（必要なら）質問は最大1つ。\n"
            "【禁止】連続質問、同じ内容の聞き返し、説教、詰問。\n"
            "\n"
            "【会話メモ（要約）】\n"
            f"{summary_text}\n"
            "【繰り返し禁止（すでに確認済み）】\n"
            f"{do_not_ask_text}\n"
            "【重要】上の『繰り返し禁止』に含まれる内容は、言い換えても再質問しない。\n"
            "すでに分かっている情報は、その前提で自然に話を進める。\n"
            "【禁止】直前のユーザー発話をそのまま繰り返す・引用する応答（例：『〜と言っていましたね』）。\n"
            "\n"
            f"【現在フェーズ】{cfg.name.value}\n"
            f"{cfg.instruction}\n"
            f"{initial_question_instruction}"
            f"{interaction_mode}\n"
        )

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)

        user_content = []
        if obs.user_text:
            user_content.append({"type": "text", "text": obs.user_text})
        else:
            user_content.append({"type": "text", "text": "(ユーザーは沈黙しています。急かさず、短い気遣いをしてください。)"})

        if base64_image:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })

        messages.append({"role": "user", "content": user_content})

        try:
            res = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                max_tokens=250,
                temperature=0.7,
            )
            return (res.choices[0].message.content or "").strip()
        except Exception as e:
            self.logger.warning("生成LLM失敗: %s", e)
            return "ごめんなさい、少し調子が悪いみたいです。落ち着いたらまた話しかけてくださいね。"

    def load_history_as_messages(self, max_messages: int = 10) -> List[dict]:
        if not os.path.exists(self.history_file):
            return []
        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                raw = f.read()
            parts = re.split(r'(\[\d{2}:\d{2}:\d{2}\] (?:User|AI):)', raw)
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
        self._banner("テキストベース会話ツール（話したい度 調査用）")
        print("  空Enter = 沈黙 / Ctrl+C = 終了")
        print(f"  最大ターン数: {self.max_total_turns}")
        print(f"  ログ出力先: {self.run_dir}/")
        if not self.static_image_b64:
            print("  ※ 画像なしで動作します（画像フェーズはスキップされる可能性があります）")
        print()

        initial_greeting = "こんにちは。たくさんお話しできたら嬉しいです！"
        self.output(initial_greeting)
        self.log_interaction("AI", initial_greeting, "")

        try:
            while True:
                self.total_turns += 1
                self.phase_mgr.turn_in_phase += 1

                user_text = self.get_input()

                if user_text:
                    self.update_conv_memory(user_text)

                if self.conv_memory.stop_intent and self.phase_mgr.phase != Phase.ENDING:
                    self.force_end = True
                    self.phase_mgr._set_phase(Phase.ENDING, reason="ユーザー終了希望")

                action_type = self.classify_action(user_text)
                minimal_reply = is_minimal_reply(user_text)

                prior = float(self.p_want_talk)
                self.update_posterior(action_type, minimal_reply=minimal_reply)
                posterior = float(self.p_want_talk)

                # 調査用ステータス表示
                self.display_status(action_type, minimal_reply, prior, posterior)

                logged_text = user_text if user_text else "(沈黙)"
                self.log_interaction("User", logged_text, action_type.value)

                memory_flag = False
                disclosure_flag = False
                note = None
                if self.phase_mgr.phase in [Phase.BRIDGE, Phase.DEEP_DIVE] and user_text:
                    memory_flag, disclosure_flag, note = self.judge_memory_and_disclosure(user_text)

                obs = Observation(
                    user_text=user_text,
                    action_type=action_type,
                    minimal_reply=minimal_reply,
                    memory_flag=memory_flag,
                    self_disclosure_flag=disclosure_flag,
                    engagement_hint=note,
                )

                if self.total_turns >= self.max_total_turns and self.phase_mgr.phase != Phase.ENDING:
                    self.force_end = True
                    self.phase_mgr._set_phase(Phase.ENDING, reason="最大ターン到達")

                if not self.force_end:
                    self.transition_policy(obs)

                img_b64 = None
                if self.phase_mgr.phase != Phase.ENDING and self.phase_mgr.phase_configs[self.phase_mgr.phase].require_image:
                    img_b64 = self.static_image_b64

                reply = self.think_and_reply(obs, img_b64)
                self.phase_mgr.notify_reply(reply)
                self.output(reply)

                self.log_interaction("AI", reply, "")

                if self.phase_mgr.phase == Phase.ENDING:
                    print("\n会話を終了します。ログは以下に保存されています:")
                    print(f"  {self.run_dir}/")
                    break

        except KeyboardInterrupt:
            print("\n\n中断しました。ログは以下に保存されています:")
            print(f"  {self.run_dir}/")


if __name__ == "__main__":
    agent = TextChatAgent(image_path="experiment_image.jpg")
    agent.run()
