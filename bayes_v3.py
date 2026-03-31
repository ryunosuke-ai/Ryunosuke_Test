import os
import sys
import io
import time
import base64
import re
import logging
from datetime import datetime
from typing import Optional, List

from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI
import cv2

# 文字化け対策（Windowsターミナル想定）
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

load_dotenv()

# 後方互換: from bayes_v3 import ActionType 等を維持
from models import ActionType, Phase, PhaseConfig, Observation, MemoryUpdate, ClassificationResult  # noqa: E402, F401
from bayes_engine import (  # noqa: E402
    update_posterior as _update_posterior,
    classify_action as _classify_action,
    judge_memory_signal as _judge_memory_signal,
    DEFAULT_LIKELIHOODS,
)
from phase_manager import PhaseManager, DEFAULT_PHASE_CONFIGS  # noqa: E402
from conv_memory import (  # noqa: E402
    detect_stop_intent,
    extract_recent_assistant_questions,
    update_conv_memory as _update_conv_memory,
)


# -----------------------------
# エージェント本体
# -----------------------------
class MultimodalAgent:
    """
    目的:
      - フェーズ（演出）を明示的に管理
      - 観測（返答/沈黙/自己開示/回想）から P(話したい) を更新（ベイズ）
      - P(話したい) とフェーズに応じて「質問する/しない」「戻る/進む」を制御
    """

    def __init__(self, image_path: str = "experiment_image.jpg"):
        self.static_image_path = image_path
        self.static_image_b64 = None

        # --- 1) 設定読み込み ---
        self.speech_key = os.getenv("AZURE_SPEECH_KEY")
        self.speech_region = os.getenv("AZURE_SPEECH_REGION")
        self.openai_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        self.openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

        missing = [k for k, v in [
            ("AZURE_SPEECH_KEY", self.speech_key),
            ("AZURE_SPEECH_REGION", self.speech_region),
            ("AZURE_OPENAI_API_KEY", self.openai_key),
            ("AZURE_OPENAI_ENDPOINT", self.openai_endpoint),
            ("AZURE_OPENAI_DEPLOYMENT_NAME", self.deployment_name),
        ] if not v]
        if missing:
            print(f"❌ エラー: .env の設定が不足しています: {', '.join(missing)}")
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
        self._load_static_image()

        self.logger.info("会話ログ: %s", self.history_file)
        self.logger.info("分析用CSV: %s", self.analysis_csv)

        # --- 3) Azure OpenAI / Speech 初期化 ---
        self.openai_client = AzureOpenAI(
            api_key=self.openai_key,
            api_version=self.openai_api_version,
            azure_endpoint=self.openai_endpoint
        )

        speech_config = speechsdk.SpeechConfig(subscription=self.speech_key, region=self.speech_region)
        speech_config.speech_recognition_language = "ja-JP"
        speech_config.speech_synthesis_voice_name = "ja-JP-NanamiNeural"

        self.initial_silence_timeout_ms = 10000
        self.segmentation_silence_timeout_ms = 1100
        speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs,
            str(self.initial_silence_timeout_ms)
        )
        speech_config.set_property(
            speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs,
            str(self.segmentation_silence_timeout_ms)
        )

        audio_in = speechsdk.audio.AudioConfig(use_default_microphone=True)
        audio_out = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
        self.recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_in)
        self.synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_out)

        # --- 4) フェーズマネージャ ---
        self.phase_mgr = PhaseManager(logger=self.logger)

        # --- 5) 状態（内部状態 + 管理変数） ---
        self.total_turns: int = 0
        self.p_want_talk: float = 0.5

        self.likelihoods = DEFAULT_LIKELIHOODS

        self.asked_initial_image_question: bool = False

        # --- 6) 会話メモ ---
        self.conv_memory: MemoryUpdate = MemoryUpdate(summary="", do_not_ask=[])
        self.force_end: bool = False
        self.max_total_turns: int = 15

    # --- 後方互換プロパティ ---
    @property
    def phase(self) -> Phase:
        return self.phase_mgr.phase

    @phase.setter
    def phase(self, value: Phase) -> None:
        self.phase_mgr.phase = value

    @property
    def turn_in_phase(self) -> int:
        return self.phase_mgr.turn_in_phase

    @turn_in_phase.setter
    def turn_in_phase(self, value: int) -> None:
        self.phase_mgr.turn_in_phase = value

    @property
    def consecutive_silence(self) -> int:
        return self.phase_mgr.consecutive_silence

    @consecutive_silence.setter
    def consecutive_silence(self, value: int) -> None:
        self.phase_mgr.consecutive_silence = value

    @property
    def bridge_fail_count(self) -> int:
        return self.phase_mgr.bridge_fail_count

    @bridge_fail_count.setter
    def bridge_fail_count(self, value: int) -> None:
        self.phase_mgr.bridge_fail_count = value

    @property
    def deep_drop_count(self) -> int:
        return self.phase_mgr.deep_drop_count

    @deep_drop_count.setter
    def deep_drop_count(self, value: int) -> None:
        self.phase_mgr.deep_drop_count = value

    @property
    def phase_configs(self):
        return self.phase_mgr.phase_configs

    @phase_configs.setter
    def phase_configs(self, value):
        self.phase_mgr.phase_configs = value

    # -----------------------------
    # ログ・表示
    # -----------------------------
    def _setup_logger(self, ts: str) -> None:
        self.logger = logging.getLogger("bayes_agent")
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
    # 音声 I/O
    # -----------------------------
    def speak(self, text: str) -> None:
        if not text or not text.strip():
            return
        print(f"\n🤖 {text}")
        self.append_to_history("AI", text)
        try:
            self.synthesizer.speak_text_async(text).get()
        except Exception as e:
            self.logger.warning("TTS失敗: %s", e)

    def listen(self) -> Optional[str]:
        wait_sec = self.initial_silence_timeout_ms // 1000
        print(f"\n🎤 マイク待機（最大 {wait_sec}s）…", flush=True)
        try:
            result = self.recognizer.recognize_once_async().get()
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                text = (result.text or "").strip()
                if text:
                    print(f"🧑 {text}")
                    self.append_to_history("User", text)
                    return text
                return None
            if result.reason == speechsdk.ResultReason.NoMatch:
                print(f"🧑 （沈黙: {wait_sec}s 発話なし）")
                return None
            return None
        except Exception as e:
            self.logger.warning("STT失敗: %s", e)
            return None

    # -----------------------------
    # 画像（実験用の静止画ロード）
    # -----------------------------
    def _load_static_image(self) -> None:
        if not os.path.exists(self.static_image_path):
            print(f"⚠️ 警告: 実験用画像が見つかりません ({self.static_image_path})")
            return

        frame = cv2.imread(self.static_image_path)
        if frame is None:
            print(f"⚠️ 警告: 画像の読み込みに失敗しました ({self.static_image_path})")
            return

        ok, buffer = cv2.imencode(".jpg", frame)
        if ok:
            self.static_image_b64 = base64.b64encode(buffer).decode("utf-8")
            print(f"✅ 実験用画像を読み込みました: {self.static_image_path}")

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
        ts = datetime.now().strftime('%H:%M:%S')
        safe_text = str(text).replace('"', '""').replace('\n', ' ')
        safe_reason = str(label_reason).replace('"', '""').replace('\n', ' ')
        try:
            with open(self.analysis_csv, "a", encoding="utf-8") as f:
                f.write(
                    f'{ts},{self.total_turns},{self.phase.name},{speaker},'
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
            prior, action_type,
            likelihoods=self.likelihoods,
        )
        print(f"📊 P(話したい) 事前 {prior:.2f} → 観測 {action_type.value} → 事後 {self.p_want_talk:.2f}")
        return self.p_want_talk

    # -----------------------------
    # 観測分類（委譲）
    # -----------------------------
    def classify_action(self, user_text: Optional[str]) -> ClassificationResult:
        return _classify_action(self.openai_client, self.deployment_name, user_text, self.logger)

    def judge_memory_signal(self, user_text: Optional[str]):
        return _judge_memory_signal(self.openai_client, self.deployment_name, user_text, self.logger)

    # -----------------------------
    # フェーズ遷移（委譲）
    # -----------------------------
    def transition_policy(self, obs: Observation) -> None:
        self.phase_mgr.transition_policy(obs, self.p_want_talk)

    def _set_phase(self, phase: Phase, reason: str = "") -> None:
        self.phase_mgr._set_phase(phase, reason)

    # -----------------------------
    # 会話メモ更新（委譲）
    # -----------------------------
    def detect_stop_intent(self, user_text: Optional[str]) -> bool:
        return detect_stop_intent(user_text)

    def _extract_recent_assistant_questions(self, max_messages: int = 12) -> List[str]:
        msgs = self.load_history_as_messages(max_messages=max_messages)
        return extract_recent_assistant_questions(msgs)

    def update_conv_memory(self, user_text: Optional[str]) -> None:
        if not user_text:
            return
        history = self.load_history_as_messages(max_messages=12)
        recent_questions = self._extract_recent_assistant_questions(max_messages=12)
        self.conv_memory = _update_conv_memory(
            self.openai_client, self.deployment_name,
            self.conv_memory, user_text, history, recent_questions,
            self.logger,
        )

    # -----------------------------
    # 返信生成（LLM）
    # -----------------------------
    def _interaction_mode_instruction(self, obs: Observation) -> str:
        return self.phase_mgr.get_interaction_mode_instruction(obs, self.p_want_talk)

    def think_and_reply(self, obs: Observation, base64_image: Optional[str], waiting_mode: bool = False) -> str:
        cfg = self.phase_configs[self.phase]

        history = self.load_history_as_messages(max_messages=10)

        summary_text = self.conv_memory.summary if self.conv_memory.summary else "（まだ要約が少ないです）"
        if self.conv_memory.do_not_ask:
            do_not_ask_text = "- " + "\n- ".join(self.conv_memory.do_not_ask)
        else:
            do_not_ask_text = "（特になし）"

        initial_question_instruction = ""
        if self.phase == Phase.SURROUNDINGS and not self.asked_initial_image_question:
            #initial_question_instruction = "【重要】このターンの発言の最初 または 最後に、必ず「なぜフード付きのシャツを着た男性とパイプを持っている男性が話していると思いますか？」という趣旨の質問を組み込んでください。\n"
            self.asked_initial_image_question = True

        waiting_instruction = ""
        if waiting_mode:
            waiting_instruction = (
                "【現在状況】ユーザーは返答内容を考えている途中です。\n"
                "【待機応答】急かさず、履歴に沿った短いひと言だけ述べて待つこと。\n"
                "【禁止】深掘りの継続、連続質問、終了誘導。\n"
                "【長さ】30〜70文字程度。\n"
            )
        length_instruction = "【長さ】30〜70文字程度。\n" if waiting_mode else "【長さ】60〜120文字程度。\n"

        system_prompt = (
            "あなたは親しみやすい会話ロボットです。\n"
            "【重要】返答は自然な日本語。できるだけカタカナ語を避ける。\n"
            f"{length_instruction}"
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
            f"{waiting_instruction if waiting_mode else self._interaction_mode_instruction(obs)}\n"
        )

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)

        user_content = []
        if obs.user_text:
            user_content.append({"type": "text", "text": obs.user_text})
        else:
            user_content.append(
                {"type": "text", "text": "(ユーザーは返答を考えています。急かさず、短い待機のひと言だけ返してください。)"}
            )

        if base64_image:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })

        messages.append({"role": "user", "content": user_content})

        try:
            print("🧠 生成中…", end="", flush=True)
            res = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                max_tokens=250,
                temperature=0.7
            )
            print("\r" + " " * 18 + "\r", end="")
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
        self._banner("会話ロボット")

        ui_flag = "ui_ready.flag"
        # 前回セッションの残存フラグを削除し、新しいUI起動を確実に待つ
        if os.path.exists(ui_flag):
            try:
                os.remove(ui_flag)
            except OSError:
                pass
        print("⏳ UI起動待機中... (streamlit run ui_display.py を実行してください)")
        while not os.path.exists(ui_flag):
            time.sleep(0.5)
        print("✅ UIが起動しました。会話を開始します。")
        try:
            os.remove(ui_flag)
        except OSError:
            pass

        initial_greeting = "こんにちは。たくさんお話しできたら嬉しいです！"
        self.speak(initial_greeting)
        self.log_interaction("AI", initial_greeting, "")

        while True:
            user_text = self.listen()
            if not user_text:
                img_b64 = None
                if self.phase != Phase.ENDING and self.phase_configs[self.phase].require_image:
                    img_b64 = self.static_image_b64
                waiting_obs = Observation(user_text=None, action_type=ActionType.MINIMAL)
                wait_reply = self.think_and_reply(waiting_obs, img_b64, waiting_mode=True)
                self.speak(wait_reply)
                if self.phase == Phase.ENDING:
                    break
                continue

            self.total_turns += 1
            self.phase_mgr.turn_in_phase += 1

            self.update_conv_memory(user_text)

            if self.conv_memory.stop_intent and self.phase != Phase.ENDING:
                self.force_end = True
                self._set_phase(Phase.ENDING, reason="ユーザー終了希望")

            classification = self.classify_action(user_text)
            action_type = classification.action_type

            self.update_posterior(action_type)

            self.log_interaction("User", user_text, action_type.value, classification.reason or "")

            memory_flag = False
            note = None
            if self.phase in [Phase.BRIDGE, Phase.DEEP_DIVE]:
                memory_flag, note = self.judge_memory_signal(user_text)

            obs = Observation(
                user_text=user_text,
                action_type=action_type,
                memory_flag=memory_flag,
                engagement_hint=note,
                label_reason=classification.reason,
            )

            if self.total_turns >= self.max_total_turns and self.phase != Phase.ENDING:
                self.force_end = True
                self._set_phase(Phase.ENDING, reason="最大ターン到達")

            if not self.force_end:
                self.transition_policy(obs)

            img_b64 = None
            if self.phase != Phase.ENDING and self.phase_configs[self.phase].require_image:
                img_b64 = self.static_image_b64

            reply = self.think_and_reply(obs, img_b64)
            self.phase_mgr.notify_reply(reply)
            self.speak(reply)

            self.log_interaction("AI", reply, "")

            if self.phase == Phase.ENDING:
                break


if __name__ == "__main__":
    agent = MultimodalAgent(image_path="experiment_image.jpg")
    agent.run()
