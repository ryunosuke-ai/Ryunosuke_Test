import os
import sys
import io
import time
import base64
import cv2
import re
import logging
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from typing import Optional, Tuple, List, Dict

from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI

# 文字化け対策（Windowsターミナル想定）
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

load_dotenv()


# -----------------------------
# データ構造
# -----------------------------
class ActionType(str, Enum):
    SILENCE = "SILENCE"
    NORMAL = "NORMAL"
    DISCLOSURE = "DISCLOSURE"


class Phase(str, Enum):
    SETUP = "事前セット"
    INTRO = "導入"
    SURROUNDINGS = "周囲の状況共有"
    BRIDGE = "連想（回想の点火）"
    DEEP_DIVE = "深掘り"
    ENDING = "エンディング"


@dataclass
class PhaseConfig:
    name: Phase
    instruction: str
    require_image: bool = False
    max_turns: int = 3


@dataclass
class Observation:
    user_text: Optional[str]
    action_type: ActionType
    memory_flag: bool = False          # 回想（過去エピソード）が出たか
    self_disclosure_flag: bool = False # 自己開示が出たか（広めに取る）
    engagement_hint: Optional[str] = None  # LLMの短い所見（任意）


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

    def __init__(self):
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
        self.image_dir = "logs/images"
        os.makedirs(self.image_dir, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.history_file = f"logs/log_{ts}.txt"
        self._setup_logger(ts)

        self.logger.info("会話ログ: %s", self.history_file)

        # --- 3) Azure OpenAI / Speech 初期化 ---
        self.openai_client = AzureOpenAI(
            api_key=self.openai_key,
            api_version=self.openai_api_version,
            azure_endpoint=self.openai_endpoint
        )

        speech_config = speechsdk.SpeechConfig(subscription=self.speech_key, region=self.speech_region)
        speech_config.speech_recognition_language = "ja-JP"
        speech_config.speech_synthesis_voice_name = "ja-JP-NanamiNeural"

        # “沈黙”を観測に使いたいので、タイムアウトを短めに
        self.initial_silence_timeout_ms = 5000
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

        # --- 4) 演出（フェーズ）定義：改善案ベース ---
        self.phase_configs: Dict[Phase, PhaseConfig] = {
            Phase.SETUP: PhaseConfig(
                name=Phase.SETUP,
                require_image=False,
                max_turns=1,
                instruction=(
                    "目的共有（超短く）：\n"
                    "・これから周りを見ながら、思い出話につながる会話をしたい\n"
                    "・無理に答えなくてよい、話したい範囲でOK\n"
                ),
            ),
            Phase.INTRO: PhaseConfig(
                name=Phase.INTRO,
                require_image=False,
                max_turns=2,
                instruction=(
                    "挨拶と関係づくり。\n"
                    "・明るく挨拶\n"
                    "・体調/気分を軽く確認（重くしない）\n"
                    "・質問は最大1つまで（圧をかけない）\n"
                ),
            ),
            Phase.SURROUNDINGS: PhaseConfig(
                name=Phase.SURROUNDINGS,
                require_image=True,
                max_turns=3,
                instruction=(
                    "共同注意（周囲共有）。\n"
                    "・画像の中の要素に短く言及して、相手が参加しやすい言い方にする\n"
                    "・『説明』に寄りすぎない（相手が入れる余白を作る）\n"
                    "・質問は最大1つ（はい/いいえ、二択など負担が小さい形が望ましい）\n"
                ),
            ),
            Phase.BRIDGE: PhaseConfig(
                name=Phase.BRIDGE,
                require_image=True,
                max_turns=4,
                instruction=(
                    "連想（回想の点火）。\n"
                    "・周囲の話題を足場に、過去の体験や好みへ“自然に”つなぐ\n"
                    "・いきなり深い記憶を聞かない（場所→季節→行事→食→人 など足場）\n"
                    "・回想が出たら深掘りへ。出なければ周囲共有へ戻るのもOK\n"
                ),
            ),
            Phase.DEEP_DIVE: PhaseConfig(
                name=Phase.DEEP_DIVE,
                require_image=False,
                max_turns=6,
                instruction=(
                    "深掘り。\n"
                    "・ユーザーの話の内容を中心に、共感→要約→（必要なら）1つだけ質問\n"
                    "・『深める（感情/意味）』と『広げる（関連記憶）』を状況で切替\n"
                    "・反応が落ちたら周囲共有に戻してよい\n"
                ),
            ),
            Phase.ENDING: PhaseConfig(
                name=Phase.ENDING,
                require_image=False,
                max_turns=2,
                instruction=(
                    "エンディング。\n"
                    "・話を続けようとせず、感想/感謝/今日のキーワードを1つ残して終える\n"
                    "・質問はしない\n"
                ),
            ),
        }

        # --- 5) 状態（内部状態 + 管理変数） ---
        self.phase: Phase = Phase.SETUP
        self.turn_in_phase: int = 0
        self.total_turns: int = 0
        self.consecutive_silence: int = 0

        # 内部状態：話したい度（事前）
        self.p_want_talk: float = 0.5

        # 尤度（観測モデル）
        # ここは実験で調整する前提。まずは「自己開示/回想が出るほど話したい確率↑」になる形。
        self.likelihoods = {
            "H1": {ActionType.SILENCE: 0.12, ActionType.NORMAL: 0.33, ActionType.DISCLOSURE: 0.55},
            "H0": {ActionType.SILENCE: 0.55, ActionType.NORMAL: 0.40, ActionType.DISCLOSURE: 0.05},
        }

        # ループ制御用
        self.bridge_fail_count: int = 0
        self.deep_drop_count: int = 0

    # -----------------------------
    # ログ・表示
    # -----------------------------
    def _setup_logger(self, ts: str) -> None:
        self.logger = logging.getLogger("bayes_agent")
        self.logger.setLevel(logging.INFO)

        fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S")
        file_handler = logging.FileHandler(f"logs/agent_{ts}.log", encoding="utf-8")
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
    # 画像
    # -----------------------------
    def capture_image(self) -> Optional[str]:
        print("📷 画像取得…", end="", flush=True)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("\r⚠️ カメラが開けませんでした")
            return None

        time.sleep(0.4)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            print("\r⚠️ 撮影に失敗しました")
            return None

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.image_dir}/img_{ts}.jpg"
        try:
            cv2.imwrite(filename, frame)
        except Exception:
            pass

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            print("\r⚠️ 画像エンコード失敗")
            return None

        b64 = base64.b64encode(buffer).decode("utf-8")
        print(f"\r✅ 画像OK（保存: {filename}）", flush=True)
        return b64

    # -----------------------------
    # ベイズ更新
    # -----------------------------
    def update_posterior(self, action_type: ActionType) -> float:
        prior = float(self.p_want_talk)
        l_h1 = float(self.likelihoods["H1"][action_type])
        l_h0 = float(self.likelihoods["H0"][action_type])

        evidence = (l_h1 * prior) + (l_h0 * (1.0 - prior))
        if evidence <= 1e-12:
            posterior = prior
        else:
            posterior = (l_h1 * prior) / evidence

        self.p_want_talk = max(0.001, min(0.999, posterior))
        print(f"📊 P(話したい) 事前 {prior:.2f} → 観測 {action_type.value} → 事後 {self.p_want_talk:.2f}")
        return self.p_want_talk

    # -----------------------------
    # 観測分類（LLM）
    # -----------------------------
    def classify_action(self, user_text: Optional[str]) -> ActionType:
        if user_text is None:
            return ActionType.SILENCE

        # 速さ優先：まずは簡易ヒューリスティック
        short = len(user_text) <= 8
        if short and any(w in user_text for w in ["はい", "うん", "そう", "ありがとう", "ええ", "なるほど"]):
            return ActionType.NORMAL

        # LLM判定（DISCLOSURE/NORMAL）
        prompt = (
            "次のユーザー発言を分類してください。\n"
            "・DISCLOSURE: 過去の思い出、個人的エピソード、好み、考え、感情が“少しでも”含まれる。\n"
            "・NORMAL: 相づち、短い返答、事実のみ、内容が浅い。\n"
            "出力は DISCLOSURE か NORMAL のどちらか1語のみ。"
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_text}
        ]
        try:
            res = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                max_tokens=6,
                temperature=0
            )
            out = (res.choices[0].message.content or "").strip().upper()
            if "DISCLOSURE" in out:
                return ActionType.DISCLOSURE
            return ActionType.NORMAL
        except Exception as e:
            self.logger.warning("分類LLM失敗: %s", e)
            return ActionType.NORMAL

    def judge_memory_and_disclosure(self, user_text: Optional[str]) -> Tuple[bool, bool, Optional[str]]:
        """
        BRIDGE/DEEP_DIVE で使う観測。
        memory_flag: 過去の出来事として語られているか（回想）
        self_disclosure_flag: 好み/考え/感情など自己開示があるか
        """
        if not user_text:
            return False, False, None

        prompt = (
            "次のユーザー発言について判定してください。\n"
            "1) memory_flag: 過去の出来事・体験として語っているなら true。\n"
            "2) self_disclosure_flag: 好み/考え/感情など自己開示が含まれるなら true。\n"
            "出力は JSON のみ: {\"memory_flag\": true/false, \"self_disclosure_flag\": true/false, \"note\": \"短い所見（任意）\"}\n"
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_text}
        ]
        try:
            res = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                max_tokens=80,
                temperature=0
            )
            text = (res.choices[0].message.content or "").strip()
            # 雑に JSON 抽出（失敗しても落とさない）
            m1 = re.search(r'"memory_flag"\s*:\s*(true|false)', text, re.IGNORECASE)
            m2 = re.search(r'"self_disclosure_flag"\s*:\s*(true|false)', text, re.IGNORECASE)
            note = None
            m3 = re.search(r'"note"\s*:\s*"([^"]*)"', text)
            if m3:
                note = m3.group(1)
            mem = (m1.group(1).lower() == "true") if m1 else False
            dis = (m2.group(1).lower() == "true") if m2 else False
            return mem, dis, note
        except Exception as e:
            self.logger.warning("回想判定LLM失敗: %s", e)
            return False, False, None

    # -----------------------------
    # フェーズ遷移（演出制御）
    # -----------------------------
    def transition_policy(self, obs: Observation) -> None:
        """
        フェーズ遷移の核：
          - BRIDGEは「回想が出るまで粘る」が、失敗が続けばSURROUNDINGSへ戻る
          - DEEP_DIVEで反応が落ちたら、周囲共有に戻す or 終了
          - 沈黙が続いたら質問を減らし、周囲共有か終了へ
        """
        # 沈黙カウント
        if obs.action_type == ActionType.SILENCE:
            self.consecutive_silence += 1
        else:
            self.consecutive_silence = 0

        # 連続沈黙が続くなら負荷を下げる
        if self.consecutive_silence >= 2 and self.phase not in [Phase.ENDING]:
            # 話したい度がかなり低いなら締める
            if self.p_want_talk < 0.20:
                self._set_phase(Phase.ENDING, reason="沈黙が続き、負担が高そう")
                return
            # それ以外は周囲共有へ戻して低負荷に
            self._set_phase(Phase.SURROUNDINGS, reason="沈黙が続いたため、周囲共有へ戻す")
            return

        cfg = self.phase_configs[self.phase]

        # フェーズ内ターン数上限
        if self.turn_in_phase >= cfg.max_turns and self.phase not in [Phase.BRIDGE, Phase.DEEP_DIVE]:
            # ざっくり次へ
            nxt = self._next_phase_linear(self.phase)
            self._set_phase(nxt, reason="フェーズ上限ターン到達")
            return

        # 個別ルール
        if self.phase == Phase.SETUP:
            self._set_phase(Phase.INTRO, reason="セット完了")
            return

        if self.phase == Phase.INTRO:
            # 返答があれば周囲共有へ
            if obs.action_type != ActionType.SILENCE:
                self._set_phase(Phase.SURROUNDINGS, reason="導入完了")
            return

        if self.phase == Phase.SURROUNDINGS:
            # ほどほどにやったら連想へ
            if self.turn_in_phase >= 1 and obs.action_type != ActionType.SILENCE:
                self._set_phase(Phase.BRIDGE, reason="共同注意ができたので連想へ")
            return

        if self.phase == Phase.BRIDGE:
            if obs.memory_flag or obs.action_type == ActionType.DISCLOSURE:
                self.bridge_fail_count = 0
                self._set_phase(Phase.DEEP_DIVE, reason="回想が出たため深掘りへ")
                return
            # 回想が出ない場合：粘るが、失敗が続けば戻す
            if obs.action_type == ActionType.SILENCE or self.p_want_talk < 0.35:
                self.bridge_fail_count += 1
            if self.bridge_fail_count >= 2:
                self.bridge_fail_count = 0
                self._set_phase(Phase.SURROUNDINGS, reason="回想が出にくいので周囲共有に戻す")
                return
            # それ以外は同フェーズ継続
            return

        if self.phase == Phase.DEEP_DIVE:
            # 深掘り中に反応が落ちたら「広げる/戻る/締める」
            if obs.action_type == ActionType.DISCLOSURE:
                self.deep_drop_count = 0
                return
            if obs.action_type in [ActionType.NORMAL, ActionType.SILENCE]:
                self.deep_drop_count += 1
            if self.deep_drop_count >= 2:
                if self.p_want_talk < 0.30:
                    self._set_phase(Phase.ENDING, reason="深掘りで反応が落ちたため終了へ")
                else:
                    self._set_phase(Phase.SURROUNDINGS, reason="深掘りで反応が落ちたため周囲共有へ戻す")
                self.deep_drop_count = 0
            return

        # ENDING は維持
        return

    def _next_phase_linear(self, phase: Phase) -> Phase:
        order = [Phase.SETUP, Phase.INTRO, Phase.SURROUNDINGS, Phase.BRIDGE, Phase.DEEP_DIVE, Phase.ENDING]
        i = order.index(phase)
        return order[min(i + 1, len(order) - 1)]

    def _set_phase(self, phase: Phase, reason: str = "") -> None:
        if self.phase != phase:
            self.logger.info("⏩ フェーズ遷移: %s → %s（%s）", self.phase.value, phase.value, reason)
            self.phase = phase
            self.turn_in_phase = 0

    # -----------------------------
    # 返信生成（LLM）
    # -----------------------------
    def _interaction_mode_instruction(self, obs: Observation) -> str:
        """
        質問をしすぎないためのモード。
        """
        p = self.p_want_talk
        if self.phase == Phase.ENDING:
            return "【モード】終了：質問禁止。感想と感謝で閉じる。"

        if obs.action_type == ActionType.SILENCE:
            return (
                "【モード】低負荷：相手を急かさない。\n"
                "・短い気遣い＋沈黙の余白\n"
                "・質問はしない（しても『大丈夫？』の1回まで）"
            )

        if p >= 0.70:
            return (
                "【モード】共感重視。\n"
                "・共感/要約/感想を中心\n"
                "・質問は最大1つ（できればしない）"
            )
        if p >= 0.40:
            return (
                "【モード】軽い誘導。\n"
                "・コメント→小さな質問1つ（はい/いいえ、二択など）\n"
                "・押し付けない"
            )
        return (
            "【モード】再点火（負担は小さく）。\n"
            "・周囲要素を1つ拾って短いコメント\n"
            "・質問するなら『二択』か『はい/いいえ』にする\n"
            "・連続質問は禁止"
        )

    def think_and_reply(self, obs: Observation, base64_image: Optional[str]) -> str:
        cfg = self.phase_configs[self.phase]

        # 会話履歴（直近のみ）
        history = self.load_history_as_messages(max_messages=10)

        system_prompt = (
            "あなたは親しみやすい会話ロボットです。\n"
            "【重要】返答は自然な日本語。できるだけカタカナ語を避ける。\n"
            "【長さ】60〜110文字程度。\n"
            "【形式】まず短いコメント→（必要なら）質問は最大1つ。\n"
            "【禁止】質問を連続で投げない。ユーザーを責めない。\n"
            "\n"
            f"【現在フェーズ】{cfg.name.value}\n"
            f"{cfg.instruction}\n"
            f"{self._interaction_mode_instruction(obs)}\n"
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
        self._banner("確率モデル付き 会話ロボット（改善演出版）")
        self.speak("こんにちは。周りを見ながら、思い出話につながるお話ができたら嬉しいです。無理のない範囲で大丈夫ですよ。")

        while True:
            self.total_turns += 1
            self.turn_in_phase += 1

            # 終了ワード（音声は句点が付く場合あり）
            user_text = self.listen()
            if user_text and any(k in user_text for k in ["終了", "終わり", "ストップ", "やめる"]):
                self._set_phase(Phase.ENDING, reason="ユーザー終了希望")

            # 観測：基本分類
            action_type = self.classify_action(user_text)

            # ベイズ更新
            self.update_posterior(action_type)

            # 追加観測（BRIDGE / DEEP_DIVEのみ）
            memory_flag = False
            disclosure_flag = False
            note = None
            if self.phase in [Phase.BRIDGE, Phase.DEEP_DIVE] and user_text:
                memory_flag, disclosure_flag, note = self.judge_memory_and_disclosure(user_text)

            obs = Observation(
                user_text=user_text,
                action_type=action_type,
                memory_flag=memory_flag,
                self_disclosure_flag=disclosure_flag,
                engagement_hint=note
            )

            # フェーズ遷移
            self.transition_policy(obs)

            # 画像取得（必要なフェーズのみ）
            img_b64 = None
            if self.phase_configs[self.phase].require_image:
                img_b64 = self.capture_image()

            # 返信生成
            reply = self.think_and_reply(obs, img_b64)
            self.speak(reply)

            # エンディング到達なら、最後の一言で閉じる
            if self.phase == Phase.ENDING:
                closing = "今日はここまでにしましょう。話してくれてありがとうございます。また少しずつ聞かせてくださいね。"
                self.speak(closing)
                break

            # 暴走防止（長くなりすぎたら終了へ）
            if self.total_turns >= 30:
                self._set_phase(Phase.ENDING, reason="最大ターン到達")
                continue


if __name__ == "__main__":
    agent = MultimodalAgent()
    agent.run()
