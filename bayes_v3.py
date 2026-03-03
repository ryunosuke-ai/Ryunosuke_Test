import os
import sys
import io
import time
import json
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
    SETUP = "初期設定"
    INTRO = "挨拶とアイスブレイク"
    SURROUNDINGS = "画像についての話題"
    BRIDGE = "連想（体験や好みへの接続）"
    DEEP_DIVE = "エピソードの深掘り"
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
    minimal_reply: bool = False        # 「はい/そうですね/ありがとうございます」など会話が広がりにくい生返事
    memory_flag: bool = False          # 回想（過去エピソード）が出たか
    self_disclosure_flag: bool = False # 自己開示が出たか（広めに取る）
    engagement_hint: Optional[str] = None  # LLMの短い所見（任意）



@dataclass
class MemoryUpdate:
    summary: str
    do_not_ask: List[str]
    stop_intent: bool = False


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
        # 実行ごとのフォルダを作成
        self.run_dir = f"logs/run_{ts}"
        os.makedirs(self.run_dir, exist_ok=True)

        # 古いカメラ保存用の処理（静止画を使うので不要になったがディレクトリ作りのため一応残すか、削除する）
        # self.image_dir = f"{self.run_dir}/images"
        # os.makedirs(self.image_dir, exist_ok=True)

        self.history_file = f"{self.run_dir}/log_{ts}.txt"
        self.analysis_csv = f"{self.run_dir}/analysis_{ts}.csv"

        # 分析用CSVの初期化
        with open(self.analysis_csv, "w", encoding="utf-8") as f:
            f.write("Timestamp,Turn,Phase,Speaker,ActionType,P_WantTalk,Text\n")

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
"・これから周りを一緒に見ながら、思い出話につながる会話をします\n"
"・答えづらい時は無理に答えなくてOK（沈黙でもOK）\n"
"・質問は基本しない（挨拶だけで十分）\n"
"・同じことを聞き返さないよう、会話メモを参照する\n"

                ),
            ),
            Phase.INTRO: PhaseConfig(
                name=Phase.INTRO,
                require_image=False,
                max_turns=2,
                instruction=(
                    "挨拶と簡単なアイスブレイク。\n"
"・明るく挨拶し、相手の様子を一言で受け止める\n"
"・確認は軽く：体調/気分を1つだけ（例：今日はどんな気分？）\n"
"・返事が短い/薄い場合は、質問を重ねずコメントで次へ\n"
"・質問は最大1つ。連続質問は禁止\n"
"・すでに聞いた項目（do_not_ask）は絶対に再質問しない\n"

                ),
            ),
            Phase.SURROUNDINGS: PhaseConfig(
                name=Phase.SURROUNDINGS,
                require_image=True,
                max_turns=3,
                instruction=(
                    "画像についての話（共同注意）。\n"
"・ユーザーの答えを受け止め、画像の中の要素を基に短いコメントを返す\n"
"・ユーザーが入りやすい余白を作る（間、共感、短い確認）\n"
"・質問するなら負担小：はい/いいえ、二択、指差しレベルの1問だけ\n"
"・会話メモにある内容（例：今日の予定）は繰り返し聞かず、言及する形でつなぐ\n"
"・反応が薄いなら、質問を減らしてコメント中心にする\n"

                ),
            ),
            Phase.BRIDGE: PhaseConfig(
                name=Phase.BRIDGE,
                require_image=True,
                max_turns=4,
                instruction=(
                    "連想（回想の点火）。\n"
"・周囲の要素→連想の足場→過去/好みへ“自然に”つなぐ\n"
"・足場例：場所→季節→行事→食べ物→人（いきなり深い話は聞かない）\n"
"・回想/自己開示が出たら深掘りへ。出なければ周囲共有に戻ってよい\n"
"・質問は最大1つ。連続質問は禁止。相手の負担が高そうなら質問を減らす\n"
"・【重要】会話メモ（すでに確認済み）にユーザーの好み（例：コーヒーが好き等）が書かれている場合、その前提で話を進めること。絶対に同じ質問（「何を飲みたいか」など）を再度ゼロから聞かないこと。\n"

                ),
            ),
            Phase.DEEP_DIVE: PhaseConfig(
                name=Phase.DEEP_DIVE,
                require_image=False,
                max_turns=6,
                instruction=(
                    "深掘り。\n"
"・基本は 共感→要約→（必要なら）質問1つ。質問より理解を優先\n"
"・深める：感情/意味（例：その時どんな気持ち？）\n"
"・広げる：関連記憶（例：似た体験は他にも？）\n"
"・整理：短い要約を返して確認（例：つまり〜ということですね）\n"
"・反応が落ちたら、深掘りをやめて周囲共有へ戻す/終盤へ移る\n"
"・過去に答えた質問を言い換えて再質問しない\n"

                ),
            ),
            Phase.ENDING: PhaseConfig(
                name=Phase.ENDING,
                require_image=False,
                max_turns=2,
                instruction=(
                    "エンディング（必ず終える）。\n"
"・ユーザーが終えたいと言ったら、最優先で丁寧に締める\n"
"・感謝→短いまとめ→終了の挨拶\n"
"・質問は禁止。新しい話題を出さない\n"
"・相手が追加で話し出しても、短く受け止めて締め直す\n"

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
        # ここは実験で調整する前提。
        # 重要：このモデルでは「生返事（特に『はい』『そうですね』のような最小応答）」を
        #       “会話が弾んでいないサイン”として強く扱い、P(話したい) を下げやすくする。
        self.likelihoods = {
            # H1 = 話したい時に出やすい観測
            "H1": {
                ActionType.SILENCE: 0.10,
                ActionType.NORMAL: 0.25,          # 0.35 -> 0.25 (NORMALはH1で出にくくする)
                ActionType.DISCLOSURE: 0.65       # 0.55 -> 0.65
            },
            # H0 = 話したくない時に出やすい観測
            "H0": {
                ActionType.SILENCE: 0.45,         # 0.55 -> 0.45
                ActionType.NORMAL: 0.50,          # 0.40 -> 0.50 (NORMALはH0で出やすくし、Pを下げる圧力を強める)
                ActionType.DISCLOSURE: 0.05
            },
        }

        # 「最小応答（生返事）」に対する尤度だけ上書きする
        self.minimal_normal_likelihood = {
            "H1": 0.05,
            "H0": 0.70,  # 0.60 -> 0.70 (より一気に下がりやすくする)
        }


        # ループ制御用
        self.bridge_fail_count: int = 0
        self.deep_drop_count: int = 0
        self.asked_initial_image_question: bool = False  # 最初の画像質問をしたか

        # --- 6) 会話メモ（繰り返し防止・要約・終了判定） ---
        self.conv_memory: MemoryUpdate = MemoryUpdate(summary="", do_not_ask=[])
        self.force_end: bool = False  # ユーザーの終了意思などで強制的に終わる
        self.max_total_turns: int = 15  # 暴走防止兼、実験の長さ上限


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
    def log_interaction(self, speaker: str, text: str, action_type: str = "") -> None:
        ts = datetime.now().strftime('%H:%M:%S')
        # 改行やダブルクォーテーションをエスケープしてCSVセーフにする
        safe_text = str(text).replace('"', '""').replace('\n', ' ')
        try:
            with open(self.analysis_csv, "a", encoding="utf-8") as f:
                f.write(f"{ts},{self.total_turns},{self.phase.name},{speaker},{action_type},{self.p_want_talk:.2f},\"{safe_text}\"\n")
        except Exception as e:
            self.logger.warning("CSV書き込み失敗: %s", e)

    # -----------------------------
    # ベイズ更新
    # -----------------------------

    def update_posterior(self, action_type: ActionType, minimal_reply: bool = False) -> float:
        prior = float(self.p_want_talk)

        # 基本の尤度
        l_h1 = float(self.likelihoods["H1"][action_type])
        l_h0 = float(self.likelihoods["H0"][action_type])

        # 「生返事（最小応答）」は別の尤度で強めに下げる
        if action_type == ActionType.NORMAL and minimal_reply:
            l_h1 = float(self.minimal_normal_likelihood["H1"])
            l_h0 = float(self.minimal_normal_likelihood["H0"])

        evidence = (l_h1 * prior) + (l_h0 * (1.0 - prior))
        if evidence <= 1e-12:
            posterior = prior
        else:
            posterior = (l_h1 * prior) / evidence

        self.p_want_talk = max(0.001, min(0.999, posterior))
        tag = f"{action_type.value}" + ("(MIN)" if (action_type == ActionType.NORMAL and minimal_reply) else "")
        print(f"📊 P(話したい) 事前 {prior:.2f} → 観測 {tag} → 事後 {self.p_want_talk:.2f}")
        return self.p_want_talk


    # -----------------------------
    # 観測分類（LLM）
    # -----------------------------

    def _is_minimal_reply(self, user_text: Optional[str]) -> bool:
        """生返事（会話が広がりにくい最小応答）かどうか。"""
        if not user_text:
            return False
        t = user_text.strip()
        # 句読点や空白を除いた短文
        t2 = re.sub(r"[\s。．、,！!？?…]+", "", t)
        if len(t2) <= 6 and re.fullmatch(r"(はい|うん|そう|そうです|そうですね|なるほど|ありがとう|ありがとうございます|ええ|うーん)", t2):
            return True
        # 「はい。」のようなパターン
        if len(t2) <= 4 and any(w == t2 for w in ["はい", "うん", "そう", "ええ"]):
            return True
        return False

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
            "・DISCLOSURE: ①過去の体験（昔〜した等）②個人の好み/価値観（好き/苦手/こだわり等）③感情（嬉しい/つらい等）"
            "④具体的なエピソード（いつ/どこで/誰と など）が含まれる。\n"
            "  ※『今日は部屋でゆっくりしてます』『特に予定ないです』のような“現在の状況だけ”は NORMAL。\n"
            "・NORMAL: 相づち、短い返答、現在状況の一言、事実のみで広がりにくい。\n"
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
    # 会話メモ更新（繰り返し防止・終了判定）
    # -----------------------------
    _STOP_PATTERNS = [
        r"(そろそろ|もう|一旦).{0,6}(終わり|終える|やめ|切り|終了)",
        r"(会話|話).{0,6}(やめ|終わり|終了)",
        r"(終わりに|終わりで|ここまで|ストップ|停止)",
        r"(また今度|またね|ばいばい|バイバイ)",
    ]

    def detect_stop_intent(self, user_text: Optional[str]) -> bool:
        if not user_text:
            return False
        t = user_text.strip()
        for pat in self._STOP_PATTERNS:
            if re.search(pat, t):
                return True
        # 「やめたい」など語尾変化も拾う
        if "やめたい" in t or "終わりたい" in t or "切りたい" in t:
            return True
        return False

    def _extract_recent_assistant_questions(self, max_messages: int = 12) -> List[str]:
        """
        直近のAI発話から「質問っぽい文」を抽出して、繰り返し防止に使う。
        厳密でなくてOK（落ちても会話は続く）。
        """
        msgs = self.load_history_as_messages(max_messages=max_messages)
        qs: List[str] = []
        for msg in msgs:
            if msg.get("role") != "assistant":
                continue
            text = str(msg.get("content", ""))
            candidates = re.split(r"[。\n]", text)
            for c in candidates:
                c = c.strip()
                if not c:
                    continue
                if ("？" in c) or ("ですか" in c) or ("でしょうか" in c) or c.endswith("か"):
                    if len(c) > 80:
                        c = c[:80] + "…"
                    qs.append(c)
        seen = set()
        uniq = []
        for q in qs:
            if q in seen:
                continue
            seen.add(q)
            uniq.append(q)
        return uniq[-8:]

    def update_conv_memory(self, user_text: Optional[str]) -> None:
        """
        会話の要約と「すでに聞いたこと（繰り返し禁止）」を更新。
        低コストで頑健にするため、失敗時はルールベースで最低限更新。
        """
        if not user_text:
            return

        stop_intent_rb = self.detect_stop_intent(user_text)

        history = self.load_history_as_messages(max_messages=12)
        recent_questions = self._extract_recent_assistant_questions(max_messages=12)

        prompt = (
            "あなたは会話ログを整理するアシスタントです。\n"
            "入力：既存の要約、直近の会話履歴、直近のAIの質問一覧、ユーザー発言。\n"
            "出力：JSONのみ。\n"
            "必ず次のキーを含める：\n"
            "  summary: これまでの会話の要約（日本語、350文字以内、簡潔に）\n"
            "  do_not_ask: すでにユーザーが答えた/確定した内容で、AIが繰り返し質問すべきでない『具体的な項目』の短いリスト（最大8件）。\n"
            "              ※「好きな飲み物（コーヒー）」「好きな場所（静かなカフェ）」のように、ジャンルと回答内容をセットで書くこと。\n"
            "              ※過去の要約と重複しないようにし、絶対に同じことをAIが聞かないように防御するためのリストです。\n"
            "  stop_intent: ユーザーが会話を終えたい意思があるなら true（今回の発言から判断）\n"
            "\n"
        )

        payload = {
            "current_summary": self.conv_memory.summary,
            "current_do_not_ask": self.conv_memory.do_not_ask,
            "recent_questions": recent_questions,
            "history": history[-12:],
            "user_text": user_text,
        }

        try:
            res = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
                ],
                max_tokens=260,
                temperature=0
            )
            text = (res.choices[0].message.content or "").strip()
            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not m:
                raise ValueError("JSONが見つかりません")
            data = json.loads(m.group(0))

            summary = str(data.get("summary", "")).strip()
            dna = data.get("do_not_ask", [])
            if not isinstance(dna, list):
                dna = []
            dna = [str(x).strip() for x in dna if str(x).strip()]
            dna = dna[:8]

            si = bool(data.get("stop_intent", False)) or stop_intent_rb

            self.conv_memory = MemoryUpdate(summary=summary, do_not_ask=dna, stop_intent=si)

        except Exception as e:
            self.logger.warning("会話メモ更新失敗: %s", e)
            self.conv_memory.stop_intent = stop_intent_rb or self.conv_memory.stop_intent
            if any(k in user_text for k in ["今日", "予定", "過ご", "する"]):
                if "今日の予定" not in self.conv_memory.do_not_ask:
                    self.conv_memory.do_not_ask.append("今日の予定")
            if any(k in user_text for k in ["好き", "好み", "ジャンル", "作家", "作品"]):
                if "好み（ジャンル/作家/作品）" not in self.conv_memory.do_not_ask:
                    self.conv_memory.do_not_ask.append("好み（ジャンル/作家/作品）")
            self.conv_memory.do_not_ask = self.conv_memory.do_not_ask[:8]

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


        # 生返事が続く時は、会話を止めないために「軽い誘導」に寄せる
        if obs.action_type == ActionType.NORMAL and obs.minimal_reply:
            return (
                "【モード】軽い誘導（生返事）。\n"
                "・短いコメント→負担の小さい質問を1つ（はい/いいえ or 二択）\n"
                "・同じことは聞き返さない"
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

        # 会話メモ（要約・繰り返し禁止）
        summary_text = self.conv_memory.summary if self.conv_memory.summary else "（まだ要約が少ないです）"
        if self.conv_memory.do_not_ask:
            do_not_ask_text = "- " + "\n- ".join(self.conv_memory.do_not_ask)
        else:
            do_not_ask_text = "（特になし）"

        # --- SURROUNDINGS フェーズの初回の特殊な質問指示を動的に追加 ---
        initial_question_instruction = ""
        if self.phase == Phase.SURROUNDINGS and not self.asked_initial_image_question:
            initial_question_instruction = "【重要】このターンの発言の最初 または 最後に、必ず「なぜフード付きのシャツを着た男性とパイプを持っている男性が話していると思いますか？」という趣旨の質問を組み込んでください。\n"
            self.asked_initial_image_question = True

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
            "すでに分かっている場合は『さっき〜と言っていましたね』と参照して話を進める。\n"
            "\n"
            f"【現在フェーズ】{cfg.name.value}\n"
            f"{cfg.instruction}\n"
            f"{initial_question_instruction}"
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
        self._banner("会話ロボット")

        # UI起動を待機（ui_display.py がフラグファイルを作成するまで）
        ui_flag = "ui_ready.flag"
        print("⏳ UI起動待機中... (streamlit run ui_display.py を実行してください)")
        while not os.path.exists(ui_flag):
            time.sleep(0.5)
        print("✅ UIが起動しました。会話を開始します。")
        try:
            os.remove(ui_flag)
        except OSError:
            pass

        # SETUP相当の一言（固定）
        initial_greeting = "こんにちは。たくさんお話しできたら嬉しいです！"
        self.speak(initial_greeting)
        self.log_interaction("AI", initial_greeting, "")

        while True:
            self.total_turns += 1
            self.turn_in_phase += 1

            user_text = self.listen()

            # 会話メモ更新（要約・繰り返し禁止・終了意思）
            if user_text:
                self.update_conv_memory(user_text)

            # ユーザーが終えたい意思を示したら、必ず終盤へ（強制終了フラグ）
            if self.conv_memory.stop_intent and self.phase != Phase.ENDING:
                self.force_end = True
                self._set_phase(Phase.ENDING, reason="ユーザー終了希望")

            # 観測：基本分類
            action_type = self.classify_action(user_text)
            minimal_reply = self._is_minimal_reply(user_text)

            # ベイズ更新（終了に入ってもログとして残す）
            self.update_posterior(action_type, minimal_reply=minimal_reply)

            # --- ここでユーザーの発話を分析用CSVに記録 ---
            logged_text = user_text if user_text else "(沈黙)"
            self.log_interaction("User", logged_text, action_type.value)

            # 追加観測（BRIDGE / DEEP_DIVEのみ）
            memory_flag = False
            disclosure_flag = False
            note = None
            if self.phase in [Phase.BRIDGE, Phase.DEEP_DIVE] and user_text:
                memory_flag, disclosure_flag, note = self.judge_memory_and_disclosure(user_text)

            obs = Observation(
                user_text=user_text,
                action_type=action_type,
                minimal_reply=minimal_reply,
                memory_flag=memory_flag,
                self_disclosure_flag=disclosure_flag,
                engagement_hint=note
            )

            # 暴走防止：一定ターンで必ず締める
            if self.total_turns >= self.max_total_turns and self.phase != Phase.ENDING:
                self.force_end = True
                self._set_phase(Phase.ENDING, reason="最大ターン到達")

            # フェーズ遷移（ただし強制終了時は固定）
            if not self.force_end:
                self.transition_policy(obs)

            # 画像取得（必要なフェーズのみ／終盤では撮らない）
            img_b64 = None
            if self.phase != Phase.ENDING and self.phase_configs[self.phase].require_image:
                img_b64 = self.static_image_b64

            # 返信生成・発話
            reply = self.think_and_reply(obs, img_b64)
            self.speak(reply)

            # --- AIの発話を分析用CSVに記録 ---
            self.log_interaction("AI", reply, "")

            # 終盤ならここで終了（追加の追い質問を出さない）
            if self.phase == Phase.ENDING:
                break



if __name__ == "__main__":
    # 実験指示: ここに添付画像のファイルパスを指定してください（デフォルト: experiment_image.jpg）
    agent = MultimodalAgent(image_path="experiment_image.jpg")
    agent.run()
