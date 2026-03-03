import os
import sys
import io
import time
import base64
import cv2
import re
from datetime import datetime
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI

# 文字化け対策
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

load_dotenv()

class MultimodalAgent:
    def __init__(self):
        # --- 1. 設定読み込み ---
        self.speech_key = os.getenv("AZURE_SPEECH_KEY")
        self.speech_region = os.getenv("AZURE_SPEECH_REGION")
        self.openai_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

        if not all([self.speech_key, self.speech_region, self.openai_key, self.openai_endpoint, self.deployment_name]):
            print("❌ エラー: .envファイルの設定が不足しています。")
            sys.exit(1)

        # --- 2. ログ・画像保存の準備 ---
        if not os.path.exists("logs"): os.makedirs("logs")
        self.image_dir = "logs/images"
        if not os.path.exists(self.image_dir): os.makedirs(self.image_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.history_file = f"logs/log_{timestamp}.txt"
        with open(self.history_file, "w", encoding="utf-8") as f: f.write("")
        
        print(f"📁 会話ログ: {self.history_file}")

        # --- 3. クライアント初期化 ---
        self.openai_client = AzureOpenAI(
            api_key=self.openai_key,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=self.openai_endpoint
        )

        speech_config = speechsdk.SpeechConfig(subscription=self.speech_key, region=self.speech_region)
        speech_config.speech_recognition_language = "ja-JP"
        speech_config.speech_synthesis_voice_name = "ja-JP-NanamiNeural"
        
        # 思考時間確保
        self.thinking_time_ms = "5000" 
        speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, self.thinking_time_ms)
        speech_config.set_property(speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, "1000")

        audio_config_in = speechsdk.audio.AudioConfig(use_default_microphone=True)
        audio_config_out = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

        self.recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config_in)
        self.synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config_out)

        # --- 4. 演出（フェーズ）の定義 ---
        self.phases = [
            {
                "name": "導入 (Intro)",
                "instruction": (
                    "まずは明るく挨拶をして、会話を始めてください。\n"
                    "この段階では、まだ周囲の環境（画像）については詳しく触れず、ユーザー自身の調子を聞くなど、人への挨拶を優先します。"
                )
            },
            {
                "name": "周囲の状況 (Surroundings)",
                "description": "環境言及フェーズ。",
                "instruction": (
                    "添付された周囲の環境の画像をよく見て、現地の様子や置いてある物についてコメントや質問をしてください。\n"
                    "「天気がいいですね」「おしゃれな時計ですね」など、ユーザーを取り巻く環境に注目して話題を提供します。"
                )
            },
            {
                "name": "連想・ブリッジ (Bridge)",
                "description": "過去への誘導フェーズ。",
                "instruction": (
                    "今の話題や環境を強引にでもきっかけにして、「ユーザーの過去の思い出」について話題を振ってください。\n"
                    "例：「この部屋を見てると、実家を思い出しませんか？」「昔はどんな部屋が好きでしたか？」"
                )
            },
            {
                "name": "深掘り (Deep Dive)",
                "description": "思い出の深掘りフェーズ。",
                "instruction": (
                    "ユーザーが話してくれた思い出話について、さらに深く質問してください。\n"
                    "【重要】画像（現在の目の前の光景）の情報よりも、ユーザーの「話の内容」に集中し、共感して深掘りを行ってください。"
                )
            },
            {
                "name": "エンディング (Ending)",
                "description": "終了フェーズ。",
                "instruction": (
                    "【最重要】会話を終了させるフェーズです。\n"
                    "直前のユーザーの話（思い出や好み）に対して、質問などで話を続けようとするのではなく、共感や素敵な感想を述べてください。\n"
                )
            }
        ]
        
        self.current_phase_index = 0

        # --- 5. 確率モデルの設定 ---
        self.p_want_talk = 0.5 

        # 尤度設定
        self.likelihoods = {
            # 話したい人(H1): 沈黙しない、普通に返す、よく自己開示する(0.6)
            "H1": {"SILENCE": 0.1, "NORMAL": 0.3, "DISCLOSURE": 0.6},
            # 話したくない人(H0): よく沈黙する、適当に返す(0.55)、自己開示しない
            "H0": {"SILENCE": 0.4, "NORMAL": 0.55, "DISCLOSURE": 0.05}
        }

    def speak(self, text):
        if not text or not text.strip(): return
        print(f"🤖 AI音声: {text}")
        result = self.synthesizer.speak_text_async(text).get()

    def listen(self):
        wait_sec = int(self.thinking_time_ms) // 1000
        print(f"\n🎤 マイク待機中... ({wait_sec}秒間 思考OK)", flush=True)
        try:
            result = self.recognizer.recognize_once_async().get()
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                print(f"あなた: {result.text}")
                return result.text
            elif result.reason == speechsdk.ResultReason.NoMatch:
                print(f"   (タイムアウト: {wait_sec}秒間発言なし -> 沈黙と判定)")
                return None
            return None
        except Exception:
            return None

    def capture_image(self):
        print("📷 撮影中...", end="", flush=True)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                return None
        time.sleep(1)
        ret, frame = cap.read()
        cap.release()
        if not ret: return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.image_dir}/img_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        _, buffer = cv2.imencode('.jpg', frame)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        print(f"\r✅ 画像取得完了 (保存: {filename})", flush=True)
        return base64_image

    def append_to_log(self, role, text):
        try:
            with open(self.history_file, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {role}: {text}\n")
        except: pass

    # --- 確率更新ロジック ---
    def update_posterior_probability(self, action_type):
        l_h1 = self.likelihoods["H1"][action_type]
        l_h0 = self.likelihoods["H0"][action_type]
        prior = self.p_want_talk
        evidence = (l_h1 * prior) + (l_h0 * (1 - prior))
        posterior = (l_h1 * prior) / evidence
        
        print(f"\n📊 [確率更新] 事前:{prior:.2f} -> 行動:{action_type} -> 事後:{posterior:.2f}")
        self.p_want_talk = posterior
        return posterior

    def classify_user_action(self, user_text):
        if user_text is None:
            return "SILENCE"
            
        print("🔍 行動分析中...", end="", flush=True)
        check_messages = [
            {"role": "system", "content": (
                "ユーザーの発言を分析し、以下のいずれかに分類してください。\n"
                "1. DISCLOSURE : 過去の思い出、個人的なエピソード、自身の好みや考えを詳しく話している。\n"
                "2. NORMAL : 単なる相槌（はい、そうです）、お礼（ありがとう）、短い事実確認。\n"
                "回答は単語のみ（DISCLOSURE または NORMAL）で出力してください。"
            )},
            {"role": "user", "content": user_text}
        ]
        try:
            response = self.openai_client.chat.completions.create(
                model=self.deployment_name, messages=check_messages, max_tokens=10
            )
            result = response.choices[0].message.content.strip().upper()
            print(f"\r{' ' * 30}\r", end="") 
            if "DISCLOSURE" in result: return "DISCLOSURE"
            return "NORMAL"
        except:
            return "NORMAL"

    def think_and_reply(self, user_text, base64_image):
        current_phase_data = self.phases[self.current_phase_index]
        phase_name = current_phase_data["name"]
        
        print(f"\n======== [現在のフェーズ: {phase_name}] ========") 
        print(f"現在P(話したい): {self.p_want_talk:.2f}")

        # --- 【修正】確率に応じた戦略の切り替えロジック ---
        strategy_instruction = ""
        
        # 確率が高い (0.6以上) -> 共感・コメント優先
        if self.p_want_talk >= 0.6:
            strategy_instruction = (
                "\n【現在の戦略：共感重視】\n"
                "ユーザーの関心度は高いです。\n"
                "無理に質問攻めにする必要はありません。「それは素敵ですね」「私もそう思います」といった"
                "『共感』や『感想』を多めにして、ユーザーの満足感を高めてください。"
            )
        # 確率が低い (0.6未満) -> 質問・話題提供優先
        else:
            strategy_instruction = (
                "\n【現在の戦略：積極誘導】\n"
                "ユーザーの関心度が低迷しています。\n"
                "会話が途切れないよう、積極的に『質問』を投げかけるか、画像の中から『新しい話題』を見つけて提供してください。"
                "「ところで…」や「そういえば…」を使って話を転換させてください。"
            )

        system_prompt = (
            "あなたはユーザーの親しみやすいAIパートナーです。"
            "返答は短く（60〜100文字程度）、話し言葉で返してください。\n"
            f"【現在のフェーズ指示】\n{current_phase_data['instruction']}\n"
            "--------------------------------------------------\n"
            f"{strategy_instruction}"
        )

        history = self.load_history_as_messages()
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        
        content = []
        if user_text: content.append({"type": "text", "text": user_text})
        if base64_image: content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
        
        if not content:
            content.append({"type": "text", "text": "(ユーザーは沈黙しています。気遣うような言葉をかけてください)"})

        messages.append({"role": "user", "content": content})

        try:
            print("🧠 思考中...", end="", flush=True)
            response = self.openai_client.chat.completions.create(
                model=self.deployment_name, messages=messages, max_tokens=250 
            )
            ai_content = response.choices[0].message.content
            print("\r" + " " * 20 + "\r", end="")
            
            if user_text: self.append_to_log("User", user_text)
            self.append_to_log("AI", ai_content)
            return ai_content
        except Exception as e:
            print(f"Error: {e}")
            return "すみません、エラーが発生しました。"

    def load_history_as_messages(self):
        messages = []
        if not os.path.exists(self.history_file): return messages
        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                parts = re.split(r'(\[\d{2}:\d{2}:\d{2}\] (?:User|AI):)', f.read())
            current_role = None
            for part in parts:
                if not part.strip(): continue
                if "User:" in part: current_role = "user"
                elif "AI:" in part: current_role = "assistant"
                else:
                    if current_role: messages.append({"role": current_role, "content": part.strip()})
            if len(messages) > 10: messages = messages[-10:]
        except: pass
        return messages

    def advance_phase(self):
        if self.current_phase_index < len(self.phases) - 1:
            self.current_phase_index += 1
            print(f"⏩ フェーズ進行: {self.phases[self.current_phase_index]['name']}")
        else:
            print("⏹️ 最終フェーズです")

    def run(self):
        print("=== 確率モデル搭載 AIエージェント (戦略切り替え版) ===")
        self.speak("こんにちは。私と一緒に楽しくお話ししてください！")

        while True:
            try:
                # 1. 音声認識 & 行動分類
                user_text = self.listen()
                action_type = self.classify_user_action(user_text)

                if user_text and "終了" in user_text:
                    self.speak("ありがとうございました。")
                    break

                # 2. 事後確率の更新
                current_prob = self.update_posterior_probability(action_type)

                # 3. フェーズ進行判定
                current_phase_name = self.phases[self.current_phase_index]["name"]
                
                # 連想フェーズでの分岐: 確率がかなり高まる(0.8)までは粘る
                if "Bridge" in current_phase_name or "連想" in current_phase_name:
                    if current_prob >= 0.8:
                        print("✨ 関心度が十分高まりました！深掘りフェーズへ進みます。")
                        self.advance_phase()
                    else:
                        print("🔒 連想フェーズを継続します (確率による戦略変動あり)")
                
                elif "Bridge" not in current_phase_name and "連想" not in current_phase_name:
                    if "Ending" not in current_phase_name and "エンディング" not in current_phase_name:
                         self.advance_phase()

                # 4. 画像取得 & 返答生成
                image_data = self.capture_image()
                if image_data:
                    reply = self.think_and_reply(user_text, image_data)
                    self.speak(reply)

                    # 5. エンディング終了判定
                    if "Ending" in self.phases[self.current_phase_index]["name"] or "エンディング" in self.phases[self.current_phase_index]["name"]:
                        closing = "今日はここまでにしておきましょう。またお話ししましょうね！"
                        self.speak(closing)
                        self.append_to_log("AI", closing)
                        break
                else:
                    print("⚠️ 画像取得エラー")

            except KeyboardInterrupt:
                break

if __name__ == "__main__":
    agent = MultimodalAgent()
    agent.run()