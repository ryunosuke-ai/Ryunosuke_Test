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

        audio_config_in = speechsdk.audio.AudioConfig(use_default_microphone=True)
        audio_config_out = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

        self.recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config_in)
        self.synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config_out)

        # --- 4. 演出（フェーズ）の定義 ---
        self.phases = [
            {
                "name": "導入 (Intro)",
                "description": "挨拶フェーズ。",
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
                    "【目的】ユーザーから「過去の思い出」を引き出すこと。\n"
                    "今の話題や環境をきっかけにして、「昔は〜でしたか？」「以前似たような経験はありますか？」と質問を投げかけてください。\n"
                    "※まだユーザーは思い出を話していません。粘り強く、しかし自然に、過去のエピソードを聞き出してください。"
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

    def speak(self, text):
        if not text or not text.strip(): return
        print(f"🤖 AI音声: {text}")
        result = self.synthesizer.speak_text_async(text).get()

    def listen(self):
        print("\n🎤 マイク待機中...", flush=True)
        result = self.recognizer.recognize_once_async().get()
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            print(f"あなた: {result.text}")
            return result.text
        return None

    def capture_image(self):
        print("📷 撮影中...", end="", flush=True)
        # USBカメラ(1) -> 内蔵(0)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                print("❌ カメラが見つかりません。")
                return None
        time.sleep(1)
        ret, frame = cap.read()
        cap.release()
        if not ret: return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.image_dir}/img_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f" (保存: {filename})", end="")
        
        _, buffer = cv2.imencode('.jpg', frame)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        print("\r✅ 画像取得完了     ", flush=True)
        return base64_image

    def load_history_as_messages(self):
        messages = []
        if not os.path.exists(self.history_file): return messages
        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                log_content = f.read()
            parts = re.split(r'(\[\d{2}:\d{2}:\d{2}\] (?:User|AI):)', log_content)
            current_role = None
            for part in parts:
                if not part.strip(): continue
                if "User:" in part: current_role = "user"
                elif "AI:" in part: current_role = "assistant"
                else:
                    content = part.strip()
                    if current_role and content:
                        messages.append({"role": current_role, "content": content})
            if len(messages) > 20: messages = messages[-20:]
        except: pass
        return messages

    def append_to_log(self, role, text):
        try:
            with open(self.history_file, "a", encoding="utf-8") as f:
                timestamp = datetime.now().strftime("%H:%M:%S")
                f.write(f"[{timestamp}] {role}: {text}\n")
        except: pass

    # --- 【追加】思い出話判定メソッド ---
    def check_memory_extracted(self, user_text):
        """ユーザーの発言に「思い出・過去のエピソード」が含まれているか判定する"""
        print("🔍 思い出判定中...", end="", flush=True)
        
        check_messages = [
            {"role": "system", "content": (
                "あなたは会話分析AIです。"
                "ユーザーの発言を分析し、「過去の思い出」「昔のエピソード」「個人的な経験」が含まれているかを判定してください。"
                "単なる相槌（「はい」「そうです」）や、現在の状況説明（「天気がいいです」）だけの場合は NO としてください。"
                "少しでも過去の話や、自身の経験に基づく好み（「昔よく〜しました」など）が含まれていれば YES としてください。"
                "回答は 'YES' または 'NO' の単語のみ出力してください。"
            )},
            {"role": "user", "content": user_text}
        ]
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=check_messages,
                max_tokens=10
            )
            result = response.choices[0].message.content.strip().upper()
            print(f"\r{' ' * 30}\r", end="") # 表示クリア
            
            if "YES" in result:
                print(f"✅ 思い出検知: YES")
                return True
            else:
                print(f"🔄 思い出検知: NO (まだ引き出せていません)")
                return False
        except:
            return False

    def think_and_reply(self, user_text, base64_image):
        print("🧠 思考中...", end="", flush=True)
        
        current_phase_data = self.phases[self.current_phase_index]
        phase_name = current_phase_data["name"]
        phase_instruction = current_phase_data["instruction"]
        
        print(f"\n======== [現在のフェーズ: {phase_name}] ========") 
        print(f"ガイドライン: {phase_instruction[:30]}...")

        system_prompt = (
            "あなたはユーザーの親しみやすいAIパートナーです。"
            "返答は短く（60〜100文字程度）、話し言葉で返してください。\n"
            "--------------------------------------------------\n"
            f"【現在の会話ガイドライン】\n{phase_instruction}"
        )

        history_messages = self.load_history_as_messages()

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history_messages)
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                }
            ]
        })

        try:
            response = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                max_tokens=250 
            )
            ai_content = response.choices[0].message.content
            print("\r" + " " * 20 + "\r", end="")

            self.append_to_log("User", user_text)
            self.append_to_log("AI", ai_content)
            
            return ai_content

        except Exception as e:
            print(f"\n❌ エラー: {e}")
            return "すみません、エラーが発生しました。"

    def advance_phase(self):
        """強制的に次のフェーズへ進める"""
        if self.current_phase_index < len(self.phases) - 1:
            self.current_phase_index += 1
            print(f"⏩ フェーズを進めました: 次は {self.phases[self.current_phase_index]['name']}")
        else:
            print("⏹️ 最終フェーズです")

    def run(self):
        print("=== 演出デモ版 AIエージェント (思い出判定付き) ===")
        self.speak("こんにちは。私と一緒に楽しくお話ししてください！")

        while True:
            try:
                user_text = self.listen()
                if not user_text: continue

                if "終了" in user_text:
                    self.speak("ありがとうございました。")
                    break
                
                image_data = self.capture_image()
                if not image_data:
                    print("⚠️ 画像取得エラーのためスキップ")
                    continue

                # --- 【ロジック変更】フェーズ進行判断 ---
                
                current_phase = self.phases[self.current_phase_index]["name"]

                # A. 現在が「連想・ブリッジ」フェーズの場合
                # ユーザーの発言に「思い出」が含まれているかチェックする
                if "Bridge" in current_phase or "連想" in current_phase:
                    has_memory = self.check_memory_extracted(user_text)
                    
                    if has_memory:
                        # 思い出が引き出せたなら、「深掘り」フェーズへ進める
                        print("✨ ユーザーが思い出を話しました！深掘りフェーズへ移行します。")
                        self.advance_phase()
                    else:
                        # まだ思い出が出ていないなら、フェーズを進めず「連想」をやり直す
                        # (advance_phase を呼ばない)
                        pass

                # ----------------------------------------

                # 1. 思考して返答
                # (上でフェーズが進んでいれば、新しいフェーズの指示で喋る)
                reply = self.think_and_reply(user_text, image_data)
                self.speak(reply)

                # --- 返答後の処理 ---
                
                # B. 終了判定
                current_phase = self.phases[self.current_phase_index]["name"]
                if "Ending" in current_phase or "エンディング" in current_phase:
                    self.speak("今日はここまでにしておきましょう。またお話ししましょうね！")
                    break
                
                # C. その他のフェーズの進行（連想以外は自動で進む）
                # 連想フェーズの場合は、上の判定ロジックで制御するのでここでは進めない
                if "Bridge" not in current_phase and "連想" not in current_phase:
                    self.advance_phase()

            except KeyboardInterrupt:
                break

if __name__ == "__main__":
    agent = MultimodalAgent()
    agent.run()