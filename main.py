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

        # --- 2. ログファイルの準備 ---
        if not os.path.exists("logs"):
            os.makedirs("logs")
        
        self.image_dir = "logs/images"
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.history_file = f"logs/log_{timestamp}.txt"
        
        with open(self.history_file, "w", encoding="utf-8") as f:
            f.write("")
            
        print(f"📁 会話ログ: {self.history_file}")
        print(f"🖼️ 画像保存先: {self.image_dir}")

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

        # --- 4. プロンプト設定 ---
        self.system_prompt = (
            "あなたはユーザーの目の代わりとなり、視覚と聴覚を持つ「好奇心旺盛な」AIパートナーです。"
            "以下のルールを厳守してください。\n"
            "1. 【視覚の活用】添付された画像を「今、目の前にある光景」として扱い、状況や雰囲気を会話に盛り込んでください。\n"
            "2. 【音声対話への最適化】Markdown記法や絵文字は使用せず、親しみやすい話し言葉で書いてください。\n"
            "3. 【簡潔さ】1回の返答は短く（60文字〜120文字程度）、テンポよく返してください。\n"
            "4. 【会話のリード（最重要）】感想だけで終わらせず、必ず最後に「質問」を投げかけてください。"
            "  - パターンA（深堀り）: ユーザーの発言に対し「それはいつ？」「詳しく教えて」など興味を持って質問する。"
            "  - パターンB（話題転換）: 話が途切れそうな場合は、画像内に映っている別の物について「あれは何？」と質問する。"
            "5. 【重複の回避】会話履歴を確認し、過去に自分が質問した内容や、すでに話した話題を繰り返さないようにしてください。"
        )

    # --- 【重要修正】 音声合成のエラーチェックを追加 ---
    def speak(self, text):
        if not text or not text.strip():
            print("⚠️ 警告: 読み上げるテキストが空でした。")
            return

        print(f"🤖 AI音声: {text}")
        
        # 音声合成を実行し、結果オブジェクトを取得
        result = self.synthesizer.speak_text_async(text).get()

        # 結果を判定
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            # 成功した場合は何もしなくてOK
            pass
        elif result.reason == speechsdk.ResultReason.Canceled:
            # 失敗した場合
            cancellation_details = result.cancellation_details
            print(f"⚠️ 音声合成エラー(Canceled): {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print(f"❌ エラー詳細: {cancellation_details.error_details}")
                print("   (ヒント: Azureのキー、リージョン、またはインターネット接続を確認してください)")
    # ----------------------------------------------------

    def listen(self):
        print("\n🎤 マイク待機中...", flush=True)
        result = self.recognizer.recognize_once_async().get()
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            print(f"あなた: {result.text}")
            return result.text
        return None

    def capture_image(self):
        print("📷 撮影中...", end="", flush=True)
        
        # Webカメラ(Index=1)を使用
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            print("❌ カメラ(1)が見つかりません。") 
            return None
            
        time.sleep(1)
        ret, frame = cap.read()
        cap.release()
        if not ret: return None
        
        # 画像保存
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
        if not os.path.exists(self.history_file):
            return messages

        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                log_content = f.read()

            parts = re.split(r'(\[\d{2}:\d{2}:\d{2}\] (?:User|AI):)', log_content)
            
            current_role = None
            
            for part in parts:
                if not part.strip(): continue
                
                if "User:" in part:
                    current_role = "user"
                elif "AI:" in part:
                    current_role = "assistant"
                else:
                    content = part.strip()
                    if current_role and content:
                        messages.append({"role": current_role, "content": content})
            
            if len(messages) > 20:
                messages = messages[-20:]

        except Exception as e:
            print(f"⚠️ 履歴読み込みエラー: {e}")
            
        return messages

    def append_to_log(self, role, text):
        try:
            with open(self.history_file, "a", encoding="utf-8") as f:
                timestamp = datetime.now().strftime("%H:%M:%S")
                f.write(f"[{timestamp}] {role}: {text}\n")
        except Exception as e:
            print(f"⚠️ ログ保存エラー: {e}")

    def think_and_reply(self, user_text, base64_image):
        print("🧠 思考中...", end="", flush=True)

        history_messages = self.load_history_as_messages()

        messages = [{"role": "system", "content": self.system_prompt}]
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

    def run(self):
        print("=== 積極会話版 AIエージェント（発話修正版） ===")
        self.speak("こんにちは！一緒に楽しく話しましょう！")

        while True:
            try:
                user_text = self.listen()
                if not user_text: continue

                if "終了" in user_text:
                    self.speak("楽しかったです。またお話ししましょう！")
                    break
                
                image_data = self.capture_image()
                if image_data:
                    reply = self.think_and_reply(user_text, image_data)
                    self.speak(reply)

            except KeyboardInterrupt:
                break

if __name__ == "__main__":
    agent = MultimodalAgent()
    agent.run()