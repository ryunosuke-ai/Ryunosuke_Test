import os
import sys
import io
import time
import base64
import cv2
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI

# 文字化け対策
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

load_dotenv()

class VisionAgent:
    def __init__(self):
        self.speech_key = os.getenv("AZURE_SPEECH_KEY")
        self.speech_region = os.getenv("AZURE_SPEECH_REGION")
        self.openai_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        
        self.openai_client = AzureOpenAI(
            api_key=self.openai_key,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=self.openai_endpoint
        )
        
        speech_config = speechsdk.SpeechConfig(subscription=self.speech_key, region=self.speech_region)
        speech_config.speech_recognition_language = "ja-JP"
        audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
        self.recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    def capture_image(self):
        """Webカメラから画像を撮影し、ファイル保存＆Base64変換を行う"""
        print("📷 カメラ撮影中...", end="", flush=True)
        
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            print("\n❌ カメラが起動できませんでした。")
            return None

        time.sleep(1) # 明るさ調整待機
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("\n❌ 画像の取得に失敗しました。")
            return None

        # --- 【追加機能】画像をファイルとして保存 ---
        filename = "captured_image.jpg"
        cv2.imwrite(filename, frame)
        print(f"\n💾 画像を保存しました: {filename}", flush=True)
        # ---------------------------------------

        _, buffer = cv2.imencode('.jpg', frame)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        
        return base64_image

    def run(self):
        print("=== Step 3.5: 画像保存機能付きエージェント ===")
        print("準備完了。", flush=True)
        print("🎤 目の前の物について聞いてみてください（例：「これ何？」）", flush=True)

        result = self.recognizer.recognize_once_async().get()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            user_text = result.text
            print("-" * 30)
            print(f"あなた: {user_text}")
            print("-" * 30)

            base64_image = self.capture_image()
            if not base64_image:
                return

            print("🧠 AIが画像を見て考えています...", end="", flush=True)

            try:
                response = self.openai_client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {
                            "role": "system", 
                            "content": "あなたはカメラの目を持つAIです。ユーザーの音声と、添付された画像に基づいて会話してください。"
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_text},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    },
                                },
                            ],
                        },
                    ],
                    max_tokens=300
                )
                
                ai_response = response.choices[0].message.content
                print("\r" + " " * 40 + "\r", end="")
                print(f"🤖 AI: {ai_response}")
                print("-" * 30)

            except Exception as e:
                print(f"\n❌ エラー: {e}")

        elif result.reason == speechsdk.ResultReason.NoMatch:
            print("⚠️ 音声を認識できませんでした。")

if __name__ == "__main__":
    agent = VisionAgent()
    agent.run()