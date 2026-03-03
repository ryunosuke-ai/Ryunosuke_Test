import os
import sys
import io
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI

# Windows文字化け対策
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

load_dotenv()

def chat_with_llm():
    print("=== Step 2: 音声認識 + LLM連携 ===")

    # --- 1. 設定の読み込み ---
    speech_key = os.getenv("AZURE_SPEECH_KEY")
    speech_region = os.getenv("AZURE_SPEECH_REGION")
    
    openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    # キーの簡易チェック
    if not all([speech_key, speech_region, openai_api_key, openai_endpoint, deployment_name]):
        print("❌ エラー: .envファイルに必要な情報が足りません。")
        return

    # --- 2. クライアントの初期化 ---
    # Speech
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    speech_config.speech_recognition_language = "ja-JP"
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    # OpenAI
    client = AzureOpenAI(
        api_key=openai_api_key,
        api_version=api_version,
        azure_endpoint=openai_endpoint
    )

    # --- 3. 音声認識パート ---
    print("準備完了。", flush=True)
    print("🎤 質問をどうぞ！（例：「面白い話をして」）", flush=True)

    result = recognizer.recognize_once_async().get()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        user_text = result.text
        print("-" * 30)
        print(f"あなた: {user_text}")
        print("-" * 30)

        # --- 4. LLM推論パート ---
        print("🧠 AIが考えています...", end="", flush=True)

        try:
            response = client.chat.completions.create(
                model=deployment_name, # .envで指定したデプロイ名
                messages=[
                    {"role": "system", "content": "あなたは親切で簡潔に話すAIアシスタントです。"},
                    {"role": "user", "content": user_text}
                ],
                max_tokens=1000 # 長すぎないように制限
            )
            
            ai_response = response.choices[0].message.content
            
            print("\r" + " " * 30 + "\r", end="") # "考えています..."を消す
            print(f"🤖 AI: {ai_response}")
            print("-" * 30)

        except Exception as e:
            print(f"\n❌ OpenAI呼び出しエラー: {e}")

    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("⚠️ 音声を認識できませんでした。")
    elif result.reason == speechsdk.ResultReason.Canceled:
        print(f"❌ 音声認識エラー: {result.cancellation_details.error_details}")

if __name__ == "__main__":
    chat_with_llm()