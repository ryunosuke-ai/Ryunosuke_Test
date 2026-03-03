import os
import sys
import io
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk

# Windowsのコンソール文字化け対策
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

load_dotenv()

def recognize_from_microphone():
    print("=== 音声認識テスト（修正版） ===")

    speech_key = os.getenv("AZURE_SPEECH_KEY")
    speech_region = os.getenv("AZURE_SPEECH_REGION")

    if not speech_key or not speech_region:
        print("❌ エラー: .envファイルの設定を確認してください。")
        return

    # 設定
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    speech_config.speech_recognition_language = "ja-JP"
    
    # マイク設定
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    # ユーザーへの案内表示
    # flush=True をつけることで、Pythonが処理待ちをしていても強制的に文字を画面に出します
    print("準備ができました。", flush=True)
    print("🎤 今、マイクに向かって話してください！", flush=True)
    
    # ここで初めてAzureに接続＆録音開始
    # 認識が終わるまでここでプログラムが待機します
    result = speech_recognizer.recognize_once_async().get()

    # 結果表示
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("-" * 30)
        print(f"📝 文字起こし: {result.text}")
        print("-" * 30)
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("⚠️ 音声を認識できませんでした。")
    elif result.reason == speechsdk.ResultReason.Canceled:
        details = result.cancellation_details
        print(f"❌ エラー: {details.error_details}")

if __name__ == "__main__":
    recognize_from_microphone()