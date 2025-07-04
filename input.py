import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import threading
import keyboard
import time
import os
from google.cloud import speech
from google.oauth2 import service_account

FILENAME = "temp_audio.wav"
RATE = 16000
CHANNELS = 1
SERVICE_ACCOUNT_FILE = "hello_world_project/helloworld-462703-8ef47f7d6b7e.json"

def key_monitor(stop_flag_container):
    input("👂 Enterキーで録音を停止します...")
    print("🛑 停止します。")
    stop_flag_container["flag"] = True


def audio_callback(indata, frames, time_info, status, recording, stop_flag_container):
    if stop_flag_container["flag"]:
        raise sd.CallbackStop()
    recording.append(indata.copy())

def transcribe_google(filename):
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
    client = speech.SpeechClient(credentials=credentials)

    with open(filename, "rb") as f:
        content = f.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="ja-JP"
    )

    print("⬆️ Google Speech-to-Text に送信中...")
    response = client.recognize(config=config, audio=audio)

    if not response.results:
        print("❌ 文字起こしできませんでした。")
        return ""
    else:
        transcript = response.results[0].alternatives[0].transcript
        print("📝 文字起こし:", transcript)
        return transcript

def voice():
    stop_flag_container = {"flag": False}
    recording = []

    print("🎙️ 録音開始")
    threading.Thread(target=key_monitor, args=(stop_flag_container,), daemon=True).start()

    try:
        def callback(indata, frames, time_info, status):
            audio_callback(indata, frames, time_info, status, recording, stop_flag_container)

        with sd.InputStream(samplerate=RATE, channels=CHANNELS, callback=callback):
            while not stop_flag_container["flag"]:
                time.sleep(0.1)

        audio_data = np.concatenate(recording, axis=0)
        audio_data_int16 = (audio_data * 32767).astype(np.int16)
        write(FILENAME, RATE, audio_data_int16)

        result = transcribe_google(FILENAME)
        return result

    except Exception as e:
        print("⚠️ エラー:", e)
        return ""

    finally:
        if os.path.exists(FILENAME):
            os.remove(FILENAME)
        print("✅ 終了しました。")

if __name__ == "__main__":
    result = voice()
    print("📄 結果:", result)
