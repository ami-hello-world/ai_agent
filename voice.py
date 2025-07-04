import requests
import sounddevice as sd
import numpy as np
from io import BytesIO
import soundfile as sf
from threading import Thread

# サーバー設定
HOST = "127.0.0.1"
PORT = "10101"
SPEAKER = "888753763"

# ✅ クエリを生成（同期）
def post_audio_query(text: str) -> dict:
    url = f"http://{HOST}:{PORT}/audio_query"
    params = {"text": text, "speaker": SPEAKER}

    res = requests.post(url, params=params)
    if res.status_code != 200:
        raise Exception(f"音声クエリ生成失敗: {res.status_code}, {res.text}")
    return res.json()


# ✅ 音声合成（同期）
def post_synthesis(query_data: dict) -> bytes:
    url = f"http://{HOST}:{PORT}/synthesis"
    params = {"speaker": SPEAKER}
    headers = {"content-type": "application/json"}

    res = requests.post(url, params=params, headers=headers, json=query_data)
    if res.status_code != 200:
        raise Exception(f"音声合成失敗: {res.status_code}, {res.text}")
    return res.content


# ✅ 音声再生をスレッドで並列処理
def play_wavfile(wav_data: bytes):
    def play():
        try:
            wav_io = BytesIO(wav_data)
            data, samplerate = sf.read(wav_io)
            sd.play(data, samplerate)
            sd.wait()
        except Exception as e:
            print(f"音声再生でエラーが発生しました: {e}")

    thread = Thread(target=play)
    thread.start()


# ✅ 音声生成＆再生（同期）
def text_to_voice(text: str):
    try:
        query_data = post_audio_query(text)
        wav_data = post_synthesis(query_data)
        play_wavfile(wav_data)
    except Exception as e:
        print(f"エラー: {e}")


# ✅ テスト実行部分
if __name__ == "__main__":
    while True:
        text = input("テキストを入力してください (qで終了): ")
        if text.lower() == "q":
            break
        text_to_voice(text)
