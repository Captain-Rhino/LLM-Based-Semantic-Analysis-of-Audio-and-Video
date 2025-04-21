#语音识别

import base64
import requests
import json

def transcribe_audio(audio_path, api_key,output_dir=None):
    with open(audio_path, 'rb') as f:
        audio_base64 = base64.b64encode(f.read()).decode('utf-8')

    url = "http://127.0.0.1:2002/asr"  # 语音识别服务的 URL
    headers = {"Content-Type": "application/json"}
    data = {
        "wav": audio_base64,  # 转码后的音频数据
        "api_key": api_key     # 你的 API Key
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.json().get("res", [])
    else:
        print("❌ 语音识别失败，状态码:", response.status_code)
        return None
