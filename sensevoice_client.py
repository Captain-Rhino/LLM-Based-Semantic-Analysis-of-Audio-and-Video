# client.py

import requests
import base64

# 文件路径
file_path = 'G:/videochat/SenseVoice/voice/test_video.mp3'  # 你的音频文件路径

# 服务器 URL
url = "http://127.0.0.1:2002/asr"  # 本地 FastAPI 服务

# 读取音频文件并转换为 base64 编码
with open(file_path, 'rb') as f:
    audio_data = base64.b64encode(f.read()).decode('utf-8')

# 发送请求
response = requests.post(url, json={"wav": audio_data})

# 处理返回的 JSON 数据
if response.status_code == 200:
    transcription_result = response.json()
    print("Transcription result:", transcription_result)
else:
    print("Error:", response.status_code)
