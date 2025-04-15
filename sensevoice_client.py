import subprocess
import os
import requests
import base64
import json

# ==== Step 1: 从 mp4 中提取音频 ====
video_path = r'G:\videochat\my_design\test_2.mp4'  # 输入视频路径
audio_path = r'G:\videochat\my_design\test_2.mp3'  # 输出音频路径

# 如果音频文件不存在，则从视频中提取音频
if not os.path.exists(audio_path):
    print("🔊 正在提取音频...")
    ffmpeg_cmd = [
        'ffmpeg', '-y',  # -y 覆盖输出文件
        '-i', video_path,
        '-vn',  # 不要视频
        '-acodec', 'libmp3lame',  # 编码为 mp3 格式
        audio_path
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    print("✅ 音频提取完成")

# ==== Step 2: 读取 mp3 音频并进行 base64 编码 ====
with open(audio_path, 'rb') as f:
    audio_data = base64.b64encode(f.read()).decode('utf-8')

# ==== Step 3: 发送请求到本地语音识别服务 ====
url = "http://127.0.0.1:2002/asr"  # FastAPI 服务地址
response = requests.post(url, json={"wav": audio_data})

# ==== Step 4: 处理返回结果 ====
if response.status_code == 200:
    transcription_result = response.json()
    print("📝 Transcription result:", transcription_result)

    results = transcription_result.get("res", [])

    # 保存为 JSON 文件
    output_json_path = 'transcription_result.json'
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)

    print(f"✅ 识别结果已保存到 {output_json_path}")
else:
    print("❌ 识别失败，状态码:", response.status_code)
