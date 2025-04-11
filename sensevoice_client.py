import requests
import base64
import json

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

    # 打印返回的文本数据（可选）
    print("Transcription result:", transcription_result)

    # 提取每个片段的开始时间、结束时间和文本
    results = transcription_result.get("res", [])

    # 保存结果到 JSON 文件
    output_json_path = 'transcription_result.json'
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)  # 保存为格式化的 JSON 文件

    print(f"识别结果已保存到 {output_json_path}")
else:
    print("Error:", response.status_code)
