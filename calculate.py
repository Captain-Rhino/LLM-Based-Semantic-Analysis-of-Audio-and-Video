import json

# 读取本地 JSON 文件
with open(r"G:\videochat\my_design\streamlit_save\test_video\test_video.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

# 计算总时长
total_duration = 0.0

for item in data:
    duration = item["end"] - item["start"]
    total_duration += duration

print(f"总时长为: {total_duration:.2f} 秒")
