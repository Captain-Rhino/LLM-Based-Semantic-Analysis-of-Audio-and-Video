import base64
import requests
import json
import time
import torch  # 导入 PyTorch

# 记录开始时间
start_time = time.time()

# 设置设备为 GPU 或 CPU
device = "cuda" if torch.cuda.is_available() else "cpu"  # 检查是否有 GPU
print(f"正在使用设备: {device}")  # 打印使用的设备

# 读取图片 +  Base64 编码
image_path = r"G:\videochat\my_design\CNCLIP_keyframes_test_movie\kf_000_00_3.79s.jpg"
with open(image_path, "rb") as img_file:
    base64_image = base64.b64encode(img_file.read()).decode('utf-8')

url = "http://localhost:11434/api/generate"
headers = {"Content-Type": "application/json"}
data = {
    "model": "llama3.2-vision",
    "prompt": "这张图片里有什么？请详细描述",
    "images": [base64_image],  # 编码后图片
    "stream": False  # 关闭流式返回
}

# 发送请求到 Ollama 本地模型（保持在 CPU 或 GPU 上）
response = requests.post(url, headers=headers, data=json.dumps(data))

# 提取 'response' 部分
response_data = response.json()['response']

# 打印结果
print(response.status_code)  # HTTP 状态码
print(response.json())  # 输出 API 响应
print(response_data)  # 输出模型返回的描述

# 记录结束时间并输出运行时间
end_time = time.time()
print(f"⏱️ 总共用时：{end_time - start_time:.2f} 秒")  # 输出模型运行的时间
