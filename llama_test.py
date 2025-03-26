import base64
import requests
import json

# 读取图片 +  Base64 编码
image_path = r"G:\videochat\my_design\K_frame\kf_003_14.40s.jpg"
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

response = requests.post(url, headers=headers, data=json.dumps(data))

# 提取 'response' 部分
response_data = response.json()['response']


print(response.status_code)  # HTTP 状态码
print(response.json())  # 输出 API 响应
print(response_data)  # 输出模型返回的描述