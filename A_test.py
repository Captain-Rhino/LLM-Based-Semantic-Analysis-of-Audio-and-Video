import base64
import requests
import json
from dashscope import MultiModalConversation

# 配置
api_key = "sk-e6f5a000ba014f92b4857a6dcd782591"
image_path = r"G:\videochat\my_design\CNCLIP_keyframes_test_video\kf_000_00_2.30s.jpg"
text = "🎼大家好，我是雷军。今年全国两会呢即将召开。"
segment_index = 0
start = 0.0
end = 4.6
timestamp = 2.3

# 构造 prompt
prompt = (
    f"当前是第 {segment_index} 段，这段的语音文本是：“{text}”，"
    f"语音起始时间是 {start} 秒，语音结束时间是 {end} 秒，"
    f"图像在该视频的第 {timestamp} 秒取得，请你理解该图片和文本，先不描述，等待后续指令。"
)

# base64 编码图像
with open(image_path, 'rb') as f:
    base64_img = base64.b64encode(f.read()).decode('utf-8')

# 构造消息体
messages = [
    {
        "role": "user",
        "content": [
            {"image": base64_img},
            {"text": prompt}
        ]
    }
]

# 调用 API
response = MultiModalConversation.call(
    api_key=api_key,
    model='llama-4-maverick-17b-128e-instruct',
    messages=messages
)

# 解析输出
try:
    content = response['output']['choices'][0]['message']['content'][0]['text']
    print("✅ 模型返回内容：\n", content)
except Exception as e:
    print("❌ 错误：模型未返回有效结果")
    print("返回内容：", response)
