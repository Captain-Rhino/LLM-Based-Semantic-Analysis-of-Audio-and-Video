import time
from dashscope import MultiModalConversation

# === 配置 ===
api_key = "sk-e6f5a000ba014f92b4857a6dcd782591"  # 替换为你的真实 API key
image_path = r"G:\videochat\my_design\CNCLIP_keyframes_no_text\visual_kf_00156.jpg"
prompt_text = "请描述这张图像的内容。"

# === 构造消息体 ===
message = [{
    "role": "user",
    "content": [
        {"image": image_path},
        {"text": prompt_text}
    ]
}]

# === 计时开始 ===
start_time = time.time()

response = MultiModalConversation.call(
    api_key=api_key,
    model="qwen-vl-plus-latest",
    messages=message
)

# === 计时结束 ===
end_time = time.time()
elapsed_time = end_time - start_time

# === 打印结果 ===
print(f"⏱️ 总耗时：{elapsed_time:.2f} 秒")

try:
    result = response['output']['choices'][0]['message']['content'][0]['text']
    print("✅ 模型返回内容：")
    print(result)
except Exception as e:
    print("❌ 返回失败")
    print("原始返回内容：", response)
