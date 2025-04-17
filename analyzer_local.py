# ✅ 需要提前安装的依赖：
# pip install transformers accelerate torchvision torch

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import json
import os

# ====== 配置模型 ======
model_name = "llava-hf/llava-llama-3-8b"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 当前使用设备：{device}")

# ====== 加载模型和处理器 ======
print("🔄 正在加载模型...")
model = LlavaForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    low_cpu_mem_usage=True
).to(device)
processor = AutoProcessor.from_pretrained(model_name)
print("✅ 模型加载完成")

# ====== 加载关键帧 + 文本数据 ======
input_json = "CNCLIP_keyframes_test_movie/test_movie_cnclip.json"  # 你之前生成的 JSON
with open(input_json, "r", encoding="utf-8") as f:
    data = json.load(f)

results = []

# ====== 执行逐帧总结 ======
for i, item in enumerate(data):
    image_path = item['image_path']
    text = item['text']
    print(f"\n🖼️ 正在分析第 {i} 段文字 + 图片：{image_path}")

    image = Image.open(image_path).convert("RGB")

    prompt = f"这是一帧视频截图，字幕为：'{text}'。请结合图片内容总结这段视频内容。"

    inputs = processor(prompt, image, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=100)
    response = processor.batch_decode(output, skip_special_tokens=True)[0]

    print("🧠 总结：", response)
    results.append({
        "segment_index": item["segment_index"],
        "image_path": image_path,
        "original_text": text,
        "summary": response
    })

# ====== 保存结果 ======
output_path = "video_summary_output.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n✅ 所有摘要已保存到：{output_path}")
