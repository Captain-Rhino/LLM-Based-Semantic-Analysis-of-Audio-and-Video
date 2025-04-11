import time
import torch
from click.core import batch
from funasr import AutoModel
from tensorflow.python.data.experimental.ops.distribute import batch_sizes_for_worker

# 🔍 检查 GPU 状态
print("🔧 CUDA 是否可用：", torch.cuda.is_available())
if torch.cuda.is_available():
    print("🚀 当前使用 GPU：", torch.cuda.get_device_name(0))
    device_type = "cuda"
else:
    print("⚠️ 未检测到 GPU，将使用 CPU 运行")
    device_type = "cpu"

# ⏱️ 开始计时
start_time = time.time()

# 初始化模型
model = AutoModel(
    model="paraformer-zh",
    model_revision="v2.0.4",
    vad_model="fsmn-vad",
    punc_model="ct-punc",
    device=device_type  # ✅ 自动根据检测结果切换
)

# 音频路径
audio_path = r"G:\videochat\my_design\test_video.aac"
result = model.generate(input=audio_path,batch_size=16)

# 📄 打印结果
print("\n📄 识别结果：")
for seg in result:
    ts = seg["timestamp"]
    if isinstance(ts[0], list):
        t_start, t_end = ts[0]
    else:
        t_start, t_end = ts
    print(f"[{t_start/1000:.2f}s - {t_end/1000:.2f}s] {seg['text']}")

# ⏱️ 计时结束
end_time = time.time()
print(f"\n⏱️ 识别耗时：{end_time - start_time:.2f} 秒")
