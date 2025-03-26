import whisper
import json
import time

#开始计时
start_time = time.time()
# 加载模型(small)
model = whisper.load_model("small")

# 视频音频文件路径
audio_path = r"G:\videochat\my_design\test_video.mp4"

# 记录结果
result = model.transcribe(
    audio_path,
    language="zh",
    initial_prompt="以下是普通话句子"
)

# 输出时间戳与对应文本
for segment in result["segments"]:
    print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text']}")

# 保存为 JSON 文件
output_json_path = r"G:\videochat\my_design\transcription.json"
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(result["segments"], f, ensure_ascii=False, indent=2)

# 结束时间
end_time = time.time()
# 运行时间
total_time = end_time - start_time
print(f"\n✅ 已保存到文件：{output_json_path}")
print(f"⏱️ 总运行时间：{total_time:.2f} 秒")