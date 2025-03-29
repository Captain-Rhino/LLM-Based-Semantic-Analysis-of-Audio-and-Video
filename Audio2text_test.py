import os
import subprocess
import whisperx
import json

# Step 1: 视频路径
video_path = "your_video.mp4"

# Step 2: WhisperX 加载模型
device = "cuda" if whisperx.utils.get_device() == "cuda" else "cpu"
model = whisperx.load_model("large-v2", device)

# Step 3: 执行语音识别
result = model.transcribe(video_path)
print("🔤 初步识别完成。")

# Step 4: 加载说话人识别模型（Diarization）
diarize_model = whisperx.DiarizationPipeline(use_auth_token="your_hf_token", device=device)
diarize_segments = diarize_model(video_path)

# Step 5: 对齐说话人标记
result_aligned = whisperx.align(
    result["segments"], model.model, model.tokenizer, video_path, device
)
result_with_speaker = whisperx.assign_speakers(result_aligned["segments"], diarize_segments)

# Step 6: 保存结构化 JSON
output_json = "multi_speaker_transcript.json"
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(result_with_speaker, f, ensure_ascii=False, indent=2)

print(f"\n✅ 多说话人语音转文字完成，结果已保存到：{output_json}")
