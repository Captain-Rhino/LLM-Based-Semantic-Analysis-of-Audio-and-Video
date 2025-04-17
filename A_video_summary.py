import os
import json
import torch
from A_audio_extractor import extract_audio_from_video
from A_audio_recognition import transcribe_audio
from A_keyframe_extractor import extract_keyframes_with_clip
from A_model_inference import generate_video_summary
from cn_clip.clip import load_from_name

def process_video(video_path, output_dir, api_key):

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 定义音频输出路径
    audio_path = video_path.replace(".mp4", ".mp3")  # 提取的音频为 mp3 格式

    # 提取音频
    audio_path = extract_audio_from_video(video_path, audio_path)
    print(f"音频提取完成，保存为：{audio_path}")

    # 语音识别
    transcription = transcribe_audio(audio_path, api_key)

    # 加载CLIP模型
    print("🧠 加载 CLIP 模型中...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_from_name("ViT-B-16", device=device)
    model.eval()

    # 提取关键帧并生成总结
    keyframes_combined = extract_keyframes_with_clip(video_path, output_dir, transcription, model, preprocess, device)

    # 保存关键帧和结果
    final_json_path = os.path.join(output_dir, f"{video_path.split('.')[0]}_cnclip.json")
    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(keyframes_combined, f, ensure_ascii=False, indent=2)

    print(f"✅ 关键帧+语音信息保存在：{final_json_path}")

    # 调用大模型分析每一帧图像和文本生成总结
    for frame_info in keyframes_combined:
        image_path = frame_info["image_path"]
        summary = generate_video_summary(image_path, frame_info["text"], api_key)
        print(f"🎬 视频内容总结：{summary}")

# 使用示例
video_path = r'G:\videochat\my_design\test_video.mp4'  # 替换为你的视频路径
output_dir = r'G:\videochat\my_design\CNCLIP_keyframes_test_video'   # 替换为输出关键帧的文件夹路径
api_key = "sk-e6f5a000ba014f92b4857a6dcd782591"  # 替换为你的 API 密钥

process_video(video_path, output_dir, api_key)
