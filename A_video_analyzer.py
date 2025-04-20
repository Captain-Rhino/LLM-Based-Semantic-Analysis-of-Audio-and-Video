import os
import json
import torch
import time
from A_audio_extractor import extract_audio_from_video
from A_audio_recognition import transcribe_audio
from A_keyframe_extractor import extract_keyframes_with_clip
from A_model_inference import summarize_video_from_all_frames,build_structured_prompt,generate_video_summary
from cn_clip.clip import load_from_name

# ===========================
# 视频分析主流程（重构版）
# ===========================
def process_video(video_path, output_dir, api_key):
    start_time = time.time()

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 音频路径设置
    audio_path = video_path.replace(".mp4", ".mp3")

    # Step 1: 提取音频
    audio_path = extract_audio_from_video(video_path, audio_path)
    print(f"🎵 音频提取完成：{audio_path}")

    # Step 2: 语音识别
    transcription = transcribe_audio(audio_path, api_key)
    print(f"📝 语音转录完成，共 {len(transcription)} 条句子")

    # Step 3: 加载 CLIP 模型
    print("🧠 正在加载 CLIP 模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_from_name("ViT-B-16", device=device)
    model.eval()

    # Step 4: 提取关键帧并匹配语音片段
    keyframes_combined = extract_keyframes_with_clip(
        video_path, output_dir, transcription, model, preprocess, device
    )

    # 保存关键帧信息
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    keyframe_json_path = os.path.join(output_dir, f"{video_name}_cnclip.json")
    with open(keyframe_json_path, "w", encoding="utf-8") as f:
        json.dump(keyframes_combined, f, ensure_ascii=False, indent=2)
    print(f"✅ 关键帧+文本信息已保存：{keyframe_json_path}")

    # Step 5: 视频总结
    summary_output_path = os.path.join(output_dir, f"{video_name}_summary.json")
    summarize_video_from_all_frames(keyframes_combined, api_key, output_summary_path=summary_output_path)
    # for idx, frame_info in enumerate(keyframes_combined):
    #     is_last = (idx == len(keyframes_combined) - 1)
    #     prompt = build_structured_prompt(frame_info, is_last=is_last)
    #     print(prompt)
    #     image_path = frame_info["image_path"]
    #     summary = generate_video_summary(image_path, prompt, api_key)
    #     print(f"🎬 视频内容总结：{summary}")

    print(f"📄 视频总结完成：{summary_output_path}")

    # Step 6: 词云或思维导图生成（接口保留）
    # from A_visualizer import generate_wordcloud_or_mindmap
    # generate_wordcloud_or_mindmap(transcription, output_dir)

    # Step 7: 视频事件定位与识别（接口保留）
    # from A_event_detector import detect_video_events
    # events = detect_video_events(transcription, keyframes_combined)
    # events_output_path = os.path.join(output_dir, f"{video_name}_events.json")
    # with open(events_output_path, "w", encoding="utf-8") as f:
    #     json.dump(events, f, ensure_ascii=False, indent=2)
    # print(f"📌 事件检测结果保存：{events_output_path}")

    end_time = time.time()
    print(f"⏱️ 全流程完成，总耗时：{end_time - start_time:.2f} 秒")


# ========== 脚本入口 ==========
if __name__ == "__main__":
    video_path = r'G:\videochat\my_design\video_without_audio.mp4'
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = f"G:/videochat/my_design/CNCLIP_keyframes_{video_name}"
    api_key = "sk-e6f5a000ba014f92b4857a6dcd782591"

    process_video(video_path, output_dir, api_key)
