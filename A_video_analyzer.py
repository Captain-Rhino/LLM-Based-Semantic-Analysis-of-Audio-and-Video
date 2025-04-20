import os
import json
import torch
import time
from A_audio_extractor import extract_audio_from_video
from A_audio_recognition import transcribe_audio
from A_keyframe_extractor import KeyframeExtractor
from A_model_inference import summarize_video_from_all_frames
from A_visualizer import generate_wordcloud
from cn_clip.clip import load_from_name
from A_clip_finetune import train_adaptor  # 新增导入


def process_video(video_path, output_dir, api_key, do_finetune=False):  # 新增do_finetune参数
    start_time = time.time()
    #新建文件路径
    os.makedirs(output_dir, exist_ok=True)

    # Step 1-3: 原有音频处理和CLIP加载不变
    #step1:提取音频mp3
    audio_path = extract_audio_from_video(video_path, video_path.replace(".mp4", ".mp3"))
    #step2:转录文本
    transcription = transcribe_audio(audio_path, api_key)
    #step3:加载clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_from_name("ViT-B-16", device=device)
    model.eval()

    # Step 4: 关键帧提取（自动保存CLIP特征）
    extractor = KeyframeExtractor(device=device)
    keyframes = extractor.extract_keyframes(
        video_path=video_path,
        output_dir=output_dir,
        asr_data=transcription
    )

    # 新增训练环节
    if do_finetune:  # 只有开启标志时才训练
        print("\n🔧 开始CLIP适配层微调...")
        train_data_path = os.path.join(output_dir, "clip_features.pth")
        adaptor = train_adaptor(
            data_path=train_data_path,
            epochs=10,
            batch_size=8
        )
        # 保存适配层权重供后续使用
        torch.save(adaptor.state_dict(), os.path.join(output_dir, "clip_adaptor.pth"))

    # Step 5: 视频总结（自动检测是否存在适配层）
    #
    adaptor_path = (
        os.path.join(output_dir, "clip_adaptor.pth")
        if do_finetune else
        "G:/videochat/my_design/adaptor_results/best_adaptor.pth"  # ← 改这里
    )

    if not os.path.exists(adaptor_path):
        adaptor_path = None  # 如果找不到路径就不传

    summary_output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_summary.json")
    summarize_video_from_all_frames(
        keyframes,
        api_key,
        adaptor_path=adaptor_path,  # 传递适配层路径
        output_summary_path=summary_output_path
    )
    # summarize_video_from_all_frames(
    #     keyframes,
    #     api_key,
    #     adaptor_path=os.path.join(output_dir, "clip_adaptor.pth") if do_finetune else None,  # 传递适配层路径
    #     output_summary_path=summary_output_path
    # )

    # Step 6: 可视化（原有逻辑不变）
    generate_wordcloud(transcription, summary_output_path, output_dir,
                       os.path.splitext(os.path.basename(video_path))[0])

    print(f"⏱️ 全流程完成，总耗时：{time.time() - start_time:.2f} 秒")


if __name__ == "__main__":
    video_path = r'G:\videochat\my_design\test_video.mp4'
    output_dir = f"G:/videochat/my_design/CNCLIP_keyframes_{os.path.splitext(os.path.basename(video_path))[0]}"
    api_key = "sk-e6f5a000ba014f92b4857a6dcd782591"

    # 新增--finetune参数控制是否训练
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune", action="store_true", help="Enable CLIP adaptor fine-tuning")
    args = parser.parse_args()

    process_video(video_path, output_dir, api_key, do_finetune=args.finetune)