import os
import json
import cv2
import torch
from PIL import Image
from tqdm import tqdm
import cn_clip.clip as clip

def extract_keyframes_with_clip(video_path, output_dir, asr_data, model, preprocess, device):
    """
    提取关键帧并与文本对齐
    :param video_path: 视频路径
    :param output_dir: 输出目录
    :param asr_data: 语音识别的 JSON 数据
    :param model: 加载的 CLIP 模型
    :param preprocess: 图像预处理
    :param device: 计算设备 (cpu 或 cuda)
    :return: keyframes_info - 提取的关键帧信息列表
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    keyframes_info = []



    # 遍历每个语音段落
    for i, seg in enumerate(tqdm(asr_data, desc="Processing segments")):
        start_time, end_time, text = seg["start"], seg["end"], seg["text"]
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        text_len = len(text)
        # 自动设定帧数：每 20 字提一帧，最少 1 帧
        num_frames_to_extract = max(1, text_len // 20)

        # 平均抽取帧 index（帧范围太小时保证至少有1帧）
        if end_frame - start_frame < num_frames_to_extract:
            frame_indices = [start_frame]
        else:
            step = max(1, (end_frame - start_frame) // (num_frames_to_extract + 1))
            frame_indices = [start_frame + step * j for j in range(1, num_frames_to_extract + 1)]

        candidate_frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = cap.read()
            if not success:
                continue
            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            candidate_frames.append((idx, image_pil, frame))

        if not candidate_frames:
            continue

        # 编码文本一次（对全部帧复用）
        text_token = clip.tokenize([text]).to(device)
        with torch.no_grad():
            text_feat = model.encode_text(text_token)
            text_feat /= text_feat.norm(dim=-1, keepdim=True)

        # 每一帧单独计算相似度并保存
        for j, (idx, img_pil, frame_raw) in enumerate(candidate_frames):
            img_tensor = preprocess(img_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                img_feat = model.encode_image(img_tensor)
                img_feat /= img_feat.norm(dim=-1, keepdim=True)

            sim = torch.cosine_similarity(text_feat, img_feat).item()
            timestamp = idx / fps
            save_path = os.path.join(output_dir, f"kf_{i:03d}_{j:02d}_{timestamp:.2f}s.jpg")
            print(f"正在保存关键帧到：{save_path}")  # 调试用，打印路径

            # 保存图像
            cv2.imwrite(save_path, frame_raw)

            keyframes_info.append({
                "segment_index": i,
                "frame_rank": j,
                "text_len": text_len,
                "start": round(start_time, 2),
                "end": round(end_time, 2),
                "text": text,
                "frame_id": idx,
                "timestamp": round(timestamp, 2),
                "image_path": save_path,
                "similarity": round(sim, 4)
            })

    cap.release()
    return keyframes_info
