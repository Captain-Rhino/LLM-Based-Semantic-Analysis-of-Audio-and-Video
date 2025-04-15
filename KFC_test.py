import os
import json
import cv2
import torch
from PIL import Image
from tqdm import tqdm
import cn_clip.clip as clip
from cn_clip.clip import load_from_name

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 CN-CLIP 模型
model, preprocess = load_from_name("ViT-B-16", device=device)
model.eval()

# 配置路径
video_path = "test_video.mp4"
asr_json_path = "transcription.json"
output_dir = "CNCLIP_keyframes"
os.makedirs(output_dir, exist_ok=True)

# 加载语音识别结果
with open(asr_json_path, "r", encoding="utf-8") as f:
    asr_data = json.load(f)

# 打开视频
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

keyframes_info = []

# 遍历每个语音段落
for i, seg in enumerate(tqdm(asr_data, desc="Processing segments")):
    start_time, end_time, text = seg["start"], seg["end"], seg["text"]
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    text_len = len(text)
    # 自动设定帧数：每 40 字提一帧，最少 1 帧
    num_frames_to_extract = max(1, text_len // 40)

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

# 保存 JSON
output_json_path = os.path.join(output_dir, "cnclip_keyframes.json")
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(keyframes_info, f, ensure_ascii=False, indent=2)

print(f"\n✅ 提取完成，关键帧信息保存在：{output_json_path}")
