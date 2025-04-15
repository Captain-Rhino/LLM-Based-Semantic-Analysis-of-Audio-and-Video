import os
import json
import time
import base64
import requests
import subprocess
import cv2
import torch
from PIL import Image
from tqdm import tqdm
import cn_clip.clip as clip
from cn_clip.clip import load_from_name

# ====== 设置视频路径======
video_path = r"G:\videochat\my_design\test_movie.mp4"
# =========================================

# 获取文件名、音频路径等
video_name = os.path.splitext(os.path.basename(video_path))[0]
audio_path = f"{video_name}.mp3"
transcription_path = f"transcription_{video_name}.json"
output_dir = f"CNCLIP_keyframes_{video_name}"
final_json_path = os.path.join(output_dir, f"{video_name}_cnclip.json")
os.makedirs(output_dir, exist_ok=True)

import threading

# Step 2：发送至语音识别服务
with open(audio_path, 'rb') as f:
    audio_base64 = base64.b64encode(f.read()).decode('utf-8')

print("🎙️ 正在调用 SenseVoice 识别服务...")

# 记录开始时间
start_time = time.time()

# 请求的响应存储
response = None

# 异步线程：请求语音识别
def send_request():
    global response
    response = requests.post("http://127.0.0.1:2002/asr", json={"wav": audio_base64})

# 开启线程并实时更新时间
request_thread = threading.Thread(target=send_request)
request_thread.start()

# 显示实时更新的时间
while response is None:
    elapsed = time.time() - start_time
    print(f"\r🧠 正在识别语音，当前用时：{elapsed:.1f} s", end="", flush=True)
    time.sleep(0.1)  # 每 0.3 秒更新一次

# 等待请求线程结束
request_thread.join()

# 处理识别结果
if response.status_code == 200:
    asr_result = response.json().get("res", [])
    with open(transcription_path, "w", encoding="utf-8") as f:
        json.dump(asr_result, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 识别结果已保存为：{transcription_path}")
else:
    print(f"❌ 语音识别失败，状态码: {response.status_code}")


# Step 3：初始化 CN-CLIP 模型
print("🧠 加载 CN-CLIP 模型中...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_from_name("ViT-B-16", device=device)
model.eval()

# Step 4：打开视频
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# Step 5：遍历每段文本，提取关键帧
keyframes_combined = []
for i, seg in enumerate(tqdm(asr_result, desc="🎞️ 正在提取关键帧")):
    start_time, end_time, text = seg["start"], seg["end"], seg["text"]
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    text_len = len(text)
    num_frames_to_extract = max(1, text_len // 20)

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

    text_token = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_feat = model.encode_text(text_token)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)

    for j, (idx, img_pil, frame_raw) in enumerate(candidate_frames):
        img_tensor = preprocess(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            img_feat = model.encode_image(img_tensor)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)

        sim = torch.cosine_similarity(text_feat, img_feat).item()
        timestamp = idx / fps
        save_path = os.path.join(output_dir, f"kf_{i:03d}_{j:02d}_{timestamp:.2f}s.jpg")
        cv2.imwrite(save_path, frame_raw)
        keyframes_combined.append({
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

# Step 6：保存合并结果
with open(final_json_path, "w", encoding="utf-8") as f:
    json.dump(keyframes_combined, f, ensure_ascii=False, indent=2)

print(f"\n✅ 合并完成，关键帧+语音信息保存在：{final_json_path}")
