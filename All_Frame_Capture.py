import cv2
import os
import json

# 视频路径与输出路径
video_path = r"G:\videochat\my_design\test_video.mp4"
output_dir = r"G:\videochat\my_design\all_frame"
os.makedirs(output_dir, exist_ok=True)

# 打开视频
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError("❌ 无法打开视频文件，请检查路径或格式是否正确！")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

# 设定截取时长（单位：秒）
clip_duration_sec = 10
max_frame_index = int(fps * clip_duration_sec)

frame_info = []
frame_index = 0

while True:
    ret, frame = cap.read()
    if not ret or frame_index > max_frame_index:
        break

    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # 当前时间戳（秒）
    output_path = os.path.join(output_dir, f"frame_{frame_index:04d}_{timestamp:.2f}s.jpg")
    cv2.imwrite(output_path, frame)

    frame_info.append({
        "frame_index": frame_index,
        "timestamp": round(timestamp, 2),
        "image_path": output_path
    })

    frame_index += 1

cap.release()

# 保存为 JSON 文件
output_json_path = os.path.join(output_dir, "all_frames_info.json")
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(frame_info, f, ensure_ascii=False, indent=2)

print(f"\n✅ 前10秒帧图像已保存至：{output_dir}")
print(f"✅ 帧信息 JSON 文件已保存为：{output_json_path}")
