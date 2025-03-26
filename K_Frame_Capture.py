import cv2
import os
import pandas as pd

# 视频路径与输出路径
video_path = r"G:\videochat\my_design\test_video.mp4"
output_dir = r"G:\videochat\my_design\K_frame"
os.makedirs(output_dir, exist_ok=True)

# 打开视频
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError("❌ 无法打开视频文件，请检查路径或格式是否正确！")

# 尝试读取第一帧
success, prev_frame = cap.read()
if not success or prev_frame is None:
    raise ValueError("❌ 视频第一帧读取失败，可能视频损坏或格式不受支持。")

# 参数设定
frame_interval = 2  #每两帧比较一次
threshold = 18
fps = cap.get(cv2.CAP_PROP_FPS)
# print(fps)    视频帧率

frame_count = 0
kf_index = 0
timestamps = []

# 灰度预处理
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) # 先前帧的灰度处理

# 遍历帧
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_interval != 0:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, gray) #对比这个diff像素差值
    score = diff.mean()

    #print(f"diff 平均值: {score:.2f}")
    #print(diff)
    #max_diff = diff.max()
    #max_gray = gray.max()
    #print(f"最大灰度值(gray_max):{max_gray},最大像素差值（diff.max）: {max_diff}")
    #print(score)

    # 手动暂停，查看当前帧信息或调试
    #input("👉 按下回车继续处理下一帧（或 Ctrl+C 退出程序）...")

    if score > threshold:
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        output_path = os.path.join(output_dir, f"kf_{kf_index:03d}_{timestamp:.2f}s.jpg")
        cv2.imwrite(output_path, frame)
        timestamps.append((kf_index, timestamp, output_path))
        kf_index += 1
        prev_gray = gray

cap.release()

# 显示结果表格
import json

# 构造结构化字典列表
keyframes_data = [
    {
        "frame_index": idx,
        "timestamp": round(ts, 2),
        "image_path": output_path
    }
    for idx, ts, output_path in timestamps
]

# 保存为 JSON 文件
output_json_path = r"G:\videochat\my_design\K_frame\keyframes_info.json"
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(keyframes_data, f, ensure_ascii=False, indent=2)

print(f"\n✅ 关键帧信息已保存为 JSON 文件：{output_json_path}")