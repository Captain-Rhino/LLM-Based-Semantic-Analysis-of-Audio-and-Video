import cv2
import os
import json
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']        # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False          # 正确显示负号

# 视频路径
video_path = r"G:\videochat\my_design\test_video.mp4"
output_dir_base = r"G:\videochat\my_design\K_frame_test_interval"
os.makedirs(output_dir_base, exist_ok=True)

# 参数设置
threshold = 15
frame_intervals = list(range(1, 31))  # [1, 2, ..., 30]
param_results = {}

for frame_interval in frame_intervals:
    # 初始化参数
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("❌ 无法打开视频文件")
    success, prev_frame = cap.read()
    if not success or prev_frame is None:
        raise ValueError("❌ 无法读取视频第一帧")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_count = 0
    kf_index = 0
    timestamps = []

    # 创建输出目录
    combo_folder = f"int{frame_interval}_thr{threshold}"
    output_dir = os.path.join(output_dir_base, combo_folder)
    os.makedirs(output_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % frame_interval != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        score = diff.mean()
        if score > threshold:
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            output_path = os.path.join(output_dir, f"kf_{kf_index:03d}_{timestamp:.2f}s.jpg")
            cv2.imwrite(output_path, frame)
            timestamps.append((kf_index, timestamp, output_path))
            kf_index += 1
            prev_gray = gray

    cap.release()
    param_results[frame_interval] = len(timestamps)

# 画图
x_labels = [str(k) for k in param_results.keys()]
y_values = list(param_results.values())

plt.figure(figsize=(10, 5))
plt.plot(x_labels, y_values, marker='o', linestyle='-')
plt.title("不同帧间隔下的关键帧数量对比（固定阈值为15）")
plt.xlabel("frame_interval")
plt.ylabel("关键帧数量")
plt.grid(True)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 保存数据为 JSON
json_path = os.path.join(output_dir_base, "frame_test_interval_results.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(param_results, f, ensure_ascii=False, indent=2)

print(f"✅ 所有帧间隔测试完成，结果已保存到：{json_path}")
