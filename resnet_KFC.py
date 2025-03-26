# semantic_keyframe_extractor.py

import cv2
import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
import numpy as np
import json
from PIL import Image

video_path = r"G:\videochat\my_design\test_video.mp4"
output_dir = r"G:\videochat\my_design\Resnet_KFC"
os.makedirs(output_dir, exist_ok=True)
json_output_path = os.path.join(output_dir, "keyframes_resnet_kmeans.json")

frame_interval = 15
num_clusters = 8

resnet = models.resnet18(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

cap = cv2.VideoCapture(video_path)
features, frames, timestamps = [], [], []
frame_index = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_index % frame_interval == 0:
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            feat = resnet(tensor).squeeze().numpy()
        features.append(feat)
        frames.append(frame)
        timestamps.append(timestamp)
    frame_index += 1

cap.release()

kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(features)
labels = kmeans.labels_
centers = kmeans.cluster_centers_

selected = []
for i in range(num_clusters):
    cluster_indices = np.where(labels == i)[0]
    center = centers[i]
    best_idx = cluster_indices[np.argmin([np.linalg.norm(features[j] - center) for j in cluster_indices])]
    selected.append(best_idx)

results = []
for i, idx in enumerate(sorted(selected)):
    frame = frames[idx]
    timestamp = timestamps[idx]
    filename = f"semantic_kf_{i:03d}_{timestamp:.2f}s.jpg"
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, frame)
    results.append({
        "semantic_kf_index": int(i),
        "source_frame_index": int(idx),
        "timestamp": float(round(timestamp, 2)),
        "image_path": filepath
    })

with open(json_output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"✅ 已保存 {len(results)} 个语义关键帧至：{output_dir}")
