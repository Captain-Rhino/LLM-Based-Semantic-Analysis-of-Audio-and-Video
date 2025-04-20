# A_event_detector.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import cv2
from PIL import Image

# 导入音频特征提取器
from feature_extractor import extract_audio_features, get_audio_segment

# --- 1. 事件类别定义 (重要：训练和推理必须一致) ---
EVENT_CLASSES = ["speech", "applause", "music", "silence", "laughter", "background"]
NUM_CLASSES = len(EVENT_CLASSES)
EVENT_TO_ID = {event: i for i, event in enumerate(EVENT_CLASSES)}
ID_TO_EVENT = {i: event for i, event in enumerate(EVENT_CLASSES)}

VISUAL_DIM = 512  # CN-CLIP ViT-B/16 输出维度
AUDIO_DIM = 128   # VGGish 输出维度
FUSED_DIM = VISUAL_DIM + AUDIO_DIM # 融合后的维度

# --- 2. 分类器模型定义 (训练和推理都需要这个结构) ---
class EventClassifier(nn.Module):
    def __init__(self, input_dim=FUSED_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- 3. 事件检测推理函数 ---
def detect_events_in_video(video_path, audio_path, trained_model_path,
                           clip_model, clip_preprocess, device,
                           segment_duration=1.0, batch_size=16):
    """
    对整个视频进行事件检测（推理）。
    加载训练好的模型，提取特征，进行预测。
    """
    print(f"🔍 开始对视频 {os.path.basename(video_path)} 进行事件检测...")
    # --- a) 加载训练好的分类器 ---
    print(f"   加载训练好的模型: {trained_model_path}")
    if not os.path.exists(trained_model_path):
        print(f"❌ 错误: 找不到模型文件 {trained_model_path}")
        return []

    model = EventClassifier(input_dim=FUSED_DIM, num_classes=NUM_CLASSES)
    try:
        model.load_state_dict(torch.load(trained_model_path, map_location=device))
        model.to(device)
        model.eval() # 设置为评估模式
        print("   模型加载成功。")
    except Exception as e:
        print(f"❌ 错误: 加载模型权重失败: {e}")
        return []

    # --- b) 获取视频信息 ---
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 错误: 无法打开视频文件 {video_path}")
            return []
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        cap.release()
        print(f"   视频时长: {duration:.2f} 秒, FPS: {fps:.2f}")
    except Exception as e:
        print(f"❌ 错误: 读取视频信息失败: {e}")
        return []

    # --- c) 准备时间戳和特征提取 ---
    results = []
    timestamps = np.arange(0, duration, segment_duration)
    all_features = []

    print(f"   按 {segment_duration}s 窗口提取音视频特征...")
    # 确保 CLIP 模型在评估模式
    clip_model.eval()

    # --- 优化：缓存 VideoCapture 对象 ---
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError("无法打开视频")

        for i in tqdm(range(len(timestamps)), desc="   特征提取进度"):
            start_time = timestamps[i]
            end_time = start_time + segment_duration

            # i) 视觉特征
            visual_feature = np.zeros(VISUAL_DIM) # 默认零向量
            try:
                mid_frame_idx = int((start_time + segment_duration / 2) * fps)
                # 检查帧索引是否有效
                if 0 <= mid_frame_idx < frame_count:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
                    success, frame = cap.read()
                    if success:
                        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        img_tensor = clip_preprocess(img_pil).unsqueeze(0).to(device)
                        with torch.no_grad():
                            visual_feature = clip_model.encode_image(img_tensor).squeeze().cpu().numpy()
                    # else: print(f"   警告: 帧 {mid_frame_idx} 读取失败")
                # else: print(f"   警告: 计算的帧索引 {mid_frame_idx} 超出范围 [0, {frame_count-1}]")

            except Exception as e:
                print(f"   警告: 提取 {start_time:.2f}s 处视觉特征时出错: {e}")
                # 保留零向量

            # ii) 音频特征
            audio_feature = np.zeros(AUDIO_DIM) # 默认零向量
            try:
                audio_segment, sr = get_audio_segment(audio_path, start_time, end_time)
                if audio_segment is not None and len(audio_segment) > 0:
                    # 使用 A_feature_extractor 中的函数
                    audio_feature_extracted = extract_audio_features(audio_segment, sr)
                    if audio_feature_extracted is not None:
                        audio_feature = audio_feature_extracted
                    # else: print(f"   警告: {start_time:.2f}s 处音频特征提取返回 None")
                # else: print(f"   警告: {start_time:.2f}s 处无法加载音频片段")

            except Exception as e:
                print(f"   警告: 提取 {start_time:.2f}s 处音频特征时出错: {e}")
                # 保留零向量

            # iii) 融合特征
            fused_feature = np.concatenate((visual_feature, audio_feature))
            all_features.append(fused_feature)

    except Exception as e:
        print(f"❌ 错误: 特征提取主循环出错: {e}")
        # 确保 VideoCapture 被释放
        if cap is not None and cap.isOpened():
            cap.release()
        return []
    finally:
        # 确保 VideoCapture 被释放
        if cap is not None and cap.isOpened():
            cap.release()

    # --- d) 批量推理 ---
    if not all_features:
        print("   没有成功提取到任何特征，无法进行事件检测。")
        return []

    feature_dataset = torch.utils.data.TensorDataset(torch.tensor(np.array(all_features), dtype=torch.float32))
    feature_loader = DataLoader(feature_dataset, batch_size=batch_size)

    all_predictions = []
    all_confidences = []
    print(f"   使用模型进行批量推理 (Batch Size: {batch_size})...")
    with torch.no_grad():
        for batch_features in tqdm(feature_loader, desc="   推理进度"):
            batch_features = batch_features[0].to(device) # DataLoader 返回的是元组
            outputs = model(batch_features)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predicted_ids = torch.max(probabilities, dim=1)
            all_predictions.extend(predicted_ids.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())

    # --- e) 整理原始结果 ---
    raw_events = []
    for i in range(len(timestamps)):
        start_time = timestamps[i]
        # 确保结束时间不超过视频总时长
        end_time = min(start_time + segment_duration, duration)
        pred_id = all_predictions[i]
        confidence = round(float(all_confidences[i]), 4)
        event_label = ID_TO_EVENT.get(pred_id, "unknown") # 使用 get 以防万一
        raw_events.append({
            "start": round(start_time, 2),
            "end": round(end_time, 2),
            "event": event_label,
            "confidence": confidence
        })
    print("   原始事件预测完成。")
    return raw_events


# --- 4. 事件后处理 (合并) ---
def merge_events(raw_events, min_confidence=0.5, min_duration=0.8):
    """
    合并相邻的、置信度高的相同事件。
    min_duration: 合并后事件的最短持续时间（秒）
    """
    if not raw_events:
        return []
    print(f"   开始合并事件 (min_confidence={min_confidence}, min_duration={min_duration}s)...")

    merged = []
    if not raw_events: return merged

    current_event_type = None
    current_event_start = -1
    current_event_end = -1
    confidences_list = []

    for i, event in enumerate(raw_events):
        is_significant = (event["event"] != "background" and event["confidence"] >= min_confidence)

        if is_significant:
            if event["event"] == current_event_type:
                # 扩展当前事件段
                current_event_end = event["end"]
                confidences_list.append(event["confidence"])
            else:
                # 结束上一个事件段（如果存在且满足最短时长）
                if current_event_type is not None and (current_event_end - current_event_start >= min_duration):
                    avg_confidence = round(np.mean(confidences_list), 4)
                    merged.append({
                        "type": current_event_type,
                        "start": round(current_event_start, 2),
                        "end": round(current_event_end, 2),
                        "avg_confidence": avg_confidence,
                        "duration": round(current_event_end - current_event_start, 2)
                    })

                # 开始新的事件段
                current_event_type = event["event"]
                current_event_start = event["start"]
                current_event_end = event["end"]
                confidences_list = [event["confidence"]]
        else:
            # 非显著事件（背景或低置信度）强制结束上一个事件段
            if current_event_type is not None and (current_event_end - current_event_start >= min_duration):
                avg_confidence = round(np.mean(confidences_list), 4)
                merged.append({
                    "type": current_event_type,
                    "start": round(current_event_start, 2),
                    "end": round(current_event_end, 2),
                    "avg_confidence": avg_confidence,
                    "duration": round(current_event_end - current_event_start, 2)
                })
            # 重置当前事件状态
            current_event_type = None
            current_event_start = -1
            current_event_end = -1
            confidences_list = []

    # 处理循环结束后可能遗留的最后一个事件段
    if current_event_type is not None and (current_event_end - current_event_start >= min_duration):
        avg_confidence = round(np.mean(confidences_list), 4)
        merged.append({
            "type": current_event_type,
            "start": round(current_event_start, 2),
            "end": round(current_event_end, 2),
            "avg_confidence": avg_confidence,
            "duration": round(current_event_end - current_event_start, 2)
        })

    print(f"   事件合并完成，得到 {len(merged)} 个事件。")
    return merged