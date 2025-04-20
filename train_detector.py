# train_detector.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import os
import cv2
from PIL import Image
import argparse # 用于命令行参数
import json # 用于加载标注数据

# 导入必要的模块
from A_feature_extractor import extract_audio_features, get_audio_segment
from A_event_detector import EventClassifier, EVENT_CLASSES, NUM_CLASSES, EVENT_TO_ID, VISUAL_DIM, AUDIO_DIM, FUSED_DIM
from cn_clip.clip import load_from_name

# --- 1. 数据集类 (与之前类似，但现在在此文件中) ---
class AudioVisualDataset(Dataset):
    def __init__(self, data_list, clip_model, clip_preprocess, device, segment_duration=1.0):
        """
        data_list: 包含标注信息的列表，每个元素是一个字典:
                   {'video_path': '...', 'audio_path': '...',
                    'start_time': float, 'event_label': 'applause'}
        clip_model: 加载的 CN-CLIP 模型
        clip_preprocess: CN-CLIP 的图像预处理
        device: 'cuda' or 'cpu'
        segment_duration: 每个片段的时长（秒）
        """
        self.data_list = data_list
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.device = device
        self.segment_duration = segment_duration
        self.visual_dim = VISUAL_DIM
        self.audio_dim = AUDIO_DIM

        # --- 优化: 预先打开 VideoCapture 对象 ---
        self.video_caps = {}
        self.video_fps = {}
        self.video_frame_count = {}
        print("预加载视频信息...")
        unique_videos = set(item['video_path'] for item in data_list)
        for video_path in tqdm(unique_videos, desc="Loading video info"):
            try:
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    self.video_caps[video_path] = cap # 保持打开状态
                    self.video_fps[video_path] = cap.get(cv2.CAP_PROP_FPS)
                    self.video_frame_count[video_path] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                else:
                    print(f"警告: 无法打开视频 {video_path}")
            except Exception as e:
                 print(f"警告: 加载视频信息 {video_path} 时出错: {e}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        video_path = item['video_path']
        audio_path = item['audio_path']
        start_time = item['start_time']
        end_time = start_time + self.segment_duration
        event_label_str = item['event_label']
        event_label_id = EVENT_TO_ID.get(event_label_str, EVENT_TO_ID['background']) # 默认背景

        # a) 提取视觉特征 (使用预加载的 cap)
        visual_feature = np.zeros(self.visual_dim)
        if video_path in self.video_caps:
            try:
                cap = self.video_caps[video_path]
                fps = self.video_fps[video_path]
                frame_count = self.video_frame_count[video_path]
                mid_frame_idx = int((start_time + self.segment_duration / 2) * fps)

                if 0 <= mid_frame_idx < frame_count:
                    # 注意：频繁 set pos 可能效率不高，但对于随机访问是必要的
                    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
                    success, frame = cap.read()
                    if success:
                        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        img_tensor = self.clip_preprocess(img_pil).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            visual_feature_extracted = self.clip_model.encode_image(img_tensor).squeeze().cpu().numpy()
                            if visual_feature_extracted is not None:
                                visual_feature = visual_feature_extracted
                    # else: print(f"警告: (Dataset) 帧 {mid_frame_idx} 读取失败 for {video_path}")
                # else: print(f"警告: (Dataset) 帧索引 {mid_frame_idx} 超出范围 for {video_path}")

            except Exception as e:
                print(f"警告: (Dataset) 提取视觉特征出错 for {video_path} at {start_time}s: {e}")
                # 保留零向量
        # else: print(f"警告: (Dataset) 找不到预加载的 VideoCapture for {video_path}")


        # b) 提取音频特征
        audio_feature = np.zeros(self.audio_dim)
        try:
            audio_segment, sr = get_audio_segment(audio_path, start_time, end_time)
            if audio_segment is not None and len(audio_segment) > 0:
                audio_feature_extracted = extract_audio_features(audio_segment, sr)
                if audio_feature_extracted is not None:
                    audio_feature = audio_feature_extracted
        except Exception as e:
            print(f"警告: (Dataset) 提取音频特征出错 for {audio_path} at {start_time}s: {e}")
            # 保留零向量

        # c) 融合特征
        fused_feature = np.concatenate((visual_feature, audio_feature))

        return torch.tensor(fused_feature, dtype=torch.float32), torch.tensor(event_label_id, dtype=torch.long)

    # --- 清理资源 ---
    def __del__(self):
        print("Releasing video captures in dataset...")
        for cap in self.video_caps.values():
            if cap.isOpened():
                cap.release()

# --- 2. 训练函数 ---
def train_event_classifier(train_data_list, val_data_list, clip_model, clip_preprocess, device,
                           model_save_path, epochs=10, batch_size=32, learning_rate=1e-4, num_workers=0): # num_workers=0 适用于 Windows/调试
    """训练事件分类器"""
    print("\n--- 开始训练事件分类器 ---")
    print(f"训练样本数: {len(train_data_list)}")
    print(f"验证样本数: {len(val_data_list)}")
    print(f"设备: {device}")
    print(f"模型保存路径: {model_save_path}")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, Learning Rate: {learning_rate}")

    # --- a) 确保 CLIP 模型在评估模式且不需要梯度 ---
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False
    print("CLIP 模型已冻结。")

    # --- b) 初始化数据集和 DataLoader ---
    print("初始化数据集...")
    # 注意：数据集初始化时会加载视频信息
    train_dataset = AudioVisualDataset(train_data_list, clip_model, clip_preprocess, device)
    val_dataset = AudioVisualDataset(val_data_list, clip_model, clip_preprocess, device)
    # pin_memory=True 如果使用 GPU 可以加速数据传输
    pin_memory = (device == 'cuda')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    print("DataLoader 初始化完成。")

    # --- c) 初始化模型、损失函数和优化器 ---
    model = EventClassifier(input_dim=FUSED_DIM, num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    # 只优化分类器参数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("分类器模型、损失函数、优化器初始化完成。")

    # --- d) 训练循环 ---
    print("\n--- 开始训练循环 ---")
    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train() # 设置为训练模式
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for features, labels in pbar_train:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            # 更新进度条显示
            pbar_train.set_postfix({'Loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total

        # --- e) 验证 ---
        model.eval() # 设置为评估模式
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        with torch.no_grad():
            for features, labels in pbar_val:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                pbar_val.set_postfix({'Loss': loss.item()})


        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        print(f"\nEpoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # --- f) 保存最佳模型 ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            try:
                torch.save(model.state_dict(), model_save_path)
                print(f"✅ 模型性能提升，已保存至 {model_save_path} (最佳验证精度: {best_val_acc:.2f}%)")
            except Exception as e:
                print(f"❌ 保存模型失败: {e}")
        else:
            print(f"   模型性能未提升 (当前最佳验证精度: {best_val_acc:.2f}%)")

    print("\n--- 训练完成 ---")
    # 释放数据集中的 VideoCapture 对象
    del train_dataset
    del val_dataset


# --- 3. 主程序入口 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练音视频事件分类器")
    parser.add_argument("--train_data", required=True, help="指向训练数据标注文件(JSON列表)的路径")
    parser.add_argument("--val_data", required=True, help="指向验证数据标注文件(JSON列表)的路径")
    parser.add_argument("--model_save_path", default="./event_classifier.pth", help="训练好的模型保存路径")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批处理大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader 使用的进程数 (Windows建议为0)")
    parser.add_argument("--device", default=None, help="指定设备 ('cuda' or 'cpu'), 默认自动检测")

    args = parser.parse_args()

    # --- a) 设置设备 ---
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # --- b) 加载 CLIP 模型 (训练需要) ---
    print("加载 CN-CLIP 模型 (ViT-B-16)...")
    try:
        clip_model, clip_preprocess = load_from_name("ViT-B-16", device=device)
        clip_model.eval() # 确保是评估模式
        print("CN-CLIP 模型加载成功。")
    except Exception as e:
        print(f"❌ 致命错误: 加载 CN-CLIP 模型失败: {e}")
        exit(1) # 无法进行特征提取，退出

    # --- c) 加载标注数据 ---
    print("加载标注数据...")
    try:
        with open(args.train_data, 'r', encoding='utf-8') as f:
            train_data_list = json.load(f)
        with open(args.val_data, 'r', encoding='utf-8') as f:
            val_data_list = json.load(f)
        print(f"加载完成: {len(train_data_list)} 个训练样本, {len(val_data_list)} 个验证样本")
        # 可以在这里添加对数据格式的校验
        if not isinstance(train_data_list, list) or not isinstance(val_data_list, list):
             raise ValueError("标注数据文件应包含 JSON 列表")
        if not train_data_list or not val_data_list:
            raise ValueError("训练或验证数据列表为空")
        # 简单检查第一个元素格式
        if not all(k in train_data_list[0] for k in ['video_path', 'audio_path', 'start_time', 'event_label']):
            raise ValueError("训练数据项缺少必要的键")

    except FileNotFoundError:
        print(f"❌ 致命错误: 找不到标注数据文件 {args.train_data} 或 {args.val_data}")
        exit(1)
    except json.JSONDecodeError:
        print(f"❌ 致命错误: 标注数据文件格式错误 (非有效 JSON)")
        exit(1)
    except ValueError as ve:
         print(f"❌ 致命错误: 标注数据内容错误: {ve}")
         exit(1)
    except Exception as e:
        print(f"❌ 致命错误: 加载标注数据时发生未知错误: {e}")
        exit(1)

    # --- d) 创建模型保存目录 ---
    model_save_dir = os.path.dirname(args.model_save_path)
    if model_save_dir and not os.path.exists(model_save_dir):
        print(f"创建模型保存目录: {model_save_dir}")
        os.makedirs(model_save_dir)

    # --- e) 调用训练函数 ---
    train_event_classifier(
        train_data_list=train_data_list,
        val_data_list=val_data_list,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        device=device,
        model_save_path=args.model_save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_workers=args.num_workers
    )