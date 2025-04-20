import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import Compose, RandomHorizontalFlip
from cn_clip.clip import load_from_name
from PIL import Image
import numpy as np


# 1. 自定义数据集类
class EventDataset(Dataset):
    def __init__(self, root_dir="dataset/train"):
        self.samples = []  # 示例: [("video1.mp4", 0, 5, "speech"), ...]
        self.transform = Compose([
            RandomHorizontalFlip(p=0.5),
            # CN-CLIP的默认预处理
            lambda x: (x - np.array([0.48145466, 0.4578275, 0.40821073])) /
                      np.array([0.26862954, 0.26130258, 0.27577711])
        ])
        self.class2id = {"speech": 0, "applause": 1}  # 根据你的类别修改

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, start, end, label = self.samples[idx]
        frames = self._extract_frames(path, start, end)  # 需实现抽帧函数
        return torch.stack([self.transform(Image.fromarray(f)) for f in frames]), self.class2id[label]

    def _extract_frames(self, path, start, end):
        # 你的抽帧逻辑（返回帧列表）
        return [np.random.rand(224, 224, 3) for _ in range(8)]  # 示例伪代码


# 2. 模型定义
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_from_name("ViT-B-16", device=device)

# 冻结CLIP主体
for param in model.parameters():
    param.requires_grad = False

# 定义分类头（修正语法错误）
model.classifier = nn.Sequential(
    nn.Linear(512, 256),
    nn.GELU(),
    nn.Linear(256, len(EventDataset().class2id))  # 类别数自动匹配
).to(device)  # 这里是要移动整个分类头到设备

# 3. 训练配置
optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


# 4. 训练循环
def train():
    dataset = EventDataset()
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    for epoch in range(10):
        for frames, labels in dataloader:
            frames, labels = frames.to(device), labels.to(device)

            # 提取特征 (B, T, C, H, W) -> (B, T, 512)
            with torch.no_grad():
                features = model.encode_image(frames.flatten(0, 1))  # 合并batch和时序
                features = features.unflatten(0, (frames.shape[0], frames.shape[1]))  # 恢复形状

            # 分类 (取时序平均)
            logits = model.classifier(features.mean(dim=1))
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        torch.save(model.classifier.state_dict(), f"event_cls_epoch{epoch}.pth")


if __name__ == "__main__":
    train()