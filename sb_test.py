import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from torch.nn.utils import clip_grad_norm_

# === 参数配置 ===
train_dir = "G:/FER/train"
val_dir = "G:/FER/test"
save_path = "G:/FER/emotion_resnet18_best.pth"
num_classes = 7
batch_size = 128
epochs = 50
patience = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 图像预处理 ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === 加载数据集 ===
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# === 模型定义 ===
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(device)

# === 损失与优化器 ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

# === EarlyStopping 参数 ===
best_val_acc = 0.0
early_stop_counter = 0
train_losses = []
val_accuracies = []
all_labels, all_preds = [], []

# === 训练循环 ===
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct_train = 0
    total_train = 0

    print(f"\n📦 Epoch {epoch + 1}/{epochs}")
    pbar = tqdm(train_loader, desc="🔄 Training", unit="batch")

    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)
        pbar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

    train_acc = 100. * correct_train / total_train
    train_losses.append(total_loss)

    # === 验证 ===
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    val_acc = 100. * correct / total
    val_accuracies.append(val_acc)
    print(f"✅ Epoch {epoch + 1} | Loss: {total_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    scheduler.step(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        early_stop_counter = 0
        torch.save(model.state_dict(), save_path)
        print(f"💾 新的最佳模型已保存：{save_path}")
    else:
        early_stop_counter += 1
        print(f"⏳ 验证准确率未提升（{early_stop_counter}/{patience}）")
        if early_stop_counter >= patience:
            print("🛑 触发 EarlyStopping，训练结束")
            break

print(f"\n🎉 训练完成，最佳验证准确率：{best_val_acc:.2f}%")

# === 混淆矩阵与分类报告 ===
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
cm = confusion_matrix(all_labels, all_preds)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=emotion_labels, yticklabels=emotion_labels)
plt.title("Normalized Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("G:/FER/confusion_matrix.png")
plt.close()

print("✅ 混淆矩阵图已保存为 G:/FER/confusion_matrix.png")

# === 保存分类报告 ===
report = classification_report(all_labels, all_preds, target_names=emotion_labels, digits=4)
with open("G:/FER/classification_report.txt", "w", encoding="utf-8") as f:
    f.write(report)
print("📄 分类报告已保存为 G:/FER/classification_report.txt")

# === 保存预测结果 CSV ===
df_result = pd.DataFrame({
    "True Label": [emotion_labels[i] for i in all_labels],
    "Predicted Label": [emotion_labels[i] for i in all_preds]
})
df_result.to_csv("G:/FER/prediction_results.csv", index=False, encoding="utf-8-sig")
print("📊 所有预测结果已保存为 G:/FER/prediction_results.csv")

# === 绘制训练损失与验证准确率图 ===
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("训练损失")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("验证准确率")
plt.legend()

plt.tight_layout()
plt.savefig("G:/FER/training_metrics.png")
plt.close()
print("📈 训练过程图已保存为 G:/FER/training_metrics.png")
