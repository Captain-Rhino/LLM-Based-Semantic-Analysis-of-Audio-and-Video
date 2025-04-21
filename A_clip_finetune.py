import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import matplotlib.pyplot as plt


class ClipAdaptor(nn.Module):
    """轻量级CLIP特征适配层
    Args:
        input_dim: CLIP特征维度（默认512）
        hidden_dim: 隐藏层维度（默认256）
        dropout: 防止过拟合（默认0.1）
    """

    def __init__(self, input_dim=512, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        if isinstance(x, list):
            x = torch.stack(x)
        return self.layers(x.to(self.device))


def train_adaptor(data_path, output_dir="adaptor_results", epochs=100, batch_size=16):
    """训练适配层的主函数
    Args:
        data_path: 包含CLIP特征的.pth文件路径
        output_dir: 输出目录（保存模型和日志）
        epochs: 训练轮次
        batch_size: 批大小
    Returns:
        trained_model: 训练好的适配层模型
    """
    # 初始化环境
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据加载与验证
    try:
        data = torch.load(data_path)
        assert "image_feats" in data and "text_feats" in data
    except Exception as e:
        raise ValueError(f"数据加载失败: {e}，请检查{data_path}格式")

    # 初始化模型和优化器
    model = ClipAdaptor().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    loss_fn = nn.CosineEmbeddingLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # 训练循环
    losses = []
    best_loss = float('inf')
    progress_bar = tqdm(range(epochs), desc="训练适配层")

    for epoch in progress_bar:
        model.train()
        epoch_loss = 0
        indices = torch.randperm(len(data["image_feats"]))

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            batch_img = data["image_feats"][batch_idx].to(device).float()
            batch_txt = data["text_feats"][batch_idx].to(device).float()

            # 前向传播
            adapted_img = model(batch_img)
            loss = loss_fn(adapted_img, batch_txt, torch.ones(batch_img.size(0)).to(device))

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        # 更新学习率和记录
        scheduler.step()
        avg_loss = epoch_loss / (len(indices) / batch_size)
        losses.append(avg_loss)
        progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "best_adaptor.pth"))

    # 保存训练曲线
    plt.figure()
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cosine Loss")
    plt.title("CLIP Adaptor Training")
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))

    # 加载最佳模型返回
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_adaptor.pth")))

    #验证？
    model.eval()
    with torch.no_grad():
        sample_feat = data["image_feats"][:5].to(device).float()
        sample_text = data["text_feats"][:5].to(device).float()

        orig_sim = torch.cosine_similarity(sample_feat, sample_text, dim=1)
        adapted_sim = torch.cosine_similarity(model(sample_feat), sample_text, dim=1)

        print("\n🔍 验证结果对比（前5条样本）：")
        print(f"原始图文相似度均值:  {orig_sim.mean().item():.4f} ± {orig_sim.std().item():.4f}")
        print(f"适配后图文相似度均值: {adapted_sim.mean().item():.4f} ± {adapted_sim.std().item():.4f}")
    return model


# if __name__ == "__main__":
#     # 示例测试代码
#     test_data = {
#         "image_feats": torch.randn(100, 512),
#         "text_feats": torch.randn(100, 512)
#     }
#     torch.save(test_data, "test_features.pth")
#
#     model = train_adaptor("test_features.pth", epochs=3)
#     print(f"训练完成，模型已保存到 adaptor_results/")