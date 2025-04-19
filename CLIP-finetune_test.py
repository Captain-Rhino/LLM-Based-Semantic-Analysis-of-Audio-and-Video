import torch
from cn_clip.clip import load_from_name, tokenize
from PIL import Image

# ==== Step 1: 加载模型结构和预处理器 ====
print("🔄 正在加载模型结构...")
model, preprocess = load_from_name("ViT-B-16", download_root="./")
model.cuda().eval()

# ==== Step 2: 加载你自己训练好的 checkpoint ====
ckpt_path = r"G:\videochat\my_design\pretrained_weights\epoch_latest.pt"  # ← ← 替换为你模型的路径
print(f"📦 正在加载权重：{ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location="cpu")

# 注意：如果模型是 DDP 保存的，需要去掉前缀 "module."
state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
model.load_state_dict(state_dict)
print("✅ 模型加载完成！")

# ==== Step 3: 准备图像输入 ====
image_path = r"G:\videochat\my_design\demo.jpg"  # ← ← 替换为你要测试的图像
image = preprocess(Image.open(image_path)).unsqueeze(0).cuda()

# ==== Step 4: 准备文本输入 ====
texts = ["一只在沙发上的猫", "一辆红色跑车", "一个穿白衣服的人"]  # 你可以随便换
text_tokens = tokenize(texts).cuda()

# ==== Step 5: 模型推理 ====
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text_tokens)

    # 归一化
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # 相似度分数
    similarity = (100.0 * image_features @ text_features.T)
    probs = similarity.softmax(dim=-1)

# ==== Step 6: 打印结果 ====
print("📊 图像 vs 文本 相似度（越高越相关）：")
for i, score in enumerate(probs[0]):
    print(f'  - "{texts[i]}"：{score.item():.4f}')
