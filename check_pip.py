import torch
print(torch.__version__)  # 输出 PyTorch 版本号
print(torch.cuda.is_available())  # True 表示 GPU 版可用
print(torch.version.cuda)  # 检查 CUDA 版本（应该是 11.7）


import torch
import torchvision

# 检查 CUDA 是否可用
print("CUDA 是否可用：", torch.cuda.is_available())  # True 表示启用了 GPU

# 获取当前使用的 GPU
print("当前 GPU：", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无 GPU")