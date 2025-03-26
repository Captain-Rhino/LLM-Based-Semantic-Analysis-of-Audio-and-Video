import torch
print(torch.__version__)  # 输出 PyTorch 版本号
print(torch.cuda.is_available())  # True 表示 GPU 版可用
print(torch.version.cuda)  # 检查 CUDA 版本（应该是 11.7）
print(torch.cuda.get_device_name(0))  # 输出你的 GPU 名称