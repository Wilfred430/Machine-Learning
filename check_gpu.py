import torch

if torch.cuda.is_available():
    print("✅ CUDA is available!")
    print("使用的 GPU 名稱:", torch.cuda.get_device_name(0))
    print("CUDA 版本:", torch.version.cuda)
else:
    print("❌ CUDA 不可用")
