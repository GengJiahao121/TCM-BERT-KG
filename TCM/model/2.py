import torch

# 假设你有一个形状为 (1274, 1024) 的 CUDA 张量
cuda_tensor = torch.randn(1274, 13, 1024).cuda()

# 计算显存大小（以字节为单位）
memory_size_bytes = cuda_tensor.element_size() * cuda_tensor.nelement()

# 转换为更大单位
memory_size_kb = memory_size_bytes / 1024
memory_size_mb = memory_size_kb / 1024
memory_size_gb = memory_size_mb / 1024

print(f"显存大小: {memory_size_bytes} 字节")
print(f"显存大小: {memory_size_kb:.2f} KB")
print(f"显存大小: {memory_size_mb:.2f} MB")
print(f"显存大小: {memory_size_gb:.2f} GB")
