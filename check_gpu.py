import torch

print("torch version :", torch.__version__)
print("CUDA runtime  :", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("arch list     :", torch.cuda.get_arch_list())

if torch.cuda.is_available():
    print("GPU           :", torch.cuda.get_device_name(0))
    print("capability    :", torch.cuda.get_device_capability(0))
    # The real test: actually run a kernel on the GPU.
    a = torch.randn(1000, 1000, device="cuda")
    b = torch.randn(1000, 1000, device="cuda")
    c = a @ b
    torch.cuda.synchronize()
    print("matmul OK, result shape:", tuple(c.shape))