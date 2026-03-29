"""Test the package structure works."""
import sys
sys.path.insert(0, r"C:\Users\salih\Desktop\py2tensor")

print("Testing package imports...")

from py2tensor import gpu, tensorize, explain, benchmark
print(f"  gpu: {type(gpu)}")
print(f"  tensorize: {tensorize}")

from py2tensor import tensorize_triton, build_pure_model, tensorize_all
print(f"  triton: {tensorize_triton}")
print(f"  pure: {build_pure_model}")
print(f"  all: {tensorize_all}")

import torch, math
device = torch.device("cuda")

@gpu
def test1(x):
    if x > 0: return x * 2
    else: return 0

print(f"\n@gpu test: {test1(torch.tensor([1,-1,3,-2], dtype=torch.float32, device=device)).tolist()}")

@gpu.fast
def test2(x):
    return math.sin(x)

print(f"@gpu.fast test: {test2(torch.tensor([0, 3.14/2], dtype=torch.float32, device=device)).tolist()}")

@gpu.triton
def test3(x):
    g = x / 2
    for i in range(5):
        g = (g + x / g) / 2
    return g

print(f"@gpu.triton test: sqrt(4) = {test3(torch.tensor([4.0], device=device)).item():.4f}")

print(f"\nPackage structure: OK")
print(f"Version: {__import__('py2tensor').__version__}")
