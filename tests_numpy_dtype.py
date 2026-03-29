"""
Py2Tensor: numpy integration + dtype + error recovery tests
"""
import torch
import numpy as np
import math
import time
import sys
sys.path.insert(0, r"C:\Users\salih\Desktop\py2tensor")
from py2tensor import tensorize, explain, benchmark

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
passed = 0
failed = 0

def check(name, expected, got, tol=1e-2):
    global passed, failed
    if isinstance(expected, (list, tuple)):
        expected = torch.tensor(expected, dtype=torch.float32)
    if isinstance(got, torch.Tensor):
        got = got.cpu().float()
    elif isinstance(got, np.ndarray):
        got = torch.tensor(got, dtype=torch.float32)
    match = torch.allclose(expected.reshape(-1), got.reshape(-1), atol=tol, rtol=1e-2)
    if match: passed += 1
    else:
        failed += 1
        print(f"  Expected: {expected.reshape(-1)[:5]}")
        print(f"  Got:      {got.reshape(-1)[:5]}")
    print(f"  [{'PASS' if match else 'FAIL'}] {name}")

# ================================================================
print("\n=== NUMPY INPUT ===")

@tensorize
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

# Numpy array input -> should auto-convert to tensor
np_data = np.array([-5, -2, 0, 2, 5], dtype=np.float32)
result = sigmoid(np_data)
expected = [sigmoid._original(float(v)) for v in np_data]
check("numpy input -> GPU", expected, result)
print(f"  Output type: {type(result)}")

# ================================================================
print("\n=== FLOAT16 DTYPE ===")

@tensorize(dtype=torch.float16)
def fast_relu(x):
    if x > 0:
        return x
    else:
        return 0

x_gpu = torch.randn(1000, device=device)
result16 = fast_relu(x_gpu)
print(f"  Output dtype: {result16.dtype}")
check("float16 relu", [max(0, float(v)) for v in x_gpu[:10].cpu()], result16[:10], tol=0.1)

# Float16 speed comparison
N = 10_000_000
x32 = torch.randn(N, device=device, dtype=torch.float32)
x16 = x32.half()

@tensorize
def complex_fn(x):
    if x > 0:
        return x * x + 2 * x + 1
    else:
        return -x * 0.5

@tensorize(dtype=torch.float16)
def complex_fn16(x):
    if x > 0:
        return x * x + 2 * x + 1
    else:
        return -x * 0.5

# Warmup
complex_fn(x32)
complex_fn16(x16)
torch.cuda.synchronize()

t0 = time.time()
for _ in range(50):
    _ = complex_fn(x32)
torch.cuda.synchronize()
t32 = (time.time() - t0) / 50

t0 = time.time()
for _ in range(50):
    _ = complex_fn16(x16)
torch.cuda.synchronize()
t16 = (time.time() - t0) / 50

print(f"\n  Float32: {t32*1000:.2f}ms ({N/t32/1e6:.0f}M/s)")
print(f"  Float16: {t16*1000:.2f}ms ({N/t16/1e6:.0f}M/s)")
print(f"  Float16 speedup: {t32/t16:.2f}x")

# ================================================================
print("\n=== ERROR RECOVERY (fallback to CPU) ===")

# This function has unsupported features - should fallback gracefully
@tensorize(fallback=True)
def normal_func(x):
    return x * 2 + 1

# Should work normally
result = normal_func(5)
print(f"  Scalar result: {result}")
check("fallback scalar", [11.0], torch.tensor([float(result)]))

result = normal_func(torch.tensor([1, 2, 3], dtype=torch.float32, device=device))
check("fallback tensor", [3.0, 5.0, 7.0], result)

# ================================================================
print("\n=== BENCHMARK API ===")

@tensorize
def quick_math(x):
    return math.sin(x) * math.exp(-x * x * 0.5)

benchmark(quick_math, 1.0, n=5_000_000)

# ================================================================
print("\n=== EXPLAIN API ===")

@tensorize
def pricing(demand, supply):
    ratio = demand / (supply + 0.001)
    if ratio > 2.0:
        return ratio * 1.5
    else:
        if ratio > 1.0:
            return ratio
        else:
            return ratio * 0.8

explain(pricing)

# ================================================================
print(f"\n{'='*50}")
print(f"RESULTS: {passed} passed, {failed} failed")
print(f"{'='*50}")
