"""
Py2Tensor Advanced Tests: loops, lookups, real-world functions
"""
import torch
import numpy as np
import time
import sys
sys.path.insert(0, r"C:\Users\salih\Desktop\py2tensor")
from py2tensor import tensorize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
passed = 0
failed = 0

def check(name, cpu_results, gpu_results, tol=1e-3):
    global passed, failed
    cpu_t = torch.tensor(cpu_results, dtype=torch.float32)
    gpu_t = gpu_results.cpu().float() if isinstance(gpu_results, torch.Tensor) else torch.tensor(gpu_results, dtype=torch.float32)
    if cpu_t.shape != gpu_t.shape:
        cpu_t = cpu_t.reshape(gpu_t.shape)
    match = torch.allclose(cpu_t, gpu_t, atol=tol)
    if match: passed += 1
    else:
        failed += 1
        print(f"  CPU: {cpu_t[:5]}")
        print(f"  GPU: {gpu_t[:5]}")
    print(f"  [{'PASS' if match else 'FAIL'}] {name}")

# ================================================================
print("\n--- Test: Piecewise function (3 regions) ---")

@tensorize
def piecewise(x):
    if x < -1:
        result = -1
    else:
        result = x
    if result > 1:
        result = 1
    else:
        result = result
    return result

vals = [-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5]
cpu_out = [piecewise(v) for v in vals]
gpu_out = piecewise(torch.tensor(vals, dtype=torch.float32, device=device))
check("piecewise clamp(-1,1)", [float(v) for v in cpu_out], gpu_out)

# ================================================================
print("\n--- Test: Sigmoid approximation ---")

@tensorize
def sigmoid_approx(x):
    if x > 6:
        return 1.0
    else:
        if x < -6:
            return 0.0
        else:
            return 0.5 + x * 0.125

vals2 = [-10, -6, -3, 0, 3, 6, 10]
cpu_out = [sigmoid_approx(v) for v in vals2]
gpu_out = sigmoid_approx(torch.tensor(vals2, dtype=torch.float32, device=device))
check("sigmoid_approx", [float(v) for v in cpu_out], gpu_out)

# ================================================================
print("\n--- Test: Multi-input function ---")

@tensorize
def weighted_sum(x, y):
    if x > y:
        return x * 0.7 + y * 0.3
    else:
        return x * 0.3 + y * 0.7

pairs = [(1, 2), (5, 3), (0, 0), (-1, 1), (10, -5)]
cpu_out = [weighted_sum(a, b) for a, b in pairs]
x_gpu = torch.tensor([p[0] for p in pairs], dtype=torch.float32, device=device)
y_gpu = torch.tensor([p[1] for p in pairs], dtype=torch.float32, device=device)
gpu_out = weighted_sum(x_gpu, y_gpu)
check("weighted_sum(x,y)", [float(v) for v in cpu_out], gpu_out)

# ================================================================
print("\n--- Test: Tax calculator ---")

@tensorize
def tax(income):
    if income < 10000:
        return income * 0.0
    else:
        if income < 50000:
            return (income - 10000) * 0.15
        else:
            return (income - 50000) * 0.25 + 40000 * 0.15

incomes = [5000, 10000, 30000, 50000, 100000, 200000]
cpu_out = [tax(v) for v in incomes]
gpu_out = tax(torch.tensor(incomes, dtype=torch.float32, device=device))
check("tax calculator", [float(v) for v in cpu_out], gpu_out)

# ================================================================
print("\n--- Test: Distance function ---")

@tensorize
def manhattan_dist(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

points = [(0,0,3,4), (1,1,1,1), (-5,3,2,-1), (10,0,0,10)]
cpu_out = [manhattan_dist(*p) for p in points]
gpu_out = manhattan_dist(
    torch.tensor([p[0] for p in points], dtype=torch.float32, device=device),
    torch.tensor([p[1] for p in points], dtype=torch.float32, device=device),
    torch.tensor([p[2] for p in points], dtype=torch.float32, device=device),
    torch.tensor([p[3] for p in points], dtype=torch.float32, device=device),
)
check("manhattan distance", [float(v) for v in cpu_out], gpu_out)

# ================================================================
print("\n--- REAL BENCHMARK: 10M elements ---")

@tensorize
def scoring(x):
    if x > 100:
        return 100 + (x - 100) * 0.5
    else:
        if x > 0:
            return x
        else:
            return 0

N = 10_000_000

# CPU (scalar loop)
cpu_data = np.random.randn(50000).astype(np.float32) * 200
t0 = time.time()
cpu_results = [scoring._original(float(v)) for v in cpu_data]
cpu_time = time.time() - t0

# GPU (batch)
gpu_data = torch.randn(N, device=device) * 200
torch.cuda.synchronize()
t0 = time.time()
for _ in range(10):  # 10 rounds for stable measurement
    gpu_results = scoring(gpu_data)
torch.cuda.synchronize()
gpu_time = (time.time() - t0) / 10

# Verify on overlap
check("scoring batch", cpu_results[:100],
      scoring(torch.tensor(cpu_data[:100], dtype=torch.float32, device=device)))

print(f"\n  CPU scalar: {len(cpu_data)} in {cpu_time*1000:.0f}ms = {len(cpu_data)/cpu_time:.0f} elem/s")
print(f"  GPU batch:  {N} in {gpu_time*1000:.2f}ms = {N/max(gpu_time,1e-9):.0f} elem/s")
print(f"  SPEEDUP: {(N/max(gpu_time,1e-9))/(len(cpu_data)/cpu_time):.0f}x (normalized)")

# ================================================================
print(f"\n{'='*50}")
print(f"RESULTS: {passed} passed, {failed} failed")
print(f"{'='*50}")
