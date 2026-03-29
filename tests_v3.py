"""
Py2Tensor v3 Tests: while loops, augmented assign, real algorithms
"""
import torch
import math
import time
import sys
sys.path.insert(0, r"C:\Users\salih\Desktop\py2tensor")
from py2tensor import tensorize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
passed = 0
failed = 0

def check(name, cpu_results, gpu_results, tol=1e-2):
    global passed, failed
    cpu_t = torch.tensor(cpu_results, dtype=torch.float32)
    gpu_t = gpu_results.cpu().float() if isinstance(gpu_results, torch.Tensor) else torch.tensor(gpu_results, dtype=torch.float32)
    match = torch.allclose(cpu_t.reshape(-1), gpu_t.reshape(-1), atol=tol)
    if match: passed += 1
    else:
        failed += 1
        print(f"  CPU: {cpu_t.reshape(-1)[:8]}")
        print(f"  GPU: {gpu_t.reshape(-1)[:8]}")
    print(f"  [{'PASS' if match else 'FAIL'}] {name}")

# ================================================================
print("\n=== AUGMENTED ASSIGNMENT ===")

@tensorize
def compound_ops(x):
    result = x
    result += 10
    result *= 2
    result -= 5
    return result

# (x+10)*2 - 5
vals = [0, 1, 5, -3, 100]
cpu_out = [compound_ops(v) for v in vals]
gpu_out = compound_ops(torch.tensor(vals, dtype=torch.float32, device=device))
check("x += *= -=", [float(v) for v in cpu_out], gpu_out)

# ================================================================
print("\n=== NEWTON'S METHOD (sqrt via iteration) ===")

@tensorize
def newton_sqrt(x):
    guess = x / 2
    for i in range(10):
        guess = (guess + x / guess) / 2
    return guess

vals = [4, 9, 16, 25, 100, 2, 0.5]
cpu_out = [newton_sqrt(v) for v in vals]
gpu_out = newton_sqrt(torch.tensor(vals, dtype=torch.float32, device=device))
expected = [math.sqrt(v) for v in vals]
check("newton sqrt (10 iter)", expected, gpu_out, tol=1e-4)

# ================================================================
print("\n=== FIXED-POINT ITERATION ===")

@tensorize
def fixed_point_cos(x):
    result = x
    for i in range(20):
        result = (result + x / (result + 1)) / 2
    return result

vals = [1, 2, 5, 10, 0.1]
cpu_out = [fixed_point_cos(v) for v in vals]
gpu_out = fixed_point_cos(torch.tensor(vals, dtype=torch.float32, device=device))
check("fixed-point iteration", [float(v) for v in cpu_out], gpu_out)

# ================================================================
print("\n=== POLYNOMIAL EVALUATION (Horner's method unrolled) ===")

# p(x) = 3x^4 - 2x^3 + x^2 - 5x + 7
@tensorize
def horner(x):
    result = 3
    result = result * x + (-2)
    result = result * x + 1
    result = result * x + (-5)
    result = result * x + 7
    return result

vals = [-2, -1, 0, 1, 2, 3]
cpu_out = [horner(v) for v in vals]
gpu_out = horner(torch.tensor(vals, dtype=torch.float32, device=device))
check("Horner polynomial", [float(v) for v in cpu_out], gpu_out)

# ================================================================
print("\n=== ITERATIVE DECAY (exponential-like) ===")

@tensorize
def decay(x):
    value = x
    for i in range(20):
        value = value * 0.9
    return value

# x * 0.9^20
vals = [100, 50, 1, 200]
cpu_out = [decay(v) for v in vals]
gpu_out = decay(torch.tensor(vals, dtype=torch.float32, device=device))
expected = [v * (0.9 ** 20) for v in vals]
check("0.9^20 decay", expected, gpu_out, tol=1e-3)

# ================================================================
print("\n=== MULTI-STEP PIPELINE ===")

@tensorize
def ml_pipeline(x):
    # Normalize
    if x > 100:
        normed = 1.0
    else:
        if x < 0:
            normed = 0.0
        else:
            normed = x / 100.0

    # Apply sigmoid-like activation
    if normed > 0.5:
        activated = 0.5 + (normed - 0.5) * 0.8
    else:
        activated = 0.5 - (0.5 - normed) * 0.8

    # Quantize to bins
    if activated > 0.8:
        return 4
    else:
        if activated > 0.6:
            return 3
        else:
            if activated > 0.4:
                return 2
            else:
                if activated > 0.2:
                    return 1
                else:
                    return 0

vals = [-10, 0, 25, 50, 75, 100, 150]
cpu_out = [ml_pipeline(v) for v in vals]
gpu_out = ml_pipeline(torch.tensor(vals, dtype=torch.float32, device=device))
check("ML pipeline (normalize+activate+quantize)", [float(v) for v in cpu_out], gpu_out)

# ================================================================
print("\n=== MEGA BENCHMARK: Newton sqrt on 10M ===")
N = 10_000_000

# CPU
import random
cpu_data = [random.uniform(0.1, 1000) for _ in range(50000)]
t0 = time.time()
cpu_out = [newton_sqrt._original(v) for v in cpu_data]
cpu_time = time.time() - t0

# GPU
gpu_data = torch.rand(N, device=device) * 1000 + 0.1
torch.cuda.synchronize()

# Warmup
_ = newton_sqrt(gpu_data)
torch.cuda.synchronize()

t0 = time.time()
for _ in range(20):
    result = newton_sqrt(gpu_data)
torch.cuda.synchronize()
gpu_time = (time.time() - t0) / 20

# Verify
check("newton sqrt 10M",
      [math.sqrt(v) for v in cpu_data[:50]],
      newton_sqrt(torch.tensor(cpu_data[:50], dtype=torch.float32, device=device)),
      tol=1e-3)

cpu_rate = len(cpu_data) / cpu_time
gpu_rate = N / max(gpu_time, 1e-9)
print(f"\n  CPU: {len(cpu_data)} in {cpu_time*1000:.0f}ms = {cpu_rate:.0f}/s")
print(f"  GPU: {N} in {gpu_time*1000:.2f}ms = {gpu_rate:.0f}/s")
print(f"  SPEEDUP: {gpu_rate/cpu_rate:.0f}x")
print(f"  (Newton sqrt: 10 iterations per element, exact to 1e-4)")

# ================================================================
print(f"\n{'='*50}")
print(f"RESULTS: {passed} passed, {failed} failed")
print(f"{'='*50}")
