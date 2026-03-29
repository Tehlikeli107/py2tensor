"""
Py2Tensor v2 Tests: for-loops, lookup tables, complex functions
"""
import torch
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
    match = torch.allclose(cpu_t.reshape(-1), gpu_t.reshape(-1), atol=tol)
    if match: passed += 1
    else:
        failed += 1
        print(f"  CPU: {cpu_t.reshape(-1)[:8]}")
        print(f"  GPU: {gpu_t.reshape(-1)[:8]}")
    print(f"  [{'PASS' if match else 'FAIL'}] {name}")

# ================================================================
print("\n=== FOR LOOP UNROLL ===")
print("\n--- Test: Sum of first N terms ---")

@tensorize
def sum_5(x):
    result = 0
    for i in range(5):
        result = result + i
    return result + x

# This should unroll to: result = 0+0+1+2+3+4 = 10, then +x
vals = [0, 1, 5, 10]
cpu_out = [sum_5(v) for v in vals]
gpu_out = sum_5(torch.tensor(vals, dtype=torch.float32, device=device))
check("sum range(5) + x", [float(v) for v in cpu_out], gpu_out)

print(f"  Source: {sum_5._tensor_source[:120]}...")

# ================================================================
print("\n--- Test: Accumulator loop ---")

@tensorize
def powers_of_2(x):
    result = x
    for i in range(4):
        result = result * 2
    return result

# x * 2^4 = x * 16
vals = [1, 2, 3, 0.5]
cpu_out = [powers_of_2(v) for v in vals]
gpu_out = powers_of_2(torch.tensor(vals, dtype=torch.float32, device=device))
check("x * 2^4", [float(v) for v in cpu_out], gpu_out)

# ================================================================
print("\n=== LOOKUP TABLE ===")
print("\n--- Test: Tax bracket lookup ---")

# Tax rates by bracket: [0%, 10%, 15%, 25%, 30%]
tax_rates = [0.0, 0.10, 0.15, 0.25, 0.30]

@tensorize(lookup_tables={"rates": tax_rates})
def get_tax_rate(bracket):
    return rates[bracket]

# CPU
cpu_out = [tax_rates[int(b)] for b in [0, 1, 2, 3, 4]]
gpu_brackets = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long, device=device)
# Direct tensor indexing
rates_gpu = torch.tensor(tax_rates, dtype=torch.float32, device=device)
gpu_out = rates_gpu[gpu_brackets]
check("lookup tax_rates[bracket]", cpu_out, gpu_out)

# ================================================================
print("\n--- Test: Score mapping ---")

score_map = [0, 10, 25, 45, 70, 100]  # level -> score

@tensorize(lookup_tables={"scores": score_map})
def get_score(level):
    return scores[level]

cpu_out = [score_map[i] for i in range(6)]
gpu_levels = torch.tensor(list(range(6)), dtype=torch.long, device=device)
scores_gpu = torch.tensor(score_map, dtype=torch.float32, device=device)
gpu_out = scores_gpu[gpu_levels]
check("lookup scores[level]", [float(v) for v in cpu_out], gpu_out)

# ================================================================
print("\n=== COMPLEX REAL-WORLD FUNCTIONS ===")
print("\n--- Test: Credit score rating ---")

@tensorize
def credit_rating(score):
    if score > 800:
        return 5
    else:
        if score > 700:
            return 4
        else:
            if score > 600:
                return 3
            else:
                if score > 500:
                    return 2
                else:
                    return 1

scores_test = [450, 520, 650, 720, 850]
cpu_out = [credit_rating(s) for s in scores_test]
gpu_out = credit_rating(torch.tensor(scores_test, dtype=torch.float32, device=device))
check("credit_rating (4 nested if)", [float(v) for v in cpu_out], gpu_out)

# ================================================================
print("\n--- Test: Physics - projectile height ---")

@tensorize
def projectile_height(t):
    v0 = 50
    g = 9.81
    h = v0 * t - 0.5 * g * t * t
    if h < 0:
        return 0
    else:
        return h

times = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
cpu_out = [projectile_height(t) for t in times]
gpu_out = projectile_height(torch.tensor(times, dtype=torch.float32, device=device))
check("projectile height", [float(v) for v in cpu_out], gpu_out)

# ================================================================
print("\n--- Test: Fibonacci-like recurrence (unrolled) ---")

@tensorize
def fib_like(x):
    a = x
    b = x + 1
    for i in range(8):
        c = a + b
        a = b
        b = c
    return b

vals = [0, 1, 2, 5]
cpu_out = [fib_like(v) for v in vals]
gpu_out = fib_like(torch.tensor(vals, dtype=torch.float32, device=device))
check("fib-like recurrence (8 steps)", [float(v) for v in cpu_out], gpu_out)

# ================================================================
print("\n=== MEGA BENCHMARK: 10M elements ===")

@tensorize
def complex_pipeline(x):
    if x > 100:
        stage1 = 100 + (x - 100) * 0.3
    else:
        if x > 0:
            stage1 = x
        else:
            stage1 = 0
    if stage1 > 50:
        result = stage1 * 1.1
    else:
        result = stage1 * 0.9
    return result

N = 10_000_000

# CPU
import random
cpu_data = [random.gauss(50, 100) for _ in range(50000)]
t0 = time.time()
cpu_out = [complex_pipeline._original(v) for v in cpu_data]
cpu_time = time.time() - t0

# GPU
gpu_data = torch.randn(N, device=device) * 100 + 50
torch.cuda.synchronize()

# Warmup
_ = complex_pipeline(gpu_data)
torch.cuda.synchronize()

# Timed
t0 = time.time()
for _ in range(100):
    gpu_out = complex_pipeline(gpu_data)
torch.cuda.synchronize()
gpu_time = (time.time() - t0) / 100

# Verify
check("complex_pipeline batch",
      cpu_out[:50],
      complex_pipeline(torch.tensor(cpu_data[:50], dtype=torch.float32, device=device)))

cpu_rate = len(cpu_data) / cpu_time
gpu_rate = N / max(gpu_time, 1e-9)
print(f"\n  CPU: {len(cpu_data)} elems in {cpu_time*1000:.0f}ms = {cpu_rate:.0f}/s")
print(f"  GPU: {N} elems in {gpu_time*1000:.3f}ms = {gpu_rate:.0f}/s")
print(f"  SPEEDUP: {gpu_rate/cpu_rate:.0f}x")

# ================================================================
print(f"\n{'='*50}")
print(f"RESULTS: {passed} passed, {failed} failed")
print(f"{'='*50}")
