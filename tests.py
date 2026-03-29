"""
Py2Tensor Tests: verify CPU==GPU for converted functions
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

def check(name, cpu_results, gpu_results, tol=1e-4):
    global passed, failed
    cpu_t = torch.tensor(cpu_results, dtype=torch.float32)
    if not isinstance(gpu_results, torch.Tensor):
        gpu_results = torch.tensor(gpu_results, dtype=torch.float32)
    gpu_t = gpu_results.cpu().float()
    match = torch.allclose(cpu_t, gpu_t, atol=tol)
    status = "PASS" if match else "FAIL"
    if match:
        passed += 1
    else:
        failed += 1
        print(f"  CPU: {cpu_t[:5]}")
        print(f"  GPU: {gpu_t[:5]}")
    print(f"  [{status}] {name}")
    return match

# ================================================================
# Test 1: Simple arithmetic
# ================================================================
print("\n--- Test 1: Simple arithmetic ---")

@tensorize
def double_plus_one(x):
    return x * 2 + 1

# CPU
test_vals = [-5, -1, 0, 1, 5, 10, 100]
cpu_out = [double_plus_one(v) for v in test_vals]

# GPU
gpu_in = torch.tensor(test_vals, dtype=torch.float32, device=device)
gpu_out = double_plus_one(gpu_in)

check("x*2+1", cpu_out, gpu_out)
print(f"  Transformed code: {double_plus_one._tensor_source[:80]}")

# ================================================================
# Test 2: If/else -> torch.where
# ================================================================
print("\n--- Test 2: If/else -> torch.where ---")

@tensorize
def relu_custom(x):
    if x > 0:
        return x
    else:
        return 0

cpu_out = [relu_custom(v) for v in test_vals]
gpu_in = torch.tensor(test_vals, dtype=torch.float32, device=device)
gpu_out = relu_custom(gpu_in)
check("custom relu", [float(v) for v in cpu_out], gpu_out)

# ================================================================
# Test 3: If/else with computation
# ================================================================
print("\n--- Test 3: If/else with computation ---")

@tensorize
def abs_double(x):
    if x > 0:
        return x * 2
    else:
        return -x * 2

cpu_out = [abs_double(v) for v in test_vals]
gpu_out = abs_double(torch.tensor(test_vals, dtype=torch.float32, device=device))
check("abs_double", [float(v) for v in cpu_out], gpu_out)

# ================================================================
# Test 4: Multiple operations
# ================================================================
print("\n--- Test 4: Polynomial ---")

@tensorize
def polynomial(x):
    return x * x * x - 2 * x * x + 3 * x - 7

cpu_out = [polynomial(v) for v in test_vals]
gpu_out = polynomial(torch.tensor(test_vals, dtype=torch.float32, device=device))
check("x^3-2x^2+3x-7", [float(v) for v in cpu_out], gpu_out)

# ================================================================
# Test 5: Nested if/else via assignment
# ================================================================
print("\n--- Test 5: Clamp function ---")

@tensorize
def clamp(x):
    if x > 10:
        result = 10
    else:
        result = x
    if result < -10:
        result = -10
    else:
        result = result
    return result

test_vals2 = [-20, -10, -5, 0, 5, 10, 20]
cpu_out = [clamp(v) for v in test_vals2]
gpu_out = clamp(torch.tensor(test_vals2, dtype=torch.float32, device=device))
check("clamp(-10,10)", [float(v) for v in cpu_out], gpu_out)

# ================================================================
# Test 6: Built-in function replacement
# ================================================================
print("\n--- Test 6: abs() ---")

@tensorize
def abs_func(x):
    return abs(x) + 1

cpu_out = [abs_func(v) for v in test_vals]
gpu_out = abs_func(torch.tensor(test_vals, dtype=torch.float32, device=device))
check("abs(x)+1", [float(v) for v in cpu_out], gpu_out)

# ================================================================
# Benchmark: Speed comparison
# ================================================================
print("\n--- BENCHMARK ---")
N = 1_000_000

@tensorize
def complex_func(x):
    if x > 0:
        return x * x + 2 * x + 1
    else:
        return -x * x + 3

# CPU scalar
test_data = [float(i - N//2) / 1000 for i in range(min(N, 50000))]
t0 = time.time()
cpu_results = [complex_func._original(v) for v in test_data]
cpu_time = time.time() - t0

# GPU batched
gpu_data = torch.tensor(test_data, dtype=torch.float32, device=device)
torch.cuda.synchronize()
t0 = time.time()
gpu_results = complex_func(gpu_data)
torch.cuda.synchronize()
gpu_time = time.time() - t0

# Verify correctness
check("complex_func batch", cpu_results[:100], gpu_results[:100])

print(f"\n  CPU: {len(test_data)} calls in {cpu_time*1000:.0f}ms ({len(test_data)/cpu_time:.0f} calls/s)")
gpu_rate = len(test_data)/max(gpu_time, 1e-9)
speedup = cpu_time/max(gpu_time, 1e-9)
print(f"  GPU: {len(test_data)} batch in {gpu_time*1000:.2f}ms ({gpu_rate:.0f} calls/s)")
print(f"  SPEEDUP: {speedup:.0f}x")

# Large batch
gpu_large = torch.randn(N, device=device)
torch.cuda.synchronize()
t0 = time.time()
_ = complex_func(gpu_large)
torch.cuda.synchronize()
large_time = time.time() - t0
print(f"\n  GPU 1M batch: {large_time*1000:.2f}ms ({N/max(large_time,1e-9):.0f} calls/s)")

# ================================================================
print(f"\n{'='*50}")
print(f"RESULTS: {passed} passed, {failed} failed")
print(f"{'='*50}")
