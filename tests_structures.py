"""
Py2Tensor Structure Tests: tuples, multiple returns, chained ops, ternary
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

def check(name, cpu_results, gpu_results, tol=1e-3):
    global passed, failed
    cpu_t = torch.tensor(cpu_results, dtype=torch.float32)
    gpu_t = gpu_results.cpu().float() if isinstance(gpu_results, torch.Tensor) else torch.tensor(gpu_results, dtype=torch.float32)
    match = torch.allclose(cpu_t.reshape(-1), gpu_t.reshape(-1), atol=tol, rtol=1e-3)
    if match: passed += 1
    else:
        failed += 1
        print(f"  CPU: {cpu_t.reshape(-1)[:5]}")
        print(f"  GPU: {gpu_t.reshape(-1)[:5]}")
    print(f"  [{'PASS' if match else 'FAIL'}] {name}")

# ================================================================
print("\n=== CHAINED COMPARISONS ===")

@tensorize
def classify(x):
    if x > 100:
        return 3
    else:
        if x > 50:
            return 2
        else:
            if x > 0:
                return 1
            else:
                return 0

vals = [-10, 0, 25, 50, 75, 100, 150]
cpu_out = [classify(v) for v in vals]
gpu_out = classify(torch.tensor(vals, dtype=torch.float32, device=device))
check("4-way classify", [float(v) for v in cpu_out], gpu_out)

# ================================================================
print("\n=== COMPLEX MATH CHAIN ===")

@tensorize
def softplus(x):
    return math.log(1.0 + math.exp(x))

vals = [-5, -2, -1, 0, 1, 2, 5]
cpu_out = [softplus(v) for v in vals]
gpu_out = softplus(torch.tensor(vals, dtype=torch.float32, device=device))
check("softplus log(1+exp(x))", [float(v) for v in cpu_out], gpu_out)

@tensorize
def gelu(x):
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))

vals = [-3, -1, -0.5, 0, 0.5, 1, 3]
cpu_out = [gelu(v) for v in vals]
gpu_out = gelu(torch.tensor(vals, dtype=torch.float32, device=device))
check("GELU activation", [float(v) for v in cpu_out], gpu_out)

# ================================================================
print("\n=== ITERATIVE ALGORITHMS ===")

@tensorize
def taylor_sin(x):
    """sin(x) via Taylor series, 7 terms"""
    result = x
    term = x
    for i in range(1, 7):
        term = -term * x * x / ((2 * i) * (2 * i + 1))
        result = result + term
    return result

vals = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]
cpu_out = [taylor_sin(v) for v in vals]
gpu_out = taylor_sin(torch.tensor(vals, dtype=torch.float32, device=device))
expected = [math.sin(v) for v in vals]
check("Taylor sin (7 terms)", expected, gpu_out, tol=0.01)

@tensorize
def taylor_exp(x):
    """exp(x) via Taylor series, 12 terms"""
    result = 1.0
    term = 1.0
    for i in range(1, 12):
        term = term * x / i
        result = result + term
    return result

vals = [-2, -1, 0, 0.5, 1, 2, 3]
cpu_out = [taylor_exp(v) for v in vals]
gpu_out = taylor_exp(torch.tensor(vals, dtype=torch.float32, device=device))
expected = [math.exp(v) for v in vals]
check("Taylor exp (12 terms)", expected, gpu_out, tol=0.01)

# ================================================================
print("\n=== CONTROL SYSTEMS ===")

@tensorize
def pid_step(error, integral, prev_error):
    Kp = 1.0
    Ki = 0.1
    Kd = 0.05
    derivative = error - prev_error
    new_integral = integral + error
    output = Kp * error + Ki * new_integral + Kd * derivative
    if output > 100:
        return 100
    else:
        if output < -100:
            return -100
        else:
            return output

errors = [10, 5, -3, 20, -15, 0]
integrals = [0, 10, 15, 12, 32, 17]
prev_errs = [0, 10, 5, -3, 20, -15]
cpu_out = [pid_step(e, i, p) for e, i, p in zip(errors, integrals, prev_errs)]
gpu_out = pid_step(
    torch.tensor(errors, dtype=torch.float32, device=device),
    torch.tensor(integrals, dtype=torch.float32, device=device),
    torch.tensor(prev_errs, dtype=torch.float32, device=device)
)
check("PID controller step", [float(v) for v in cpu_out], gpu_out)

# ================================================================
print("\n=== FINANCIAL: Compound with variable rates ===")

@tensorize
def variable_rate_growth(principal, year):
    """Different growth rates per period."""
    if year < 5:
        rate = 0.03
    else:
        if year < 10:
            rate = 0.05
        else:
            rate = 0.07
    return principal * (1 + rate)

principals = [1000, 2000, 5000, 10000]
years = [2, 7, 12, 3]
cpu_out = [variable_rate_growth(p, y) for p, y in zip(principals, years)]
gpu_out = variable_rate_growth(
    torch.tensor(principals, dtype=torch.float32, device=device),
    torch.tensor(years, dtype=torch.float32, device=device)
)
check("variable rate growth", [float(v) for v in cpu_out], gpu_out)

# ================================================================
print("\n=== MEGA BENCHMARK: Taylor sin on 10M ===")

N = 10_000_000
cpu_data = [float(i) / 1000000 - 5 for i in range(50000)]
t0 = time.time()
cpu_out = [taylor_sin._original(v) for v in cpu_data]
cpu_time = time.time() - t0

gpu_data = torch.linspace(-5, 5, N, device=device)
torch.cuda.synchronize()
_ = taylor_sin(gpu_data)
torch.cuda.synchronize()

t0 = time.time()
for _ in range(20):
    result = taylor_sin(gpu_data)
torch.cuda.synchronize()
gpu_time = (time.time() - t0) / 20

# Verify
check("Taylor sin batch",
      [math.sin(v) for v in cpu_data[:50]],
      taylor_sin(torch.tensor(cpu_data[:50], dtype=torch.float32, device=device)),
      tol=0.01)

cpu_rate = len(cpu_data) / cpu_time
gpu_rate = N / max(gpu_time, 1e-9)
print(f"\n  CPU: {cpu_rate/1e6:.1f}M/s")
print(f"  GPU: {gpu_rate/1e6:.0f}M/s (Taylor sin, 7 terms, 10M elements)")
print(f"  SPEEDUP: {gpu_rate/cpu_rate:.0f}x")

# GELU benchmark
print("\n=== MEGA BENCHMARK: GELU on 10M ===")
torch.cuda.synchronize()
_ = gelu(gpu_data)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(20):
    result = gelu(gpu_data)
torch.cuda.synchronize()
gpu_time = (time.time() - t0) / 20
print(f"  GPU GELU: {N/max(gpu_time,1e-9)/1e6:.0f}M/s")

# ================================================================
print(f"\n{'='*50}")
print(f"RESULTS: {passed} passed, {failed} failed")
print(f"{'='*50}")
