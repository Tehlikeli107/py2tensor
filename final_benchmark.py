"""
FINAL BENCHMARK: 5-way comparison
==================================
For each function, compare:
1. Python scalar loop
2. NumPy vectorized
3. @tensorize (GPU)
4. @tensorize(compile=True) (GPU + kernel fusion)
5. Hand-written PyTorch (gold standard)
"""
import torch
import numpy as np
import math
import time
import sys
sys.path.insert(0, r"C:\Users\salih\Desktop\py2tensor")
from py2tensor import tensorize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"GPU: {torch.cuda.get_device_name()}")
print(f"PyTorch: {torch.__version__}")

N = 10_000_000
WARMUP = 3
ROUNDS = 30

def bench_python(name, fn, data):
    t0 = time.time()
    out = [fn(v) for v in data]
    return (time.time() - t0), out

def bench_numpy(name, fn, data):
    arr = np.array(data, dtype=np.float32)
    t0 = time.time()
    out = fn(arr)
    return (time.time() - t0), out

def bench_gpu(name, fn, data_gpu):
    torch.cuda.synchronize()
    for _ in range(WARMUP): fn(data_gpu)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(ROUNDS): out = fn(data_gpu)
    torch.cuda.synchronize()
    return (time.time() - t0) / ROUNDS, out

results = []

def run_comparison(title, py_fn, np_fn, tensor_fn, tensor_compile_fn, hand_fn):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

    # Data
    py_data = [float(i)/N*20 - 10 for i in range(min(N, 50000))]
    gpu_data = torch.linspace(-10, 10, N, device=device)
    np_data = np.linspace(-10, 10, min(N, 50000), dtype=np.float32)

    # Python
    t_py, _ = bench_python("Python", py_fn, py_data)
    r_py = len(py_data) / t_py

    # NumPy (run multiple times for stable measurement)
    np_big = np.linspace(-10, 10, N, dtype=np.float32)
    t0 = time.time()
    for _ in range(10):
        _ = np_fn(np_big)
    t_np = (time.time() - t0) / 10
    r_np = N / max(t_np, 1e-12)

    # @tensorize
    t_t, out_t = bench_gpu("@tensorize", tensor_fn, gpu_data)
    r_t = N / t_t

    # @tensorize(compile=True)
    t_tc, out_tc = bench_gpu("@tensorize+compile", tensor_compile_fn, gpu_data)
    r_tc = N / t_tc

    # Hand PyTorch
    t_h, out_h = bench_gpu("Hand PyTorch", hand_fn, gpu_data)
    r_h = N / t_h

    # Verify
    match = torch.allclose(out_t, out_h, atol=1e-3, rtol=1e-3)

    print(f"  {'Method':<25} {'Rate':>12} {'vs Python':>10} {'vs Hand':>10}")
    print(f"  {'-'*57}")
    print(f"  {'Python scalar':<25} {r_py/1e6:>9.1f}M/s {'1x':>10} {r_py/r_h:>9.3f}x")
    print(f"  {'NumPy vectorized':<25} {r_np/1e6:>9.1f}M/s {r_np/r_py:>9.0f}x {r_np/r_h:>9.3f}x")
    print(f"  {'@tensorize':<25} {r_t/1e6:>9.0f}M/s {r_t/r_py:>9.0f}x {r_t/r_h:>9.3f}x")
    print(f"  {'@tensorize(compile)':<25} {r_tc/1e6:>9.0f}M/s {r_tc/r_py:>9.0f}x {r_tc/r_h:>9.3f}x")
    print(f"  {'Hand-written PyTorch':<25} {r_h/1e6:>9.0f}M/s {r_h/r_py:>9.0f}x {'1.000x':>10}")
    print(f"  Match: {match}")

    results.append({
        'name': title,
        'python': r_py, 'numpy': r_np,
        'tensorize': r_t, 'compile': r_tc, 'hand': r_h,
    })

# ================================================================
# BENCHMARK 1: Simple arithmetic
# ================================================================
@tensorize
def arith_t(x):
    return x * x + 2 * x + 1

@tensorize(compile=True)
def arith_tc(x):
    return x * x + 2 * x + 1

def arith_py(x): return x*x + 2*x + 1
def arith_np(x): return x*x + 2*x + 1
def arith_hand(x): return x*x + 2*x + 1

run_comparison("Polynomial x^2+2x+1",
               arith_py, arith_np, arith_t, arith_tc, arith_hand)

# ================================================================
# BENCHMARK 2: Gaussian PDF (exp + sqrt + pi)
# ================================================================
@tensorize
def gauss_t(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

@tensorize(compile=True)
def gauss_tc(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

def gauss_py(x): return math.exp(-0.5*x*x) / math.sqrt(2*math.pi)
def gauss_np(x): return np.exp(-0.5*x*x) / np.sqrt(2*np.pi)
def gauss_hand(x): return torch.exp(-0.5*x*x) / math.sqrt(2*math.pi)

run_comparison("Gaussian PDF exp(-x^2/2)/sqrt(2pi)",
               gauss_py, gauss_np, gauss_t, gauss_tc, gauss_hand)

# ================================================================
# BENCHMARK 3: Branching (if/else)
# ================================================================
@tensorize
def branch_t(x):
    if x > 0:
        return x * x
    else:
        return -x * 0.5

@tensorize(compile=True)
def branch_tc(x):
    if x > 0:
        return x * x
    else:
        return -x * 0.5

def branch_py(x): return x*x if x > 0 else -x*0.5
def branch_np(x): return np.where(x > 0, x*x, -x*0.5)
def branch_hand(x): return torch.where(x > 0, x*x, -x*0.5)

run_comparison("Branching if/else",
               branch_py, branch_np, branch_t, branch_tc, branch_hand)

# ================================================================
# BENCHMARK 4: Sigmoid
# ================================================================
@tensorize
def sig_t(x):
    return 1.0 / (1.0 + math.exp(-x))

@tensorize(compile=True)
def sig_tc(x):
    return 1.0 / (1.0 + math.exp(-x))

def sig_py(x): return 1.0 / (1.0 + math.exp(-x))
def sig_np(x): return 1.0 / (1.0 + np.exp(-x))
def sig_hand(x): return 1.0 / (1.0 + torch.exp(-x))

run_comparison("Sigmoid 1/(1+exp(-x))",
               sig_py, sig_np, sig_t, sig_tc, sig_hand)

# ================================================================
# BENCHMARK 5: Newton sqrt (10 iterations)
# ================================================================
@tensorize
def newton_t(x):
    g = x / 2
    for i in range(10):
        g = (g + x / g) / 2
    return g

@tensorize(compile=True)
def newton_tc(x):
    g = x / 2
    for i in range(10):
        g = (g + x / g) / 2
    return g

def newton_py(x):
    g = x / 2
    for _ in range(10): g = (g + x / g) / 2
    return g

def newton_np(x):
    g = x / 2
    for _ in range(10): g = (g + x / g) / 2
    return g

def newton_hand(x):
    g = x / 2
    for _ in range(10): g = (g + x / g) / 2
    return g

run_comparison("Newton sqrt (10 iterations)",
               newton_py, newton_np, newton_t, newton_tc, newton_hand)

# ================================================================
# GRAND SUMMARY
# ================================================================
print(f"\n{'='*70}")
print("GRAND SUMMARY: @tensorize vs alternatives (10M elements)")
print(f"{'='*70}")

print(f"\n{'Function':<30} {'Python':>8} {'NumPy':>8} {'@tensor':>8} {'+compile':>8} {'Hand PT':>8}")
print(f"{'-'*70}")
for r in results:
    def fmt(v):
        if v > 1e9: return f"{v/1e9:.1f}B"
        if v > 1e6: return f"{v/1e6:.0f}M"
        return f"{v/1e3:.0f}K"
    print(f"{r['name']:<30} {fmt(r['python']):>8} {fmt(r['numpy']):>8} {fmt(r['tensorize']):>8} {fmt(r['compile']):>8} {fmt(r['hand']):>8}")

print(f"\n@tensorize vs Hand-written PyTorch ratio:")
for r in results:
    ratio = r['tensorize'] / r['hand']
    print(f"  {r['name']:<30} {ratio:.3f}x {'(SAME)' if 0.9 < ratio < 1.1 else ''}")

avg_ratio = sum(r['tensorize']/r['hand'] for r in results) / len(results)
print(f"\n  AVERAGE: {avg_ratio:.3f}x (1.000 = identical to hand-written)")
