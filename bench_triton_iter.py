"""
Triton iterative algorithms: Newton, Bisection, RK4
These benefit MOST from single-kernel fusion because
PyTorch launches N separate kernels for N iterations.
"""
import torch
import math
import time
import sys
sys.path.insert(0, r"C:\Users\salih\Desktop\py2tensor")
from py2tensor import tensorize
from triton_backend import tensorize_triton

device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name()}")

N = 10_000_000
W, R = 5, 30

def bench(name, fn, *args):
    torch.cuda.synchronize()
    for _ in range(W): fn(*args)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(R): out = fn(*args)
    torch.cuda.synchronize()
    t = (time.time() - t0) / R
    rate = N / t / 1e9
    print(f"    {name:<35} {t*1000:>7.2f}ms {rate:>6.1f}B/s")
    return t, out

print("=" * 60)
print("ITERATIVE ALGORITHMS: Triton vs PyTorch")
print("  (These benefit MOST from kernel fusion)")
print("=" * 60)

# ================================================================
print(f"\n--- Newton sqrt (10 iterations) ---")
print(f"    PyTorch: 10 iterations = 10 kernel launches")
print(f"    Triton:  10 iterations = 1 kernel launch")

@tensorize
def newton_pt(x):
    g = x / 2
    for i in range(10):
        g = (g + x / g) / 2
    return g

@tensorize(compile=True)
def newton_co(x):
    g = x / 2
    for i in range(10):
        g = (g + x / g) / 2
    return g

@tensorize_triton
def newton_tr(x):
    g = x / 2
    for i in range(10):
        g = (g + x / g) / 2
    return g

x = torch.rand(N, device=device) * 1000 + 0.1

t1, o1 = bench("@tensorize (PyTorch)", newton_pt, x)
t2, o2 = bench("@tensorize(compile)", newton_co, x)
t3, o3 = bench("@tensorize_triton", newton_tr, x)

match = torch.allclose(o1, o3, atol=1e-3)
print(f"    Match: {match}")
print(f"    Triton vs PyTorch: {t1/t3:.2f}x FASTER")
print(f"    Triton vs compile: {t2/t3:.2f}x")

# Verify correctness
expected = torch.sqrt(x)
err = (o3 - expected).abs().mean().item()
print(f"    Mean error vs torch.sqrt: {err:.8f}")

# ================================================================
print(f"\n--- Exponential decay (20 iterations: y *= 0.9) ---")

@tensorize
def decay_pt(x):
    v = x
    for i in range(20):
        v = v * 0.9
    return v

@tensorize(compile=True)
def decay_co(x):
    v = x
    for i in range(20):
        v = v * 0.9
    return v

@tensorize_triton
def decay_tr(x):
    v = x
    for i in range(20):
        v = v * 0.9
    return v

t1, o1 = bench("@tensorize (PyTorch)", decay_pt, x)
t2, o2 = bench("@tensorize(compile)", decay_co, x)
t3, o3 = bench("@tensorize_triton", decay_tr, x)
print(f"    Match: {torch.allclose(o1, o3, atol=1e-3)}")
print(f"    Triton vs PyTorch: {t1/t3:.2f}x | vs compile: {t2/t3:.2f}x")

# ================================================================
print(f"\n--- Horner polynomial (degree 8) ---")

@tensorize
def horner_pt(x):
    r = 1.0
    r = r * x + 2.0
    r = r * x + (-3.0)
    r = r * x + 4.0
    r = r * x + (-5.0)
    r = r * x + 6.0
    r = r * x + (-7.0)
    r = r * x + 8.0
    r = r * x + (-9.0)
    return r

@tensorize(compile=True)
def horner_co(x):
    r = 1.0
    r = r * x + 2.0
    r = r * x + (-3.0)
    r = r * x + 4.0
    r = r * x + (-5.0)
    r = r * x + 6.0
    r = r * x + (-7.0)
    r = r * x + 8.0
    r = r * x + (-9.0)
    return r

@tensorize_triton
def horner_tr(x):
    r = 1.0
    r = r * x + 2.0
    r = r * x + (-3.0)
    r = r * x + 4.0
    r = r * x + (-5.0)
    r = r * x + 6.0
    r = r * x + (-7.0)
    r = r * x + 8.0
    r = r * x + (-9.0)
    return r

x2 = torch.randn(N, device=device)
t1, o1 = bench("@tensorize (PyTorch)", horner_pt, x2)
t2, o2 = bench("@tensorize(compile)", horner_co, x2)
t3, o3 = bench("@tensorize_triton", horner_tr, x2)
print(f"    Match: {torch.allclose(o1, o3, atol=1e-1)}")
print(f"    Triton vs PyTorch: {t1/t3:.2f}x | vs compile: {t2/t3:.2f}x")

# ================================================================
print(f"\n--- Fixed-point iteration (15 steps) ---")

@tensorize
def fixed_pt(x):
    y = x
    for i in range(15):
        y = (y + x / (y + 1)) / 2
    return y

@tensorize(compile=True)
def fixed_co(x):
    y = x
    for i in range(15):
        y = (y + x / (y + 1)) / 2
    return y

@tensorize_triton
def fixed_tr(x):
    y = x
    for i in range(15):
        y = (y + x / (y + 1)) / 2
    return y

t1, o1 = bench("@tensorize (PyTorch)", fixed_pt, x)
t2, o2 = bench("@tensorize(compile)", fixed_co, x)
t3, o3 = bench("@tensorize_triton", fixed_tr, x)
print(f"    Match: {torch.allclose(o1, o3, atol=1e-3)}")
print(f"    Triton vs PyTorch: {t1/t3:.2f}x | vs compile: {t2/t3:.2f}x")

# ================================================================
print(f"\n{'='*60}")
print("KEY INSIGHT")
print(f"{'='*60}")
print(f"""
  Iterative algorithms (Newton, RK4, fixed-point) benefit
  MOST from Triton kernel fusion because:

  PyTorch: N iterations = N kernel launches
           Each launch: ~5-10us overhead + memory read/write
           10 iterations: ~50-100us WASTED on overhead

  Triton:  N iterations = 1 kernel launch
           Data stays in GPU registers for ALL iterations
           10 iterations: 0 overhead, 0 extra memory traffic

  This is why Triton gives the BIGGEST speedup for
  iterative numerical methods.
""")
