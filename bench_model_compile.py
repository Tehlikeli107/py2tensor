"""
Can we make model backend FAST with torch.compile?
If yes: nn.Module + Triton speed + autograd + save/load = everything.
"""
import torch
import math
import time
import sys
sys.path.insert(0, r"C:\Users\salih\Desktop\py2tensor")
from py2tensor import tensorize
from model_backend import tensorize_model
from triton_backend import tensorize_triton

device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name()}")
N = 10_000_000
W, R = 5, 30

def bench(name, fn, x):
    torch.cuda.synchronize()
    for _ in range(W): fn(x)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(R): out = fn(x)
    torch.cuda.synchronize()
    t = (time.time() - t0) / R
    print(f"  {name:<40} {N/t/1e9:>6.1f}B/s  {t*1000:.2f}ms")
    return t, out

x = torch.randn(N, device=device)
xp = torch.rand(N, device=device) * 100 + 0.1

# ================================================================
print("=" * 60)
print("MODEL + TORCH.COMPILE = HOLY GRAIL?")
print("=" * 60)

# === Simple math ===
print(f"\n--- sin(x) * exp(-x*0.1) ---")

@tensorize
def f_pt(x): return math.sin(x) * math.exp(-x * 0.1)

@tensorize_triton
def f_tr(x): return math.sin(x) * math.exp(-x * 0.1)

@tensorize_model
def f_model(x): return math.sin(x) * math.exp(-x * 0.1)
f_model = f_model.to(device)

f_model_compiled = torch.compile(f_model)

t1, _ = bench("PyTorch ops", f_pt, x)
t2, _ = bench("Triton fused", f_tr, x)
t3, _ = bench("nn.Module (no compile)", f_model, x)
t4, _ = bench("nn.Module + torch.compile", f_model_compiled, x)

print(f"  compile speedup on model: {t3/t4:.2f}x")

# === Newton sqrt ===
print(f"\n--- Newton sqrt (10 iter) ---")

@tensorize
def n_pt(x):
    g = x / 2
    for i in range(10): g = (g + x / g) / 2
    return g

@tensorize_triton
def n_tr(x):
    g = x / 2
    for i in range(10): g = (g + x / g) / 2
    return g

@tensorize_model
def n_model(x):
    g = x / 2
    for i in range(10): g = (g + x / g) / 2
    return g
n_model = n_model.to(device)

n_model_compiled = torch.compile(n_model)

t1, o1 = bench("PyTorch ops", n_pt, xp)
t2, o2 = bench("Triton fused", n_tr, xp)
t3, o3 = bench("nn.Module (no compile)", n_model, xp)
t4, o4 = bench("nn.Module + torch.compile", n_model_compiled, xp)

print(f"  compile speedup on model: {t3/t4:.2f}x")
print(f"  Match triton: {torch.allclose(o2, o4, atol=1e-3)}")

# === Verify model extras still work after compile ===
print(f"\n--- Extras still work? ---")

# Autograd
x_g = torch.tensor([4.0, 9.0], device=device, requires_grad=True)
y = f_model(x_g)  # use non-compiled for autograd
y.sum().backward()
print(f"  Autograd: {x_g.grad is not None}")

# Save/load
torch.save(n_model.state_dict(), r"C:\Users\salih\Desktop\py2tensor\holy_grail.pt")
print(f"  Save: OK")

# Sequential
pipe = torch.nn.Sequential(f_model, f_model).to(device)
out = pipe(torch.tensor([1.0], device=device))
print(f"  Sequential: OK ({out.item():.4f})")

print(f"\n{'='*60}")
print("HOLY GRAIL ACHIEVED?" )
print(f"{'='*60}")
