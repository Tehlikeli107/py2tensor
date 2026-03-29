"""Test profile + tuple return + comprehensive patterns."""
import torch, math, sys
sys.path.insert(0, r"C:\Users\salih\Desktop\py2tensor")
from py2tensor import tensorize, explain, profile
from triton_backend import tensorize_triton

device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name()}\n")
passed = failed = 0

def check(name, cpu, gpu, tol=1e-2):
    global passed, failed
    if isinstance(cpu, tuple):
        ok = all(torch.allclose(
            torch.tensor(c, dtype=torch.float32),
            g.cpu().float(), atol=tol, rtol=0.1
        ) for c, g in zip(cpu, gpu))
    else:
        c = torch.tensor(cpu, dtype=torch.float32)
        g = gpu.cpu().float() if isinstance(gpu, torch.Tensor) else torch.tensor(gpu)
        ok = torch.allclose(c.reshape(-1), g.reshape(-1), atol=tol, rtol=0.1)
    if ok: passed += 1
    else: failed += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

# === Tuple return ===
print("=== Tuple Return ===")

@tensorize
def polar(x, y):
    r = math.sqrt(x * x + y * y)
    theta = math.atan2(y, x)
    return r, theta

x = torch.tensor([3.0, 0.0, 1.0], device=device)
y = torch.tensor([4.0, 5.0, 1.0], device=device)
r, theta = polar(x, y)

cpu_r = [math.sqrt(3**2+4**2), math.sqrt(0+25), math.sqrt(2)]
cpu_t = [math.atan2(4,3), math.atan2(5,0), math.atan2(1,1)]
check("polar r", cpu_r, r)
check("polar theta", cpu_t, theta)

# === Profile demos ===
print("\n=== Profile ===")

@tensorize
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

profile(sigmoid, 1.0)

print()

@tensorize
def complex_scoring(x):
    if x > 100:
        return 100
    else:
        if x > 0:
            return x * math.log(x + 1)
        else:
            return 0

profile(complex_scoring, 50.0)

print()

@tensorize_triton
def triton_gauss(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

profile(triton_gauss, 1.0)

# === Explain all backends ===
print("\n=== Explain All Backends ===")

@tensorize
def f_pt(x):
    if x > 0:
        return math.sin(x)
    else:
        return 0

@tensorize_triton
def f_tr(x):
    if x > 0:
        return math.sin(x)
    else:
        return 0

from model_backend import tensorize_model
@tensorize_model
def f_mo(x):
    if x > 0:
        return math.sin(x)
    else:
        return 0

print("\nPyTorch:")
explain(f_pt)
print("\nTriton:")
explain(f_tr)
print("\nModel:")
explain(f_mo)

print(f"\n{'='*50}")
print(f"RESULTS: {passed} passed, {failed} failed")
