"""Test while loops, min/max 2-arg, nested tensorize calls."""
import torch, math, time, sys
sys.path.insert(0, r"C:\Users\salih\Desktop\py2tensor")
from py2tensor import tensorize, benchmark

device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name()}\n")
passed = failed = 0

def check(name, cpu, gpu, tol=1e-2):
    global passed, failed
    c = torch.tensor(cpu, dtype=torch.float32)
    g = gpu.cpu().float() if isinstance(gpu, torch.Tensor) else torch.tensor(gpu)
    ok = torch.allclose(c.reshape(-1), g.reshape(-1), atol=tol, rtol=0.1)
    if ok: passed += 1
    else:
        failed += 1
        print(f"  C:{c.reshape(-1)[:5]} G:{g.reshape(-1)[:5]}")
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

# === min/max with 2 args ===
print("=== min/max 2 args ===")

@tensorize
def clamp_func(x):
    return max(min(x, 10), -10)

vals = [-20, -5, 0, 5, 20]
cpu = [clamp_func._original(v) for v in vals]
gpu = clamp_func(torch.tensor(vals, dtype=torch.float32, device=device))
check("max(min(x,10),-10)", [float(v) for v in cpu], gpu)

@tensorize
def smooth_max(a, b):
    return max(a, b) - min(a, b)

pairs = [(1,2), (5,3), (0,0), (-1,1)]
cpu = [smooth_max._original(a,b) for a,b in pairs]
gpu = smooth_max(
    torch.tensor([p[0] for p in pairs], dtype=torch.float32, device=device),
    torch.tensor([p[1] for p in pairs], dtype=torch.float32, device=device)
)
check("max(a,b)-min(a,b) = abs diff", [float(v) for v in cpu], gpu)

# === Nested @tensorize calls ===
print("\n=== Nested Tensorize ===")

@tensorize
def normalize(x):
    if x > 100:
        return 1.0
    else:
        if x < 0:
            return 0.0
        else:
            return x / 100.0

@tensorize
def activate(x):
    return 1.0 / (1.0 + math.exp(-5 * (x - 0.5)))

# Chain: normalize -> activate
vals = [-10, 0, 25, 50, 75, 100, 150]
cpu = [activate._original(normalize._original(v)) for v in vals]
x = torch.tensor(vals, dtype=torch.float32, device=device)
gpu = activate(normalize(x))
check("chained normalize->activate", [float(v) for v in cpu], gpu)

# === Modular arithmetic ===
print("\n=== Modular Arithmetic ===")

@tensorize
def sawtooth(x):
    return x - 10 * (x / 10)  # approximate modulo

vals = [0, 5, 10, 15, 25, -5]
cpu = [sawtooth._original(float(v)) for v in vals]
gpu = sawtooth(torch.tensor(vals, dtype=torch.float32, device=device))
check("sawtooth (approx mod)", [float(v) for v in cpu], gpu, tol=0.5)

# === Complex multi-step ===
print("\n=== Complex Multi-Step ===")

@tensorize
def insurance_premium(age, bmi, smoker):
    """Calculate insurance premium."""
    base = 200
    if age > 50:
        age_factor = 2.0
    else:
        if age > 30:
            age_factor = 1.5
        else:
            age_factor = 1.0
    if bmi > 30:
        bmi_factor = 1.8
    else:
        if bmi > 25:
            bmi_factor = 1.3
        else:
            bmi_factor = 1.0
    if smoker > 0.5:
        smoke_factor = 2.5
    else:
        smoke_factor = 1.0
    return base * age_factor * bmi_factor * smoke_factor

ages = [25, 35, 55, 40]
bmis = [22, 28, 32, 24]
smokers = [0, 0, 1, 0]
cpu = [insurance_premium._original(a,b,s) for a,b,s in zip(ages, bmis, smokers)]
gpu = insurance_premium(
    torch.tensor(ages, dtype=torch.float32, device=device),
    torch.tensor(bmis, dtype=torch.float32, device=device),
    torch.tensor(smokers, dtype=torch.float32, device=device)
)
check("insurance premium (3 factors)", [float(v) for v in cpu], gpu)

# === Benchmark ===
print("\n=== Benchmarks ===")
benchmark(clamp_func, 5.0)
print()
benchmark(insurance_premium, 35.0, 28.0, 0.0)

print(f"\n{'='*50}")
print(f"RESULTS: {passed} passed, {failed} failed")
