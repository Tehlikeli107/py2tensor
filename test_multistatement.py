"""Test multi-statement if/else body (assigns + return)."""
import torch, math, sys
sys.path.insert(0, r"C:\Users\salih\Desktop\py2tensor")
from py2tensor import tensorize, explain

device = torch.device("cuda")
passed = 0
failed = 0

def check(name, cpu, gpu, tol=1e-1):
    global passed, failed
    c = torch.tensor(cpu, dtype=torch.float32)
    g = gpu.cpu().float() if isinstance(gpu, torch.Tensor) else torch.tensor(gpu, dtype=torch.float32)
    ok = torch.allclose(c.reshape(-1), g.reshape(-1), atol=tol, rtol=0.1)
    if ok: passed += 1
    else:
        failed += 1
        print(f"  CPU: {c.reshape(-1)[:5]}")
        print(f"  GPU: {g.reshape(-1)[:5]}")
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

# Multi-assign + return in if/else
@tensorize
def compute(x):
    if x > 0:
        a = x * 2
        b = x + 10
        return a + b
    else:
        a = 0
        b = -x
        return a + b

vals = [-5, -1, 0, 1, 5]
cpu_out = [compute._original(v) for v in vals]
gpu_out = compute(torch.tensor(vals, dtype=torch.float32, device=device))
check("multi-assign + return", [float(v) for v in cpu_out], gpu_out)
explain(compute)

# Complex physics
@tensorize
def rocket(thrust, angle, burn):
    g = 9.81
    mass = 1000
    vy = thrust * math.sin(angle * 3.14159 / 180) / mass * burn - g * burn
    if vy > 0:
        t_peak = vy / g
        h = 0.5 * vy * t_peak
        return h
    else:
        return 0

vals_t = [20000, 30000, 50000]
vals_a = [45, 60, 80]
vals_b = [30, 30, 30]
cpu_out = [rocket._original(t, a, b) for t, a, b in zip(vals_t, vals_a, vals_b)]
gpu_out = rocket(
    torch.tensor(vals_t, dtype=torch.float32, device=device),
    torch.tensor(vals_a, dtype=torch.float32, device=device),
    torch.tensor(vals_b, dtype=torch.float32, device=device)
)
check("rocket (multi-assign in if + return)", [float(v) for v in cpu_out], gpu_out, tol=100)
explain(rocket)

# Pharmacokinetics
@tensorize
def pk_model(dose, weight):
    conc = dose / (weight * 0.5)
    if conc > 5:
        efficacy = 1.0
        toxicity = (conc - 5) * 0.1
        return efficacy - toxicity
    else:
        efficacy = conc / 5.0
        toxicity = 0
        return efficacy - toxicity

vals_d = [100, 500, 1000, 50]
vals_w = [70, 70, 70, 70]
cpu_out = [pk_model._original(d, w) for d, w in zip(vals_d, vals_w)]
gpu_out = pk_model(
    torch.tensor(vals_d, dtype=torch.float32, device=device),
    torch.tensor(vals_w, dtype=torch.float32, device=device)
)
check("pharmacokinetics (3-assign + return per branch)", [float(v) for v in cpu_out], gpu_out)
explain(pk_model)

print(f"\n{'='*50}")
print(f"RESULTS: {passed} passed, {failed} failed")
