"""
Real-world pattern tests: early return, comparison chains, diverse code styles
"""
import torch, math, time, sys
sys.path.insert(0, r"C:\Users\salih\Desktop\py2tensor")
from py2tensor import tensorize, explain, benchmark

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

# === Comparison chain ===
print("=== Comparison Chains ===")

@tensorize
def in_range(x):
    if 0 < x:
        if x < 100:
            return x
        else:
            return 100
    else:
        return 0

vals = [-5, 0, 50, 100, 150]
cpu = [in_range._original(v) for v in vals]
gpu = in_range(torch.tensor(vals, dtype=torch.float32, device=device))
check("0 < x < 100 clamp", [float(v) for v in cpu], gpu)

# === Complex scoring ===
print("\n=== Complex Scoring ===")

@tensorize
def credit_score(income, debt, age):
    ratio = debt / (income + 1)
    if ratio > 0.5:
        base = 300
    else:
        if ratio > 0.3:
            base = 500
        else:
            base = 700
    if age > 30:
        bonus = 50
    else:
        bonus = 0
    return base + bonus

incomes = [50000, 80000, 30000, 100000]
debts =   [30000, 20000, 20000, 10000]
ages =    [25, 35, 28, 45]
cpu = [credit_score._original(i,d,a) for i,d,a in zip(incomes, debts, ages)]
gpu = credit_score(
    torch.tensor(incomes, dtype=torch.float32, device=device),
    torch.tensor(debts, dtype=torch.float32, device=device),
    torch.tensor(ages, dtype=torch.float32, device=device)
)
check("3-input credit score", [float(v) for v in cpu], gpu)

# === Iterative with convergence ===
print("\n=== Iterative Convergence ===")

@tensorize
def golden_ratio(x):
    """Compute golden ratio via continued fraction."""
    phi = x
    for i in range(30):
        phi = 1.0 + 1.0 / (phi + 0.001)
    return phi

vals = [1.0, 2.0, 0.5, 10.0]
cpu = [golden_ratio._original(v) for v in vals]
gpu = golden_ratio(torch.tensor(vals, dtype=torch.float32, device=device))
check("golden ratio (30 iter)", [float(v) for v in cpu], gpu)

# === Compound operations ===
print("\n=== Compound Operations ===")

@tensorize
def compound(x):
    a = x
    a += 5
    a *= 2
    a -= 3
    if a > 20:
        a = 20
    else:
        a = a
    return a

vals = [0, 5, 10, -5]
cpu = [compound._original(v) for v in vals]
gpu = compound(torch.tensor(vals, dtype=torch.float32, device=device))
check("compound +=*=-= with clamp", [float(v) for v in cpu], gpu)

# === Multi-input physics ===
print("\n=== Multi-Input Physics ===")

@tensorize
def kinetic_energy(mass, velocity):
    return 0.5 * mass * velocity * velocity

@tensorize
def potential_energy(mass, height):
    return mass * 9.81 * height

@tensorize
def total_energy(mass, velocity, height):
    ke = 0.5 * mass * velocity * velocity
    pe = mass * 9.81 * height
    return ke + pe

masses = [1, 10, 100, 0.5]
vels = [10, 5, 1, 20]
heights = [100, 50, 10, 200]
cpu = [total_energy._original(m,v,h) for m,v,h in zip(masses, vels, heights)]
gpu = total_energy(
    torch.tensor(masses, dtype=torch.float32, device=device),
    torch.tensor(vels, dtype=torch.float32, device=device),
    torch.tensor(heights, dtype=torch.float32, device=device)
)
check("total energy (KE+PE)", [float(v) for v in cpu], gpu)

# === Batch benchmark ===
print("\n=== BATCH BENCHMARK ===")
benchmark(credit_score, 50000.0, 20000.0, 35.0)
print()
benchmark(golden_ratio, 1.0)
print()
benchmark(total_energy, 10.0, 5.0, 100.0)

print(f"\n{'='*50}")
print(f"RESULTS: {passed} passed, {failed} failed")
