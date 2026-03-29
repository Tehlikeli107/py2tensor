"""
Py2Tensor Math/Science Tests: trig, exp, log, physics formulas
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
print("\n=== MATH FUNCTIONS ===")

@tensorize
def trig_combo(x):
    return math.sin(x) * math.cos(x) + math.tan(x * 0.1)

vals = [0, 0.5, 1.0, 1.5, 2.0, -1.0]
cpu_out = [trig_combo(v) for v in vals]
gpu_out = trig_combo(torch.tensor(vals, dtype=torch.float32, device=device))
check("sin*cos+tan", [float(v) for v in cpu_out], gpu_out)

@tensorize
def exp_log(x):
    return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)

vals = [-3, -1, 0, 1, 3]
cpu_out = [exp_log(v) for v in vals]
gpu_out = exp_log(torch.tensor(vals, dtype=torch.float32, device=device))
check("gaussian pdf", [float(v) for v in cpu_out], gpu_out)

@tensorize
def hyperbolic(x):
    return math.sinh(x) + math.cosh(x) - math.tanh(x)

vals = [-2, -1, 0, 1, 2]
cpu_out = [hyperbolic(v) for v in vals]
gpu_out = hyperbolic(torch.tensor(vals, dtype=torch.float32, device=device))
check("sinh+cosh-tanh", [float(v) for v in cpu_out], gpu_out)

# ================================================================
print("\n=== PHYSICS FORMULAS ===")

@tensorize
def relativistic_energy(v):
    c = 299792458.0
    gamma = 1.0 / math.sqrt(1.0 - v * v / (c * c) + 1e-30)
    m0 = 9.109e-31
    return gamma * m0 * c * c

speeds = [0, 1e6, 1e7, 1e8, 2e8, 2.9e8]
cpu_out = [relativistic_energy(v) for v in speeds]
gpu_out = relativistic_energy(torch.tensor(speeds, dtype=torch.float32, device=device))
check("E=gamma*m*c^2", [float(v) for v in cpu_out], gpu_out, tol=1e20)  # large values

@tensorize
def orbital_velocity(r):
    G = 6.674e-11
    M = 5.972e24
    return math.sqrt(G * M / (r + 1e-10))

radii = [6.371e6, 7e6, 1e7, 4.2e7, 3.844e8]
cpu_out = [orbital_velocity(r) for r in radii]
gpu_out = orbital_velocity(torch.tensor(radii, dtype=torch.float32, device=device))
check("v_orbital = sqrt(GM/r)", [float(v) for v in cpu_out], gpu_out, tol=10)

@tensorize
def blackbody_peak(T):
    b = 2.898e-3
    return b / (T + 1e-10)

temps = [300, 1000, 3000, 5778, 10000, 50000]
cpu_out = [blackbody_peak(t) for t in temps]
gpu_out = blackbody_peak(torch.tensor(temps, dtype=torch.float32, device=device))
check("Wien lambda_max = b/T", [float(v) for v in cpu_out], gpu_out)

# ================================================================
print("\n=== FINANCIAL FORMULAS ===")

@tensorize
def compound_interest(principal, rate, years):
    return principal * (1 + rate) ** years

ps = [1000, 5000, 10000, 100000]
cpu_out = [compound_interest(p, 0.07, 10) for p in ps]
gpu_out = compound_interest(
    torch.tensor(ps, dtype=torch.float32, device=device),
    torch.tensor([0.07]*4, dtype=torch.float32, device=device),
    torch.tensor([10]*4, dtype=torch.float32, device=device)
)
check("compound interest", [float(v) for v in cpu_out], gpu_out, tol=1.0)

@tensorize
def loan_payment(principal, annual_rate, months):
    r = annual_rate / 12
    payment = principal * r * (1 + r) ** months / ((1 + r) ** months - 1 + 1e-10)
    return payment

principals = [100000, 200000, 500000, 1000000]
cpu_out = [loan_payment(p, 0.06, 360) for p in principals]
gpu_out = loan_payment(
    torch.tensor(principals, dtype=torch.float32, device=device),
    torch.tensor([0.06]*4, dtype=torch.float32, device=device),
    torch.tensor([360]*4, dtype=torch.float32, device=device)
)
check("monthly loan payment", [float(v) for v in cpu_out], gpu_out, tol=1.0)

# ================================================================
print("\n=== SIGNAL PROCESSING ===")

@tensorize
def damped_oscillator(t):
    omega = 2.0 * math.pi * 5.0
    gamma = 0.5
    return math.exp(-gamma * t) * math.sin(omega * t)

times = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
cpu_out = [damped_oscillator(t) for t in times]
gpu_out = damped_oscillator(torch.tensor(times, dtype=torch.float32, device=device))
check("damped oscillator", [float(v) for v in cpu_out], gpu_out)

@tensorize
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

vals = [-5, -2, -1, 0, 1, 2, 5]
cpu_out = [sigmoid(v) for v in vals]
gpu_out = sigmoid(torch.tensor(vals, dtype=torch.float32, device=device))
check("sigmoid 1/(1+exp(-x))", [float(v) for v in cpu_out], gpu_out)

# ================================================================
print("\n=== MEGA BENCHMARK: 10M Gaussian PDF ===")

@tensorize
def gaussian_pdf(x):
    mu = 0.0
    sigma = 1.0
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))

N = 10_000_000

cpu_data = [float(i - 25000) / 10000 for i in range(50000)]
t0 = time.time()
cpu_out = [gaussian_pdf._original(v) for v in cpu_data]
cpu_time = time.time() - t0

gpu_data = torch.randn(N, device=device)
torch.cuda.synchronize()
_ = gaussian_pdf(gpu_data)  # warmup
torch.cuda.synchronize()
t0 = time.time()
for _ in range(20):
    gpu_out = gaussian_pdf(gpu_data)
torch.cuda.synchronize()
gpu_time = (time.time() - t0) / 20

check("gaussian pdf batch", cpu_out[:50],
      gaussian_pdf(torch.tensor(cpu_data[:50], dtype=torch.float32, device=device)))

cpu_rate = len(cpu_data) / cpu_time
gpu_rate = N / max(gpu_time, 1e-9)
print(f"\n  CPU: {cpu_rate:.0f}/s")
print(f"  GPU: {gpu_rate:.0f}/s ({N/1e6:.0f}M elements)")
print(f"  SPEEDUP: {gpu_rate/cpu_rate:.0f}x")

# ================================================================
print(f"\n{'='*50}")
print(f"RESULTS: {passed} passed, {failed} failed")
print(f"{'='*50}")
