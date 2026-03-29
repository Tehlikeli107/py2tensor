"""
Py2Tensor Advanced Tests v2: multiple returns, chained functions,
complex pipelines, Monte Carlo simulation
"""
import torch
import math
import time
import sys
sys.path.insert(0, r"C:\Users\salih\Desktop\py2tensor")
from py2tensor import tensorize, benchmark

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
passed = 0
failed = 0

def check(name, expected, got, tol=1e-2):
    global passed, failed
    if isinstance(expected, (list, tuple)):
        expected = torch.tensor(expected, dtype=torch.float32)
    if isinstance(got, torch.Tensor):
        got = got.cpu().float()
    match = torch.allclose(expected.reshape(-1), got.reshape(-1), atol=tol, rtol=1e-2)
    if match: passed += 1
    else:
        failed += 1
        print(f"  Expected: {expected.reshape(-1)[:5]}")
        print(f"  Got:      {got.reshape(-1)[:5]}")
    print(f"  [{'PASS' if match else 'FAIL'}] {name}")

# ================================================================
print("\n=== CHAINED @tensorize FUNCTIONS ===")

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
    return 1.0 / (1.0 + math.exp(-10 * (x - 0.5)))

@tensorize
def quantize(x):
    if x > 0.75:
        return 3
    else:
        if x > 0.5:
            return 2
        else:
            if x > 0.25:
                return 1
            else:
                return 0

# Chain them: normalize -> activate -> quantize
vals = [-10, 0, 25, 50, 75, 100, 150]

# CPU
cpu_out = []
for v in vals:
    n = normalize(v)
    a = activate(n)
    q = quantize(a)
    cpu_out.append(float(q))

# GPU (chained)
x = torch.tensor(vals, dtype=torch.float32, device=device)
n_gpu = normalize(x)
a_gpu = activate(n_gpu)
q_gpu = quantize(a_gpu)
check("chained: normalize->activate->quantize", cpu_out, q_gpu)

# ================================================================
print("\n=== MULTIPLE RETURN VALUES (tuples) ===")

@tensorize
def polar_coords(x, y):
    r = math.sqrt(x * x + y * y)
    theta = math.atan2(y, x)
    return r, theta

# CPU
x_vals = [3, 0, -1, 1]
y_vals = [4, 5, 0, 1]
cpu_r = [math.sqrt(x**2 + y**2) for x, y in zip(x_vals, y_vals)]
cpu_t = [math.atan2(y, x) for x, y in zip(x_vals, y_vals)]

# GPU
x_g = torch.tensor(x_vals, dtype=torch.float32, device=device)
y_g = torch.tensor(y_vals, dtype=torch.float32, device=device)
r_g, theta_g = polar_coords(x_g, y_g)
check("polar r", cpu_r, r_g)
check("polar theta", cpu_t, theta_g)

# ================================================================
print("\n=== COMPLEX NUMERICAL PIPELINE ===")

@tensorize
def runge_kutta_step(y, t, dt):
    """Single RK4 step for dy/dt = -y (exponential decay)"""
    k1 = -y
    k2 = -(y + 0.5 * dt * k1)
    k3 = -(y + 0.5 * dt * k2)
    k4 = -(y + dt * k3)
    return y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

# Simulate exponential decay y(t) = exp(-t)
# RK4 for N steps
@tensorize
def decay_rk4(y0):
    y = y0
    dt = 0.1
    for i in range(10):
        k1 = -y
        k2 = -(y + 0.5 * dt * k1)
        k3 = -(y + 0.5 * dt * k2)
        k4 = -(y + dt * k3)
        y = y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y

# After 10 steps of dt=0.1, t=1.0, exact: exp(-1) = 0.3679
y0_vals = [1.0, 2.0, 5.0, 0.5]
cpu_out = [decay_rk4._original(v) for v in y0_vals]
gpu_out = decay_rk4(torch.tensor(y0_vals, dtype=torch.float32, device=device))
expected = [v * math.exp(-1.0) for v in y0_vals]
check("RK4 decay (10 steps)", expected, gpu_out, tol=0.01)

# ================================================================
print("\n=== MONTE CARLO OPTION PRICING (full pipeline) ===")

@tensorize
def geometric_brownian_step(S, r, sigma, dt, z):
    """Single GBM step: S_{t+1} = S_t * exp((r-sigma^2/2)*dt + sigma*sqrt(dt)*z)"""
    drift = (r - 0.5 * sigma * sigma) * dt
    diffusion = sigma * math.sqrt(dt) * z
    return S * math.exp(drift + diffusion)

@tensorize
def option_payoff(S_final, K):
    """European call payoff: max(S-K, 0)"""
    payoff = S_final - K
    if payoff > 0:
        return payoff
    else:
        return 0

# Price a European call option via Monte Carlo
N_PATHS = 1_000_000
S0 = 100.0
K = 105.0
r = 0.05
sigma = 0.2
T = 1.0
N_STEPS = 12  # monthly steps
dt = T / N_STEPS

print(f"\n  Monte Carlo: {N_PATHS/1e6:.0f}M paths, {N_STEPS} steps each")

torch.cuda.synchronize()
t0 = time.time()

# Simulate paths
S = torch.full((N_PATHS,), S0, device=device)
for step in range(N_STEPS):
    z = torch.randn(N_PATHS, device=device)
    S = geometric_brownian_step(S, r, sigma, dt, z)

# Compute payoffs
payoffs = option_payoff(S, K)

# Discount
discount = math.exp(-r * T)
price = discount * payoffs.mean().item()

torch.cuda.synchronize()
mc_time = time.time() - t0

# Black-Scholes analytical
from math import log, sqrt, erf
d1 = (log(S0/K) + (r + sigma**2/2)*T) / (sigma*sqrt(T))
d2 = d1 - sigma*sqrt(T)
N_d1 = 0.5 * (1 + erf(d1/sqrt(2)))
N_d2 = 0.5 * (1 + erf(d2/sqrt(2)))
bs_price = S0 * N_d1 - K * math.exp(-r*T) * N_d2

print(f"  GPU MC price:    ${price:.4f}")
print(f"  BS analytical:   ${bs_price:.4f}")
print(f"  Difference:      ${abs(price - bs_price):.4f}")
print(f"  Time: {mc_time*1000:.0f}ms for {N_PATHS/1e6:.0f}M paths x {N_STEPS} steps")
print(f"  Throughput: {N_PATHS * N_STEPS / mc_time / 1e6:.0f}M path-steps/s")

mc_match = abs(price - bs_price) < 0.5
if mc_match: passed += 1
else: failed += 1
print(f"  [{'PASS' if mc_match else 'FAIL'}] MC price within $0.50 of BS")

# ================================================================
print("\n=== PHYSICS: N-body gravitational force ===")

@tensorize
def gravitational_force(m1, m2, r):
    G = 6.674e-11
    return G * m1 * m2 / (r * r + 1e-20)

masses1 = [1e30, 5.972e24, 1.989e30]
masses2 = [1e30, 7.348e22, 5.972e24]
distances = [1e10, 3.844e8, 1.496e11]

cpu_out = [gravitational_force._original(m1, m2, d) for m1, m2, d in zip(masses1, masses2, distances)]
gpu_out = gravitational_force(
    torch.tensor(masses1, dtype=torch.float32, device=device),
    torch.tensor(masses2, dtype=torch.float32, device=device),
    torch.tensor(distances, dtype=torch.float32, device=device),
)
# Large values, check relative
check("gravitational force", cpu_out, gpu_out, tol=max(abs(v) * 0.01 for v in cpu_out))

# ================================================================
print("\n=== MEGA BENCHMARK: RK4 on 10M ===")

N = 10_000_000
benchmark(decay_rk4, 1.0, n=N)

print("\n=== MEGA BENCHMARK: GBM step on 10M ===")
benchmark(geometric_brownian_step, 100.0, 0.05, 0.2, 0.083, 0.0, n=N)

# ================================================================
print(f"\n{'='*50}")
print(f"RESULTS: {passed} passed, {failed} failed")
print(f"{'='*50}")
