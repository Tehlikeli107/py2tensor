"""
Py2Tensor Ternary + advanced ops tests
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
print("\n=== TERNARY EXPRESSIONS ===")

@tensorize
def ternary_relu(x):
    return x if x > 0 else 0

vals = [-5, -1, 0, 1, 5]
cpu_out = [ternary_relu(v) for v in vals]
gpu_out = ternary_relu(torch.tensor(vals, dtype=torch.float32, device=device))
check("ternary relu", [float(v) for v in cpu_out], gpu_out)

@tensorize
def ternary_abs(x):
    return x if x > 0 else -x

vals = [-5, -1, 0, 1, 5]
cpu_out = [ternary_abs(v) for v in vals]
gpu_out = ternary_abs(torch.tensor(vals, dtype=torch.float32, device=device))
check("ternary abs", [float(v) for v in cpu_out], gpu_out)

@tensorize
def ternary_sign(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)

vals = [-5, -1, 0, 1, 5]
cpu_out = [ternary_sign(v) for v in vals]
gpu_out = ternary_sign(torch.tensor(vals, dtype=torch.float32, device=device))
check("nested ternary sign", [float(v) for v in cpu_out], gpu_out)

# ================================================================
print("\n=== UNARY OPS ===")

@tensorize
def negate(x):
    return -x + 1

vals = [-5, 0, 5]
cpu_out = [negate(v) for v in vals]
gpu_out = negate(torch.tensor(vals, dtype=torch.float32, device=device))
check("unary negate", [float(v) for v in cpu_out], gpu_out)

# ================================================================
print("\n=== POWER OPERATOR ===")

@tensorize
def power_func(x):
    return x ** 2 + x ** 0.5

vals = [1, 4, 9, 16, 25]
cpu_out = [power_func(v) for v in vals]
gpu_out = power_func(torch.tensor(vals, dtype=torch.float32, device=device))
check("x^2 + x^0.5", [float(v) for v in cpu_out], gpu_out)

# ================================================================
print("\n=== COMPLEX REAL-WORLD: Neural network forward pass ===")

@tensorize
def neural_layer(x):
    """Simulated single neuron: wx + b -> activation"""
    w1 = 0.5
    w2 = -0.3
    b = 0.1
    z = w1 * x + b
    # ReLU
    a = z if z > 0 else 0
    # Second transform
    out = w2 * a + 0.5
    return 1.0 / (1.0 + math.exp(-out))  # sigmoid

vals = [-5, -2, 0, 2, 5, 10]
cpu_out = [neural_layer(v) for v in vals]
gpu_out = neural_layer(torch.tensor(vals, dtype=torch.float32, device=device))
check("neural layer (linear+relu+sigmoid)", [float(v) for v in cpu_out], gpu_out)

# ================================================================
print("\n=== COMPLEX REAL-WORLD: Pricing with multiple conditions ===")

@tensorize
def dynamic_pricing(demand, supply, base_price):
    ratio = demand / (supply + 0.001)
    surge = ratio ** 1.5 if ratio > 1 else 1.0
    discount = 0.8 if ratio < 0.5 else 1.0
    price = base_price * surge * discount
    price = 10.0 if price < 10 else (1000.0 if price > 1000 else price)
    return price

demands = [100, 50, 200, 10, 150]
supplies = [100, 100, 50, 100, 100]
prices = [50, 50, 50, 50, 50]
cpu_out = [dynamic_pricing(d, s, p) for d, s, p in zip(demands, supplies, prices)]
gpu_out = dynamic_pricing(
    torch.tensor(demands, dtype=torch.float32, device=device),
    torch.tensor(supplies, dtype=torch.float32, device=device),
    torch.tensor(prices, dtype=torch.float32, device=device),
)
check("dynamic pricing", [float(v) for v in cpu_out], gpu_out)

# ================================================================
print("\n=== MEGA BENCHMARK: Neural layer on 10M ===")
N = 10_000_000

cpu_data = [float(i)/1000000 - 5 for i in range(50000)]
t0 = time.time()
cpu_out = [neural_layer._original(v) for v in cpu_data]
cpu_time = time.time() - t0

gpu_data = torch.randn(N, device=device) * 5
torch.cuda.synchronize()
_ = neural_layer(gpu_data)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(20):
    _ = neural_layer(gpu_data)
torch.cuda.synchronize()
gpu_time = (time.time() - t0) / 20

cpu_rate = len(cpu_data) / cpu_time
gpu_rate = N / max(gpu_time, 1e-9)
print(f"  CPU: {cpu_rate/1e6:.1f}M/s")
print(f"  GPU: {gpu_rate/1e6:.0f}M/s")
print(f"  SPEEDUP: {gpu_rate/cpu_rate:.0f}x")

# ================================================================
print(f"\n{'='*50}")
print(f"RESULTS: {passed} passed, {failed} failed")
print(f"{'='*50}")
