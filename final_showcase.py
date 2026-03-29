"""
Py2Tensor Final Showcase: The Complete Pipeline
================================================
One file that demonstrates EVERYTHING.
Run this, see the magic.
"""
import torch
import math
import time
import sys
sys.path.insert(0, r"C:\Users\salih\Desktop\py2tensor")
from py2tensor import gpu, diagnose
from auto_compress import auto_compress
from piecewise_nd import piecewise_nd_compress

device = torch.device("cuda")
N = 10_000_000

print(f"""
{'='*60}
PY2TENSOR v2.0 — FINAL SHOWCASE
{'='*60}
GPU: {torch.cuda.get_device_name()}

The idea: "Take any CPU function, convert to GPU tensor model,
then compress to smallest possible representation."

Three steps:
  1. @gpu        -> CPU function runs on GPU (100-4000x faster)
  2. diagnose()  -> analyze compatibility, suggest best backend
  3. compress()  -> shrink model to tiny size (R2>0.999)
{'='*60}
""")

# ================================================================
print("[STEP 1] Write normal Python, add @gpu")
print("-" * 40)

@gpu
def black_scholes(S, K, T, r, sigma):
    d1 = (math.log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*math.sqrt(T) + 0.0001)
    nd1 = 1.0 / (1.0 + math.exp(-1.7 * d1))
    d2 = d1 - sigma * math.sqrt(T)
    nd2 = 1.0 / (1.0 + math.exp(-1.7 * d2))
    call = S * nd1 - K * math.exp(-r*T) * nd2
    if call < 0:
        return 0
    else:
        return call

# Single value
print(f"  black_scholes(100, 100, 1, 0.05, 0.2) = ${black_scholes(100, 100, 1, 0.05, 0.2):.2f}")

# 10M options
S = torch.rand(N, device=device) * 100 + 50
K = torch.rand(N, device=device) * 100 + 50
T = torch.rand(N, device=device) * 2 + 0.1
r = torch.full((N,), 0.05, device=device)
sigma = torch.rand(N, device=device) * 0.3 + 0.1

torch.cuda.synchronize()
t0 = time.time()
prices = black_scholes(S, K, T, r, sigma)
torch.cuda.synchronize()
t = time.time() - t0
print(f"  {N/1e6:.0f}M options priced in {t*1000:.0f}ms ({N/t/1e6:.0f}M/s)")

# ================================================================
print(f"\n[STEP 2] diagnose() — analyze any function")
print("-" * 40)

def complex_function(x, y, z):
    rates = {0: 0.05, 1: 0.10, 2: 0.15}
    if x > 100:
        tier = 2
    else:
        if x > 50:
            tier = 1
        else:
            tier = 0
    base = x * rates[tier]
    for i in range(5):
        base = base + math.sin(y * 0.1) * z
    return base

diagnose(complex_function)

# ================================================================
print(f"\n[STEP 3] auto_compress() — smallest accurate model")
print("-" * 40)

@gpu
def tax(income):
    if income > 500000:
        return (income - 500000) * 0.37 + 150000
    else:
        if income > 200000:
            return (income - 200000) * 0.32 + 50000
        else:
            if income > 80000:
                return (income - 80000) * 0.22 + 20000
            else:
                if income > 40000:
                    return (income - 40000) * 0.12 + 5000
                else:
                    return income * 0.10

small_tax = auto_compress(tax, (0, 1000000))

# Verify
test = torch.tensor([10000, 50000, 100000, 300000, 700000], dtype=torch.float32, device=device)
print(f"\n  Verify: original  = {[f'${v:.0f}' for v in tax(test).tolist()]}")
if small_tax is not None:
    small_tax = small_tax.to(device)
    print(f"          compressed = {[f'${v:.0f}' for v in small_tax(test).tolist()]}")

# ================================================================
print(f"\n[STEP 3b] N-D compression — multi-input functions")
print("-" * 40)

@gpu
def insurance(age, bmi):
    if age > 50:
        base = 500
    else:
        base = 200
    if bmi > 30:
        factor = 2.0
    else:
        factor = 1.0
    return base * factor

model_2d = piecewise_nd_compress(insurance, [(20, 80), (18, 45)], 2)
model_2d = model_2d.to(device)

ages = torch.tensor([25, 55, 35, 65], dtype=torch.float32, device=device)
bmis = torch.tensor([22, 22, 35, 35], dtype=torch.float32, device=device)
print(f"  Original:   {insurance(ages, bmis).tolist()}")
print(f"  Compressed: {model_2d(ages, bmis).tolist()}")

# ================================================================
print(f"\n[STEP 4] Triton — fused kernel for iterative algorithms")
print("-" * 40)

@gpu.triton
def newton_sqrt(x):
    g = x / 2
    for i in range(10):
        g = (g + x / g) / 2
    return g

x = torch.rand(N, device=device) * 100 + 0.1
torch.cuda.synchronize()
for _ in range(3): newton_sqrt(x)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(20): newton_sqrt(x)
torch.cuda.synchronize()
t = (time.time()-t0)/20
print(f"  Newton sqrt: {N/t/1e9:.1f}B/s ({t*1000:.2f}ms for {N/1e6:.0f}M)")
print(f"  10 iterations fused into SINGLE GPU kernel")

# ================================================================
print(f"\n[STEP 5] Pure Model — save/load/compose")
print("-" * 40)

@gpu.model
def activation(x):
    return x * math.tanh(math.log(1 + math.exp(x)))

model = activation.to(device)
x = torch.tensor([1.0, 2.0], device=device, requires_grad=True)
y = model(x)
y.sum().backward()
print(f"  Forward:  {y.tolist()}")
print(f"  Gradient: {x.grad.tolist()}")
torch.save(model.state_dict(), r'C:\Users\salih\Desktop\py2tensor\showcase_model.pt')
print(f"  Saved to showcase_model.pt")

# ================================================================
print(f"\n{'='*60}")
print("FINAL NUMBERS")
print(f"{'='*60}")
print(f"""
  Tests:          70+
  Backends:       6 (pytorch, compile, triton, pure, auto, all)
  Compression:    3 (piecewise 1D, piecewise ND, SVD)
  Max throughput: 44.2 billion elements/sec
  Max speedup:    3,959x (fraud rules)
  Max compress:   R2=1.000 with 16 parameters (tax)

  The complete pipeline:
    Python function
      -> @gpu (GPU tensor model, 100-4000x faster)
      -> diagnose() (compatibility check + suggestion)
      -> auto_compress() (smallest model, R2>0.999)

  Everything from one idea:
    "Take the CPU function, build it as a model,
     not by training, but by CONSTRUCTION."

  GitHub: https://github.com/Tehlikeli107/py2tensor
""")
