"""
Py2Tensor Practical Demo: Real Data Processing
===============================================
Shows py2tensor on realistic data workflows:
1. Financial: process 10M stock prices
2. Scientific: batch compute physics formulas
3. ML: custom activation + loss functions
4. Data: pandas DataFrame processing
"""
import torch
import numpy as np
import math
import time
import sys
sys.path.insert(0, r"C:\Users\salih\Desktop\py2tensor")
from py2tensor import tensorize, benchmark

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("PY2TENSOR PRACTICAL DEMO")
print("=" * 60)

# ================================================================
print("\n[1] FINANCE: Compute technical indicators for 10M price points")
print("-" * 60)

@tensorize
def rsi_signal(price, prev_price):
    """Relative Strength Index signal: simplified"""
    change = price - prev_price
    # gain = max(change, 0), loss = max(-change, 0)
    if change > 0:
        gain = change
    else:
        gain = 0
    if change < 0:
        loss = -change
    else:
        loss = 0
    # Signal: +1 buy, -1 sell, 0 hold
    ratio = gain / (loss + 0.001)
    if ratio > 2.0:
        return 1
    else:
        if ratio < 0.5:
            return -1
        else:
            return 0

@tensorize
def bollinger_signal(price, mean, std):
    """Bollinger band signal"""
    upper = mean + 2 * std
    lower = mean - 2 * std
    if price > upper:
        return -1
    else:
        if price < lower:
            return 1
        else:
            return 0

@tensorize
def combined_signal(rsi, boll, momentum):
    """Combine signals: -3 to +3"""
    total = rsi + boll + momentum
    if total > 2:
        result = 3
    else:
        result = total
    if result < -2:
        result = -3
    else:
        result = result
    return result

N = 10_000_000
prices = torch.rand(N, device=device) * 100 + 50
prev_prices = prices + torch.randn(N, device=device) * 2
means = torch.full((N,), 75.0, device=device)
stds = torch.full((N,), 10.0, device=device)

torch.cuda.synchronize()
t0 = time.time()
rsi = rsi_signal(prices, prev_prices)
boll = bollinger_signal(prices, means, stds)
momentum = torch.sign(prices - prev_prices)
signal = combined_signal(rsi, boll, momentum)
torch.cuda.synchronize()
elapsed = time.time() - t0

buys = (signal > 0).sum().item()
sells = (signal < 0).sum().item()
holds = (signal == 0).sum().item()

print(f"  {N/1e6:.0f}M price points processed in {elapsed*1000:.0f}ms")
print(f"  Rate: {N/elapsed/1e6:.0f}M signals/s")
print(f"  Buy: {buys} ({100*buys/N:.1f}%)")
print(f"  Sell: {sells} ({100*sells/N:.1f}%)")
print(f"  Hold: {holds} ({100*holds/N:.1f}%)")

# ================================================================
print(f"\n[2] PHYSICS: Batch orbital mechanics for 1M satellites")
print("-" * 60)

@tensorize
def orbital_params(r, v, M):
    """Compute orbital energy and period from radius, velocity, central mass"""
    G = 6.674e-11
    E = 0.5 * v * v - G * M / r
    # Semi-major axis
    a = -G * M / (2 * E + 1e-30)
    # Period (Kepler's 3rd law)
    T = 2 * math.pi * math.sqrt(a * a * a / (G * M + 1e-30))
    return T

N_sat = 1_000_000
radii = torch.rand(N_sat, device=device) * 4e7 + 6.5e6  # 6500-46500 km
velocities = torch.rand(N_sat, device=device) * 5000 + 3000  # 3-8 km/s
M_earth = torch.full((N_sat,), 5.972e24, device=device)

torch.cuda.synchronize()
t0 = time.time()
periods = orbital_params(radii, velocities, M_earth)
torch.cuda.synchronize()
elapsed = time.time() - t0

print(f"  {N_sat/1e6:.0f}M satellite orbits computed in {elapsed*1000:.0f}ms")
print(f"  Rate: {N_sat/elapsed/1e6:.0f}M orbits/s")
print(f"  Period range: {periods.min().item()/3600:.0f} - {periods.max().item()/3600:.0f} hours")

# ================================================================
print(f"\n[3] ML: Custom activation + loss on 10M")
print("-" * 60)

@tensorize(compile=True)
def swish(x):
    """Swish activation: x * sigmoid(x)"""
    return x / (1.0 + math.exp(-x))

@tensorize(compile=True)
def focal_loss(pred, target):
    """Simplified focal loss for hard example mining"""
    p = 1.0 / (1.0 + math.exp(-pred))
    if target > 0.5:
        return -(1 - p) * (1 - p) * math.log(p + 1e-8)
    else:
        return -p * p * math.log(1 - p + 1e-8)

N_ml = 10_000_000
x_ml = torch.randn(N_ml, device=device)
targets = (torch.rand(N_ml, device=device) > 0.5).float()

torch.cuda.synchronize()
t0 = time.time()
activated = swish(x_ml)
losses = focal_loss(x_ml, targets)
torch.cuda.synchronize()
elapsed = time.time() - t0

print(f"  {N_ml/1e6:.0f}M: Swish + Focal Loss in {elapsed*1000:.0f}ms")
print(f"  Rate: {N_ml/elapsed/1e6:.0f}M/s")
print(f"  Mean loss: {losses.mean().item():.4f}")

# ================================================================
print(f"\n[4] PANDAS: Process DataFrame on GPU")
print("-" * 60)

try:
    import pandas as pd

    # Create 1M-row DataFrame
    N_pd = 1_000_000
    df = pd.DataFrame({
        'temperature': np.random.randn(N_pd) * 10 + 20,
        'humidity': np.random.rand(N_pd) * 100,
        'pressure': np.random.randn(N_pd) * 5 + 1013,
    })

    @tensorize
    def heat_index(temp, humidity):
        """Simplified heat index"""
        hi = temp + 0.5 * humidity * 0.01 * (temp - 14)
        if temp > 27:
            result = hi
        else:
            result = temp
        if result > 40:
            return 40
        else:
            return result

    @tensorize
    def weather_risk(hi, pressure):
        if hi > 35:
            base = 3
        else:
            if hi > 30:
                base = 2
            else:
                base = 1
        if pressure < 1005:
            return base + 1
        else:
            return base

    # Process DataFrame columns directly
    t0 = time.time()
    hi = heat_index(df['temperature'], df['humidity'])
    risk = weather_risk(hi, df['pressure'])
    elapsed = time.time() - t0

    # Convert back to pandas
    df['heat_index'] = hi.cpu().numpy()
    df['risk_level'] = risk.cpu().numpy()

    print(f"  {N_pd/1e6:.0f}M rows: heat_index + risk in {elapsed*1000:.0f}ms")
    print(f"  Rate: {N_pd/elapsed/1e6:.0f}M rows/s")
    print(f"  High risk (>=3): {(df['risk_level'] >= 3).sum()} ({100*(df['risk_level'] >= 3).mean():.1f}%)")
    print(f"  Sample output:")
    print(df.head(3).to_string(index=False))

except ImportError:
    print("  pandas not installed, skipping")

# ================================================================
print(f"\n[5] COMBINED: Full pipeline benchmark")
print("-" * 60)

@tensorize(compile=True)
def full_pipeline(x):
    # Normalize
    if x > 100:
        n = 1.0
    else:
        if x < 0:
            n = 0.0
        else:
            n = x / 100.0
    # Non-linear transform
    t = math.tanh(3.0 * (n - 0.5))
    # Sigmoid output
    s = 1.0 / (1.0 + math.exp(-5.0 * t))
    # Quantize to 10 levels
    level = s * 10.0
    if level > 9.5:
        return 9
    else:
        return level

N_full = 10_000_000
data = torch.randn(N_full, device=device) * 50 + 50

torch.cuda.synchronize()
_ = full_pipeline(data)  # warmup compile
torch.cuda.synchronize()

t0 = time.time()
for _ in range(50):
    result = full_pipeline(data)
torch.cuda.synchronize()
elapsed = (time.time() - t0) / 50

print(f"  Full pipeline (normalize+tanh+sigmoid+quantize)")
print(f"  {N_full/1e6:.0f}M elements in {elapsed*1000:.2f}ms")
print(f"  Rate: {N_full/elapsed/1e6:.0f}M/s")
print(f"  Output range: {result.min().item():.2f} - {result.max().item():.2f}")

# ================================================================
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"""
  All computations done with @tensorize decorator.
  No hand-written CUDA. No GPU programming knowledge needed.
  Just write Python, add @tensorize, get GPU speed.

  Finance:  {N/1e6:.0f}M signals in <{elapsed*1000:.0f}ms
  Physics:  1M orbital calculations
  ML:       10M custom loss computations
  Pandas:   DataFrame columns processed on GPU
  Pipeline: {N_full/elapsed/1e6:.0f}M elements/s with compile=True
""")
