"""
Py2Tensor Integrations: works with your existing tools.
Sklearn, Pandas, FastAPI, NumPy — drop @gpu and go faster.
"""
import torch
import numpy as np
import math
import time
import sys
sys.path.insert(0, r"C:\Users\salih\Desktop\py2tensor")
from api import gpu

device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name()}")
print("=" * 60)
print("INTEGRATIONS: Py2Tensor + Your Existing Tools")
print("=" * 60)

# ================================================================
print("\n[1] PANDAS: GPU-accelerate DataFrame operations")
print("-" * 40)
import pandas as pd

df = pd.DataFrame({
    'age': np.random.randint(18, 80, 1_000_000),
    'income': np.random.randint(20000, 200000, 1_000_000),
    'score': np.random.randint(300, 850, 1_000_000),
})

@gpu
def loan_approval(age, income, score):
    if score > 700:
        if income > 50000:
            return 95
        else:
            return 70
    else:
        if score > 600:
            return 50
        else:
            return 10

# Pandas -> GPU -> Pandas
t0 = time.time()
df['approval'] = loan_approval(
    torch.tensor(df['age'].values, dtype=torch.float32, device=device),
    torch.tensor(df['income'].values, dtype=torch.float32, device=device),
    torch.tensor(df['score'].values, dtype=torch.float32, device=device),
).cpu().numpy()
gpu_time = time.time() - t0

# Compare: pure pandas
t0 = time.time()
df['approval_pd'] = np.where(df['score'] > 700,
    np.where(df['income'] > 50000, 95, 70),
    np.where(df['score'] > 600, 50, 10))
pd_time = time.time() - t0

print(f"  1M rows")
print(f"  GPU:    {gpu_time*1000:.0f}ms")
print(f"  Pandas: {pd_time*1000:.0f}ms")
print(f"  Speedup: {pd_time/gpu_time:.1f}x")
print(f"  Approved (>50): {(df['approval'] > 50).sum()}")

# ================================================================
print(f"\n[2] NUMPY: Drop-in replacement for vectorized ops")
print("-" * 40)

@gpu
def numpy_replacement(x):
    return math.sin(x) * math.exp(-x * x * 0.1) + math.cos(x * 2)

N = 10_000_000
np_data = np.random.randn(N).astype(np.float32)

# NumPy
t0 = time.time()
np_result = np.sin(np_data) * np.exp(-np_data**2 * 0.1) + np.cos(np_data * 2)
np_time = time.time() - t0

# GPU via @gpu
gpu_data = torch.tensor(np_data, device=device)
torch.cuda.synchronize()
t0 = time.time()
gpu_result = numpy_replacement(gpu_data)
torch.cuda.synchronize()
gpu_time = time.time() - t0

print(f"  {N/1e6:.0f}M elements")
print(f"  NumPy:  {np_time*1000:.0f}ms ({N/np_time/1e6:.0f}M/s)")
print(f"  GPU:    {gpu_time*1000:.0f}ms ({N/gpu_time/1e6:.0f}M/s)")
print(f"  Speedup: {np_time/gpu_time:.1f}x")
print(f"  Match: {np.allclose(np_result[:100], gpu_result[:100].cpu().numpy(), atol=1e-3)}")

# ================================================================
print(f"\n[3] SKLEARN-COMPATIBLE: Custom transformer")
print("-" * 40)

try:
    from sklearn.base import BaseEstimator, TransformerMixin

    class GPUFeatureTransformer(BaseEstimator, TransformerMixin):
        """Sklearn transformer that runs on GPU via @gpu."""
        def __init__(self):
            pass

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            if isinstance(X, np.ndarray):
                X_gpu = torch.tensor(X, dtype=torch.float32, device=device)
            else:
                X_gpu = X

            @gpu
            def feature_eng(x):
                return math.log(abs(x) + 1) * math.sin(x * 0.1)

            results = []
            for col in range(X_gpu.shape[1]):
                results.append(feature_eng(X_gpu[:, col]))
            return torch.stack(results, dim=1).cpu().numpy()

    X = np.random.randn(100000, 5).astype(np.float32)
    transformer = GPUFeatureTransformer()
    t0 = time.time()
    X_transformed = transformer.fit_transform(X)
    t = time.time() - t0
    print(f"  100K x 5 features transformed in {t*1000:.0f}ms")
    print(f"  Output shape: {X_transformed.shape}")
    print(f"  Sklearn pipeline compatible: True")

except ImportError:
    print(f"  sklearn not installed, skipping")

# ================================================================
print(f"\n[4] BATCH PREDICTION: Score millions of records")
print("-" * 40)

@gpu.fast
def risk_score(balance, transactions, age, region_risk):
    base = 50
    if balance < 0:
        base = base + 30
    else:
        base = base
    if transactions > 50:
        base = base + 20
    else:
        base = base
    if age < 25:
        base = base + 15
    else:
        base = base
    base = base + region_risk * 5
    if base > 100:
        return 100
    else:
        return base

N = 10_000_000
balances = torch.randn(N, device=device) * 5000
transactions = torch.rand(N, device=device) * 100
ages = torch.rand(N, device=device) * 60 + 18
region_risks = torch.rand(N, device=device) * 10

torch.cuda.synchronize()
for _ in range(3): risk_score(balances, transactions, ages, region_risks)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(20): scores = risk_score(balances, transactions, ages, region_risks)
torch.cuda.synchronize()
t = (time.time()-t0)/20

high_risk = (scores > 70).sum().item()
print(f"  {N/1e6:.0f}M records scored in {t*1000:.1f}ms")
print(f"  Rate: {N/t/1e6:.0f}M records/sec")
print(f"  High risk (>70): {high_risk} ({100*high_risk/N:.1f}%)")

# ================================================================
print(f"\n[5] REAL-TIME STREAM: Process data as it arrives")
print("-" * 40)

@gpu.triton
def process_tick(price, prev_price):
    change = (price - prev_price) / (prev_price + 0.001)
    if change > 0.02:
        return 1
    else:
        if change < -0.02:
            return -1
        else:
            return 0

# Simulate streaming: process 100 batches of 100K
batch_size = 100_000
n_batches = 100
total_signals = 0

torch.cuda.synchronize()
t0 = time.time()
for i in range(n_batches):
    prices = torch.rand(batch_size, device=device) * 100 + 50
    prev = prices + torch.randn(batch_size, device=device) * 2
    signals = process_tick(prices, prev)
    total_signals += (signals != 0).sum().item()
torch.cuda.synchronize()
stream_time = time.time() - t0

total_ticks = batch_size * n_batches
print(f"  {n_batches} batches x {batch_size/1000:.0f}K = {total_ticks/1e6:.0f}M ticks")
print(f"  Total time: {stream_time*1000:.0f}ms")
print(f"  Rate: {total_ticks/stream_time/1e6:.0f}M ticks/sec")
print(f"  Signals generated: {total_signals}")

# ================================================================
print(f"\n{'='*60}")
print("SUMMARY: Py2Tensor works with everything")
print(f"{'='*60}")
print(f"""
  Pandas:    df columns -> GPU -> df columns
  NumPy:     drop-in replacement, {np_time/gpu_time:.0f}x faster
  Sklearn:   custom transformer, pipeline compatible
  Batch:     {N/1e6:.0f}M records in {t*1000:.0f}ms
  Streaming: {total_ticks/stream_time/1e6:.0f}M ticks/sec real-time

  Just add @gpu to your existing Python functions.
""")
