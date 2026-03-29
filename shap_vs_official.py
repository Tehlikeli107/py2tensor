"""
GPU SHAP vs Official SHAP library: head-to-head comparison.
"""
import torch
import numpy as np
import time
import sys
sys.path.insert(0, r"C:\Users\salih\Desktop\py2tensor")
from shap_gpu import shap_gpu_ultra
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn_to_gpu import convert_rf

device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name()}")

try:
    import shap
    print(f"SHAP: {shap.__version__}")
except ImportError:
    print("shap not installed. pip install shap")
    sys.exit(1)

print("=" * 60)
print("GPU SHAP vs Official SHAP: Head-to-Head")
print("=" * 60)

# Train model
X, y = make_classification(n_samples=5000, n_features=10, n_informative=4, random_state=42)
rf = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42)
rf.fit(X, y)

gpu_rf = convert_rf(rf).to(device)
def rf_gpu(X_t):
    return gpu_rf(X_t).float()

# ================================================================
print(f"\n[1] 50 instances — Official KernelSHAP vs GPU SHAP")
print("-" * 40)

X_test = X[:50].astype(np.float32)
background = X[:100]

# Official SHAP
print("  Running official KernelSHAP...")
t0 = time.time()
explainer = shap.KernelExplainer(lambda x: rf.predict_proba(x)[:, 1], background[:50])
official_values = explainer.shap_values(X_test, nsamples=200, silent=True)
official_time = time.time() - t0
print(f"  Official SHAP: {official_time:.1f}s")

# GPU SHAP
X_gpu = torch.tensor(X_test, device=device)
bg_gpu = torch.tensor(background[:50].mean(axis=0), dtype=torch.float32, device=device).unsqueeze(0)

torch.cuda.synchronize()
t0 = time.time()
gpu_values = shap_gpu_ultra(rf_gpu, X_gpu, n_samples=500, background=bg_gpu)
torch.cuda.synchronize()
gpu_time = time.time() - t0
print(f"  GPU SHAP:      {gpu_time:.1f}s")
print(f"  SPEEDUP:       {official_time/gpu_time:.0f}x")

# Compare values
gpu_np = gpu_values.cpu().numpy()
if isinstance(official_values, list):
    official_np = official_values[0] if len(official_values) > 0 else official_values
elif isinstance(official_values, np.ndarray):
    official_np = official_values
else:
    official_np = np.array(official_values)

# Flatten and correlate
if official_np.shape == gpu_np.shape:
    corr = np.corrcoef(official_np.flatten(), gpu_np.flatten())[0, 1]
    print(f"  Correlation:   {corr:.4f}")
else:
    print(f"  Shape mismatch: official={official_np.shape}, gpu={gpu_np.shape}")
    # Try to match
    min_shape = min(official_np.shape[0], gpu_np.shape[0])
    corr = np.corrcoef(official_np[:min_shape].flatten(), gpu_np[:min_shape].flatten())[0, 1]
    print(f"  Correlation (matched): {corr:.4f}")

# Feature ranking comparison
off_importance = np.abs(official_np).mean(axis=0)
gpu_importance = np.abs(gpu_np).mean(axis=0)

off_rank = off_importance.argsort()[::-1]
gpu_rank = gpu_importance.argsort()[::-1]

print(f"\n  Feature ranking:")
print(f"  {'Rank':<6} {'Official':>10} {'GPU':>10}")
for i in range(min(10, len(off_rank))):
    match = "=" if off_rank[i] == gpu_rank[i] else ""
    print(f"  {i+1:<6} Feature {off_rank[i]:<4} Feature {gpu_rank[i]:<4} {match}")

top5_off = set(off_rank[:5])
top5_gpu = set(gpu_rank[:5])
overlap = len(top5_off & top5_gpu)
print(f"\n  Top 5 overlap: {overlap}/5")

# ================================================================
print(f"\n[2] Scale comparison: 200 instances")
print("-" * 40)

X_test2 = X[:200].astype(np.float32)

# Official (smaller nsamples for speed)
print("  Running official KernelSHAP (200 instances)...")
t0 = time.time()
explainer2 = shap.KernelExplainer(lambda x: rf.predict_proba(x)[:, 1], background[:50])
official2 = explainer2.shap_values(X_test2, nsamples=100, silent=True)
official_time2 = time.time() - t0
print(f"  Official: {official_time2:.1f}s")

X_gpu2 = torch.tensor(X_test2, device=device)
torch.cuda.synchronize()
t0 = time.time()
gpu2 = shap_gpu_ultra(rf_gpu, X_gpu2, n_samples=300, background=bg_gpu)
torch.cuda.synchronize()
gpu_time2 = time.time() - t0
print(f"  GPU:      {gpu_time2:.1f}s")
print(f"  SPEEDUP:  {official_time2/gpu_time2:.0f}x")

# ================================================================
print(f"\n{'='*60}")
print("VERDICT")
print(f"{'='*60}")
print(f"""
  Official SHAP (KernelSHAP):
    50 instances:  {official_time:.1f}s
    200 instances: {official_time2:.1f}s

  GPU SHAP (py2tensor):
    50 instances:  {gpu_time:.1f}s
    200 instances: {gpu_time2:.1f}s

  Speedup: {official_time/gpu_time:.0f}x (50 inst), {official_time2/gpu_time2:.0f}x (200 inst)
  Top 5 feature overlap: {overlap}/5

  GPU SHAP scales BETTER:
    Official scales linearly with instances.
    GPU batches all instances together.
""")
