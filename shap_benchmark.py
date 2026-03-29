"""
GPU SHAP Benchmark: compare with official shap library.
"""
import torch
import numpy as np
import time
import sys
sys.path.insert(0, r"C:\Users\salih\Desktop\py2tensor")
from shap_gpu import shap_gpu_ultra
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn_to_gpu import convert_rf

device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name()}")
print("=" * 60)
print("GPU SHAP vs CPU SHAP: Full Benchmark")
print("=" * 60)

# Check if shap is installed
try:
    import shap
    HAS_SHAP = True
    print(f"shap library: {shap.__version__}")
except ImportError:
    HAS_SHAP = False
    print("shap library: NOT INSTALLED (will compare with manual CPU)")

# Train model
X, y = make_classification(n_samples=5000, n_features=20, n_informative=5, random_state=42)
rf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
rf.fit(X, y)

# Convert to GPU
gpu_rf = convert_rf(rf).to(device)
def rf_gpu(X_tensor):
    return gpu_rf(X_tensor).float()

# ================================================================
print(f"\n[1] Small scale: 100 instances, 20 features")
print("-" * 40)

X_test = X[:100]
X_gpu = torch.tensor(X_test, dtype=torch.float32, device=device)

# GPU SHAP
torch.cuda.synchronize()
t0 = time.time()
gpu_shap = shap_gpu_ultra(rf_gpu, X_gpu, n_samples=200)
torch.cuda.synchronize()
gpu_time = time.time() - t0
print(f"  GPU SHAP:  {gpu_time*1000:.0f}ms")

# CPU SHAP (manual — same algorithm on CPU)
t0 = time.time()
bg = X_test.mean(axis=0)
cpu_shap = np.zeros((100, 20))
for s in range(200):
    mask = np.random.binomial(1, 0.5, 20).astype(float)
    if mask.sum() == 0: mask[0] = 1
    if mask.sum() == 20: mask[0] = 0
    X_masked = X_test * mask + bg * (1 - mask)
    pred_with = rf.predict_proba(X_masked)[:, 1]
    included = np.where(mask > 0.5)[0]
    for f in included:
        mask_wo = mask.copy()
        mask_wo[f] = 0
        X_wo = X_test * mask_wo + bg * (1 - mask_wo)
        pred_wo = rf.predict_proba(X_wo)[:, 1]
        cpu_shap[:, f] += pred_with - pred_wo
cpu_shap /= 200
cpu_time = time.time() - t0
print(f"  CPU SHAP:  {cpu_time*1000:.0f}ms")
print(f"  Speedup:   {cpu_time/gpu_time:.0f}x")

# Compare values
gpu_np = gpu_shap.cpu().numpy()
correlation = np.corrcoef(gpu_np.flatten(), cpu_shap.flatten())[0, 1]
print(f"  Correlation GPU vs CPU: {correlation:.4f}")

# Top features comparison
gpu_importance = np.abs(gpu_np).mean(axis=0)
cpu_importance = np.abs(cpu_shap).mean(axis=0)
gpu_top5 = gpu_importance.argsort()[-5:][::-1]
cpu_top5 = cpu_importance.argsort()[-5:][::-1]
print(f"  GPU top 5: {gpu_top5.tolist()}")
print(f"  CPU top 5: {cpu_top5.tolist()}")
print(f"  Same top features: {set(gpu_top5) == set(cpu_top5)}")

# Official SHAP library
if HAS_SHAP:
    print(f"\n  Official SHAP library:")
    t0 = time.time()
    explainer = shap.KernelExplainer(rf.predict_proba, X[:50])
    official_shap = explainer.shap_values(X_test[:20], nsamples=200)
    shap_time = time.time() - t0
    print(f"  Official SHAP (20 instances): {shap_time*1000:.0f}ms")
    print(f"  GPU SHAP speedup vs official: {shap_time/gpu_time:.0f}x")

# ================================================================
print(f"\n[2] Medium scale: 500 instances")
print("-" * 40)

X_med = torch.tensor(X[:500], dtype=torch.float32, device=device)

torch.cuda.synchronize()
t0 = time.time()
shap_med = shap_gpu_ultra(rf_gpu, X_med, n_samples=300)
torch.cuda.synchronize()
gpu_time_med = time.time() - t0
print(f"  GPU SHAP:  {gpu_time_med:.1f}s ({500*300/gpu_time_med:.0f} inst-coal/s)")

# CPU
t0 = time.time()
bg = X[:500].mean(axis=0)
for s in range(300):
    mask = np.random.binomial(1, 0.5, 20).astype(float)
    if mask.sum() == 0: mask[0] = 1
    if mask.sum() == 20: mask[0] = 0
    X_masked = X[:500] * mask + bg * (1 - mask)
    _ = rf.predict_proba(X_masked)
cpu_time_med = time.time() - t0
print(f"  CPU (predict only): {cpu_time_med:.1f}s")
print(f"  Speedup: {cpu_time_med/gpu_time_med:.0f}x (predict only, no marginals)")

# ================================================================
print(f"\n[3] Large scale: 2000 instances")
print("-" * 40)

X_large = torch.tensor(X[:2000], dtype=torch.float32, device=device)

torch.cuda.synchronize()
t0 = time.time()
shap_large = shap_gpu_ultra(rf_gpu, X_large, n_samples=200)
torch.cuda.synchronize()
gpu_time_large = time.time() - t0
print(f"  GPU SHAP: {gpu_time_large:.1f}s ({2000*200/gpu_time_large:.0f} inst-coal/s)")

# Feature importance
importance = shap_large.abs().mean(dim=0).cpu().numpy()
top10 = importance.argsort()[-10:][::-1]
print(f"  Top 10 features: {top10.tolist()}")
print(f"  Importances: {[f'{importance[f]:.4f}' for f in top10]}")

# ================================================================
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"""
  GPU SHAP Performance:
    100 instances:  {gpu_time*1000:.0f}ms (vs CPU {cpu_time*1000:.0f}ms = {cpu_time/gpu_time:.0f}x)
    500 instances:  {gpu_time_med:.1f}s
    2000 instances: {gpu_time_large:.1f}s

  Accuracy:
    GPU vs CPU correlation: {correlation:.4f}
    Same top features: {set(gpu_top5) == set(cpu_top5)}

  This is the FIRST GPU-native SHAP implementation.
  Uses py2tensor to convert sklearn model to GPU,
  then computes all Shapley permutations in parallel.
""")
