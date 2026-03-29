"""
KILLER APP: GPU Hyperparameter Search
======================================
Search 10M parameter combinations in seconds.
Normal grid search: hours. Py2Tensor: seconds.

Use case: any function you want to optimize over a parameter space.
"""
import torch
import math
import time
import sys
sys.path.insert(0, r"C:\Users\salih\Desktop\py2tensor")
from py2tensor import tensorize

device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name()}")
print("=" * 60)
print("GPU HYPERPARAMETER SEARCH")
print("=" * 60)

# ================================================================
# EXAMPLE 1: Find optimal trading strategy parameters
# ================================================================
print(f"\n[1] Trading Strategy Optimization")
print("-" * 40)

@tensorize(backend="auto")
def trading_score(fast_window, slow_window, threshold):
    """Score a moving average crossover strategy.
    Higher score = better backtest performance (simplified)."""
    # Simulate: signal quality depends on window ratio and threshold
    ratio = fast_window / (slow_window + 0.001)
    # Sweet spot: ratio around 0.3-0.5, threshold around 0.02
    signal_quality = math.exp(-10 * (ratio - 0.4) * (ratio - 0.4))
    threshold_quality = math.exp(-1000 * (threshold - 0.02) * (threshold - 0.02))
    # Penalty for too-similar windows
    if fast_window > slow_window * 0.8:
        penalty = 0.1
    else:
        penalty = 1.0
    return signal_quality * threshold_quality * penalty

# Search space: 1000 x 1000 x 100 = 100M combinations
# But let's do 10M for speed
N = 10_000_000
fast = torch.rand(N, device=device) * 50 + 5      # 5-55
slow = torch.rand(N, device=device) * 200 + 20    # 20-220
thresh = torch.rand(N, device=device) * 0.1        # 0-0.1

torch.cuda.synchronize()
t0 = time.time()
scores = trading_score(fast, slow, thresh)
torch.cuda.synchronize()
search_time = time.time() - t0

best_idx = scores.argmax().item()
print(f"  Searched {N/1e6:.0f}M combinations in {search_time*1000:.0f}ms")
print(f"  Rate: {N/search_time/1e6:.0f}M evaluations/sec")
print(f"  Best score: {scores[best_idx].item():.6f}")
print(f"  Best params: fast={fast[best_idx].item():.1f}, slow={slow[best_idx].item():.1f}, thresh={thresh[best_idx].item():.4f}")

# CPU comparison
cpu_data = [(float(fast[i]), float(slow[i]), float(thresh[i])) for i in range(10000)]
t0 = time.time()
cpu_scores = [trading_score._original(*d) for d in cpu_data]
cpu_time = time.time() - t0
cpu_rate = len(cpu_data) / cpu_time
gpu_rate = N / search_time
print(f"  CPU: {cpu_rate:.0f}/s | GPU: {gpu_rate:.0f}/s | Speedup: {gpu_rate/cpu_rate:.0f}x")

# ================================================================
# EXAMPLE 2: Physics simulation parameter sweep
# ================================================================
print(f"\n[2] Physics: Optimal Rocket Trajectory")
print("-" * 40)

@tensorize(compile=True)
def rocket_altitude(thrust, angle, burn_time):
    """Max altitude of a rocket with given parameters."""
    g = 9.81
    mass = 1000
    vy = thrust * math.sin(angle * 3.14159 / 180) / mass * burn_time - g * burn_time
    t_peak = vy / g
    h_burn = 0.5 * (thrust * math.sin(angle * 3.14159 / 180) / mass - g) * burn_time * burn_time
    h_coast = vy * t_peak - 0.5 * g * t_peak * t_peak
    h = h_burn + h_coast
    if h < 0:
        return 0
    else:
        return h

N = 5_000_000
thrust = torch.rand(N, device=device) * 50000 + 5000    # 5K-55K N
angle = torch.rand(N, device=device) * 90                 # 0-90 degrees
burn = torch.rand(N, device=device) * 60 + 5              # 5-65 seconds

torch.cuda.synchronize()
t0 = time.time()
altitudes = rocket_altitude(thrust, angle, burn)
torch.cuda.synchronize()
t_rocket = time.time() - t0

best = altitudes.argmax().item()
print(f"  Searched {N/1e6:.0f}M trajectories in {t_rocket*1000:.0f}ms")
print(f"  Best altitude: {altitudes[best].item()/1000:.1f} km")
print(f"  Params: thrust={thrust[best].item():.0f}N, angle={angle[best].item():.1f}deg, burn={burn[best].item():.1f}s")

# ================================================================
# EXAMPLE 3: ML Loss landscape exploration
# ================================================================
print(f"\n[3] ML: Loss Landscape Exploration")
print("-" * 40)

@tensorize(backend="auto")
def loss_landscape(w1, w2, b):
    """Compute loss for a simple model at given weights."""
    # Simulated loss surface with multiple local minima
    pred = math.sin(w1 * 3) * math.cos(w2 * 2) + b
    target = 0.5
    mse = (pred - target) * (pred - target)
    # Regularization
    reg = 0.01 * (w1 * w1 + w2 * w2 + b * b)
    return mse + reg

N = 10_000_000
w1 = torch.rand(N, device=device) * 6 - 3    # -3 to 3
w2 = torch.rand(N, device=device) * 6 - 3
b = torch.rand(N, device=device) * 2 - 1      # -1 to 1

torch.cuda.synchronize()
t0 = time.time()
losses = loss_landscape(w1, w2, b)
torch.cuda.synchronize()
t_loss = time.time() - t0

best = losses.argmin().item()
print(f"  Explored {N/1e6:.0f}M weight combinations in {t_loss*1000:.0f}ms")
print(f"  Best loss: {losses[best].item():.6f}")
print(f"  Best weights: w1={w1[best].item():.4f}, w2={w2[best].item():.4f}, b={b[best].item():.4f}")
print(f"  Rate: {N/t_loss/1e6:.0f}M evaluations/sec")

# ================================================================
# EXAMPLE 4: Drug dosage optimization
# ================================================================
print(f"\n[4] Pharma: Drug Dosage Optimization")
print("-" * 40)

@tensorize(compile=True)
def drug_efficacy(dose, frequency, body_weight):
    """Model drug concentration and efficacy."""
    concentration = dose * frequency / (body_weight * 0.1 + 1)
    steady_state = concentration * 6.0 / (0.693 * 24 / frequency)
    # Therapeutic window
    if steady_state < 10:
        efficacy = steady_state / 10
    else:
        efficacy = 1.0
    if steady_state > 50:
        efficacy = 50 / steady_state
    else:
        efficacy = efficacy
    toxicity = math.exp(-0.01 * steady_state)
    return efficacy * (1 - 0.3 * (1 - toxicity))

N = 5_000_000
dose = torch.rand(N, device=device) * 500 + 10       # 10-510 mg
freq = torch.rand(N, device=device) * 4 + 1           # 1-5 times/day
weight = torch.rand(N, device=device) * 80 + 40       # 40-120 kg

torch.cuda.synchronize()
t0 = time.time()
efficacies = drug_efficacy(dose, freq, weight)
torch.cuda.synchronize()
t_drug = time.time() - t0

best = efficacies.argmax().item()
print(f"  Tested {N/1e6:.0f}M dosing regimens in {t_drug*1000:.0f}ms")
print(f"  Best efficacy: {efficacies[best].item():.4f}")
print(f"  Optimal: dose={dose[best].item():.0f}mg, freq={freq[best].item():.1f}x/day, weight={weight[best].item():.0f}kg")

# ================================================================
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
total_evals = 30_000_000
total_time = search_time + t_rocket + t_loss + t_drug
print(f"""
  4 optimization problems, {total_evals/1e6:.0f}M total evaluations
  Total time: {total_time:.1f}s
  Average: {total_evals/total_time/1e6:.0f}M evaluations/sec

  Same functions on CPU would take {total_evals/cpu_rate:.0f} seconds ({total_evals/cpu_rate/60:.0f} minutes).
  GPU speedup: {total_evals/total_time/cpu_rate:.0f}x

  All you need: write a Python function + @tensorize + search.
""")
