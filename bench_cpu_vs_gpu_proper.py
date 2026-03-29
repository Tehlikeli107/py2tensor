"""Proper CPU vs GPU comparison with correct timing."""
import torch, math, time, sys
sys.path.insert(0, r"C:\Users\salih\Desktop\py2tensor")
from pure_model import build_pure_model

device = torch.device("cuda")
N = 10_000_000

@build_pure_model
def decision_tree(age, income, score):
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

@build_pure_model
def fraud_score(amount, frequency, country_risk, hour):
    score = 0
    if amount > 10000:
        score = score + 30
    else:
        score = score + 0
    if amount > 50000:
        score = score + 20
    else:
        score = score + 0
    if frequency > 10:
        score = score + 25
    else:
        score = score + 0
    if country_risk > 7:
        score = score + 20
    else:
        score = score + 0
    if hour > 22:
        score = score + 15
    else:
        score = score + 0
    return score

@build_pure_model
def piecewise(kwh):
    if kwh > 1000:
        return 1000 * 0.05 + (kwh - 1000) * 0.15
    else:
        if kwh > 500:
            return 500 * 0.03 + (kwh - 500) * 0.05
        else:
            if kwh > 200:
                return 200 * 0.02 + (kwh - 200) * 0.03
            else:
                return kwh * 0.01

print(f"GPU: {torch.cuda.get_device_name()}")
print("=" * 60)

for name, fn, n_args in [
    ("Decision Tree", decision_tree, 3),
    ("Fraud Rules", fraud_score, 4),
    ("Piecewise Tariff", piecewise, 1),
]:
    fn_gpu = torch.compile(fn.to(device))

    # CPU: loop over N elements
    N_cpu = 100000
    if n_args == 1:
        cpu_data = [float(i) for i in range(N_cpu)]
    elif n_args == 3:
        cpu_data = [(float(i % 50 + 20), float(i % 100000 + 20000), float(i % 400 + 400)) for i in range(N_cpu)]
    else:
        cpu_data = [(float(i % 100000), float(i % 20), float(i % 10), float(i % 24)) for i in range(N_cpu)]

    t0 = time.time()
    if n_args == 1:
        for v in cpu_data: fn._original(v)
    elif n_args == 3:
        for a, b, c in cpu_data: fn._original(a, b, c)
    else:
        for a, b, c, d in cpu_data: fn._original(a, b, c, d)
    cpu_time = time.time() - t0
    cpu_rate = N_cpu / cpu_time

    # GPU: batch N elements
    if n_args == 1:
        gpu_args = [torch.rand(N, device=device) * 2000]
    elif n_args == 3:
        gpu_args = [torch.rand(N, device=device)*40+20, torch.rand(N, device=device)*150000+20000, torch.rand(N, device=device)*400+400]
    else:
        gpu_args = [torch.rand(N, device=device)*100000, torch.rand(N, device=device)*20, torch.rand(N, device=device)*10, torch.rand(N, device=device)*24]

    torch.cuda.synchronize()
    for _ in range(3): fn_gpu(*gpu_args)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(30): fn_gpu(*gpu_args)
    torch.cuda.synchronize()
    gpu_time = (time.time() - t0) / 30
    gpu_rate = N / gpu_time

    speedup = gpu_rate / cpu_rate
    print(f"\n{name}:")
    print(f"  CPU: {N_cpu} elems in {cpu_time*1000:.0f}ms = {cpu_rate/1e6:.1f}M/s")
    print(f"  GPU: {N/1e6:.0f}M elems in {gpu_time*1000:.1f}ms = {gpu_rate/1e6:.0f}M/s")
    print(f"  SPEEDUP: {speedup:.0f}x")
