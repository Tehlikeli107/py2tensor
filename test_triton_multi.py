"""Test Triton multi-statement if/else + comparison chains."""
import torch, math, time, sys
sys.path.insert(0, r"C:\Users\salih\Desktop\py2tensor")
from py2tensor import tensorize
from triton_backend import tensorize_triton

device = torch.device("cuda")
N = 10_000_000

# Triton: rocket with multi-assign + return
@tensorize_triton
def rocket_tr(thrust, angle, burn):
    g = 9.81
    mass = 1000
    vy = thrust * math.sin(angle * 3.14159 / 180) / mass * burn - g * burn
    if vy > 0:
        t_peak = vy / g
        h = 0.5 * vy * t_peak
        return h
    else:
        return 0

# PyTorch version for comparison
@tensorize
def rocket_pt(thrust, angle, burn):
    g = 9.81
    mass = 1000
    vy = thrust * math.sin(angle * 3.14159 / 180) / mass * burn - g * burn
    if vy > 0:
        t_peak = vy / g
        h = 0.5 * vy * t_peak
        return h
    else:
        return 0

thrust = torch.rand(N, device=device) * 50000 + 5000
angle = torch.rand(N, device=device) * 90
burn = torch.rand(N, device=device) * 60 + 5

print(f"GPU: {torch.cuda.get_device_name()}\n")

# Test correctness
out_pt = rocket_pt(thrust, angle, burn)
out_tr = rocket_tr(thrust, angle, burn)
print(f"Match: {torch.allclose(out_pt, out_tr, atol=1e-1)}")
print(f"PT sample: {out_pt[:5].tolist()}")
print(f"TR sample: {out_tr[:5].tolist()}")

# Benchmark
torch.cuda.synchronize()
for _ in range(3): rocket_pt(thrust, angle, burn)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(20): rocket_pt(thrust, angle, burn)
torch.cuda.synchronize()
t_pt = (time.time() - t0) / 20

torch.cuda.synchronize()
for _ in range(3): rocket_tr(thrust, angle, burn)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(20): rocket_tr(thrust, angle, burn)
torch.cuda.synchronize()
t_tr = (time.time() - t0) / 20

print(f"\nPyTorch: {N/t_pt/1e9:.1f}B/s ({t_pt*1000:.1f}ms)")
print(f"Triton:  {N/t_tr/1e9:.1f}B/s ({t_tr*1000:.1f}ms)")
print(f"Speedup: {t_pt/t_tr:.1f}x")

# Comparison chain test
@tensorize
def clamp_chain(x):
    if 0 < x:
        if x < 10:
            return x
        else:
            return 10
    else:
        return 0

vals = [-5, 0, 5, 10, 15]
cpu_out = [clamp_chain._original(v) for v in vals]
gpu_out = clamp_chain(torch.tensor(vals, dtype=torch.float32, device=device))
print(f"\nChain clamp: CPU={[float(v) for v in cpu_out]}, GPU={gpu_out.cpu().tolist()}")
print(f"Match: {torch.allclose(torch.tensor(cpu_out, dtype=torch.float32), gpu_out.cpu(), atol=1e-3)}")

print(f"\nTriton kernel source:")
print(rocket_tr._triton_source)
