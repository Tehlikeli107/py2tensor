import sys
print("Starting...", flush=True)
try:
    import triton
    print(f"Triton: {triton.__version__}", flush=True)
    import triton.language as tl
    print("Triton OK", flush=True)
except Exception as e:
    print(f"Triton import error: {e}", flush=True)
    sys.exit(1)

try:
    from triton_backend import tensorize_triton
    print("Backend imported", flush=True)
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)

import torch, math, time
print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

try:
    @tensorize_triton
    def test_fn(x):
        return math.sin(x) * math.exp(-x * 0.1)
    print("Kernel generated", flush=True)
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
print(f"Source:\n{test_fn._triton_source}", flush=True)

x = torch.randn(1000, device='cuda')
out = test_fn(x)
print(f"Output shape: {out.shape}, mean: {out.mean().item():.4f}", flush=True)

# Benchmark
N = 10_000_000
x = torch.randn(N, device='cuda')
torch.cuda.synchronize()
for _ in range(3): test_fn(x)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(50): test_fn(x)
torch.cuda.synchronize()
t = (time.time() - t0) / 50
triton_rate = N/t/1e9
print(f"Triton: {triton_rate:.1f}B/s ({t*1000:.2f}ms for {N/1e6:.0f}M)", flush=True)

# Compare with PyTorch ops
from py2tensor import tensorize

@tensorize
def test_pt(x):
    return math.sin(x) * math.exp(-x * 0.1)

@tensorize(compile=True)
def test_co(x):
    return math.sin(x) * math.exp(-x * 0.1)

# PyTorch ops
torch.cuda.synchronize()
for _ in range(3): test_pt(x)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(50): test_pt(x)
torch.cuda.synchronize()
t_pt = (time.time() - t0) / 50
pt_rate = N/t_pt/1e9

# torch.compile
torch.cuda.synchronize()
for _ in range(3): test_co(x)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(50): test_co(x)
torch.cuda.synchronize()
t_co = (time.time() - t0) / 50
co_rate = N/t_co/1e9

print(f"\n{'='*60}", flush=True)
print(f"3-WAY COMPARISON (10M elements, sin(x)*exp(-x*0.1)):", flush=True)
print(f"{'='*60}", flush=True)
print(f"  @tensorize (PyTorch ops): {pt_rate:.1f}B/s  ({t_pt*1000:.2f}ms)", flush=True)
print(f"  @tensorize(compile):      {co_rate:.1f}B/s  ({t_co*1000:.2f}ms)", flush=True)
print(f"  @tensorize_triton:        {triton_rate:.1f}B/s  ({t*1000:.2f}ms)", flush=True)
print(f"", flush=True)
print(f"  Triton vs PyTorch ops: {t_pt/t:.2f}x FASTER", flush=True)
print(f"  Triton vs compile:     {t_co/t:.2f}x {'FASTER' if t < t_co else 'same'}", flush=True)
print(f"", flush=True)
print(f"  Match: {torch.allclose(test_pt(x), test_fn(x), atol=1e-4)}", flush=True)
