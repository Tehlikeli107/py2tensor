"""Test auto backend selection."""
import torch, math, time, sys
sys.path.insert(0, r"C:\Users\salih\Desktop\py2tensor")
from py2tensor import tensorize

device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name()}\n")

# No loop -> PyTorch backend
@tensorize(backend="auto")
def simple(x):
    return math.sin(x) * math.exp(-x)

# Has loop -> Triton backend (auto-detected)
@tensorize(backend="auto")
def newton(x):
    g = x / 2
    for i in range(10):
        g = (g + x / g) / 2
    return g

# Force Triton
@tensorize(backend="triton")
def forced_triton(x):
    return x * x + 1

N = 10_000_000
x = torch.rand(N, device=device) * 100 + 0.1

# Test simple (should use PyTorch)
print("simple(x) = sin(x)*exp(-x):")
print(f"  Backend: {'Triton' if hasattr(simple, '_triton_source') else 'PyTorch'}")
torch.cuda.synchronize()
for _ in range(3): simple(x)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(30): simple(x)
torch.cuda.synchronize()
t = (time.time() - t0) / 30
print(f"  Speed: {N/t/1e9:.1f}B/s")

# Test newton (should auto-select Triton)
print(f"\nnewton(x) = sqrt via 10 iterations:")
print(f"  Backend: {'Triton' if hasattr(newton, '_triton_source') else 'PyTorch'}")
torch.cuda.synchronize()
for _ in range(3): newton(x)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(30): newton(x)
torch.cuda.synchronize()
t = (time.time() - t0) / 30
print(f"  Speed: {N/t/1e9:.1f}B/s")

# Verify
expected = torch.sqrt(x)
result = newton(x)
print(f"  Correct: {torch.allclose(result, expected, atol=1e-3)}")

# Force triton
print(f"\nforced_triton(x) = x^2 + 1:")
print(f"  Backend: {'Triton' if hasattr(forced_triton, '_triton_source') else 'PyTorch'}")
torch.cuda.synchronize()
for _ in range(3): forced_triton(x)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(30): forced_triton(x)
torch.cuda.synchronize()
t = (time.time() - t0) / 30
print(f"  Speed: {N/t/1e9:.1f}B/s")

print(f"\nAuto-backend: loops->Triton, simple->PyTorch. Best of both worlds.")
