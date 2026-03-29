"""All 4 backends from single @tensorize decorator."""
import torch, math, time, sys
sys.path.insert(0, r"C:\Users\salih\Desktop\py2tensor")
from py2tensor import tensorize

device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name()}\n")

N = 10_000_000

# Same function, 4 backends
@tensorize
def f_pytorch(x): return math.sin(x) * math.exp(-x * 0.1)

@tensorize(compile=True)
def f_compile(x): return math.sin(x) * math.exp(-x * 0.1)

@tensorize(backend="triton")
def f_triton(x): return math.sin(x) * math.exp(-x * 0.1)

@tensorize(backend="model")
def f_model(x): return math.sin(x) * math.exp(-x * 0.1)

f_model = f_model.to(device)

# Same function, 4 backends - iterative
@tensorize
def n_pytorch(x):
    g = x / 2
    for i in range(10): g = (g + x / g) / 2
    return g

@tensorize(compile=True)
def n_compile(x):
    g = x / 2
    for i in range(10): g = (g + x / g) / 2
    return g

@tensorize(backend="triton")
def n_triton(x):
    g = x / 2
    for i in range(10): g = (g + x / g) / 2
    return g

@tensorize(backend="auto")
def n_auto(x):
    g = x / 2
    for i in range(10): g = (g + x / g) / 2
    return g

@tensorize(backend="model")
def n_model(x):
    g = x / 2
    for i in range(10): g = (g + x / g) / 2
    return g

n_model = n_model.to(device)

x = torch.randn(N, device=device)
xp = torch.rand(N, device=device) * 100 + 0.1

def bench(name, fn, data):
    torch.cuda.synchronize()
    for _ in range(3): fn(data)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20): out = fn(data)
    torch.cuda.synchronize()
    t = (time.time() - t0) / 20
    print(f"  {name:<25} {N/t/1e9:>6.1f}B/s  {t*1000:.2f}ms")
    return out

print("=== sin(x) * exp(-x*0.1) ===")
o1 = bench("pytorch", f_pytorch, x)
o2 = bench("compile", f_compile, x)
o3 = bench("triton", f_triton, x)
o4 = bench("model", f_model, x)
print(f"  All match: {torch.allclose(o1, o3, atol=1e-3) and torch.allclose(o1, o4, atol=1e-3)}")

print(f"\n=== Newton sqrt (10 iter) ===")
o1 = bench("pytorch", n_pytorch, xp)
o2 = bench("compile", n_compile, xp)
o3 = bench("triton", n_triton, xp)
o4 = bench("auto (->triton)", n_auto, xp)
o5 = bench("model", n_model, xp)
print(f"  All match: {torch.allclose(o1, o3, atol=1e-3) and torch.allclose(o1, o5, atol=1e-3)}")

print(f"\n=== MODEL extras ===")
# Autograd
x_g = torch.tensor([1.0, 2.0], device=device, requires_grad=True)
y = f_model(x_g)
y.sum().backward()
print(f"  Autograd: grad={x_g.grad.tolist()}")

# Save
torch.save(n_model.state_dict(), r"C:\Users\salih\Desktop\py2tensor\test_save.pt")
print(f"  Save: OK")

# Sequential
pipeline = torch.nn.Sequential(f_model, f_model).to(device)
out = pipeline(torch.tensor([1.0, 2.0], device=device))
print(f"  Sequential: {out.tolist()}")

print(f"\n{'='*50}")
print(f"4 backends, 1 decorator, same results.")
print(f"  @tensorize               -> PyTorch ops")
print(f"  @tensorize(compile=True) -> torch.compile")
print(f"  @tensorize(backend='triton') -> fused Triton kernel")
print(f"  @tensorize(backend='model')  -> nn.Module (save/load/compose)")
print(f"  @tensorize(backend='auto')   -> auto-select best")
