"""5-Backend Comparison."""
import torch, math, time, sys
sys.path.insert(0, r"C:\Users\salih\Desktop\py2tensor")
from py2tensor import tensorize
from triton_backend import tensorize_triton
from pure_model import build_pure_model

device = torch.device("cuda")
N = 10_000_000
W, R = 5, 50

def b(fn, x):
    torch.cuda.synchronize()
    for _ in range(W): fn(x)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(R): fn(x)
    torch.cuda.synchronize()
    return (time.time()-t0)/R

print(f"GPU: {torch.cuda.get_device_name()}")
print("=" * 70)
print("5-BACKEND COMPARISON (10M elements)")
print("=" * 70)

x = torch.randn(N, device=device)
xp = torch.rand(N, device=device) * 100 + 0.1

# === Gaussian PDF ===
print("\n--- Gaussian PDF ---")
@tensorize
def g1(x): return math.exp(-0.5*x*x)/math.sqrt(2*math.pi)
@tensorize(compile=True)
def g2(x): return math.exp(-0.5*x*x)/math.sqrt(2*math.pi)
@tensorize_triton
def g3(x): return math.exp(-0.5*x*x)/math.sqrt(2*math.pi)
@build_pure_model
def g4(x): return math.exp(-0.5*x*x)/math.sqrt(2*math.pi)
g4 = g4.to(device)
g5 = torch.compile(g4)

for name, fn in [("PT",g1),("CO",g2),("TR",g3),("PM",g4),("PM+C",g5)]:
    try:
        t = b(fn, x)
        print(f"  {name:<6} {N/t/1e9:>6.1f}B/s")
    except Exception as e:
        print(f"  {name:<6} ERROR: {str(e)[:60]}")

# === If/else ===
print("\n--- if/else branch ---")
@tensorize
def b1(x):
    if x > 0: return x*x
    else: return -x*0.5
@tensorize(compile=True)
def b2(x):
    if x > 0: return x*x
    else: return -x*0.5
@tensorize_triton
def b3(x):
    if x > 0: return x*x
    else: return -x*0.5
@build_pure_model
def b4(x):
    if x > 0: return x*x
    else: return -x*0.5
b4 = b4.to(device)
b5 = torch.compile(b4)

for name, fn in [("PT",b1),("CO",b2),("TR",b3),("PM",b4),("PM+C",b5)]:
    try:
        t = b(fn, x)
        print(f"  {name:<6} {N/t/1e9:>6.1f}B/s")
    except Exception as e:
        print(f"  {name:<6} ERROR: {str(e)[:60]}")

# === Newton sqrt ===
print("\n--- Newton sqrt (10 iter) ---")
@tensorize
def n1(x):
    g=x/2
    for i in range(10): g=(g+x/g)/2
    return g
@tensorize(compile=True)
def n2(x):
    g=x/2
    for i in range(10): g=(g+x/g)/2
    return g
@tensorize_triton
def n3(x):
    g=x/2
    for i in range(10): g=(g+x/g)/2
    return g
@build_pure_model
def n4(x):
    g=x/2
    for i in range(10): g=(g+x/g)/2
    return g
n4 = n4.to(device)
n5 = torch.compile(n4)

for name, fn in [("PT",n1),("CO",n2),("TR",n3),("PM",n4),("PM+C",n5)]:
    try:
        t = b(fn, xp)
        print(f"  {name:<6} {N/t/1e9:>6.1f}B/s")
    except Exception as e:
        print(f"  {name:<6} ERROR: {str(e)[:60]}")

print(f"\n{'='*70}")
print("PT=PyTorch | CO=compile | TR=Triton | PM=PureModel | PM+C=PureModel+compile")
print("Pure Model = YOUR IDEA: function as nn.Module, zero Python in forward")
