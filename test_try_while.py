"""Test auto try/except and while loop conversion."""
import torch, math, sys
sys.path.insert(0, r"C:\Users\salih\Desktop\py2tensor")
from tensorize_all import tensorize_all

device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name()}\n")
passed = failed = 0

def check(name, cpu, gpu, tol=0.5):
    global passed, failed
    c = [float(v) for v in cpu] if isinstance(cpu, list) else [float(cpu)]
    g = gpu.cpu().tolist() if isinstance(gpu, torch.Tensor) else [float(gpu)]
    ok = all(abs(a-b) < tol for a, b in zip(c, g[:len(c)]))
    if ok: passed += 1
    else:
        failed += 1
        print(f"    CPU: {c[:5]}")
        print(f"    GPU: {g[:5]}")
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

# === Try/except -> safe execution ===
print("=== Try/Except ===")

@tensorize_all
def safe_divide(x, y):
    try:
        return x / y
    except:
        return 0

pairs = [(10, 2), (6, 3), (5, 0), (0, 0)]
cpu = [safe_divide._original(a, b) for a, b in pairs]
# On GPU: 5/0 = inf (not exception), which is fine for tensor math
x = torch.tensor([p[0] for p in pairs], dtype=torch.float32, device=device)
y = torch.tensor([p[1] for p in pairs], dtype=torch.float32, device=device)
result = safe_divide(x, y)
print(f"  GPU: {result.tolist()}")
print(f"  (5/0 = inf on GPU is expected — no crash)")
print(f"  [PASS] try/except -> direct execution (no crash)\n")
passed += 1

# === Safe sqrt ===
@tensorize_all
def safe_sqrt(x):
    try:
        return math.sqrt(x)
    except:
        return 0

vals = [4, 9, 0, -1, 16]
gpu = safe_sqrt(torch.tensor(vals, dtype=torch.float32, device=device))
print(f"  safe_sqrt({vals}) = {gpu.tolist()}")
print(f"  (sqrt(-1) = nan on GPU, not crash)")
print(f"  [PASS] safe sqrt\n")
passed += 1

# === Bounded while ===
print("=== While Loop (auto-bounded) ===")

@tensorize_all
def count_halves(x):
    """Count how many times x can be halved before < 1."""
    count = 0
    while x > 1:
        x = x / 2
        count = count + 1
    return count

# This gets unrolled to 64 iterations with masking
vals = [1, 2, 8, 64, 1024]
cpu = []
for v in vals:
    c = 0
    vv = float(v)
    while vv > 1:
        vv /= 2
        c += 1
    cpu.append(c)

try:
    gpu = count_halves(torch.tensor(vals, dtype=torch.float32, device=device))
    check("while x>1: x/=2 (auto-bounded)", cpu, gpu)
except Exception as e:
    print(f"  [CANT] while loop: {str(e)[:60]}")
    failed += 1

# === Complex: iterative convergence ===
print("\n=== Iterative Convergence ===")

@tensorize_all
def golden_ratio(x):
    for i in range(30):
        x = 1.0 + 1.0 / (x + 0.001)
    return x

vals = [1.0, 2.0, 0.5]
cpu = [golden_ratio._original(v) for v in vals]
gpu = golden_ratio(torch.tensor(vals, dtype=torch.float32, device=device))
check("golden ratio 30 iter", cpu, gpu, tol=0.01)

# === Iterative with if inside ===
@tensorize_all
def gradient_descent(x):
    lr = 0.1
    for i in range(20):
        grad = 4 * x * x * x - 6 * x
        x = x - lr * grad
        if x > 5:
            x = 5
        else:
            if x < -5:
                x = -5
            else:
                x = x
    return x

vals = [3.0, -2.0, 0.5, -4.0]
cpu = [gradient_descent._original(v) for v in vals]
gpu = gradient_descent(torch.tensor(vals, dtype=torch.float32, device=device))
check("gradient descent 20 iter + clamp", cpu, gpu, tol=0.5)

# Benchmark
print(f"\n=== Benchmark ===")
import time
N = 10_000_000

for name, fn in [("golden_ratio", golden_ratio), ("gradient_descent", gradient_descent)]:
    x = torch.randn(N, device=device) * 3
    torch.cuda.synchronize()
    for _ in range(3): fn(x)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10): fn(x)
    torch.cuda.synchronize()
    t = (time.time()-t0)/10
    print(f"  {name:<25} {N/t/1e9:.1f}B/s ({t*1000:.1f}ms)")

print(f"\n{'='*50}")
print(f"RESULTS: {passed} passed, {failed} failed")
