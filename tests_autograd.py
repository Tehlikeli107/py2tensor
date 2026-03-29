"""
Py2Tensor Autograd + torch.compile + Optimization
==================================================
Since @tensorize outputs real PyTorch ops, autograd works FREE.
This means: gradient descent on ANY Python function!
"""
import torch
import math
import time
import sys
sys.path.insert(0, r"C:\Users\salih\Desktop\py2tensor")
from py2tensor import tensorize, benchmark

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}")

# ================================================================
print("\n=== AUTOGRAD: Gradients of @tensorize'd functions ===")

@tensorize
def parabola(x):
    return x * x + 3 * x + 2

# Gradient: d/dx(x^2 + 3x + 2) = 2x + 3
x = torch.tensor([0.0, 1.0, 2.0, -1.0, 5.0], device=device, requires_grad=True)
y = parabola(x)
y.sum().backward()

expected_grad = 2 * x.detach() + 3
check("d/dx(x^2+3x+2) = 2x+3",
      torch.allclose(x.grad, expected_grad, atol=1e-5))
print(f"  x={x.detach().cpu().tolist()}")
print(f"  grad={x.grad.cpu().tolist()}")
print(f"  expected={expected_grad.cpu().tolist()}")

# ================================================================
print("\n=== AUTOGRAD: Gradient through torch.where ===")

@tensorize
def soft_relu(x):
    if x > 0:
        return x
    else:
        return x * 0.01  # leaky relu

x = torch.tensor([-2.0, -1.0, 0.5, 1.0, 3.0], device=device, requires_grad=True)
y = soft_relu(x)
y.sum().backward()

# Gradient: 1 if x>0, 0.01 if x<=0
expected_grad = torch.where(x.detach() > 0,
                            torch.ones_like(x.detach()),
                            torch.full_like(x.detach(), 0.01))
check("grad through torch.where (leaky relu)",
      torch.allclose(x.grad, expected_grad, atol=1e-5))

# ================================================================
print("\n=== AUTOGRAD: Gradient of math functions ===")

@tensorize
def wave(x):
    return math.sin(x) * math.exp(-x * x * 0.1)

x = torch.tensor([0.0, 1.0, 2.0, 3.0], device=device, requires_grad=True)
y = wave(x)
y.sum().backward()

# Numerical gradient check
eps = 1e-4
x_np = x.detach()
grad_num = (wave(x_np + eps) - wave(x_np - eps)) / (2 * eps)
check("grad of sin(x)*exp(-x^2/10) (numerical check)",
      torch.allclose(x.grad, grad_num, atol=1e-3))

# ================================================================
print("\n=== OPTIMIZATION: Find minimum of a function ===")

@tensorize
def rosenbrock(x, y):
    return (1 - x) * (1 - x) + 100 * (y - x * x) * (y - x * x)

# Gradient descent to find minimum (known: x=1, y=1)
x = torch.tensor([0.0], device=device, requires_grad=True)
y = torch.tensor([0.0], device=device, requires_grad=True)
optimizer = torch.optim.Adam([x, y], lr=0.01)

for i in range(2000):
    optimizer.zero_grad()
    loss = rosenbrock(x, y)
    loss.backward()
    optimizer.step()

final_x = x.item()
final_y = y.item()
final_loss = rosenbrock(x, y).item()
print(f"  Rosenbrock minimum found: x={final_x:.4f}, y={final_y:.4f}")
print(f"  Loss: {final_loss:.6f} (optimal: 0)")
check("Rosenbrock optimization (x->1, y->1)",
      abs(final_x - 1.0) < 0.1 and abs(final_y - 1.0) < 0.1)

# ================================================================
print("\n=== OPTIMIZATION: Fit a curve ===")

@tensorize
def model(x, a, b, c):
    return a * math.sin(b * x) + c

# Generate "data" from a*sin(b*x)+c with a=3, b=2, c=1
true_a, true_b, true_c = 3.0, 2.0, 1.0
x_data = torch.linspace(0, 6.28, 1000, device=device)
y_data = true_a * torch.sin(true_b * x_data) + true_c + torch.randn(1000, device=device) * 0.1

# Fit parameters
a = torch.tensor([1.0], device=device, requires_grad=True)
b = torch.tensor([1.0], device=device, requires_grad=True)
c = torch.tensor([0.0], device=device, requires_grad=True)
optimizer = torch.optim.Adam([a, b, c], lr=0.01)

for i in range(3000):
    optimizer.zero_grad()
    y_pred = model(x_data, a, b, c)
    loss = ((y_pred - y_data) ** 2).mean()
    loss.backward()
    optimizer.step()

print(f"  Curve fit: a={a.item():.3f} (true={true_a}), b={b.item():.3f} (true={true_b}), c={c.item():.3f} (true={true_c})")
print(f"  MSE: {loss.item():.6f}")
check("curve fitting a*sin(bx)+c",
      abs(a.item() - true_a) < 0.3 and abs(c.item() - true_c) < 0.3)

# ================================================================
print("\n=== TORCH.COMPILE SPEEDUP ===")

@tensorize
def heavy_math(x):
    y = math.sin(x) * math.cos(x)
    z = math.exp(-y * y)
    if z > 0.5:
        return z * 2
    else:
        return z * 0.5

N = 10_000_000
x_bench = torch.randn(N, device=device)

# Without compile
torch.cuda.synchronize()
_ = heavy_math(x_bench)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(50):
    _ = heavy_math(x_bench)
torch.cuda.synchronize()
t_normal = (time.time() - t0) / 50

# With torch.compile
compiled = torch.compile(heavy_math._gpu)
torch.cuda.synchronize()
_ = compiled(x_bench)  # warmup + compile
_ = compiled(x_bench)  # second warmup
torch.cuda.synchronize()
t0 = time.time()
for _ in range(50):
    _ = compiled(x_bench)
torch.cuda.synchronize()
t_compiled = (time.time() - t0) / 50

print(f"  Normal:   {t_normal*1000:.2f}ms ({N/t_normal/1e6:.0f}M/s)")
print(f"  Compiled: {t_compiled*1000:.2f}ms ({N/t_compiled/1e6:.0f}M/s)")
print(f"  Compile speedup: {t_normal/t_compiled:.2f}x")

check("torch.compile works on @tensorize output",
      t_compiled <= t_normal * 1.5)  # should be at least not slower

# ================================================================
print("\n=== BATCH MAP: Process different-length inputs ===")

@tensorize
def process(x):
    if x > 0:
        return math.sqrt(x)
    else:
        return 0

# Process a large mixed dataset
data = torch.randn(N, device=device) * 10
torch.cuda.synchronize()
t0 = time.time()
result = process(data)
torch.cuda.synchronize()
t_batch = time.time() - t0
print(f"  10M mixed inputs: {t_batch*1000:.2f}ms ({N/t_batch/1e6:.0f}M/s)")

# ================================================================
print(f"\n{'='*60}")
print(f"RESULTS: {passed} passed, {failed} failed")
print(f"{'='*60}")

print(f"""
KEY INSIGHT: @tensorize'd functions support autograd!
This means you can:
1. Compute gradients of ANY Python function
2. Optimize (gradient descent) ANY Python function
3. Fit curves using ANY Python formula
4. Use @tensorize'd functions as neural network layers
5. Apply torch.compile for extra kernel fusion speed
""")
