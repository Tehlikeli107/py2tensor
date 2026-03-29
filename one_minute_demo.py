"""
Py2Tensor: 1-Minute Demo
=========================
Copy this file, run it, see the magic.
Requires: pip install torch (with CUDA)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from py2tensor import tensorize, explain, benchmark
import torch, math

print("Py2Tensor: Convert Python functions to GPU. No training. Exact.")
print(f"GPU: {torch.cuda.get_device_name()}\n")

# === 1. Write a normal Python function ===
@tensorize
def my_function(x):
    if x > 0:
        return math.sin(x) * math.exp(-x * 0.1)
    else:
        return 0

# === 2. See what it generates ===
print("YOUR CODE:")
print("  def my_function(x):")
print("      if x > 0: return math.sin(x) * math.exp(-x * 0.1)")
print("      else: return 0")
print()
print("GENERATED GPU CODE:")
explain(my_function)

# === 3. Benchmark: CPU vs GPU ===
print()
benchmark(my_function, 1.0)

# === 4. Works with autograd! ===
print("\nAUTOGRAD: compute gradient of ANY function")
x = torch.tensor([1.0, 2.0, 3.0], device='cuda', requires_grad=True)
y = my_function(x)
y.sum().backward()
print(f"  x = {x.data.tolist()}")
print(f"  dy/dx = {x.grad.tolist()}")

# === 5. Optimize ANY function with gradient descent ===
print("\nOPTIMIZE: find maximum of my_function")
x = torch.tensor([0.5], device='cuda', requires_grad=True)
opt = torch.optim.Adam([x], lr=0.1)
for _ in range(200):
    opt.zero_grad()
    loss = -my_function(x)  # negate for maximization
    loss.backward()
    opt.step()
print(f"  Maximum at x = {x.item():.4f}, f(x) = {my_function(x).item():.4f}")

print(f"\nDone. github.com/Tehlikeli107/py2tensor")
