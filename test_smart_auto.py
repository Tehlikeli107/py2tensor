"""Test @gpu.auto smart backend selection."""
import torch, math, sys
sys.path.insert(0, r"C:\Users\salih\Desktop\py2tensor")
from py2tensor import gpu

device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name()}\n")

# Simple -> should pick compile
try:
    @gpu.auto
    def simple(x):
        return math.sin(x) * math.exp(-x)
    print(f"simple: OK, test={simple(torch.tensor([1.0], device=device)).item():.4f}")
except Exception as e:
    print(f"simple: ERROR {e}")

# Loop -> should pick triton
try:
    @gpu.auto
    def newton(x):
        g = x / 2
        for i in range(10):
            g = (g + x / g) / 2
        return g
    print(f"newton: OK, sqrt(9)={newton(torch.tensor([9.0], device=device)).item():.4f}")
except Exception as e:
    print(f"newton: ERROR {e}")

# Dict -> should pick tensorize_all
try:
    @gpu.auto
    def pricing(x):
        rates = {0: 0, 1: 10, 2: 25, 3: 50}
        if x > 50:
            return 50
        else:
            return 0
    vals = torch.tensor([10, 60], dtype=torch.float32, device=device)
    print(f"pricing: OK, test={pricing(vals).tolist()}")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"pricing: ERROR {e}")

print("\nDone.")
