"""
Py2Tensor Unified API — Single import, everything works.

Usage:
    from api import gpu

    @gpu
    def my_func(x):
        if x > 0: return x * 2
        else: return 0

    # Scalar: my_func(5) -> 10
    # GPU:    my_func(torch.randn(10M, device='cuda')) -> 10M results in 1ms
    # NumPy:  my_func(np.array([1,2,3])) -> auto GPU
    # Pandas: my_func(df['column']) -> auto GPU

    # Advanced:
    @gpu.fast        # torch.compile (30B/s)
    @gpu.triton      # Triton fused kernel (29B/s, best for loops)
    @gpu.model       # nn.Module (save/load/compose)
    @gpu.all         # auto dict/list/string conversion
"""
from .core import tensorize, explain, benchmark, profile

try:
    from .triton import tensorize_triton
except ImportError:
    tensorize_triton = None

try:
    from .pure import build_pure_model
except ImportError:
    build_pure_model = None

try:
    from .all import tensorize_all
except ImportError:
    tensorize_all = None


class GPU:
    """Unified GPU decorator with multiple backends."""

    def __call__(self, fn):
        """Default: @gpu -> @tensorize (PyTorch ops, most compatible)."""
        return tensorize(fn)

    @staticmethod
    def fast(fn):
        """@gpu.fast -> @tensorize(compile=True) for 5-30x extra speed."""
        return tensorize(fn, compile=True)

    @staticmethod
    def triton(fn):
        """@gpu.triton -> Triton fused kernel, best for iterative algorithms."""
        if tensorize_triton:
            return tensorize_triton(fn)
        return tensorize(fn, compile=True)  # fallback

    @staticmethod
    def model(fn):
        """@gpu.model -> nn.Module, supports save/load/compose/Sequential."""
        if build_pure_model:
            return build_pure_model(fn)
        return tensorize(fn)  # fallback

    @staticmethod
    def all(fn):
        """@gpu.all -> auto-converts dict, list, string, try/except."""
        if tensorize_all:
            return tensorize_all(fn)
        return tensorize(fn)  # fallback

    @staticmethod
    def auto(fn):
        """@gpu.auto -> selects best backend based on function analysis."""
        from .diagnostics import FunctionAnalyzer
        import ast, inspect, textwrap

        source = inspect.getsource(fn)
        source = textwrap.dedent(source)
        clean = []
        skip = True
        for line in source.split('\n'):
            if skip and (line.strip().startswith('@') or line.strip() == ''):
                continue
            skip = False
            clean.append(line)

        try:
            tree = ast.parse('\n'.join(clean))
            analyzer = FunctionAnalyzer()
            analyzer.visit(tree)

            has_dict = analyzer.has_dict_literal
            has_list = analyzer.has_list_literal
            has_try = analyzer.has_try
            has_loop = analyzer.has_for and analyzer.for_count > 0
            has_while = analyzer.has_while

            # Route to best backend
            if has_dict or has_list or has_try or has_while:
                if tensorize_all:
                    return tensorize_all(fn)
            if has_loop:
                if tensorize_triton:
                    return tensorize_triton(fn)
            return tensorize(fn, compile=True)  # default: compile for speed
        except Exception:
            return tensorize(fn)  # fallback


gpu = GPU()


# ================================================================
if __name__ == '__main__':
    import torch
    import math
    import time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"GPU: {torch.cuda.get_device_name() if device.type == 'cuda' else 'CPU'}")
    print("=" * 60)
    print("Py2Tensor Unified API Demo")
    print("=" * 60)

    # Basic
    @gpu
    def sigmoid(x):
        return 1.0 / (1.0 + math.exp(-x))

    print(f"\n@gpu sigmoid(2.0) = {sigmoid(2.0):.4f}")
    x = torch.randn(10_000_000, device=device)
    benchmark(sigmoid, 1.0)

    # Fast
    @gpu.fast
    def gaussian(x):
        return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

    print(f"\n@gpu.fast gaussian(0) = {gaussian(0.0):.4f}")
    benchmark(gaussian, 0.0)

    # Triton
    @gpu.triton
    def newton(x):
        g = x / 2
        for i in range(10):
            g = (g + x / g) / 2
        return g

    print(f"\n@gpu.triton newton(9) = {newton(torch.tensor([9.0], device=device)).item():.4f}")
    xp = torch.rand(10_000_000, device=device) * 100 + 0.1
    torch.cuda.synchronize()
    for _ in range(3): newton(xp)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20): newton(xp)
    torch.cuda.synchronize()
    t = (time.time() - t0) / 20
    print(f"  Triton Newton: {10_000_000/t/1e9:.1f}B/s")

    # Model
    @gpu.model
    def simple(x):
        return x * x + 1

    model = simple.to(device)
    x = torch.tensor([1.0, 2.0, 3.0], device=device, requires_grad=True)
    y = model(x)
    y.sum().backward()
    print(f"\n@gpu.model: output={y.tolist()}, grad={x.grad.tolist()}")
    torch.save(model.state_dict(), r'C:\Users\salih\Desktop\py2tensor\api_model.pt')
    print(f"  Saved model to api_model.pt")

    # All
    @gpu.all
    def pricing(x):
        tiers = {0: 0, 1: 9.99, 2: 29.99, 3: 99.99}
        if x > 100:
            tier = 3
        else:
            if x > 50:
                tier = 2
            else:
                if x > 20:
                    tier = 1
                else:
                    tier = 0
        return tiers[tier]

    print(f"\n@gpu.all pricing(60) = {pricing(60.0):.2f}")
    vals = torch.tensor([10, 30, 60, 150], dtype=torch.float32, device=device)
    print(f"  pricing([10,30,60,150]) = {pricing(vals).tolist()}")

    print(f"\n{'='*60}")
    print("API: from api import gpu")
    print("  @gpu         -> basic (compatible)")
    print("  @gpu.fast    -> torch.compile (5-30x faster)")
    print("  @gpu.triton  -> fused kernel (best for loops)")
    print("  @gpu.model   -> nn.Module (save/load/compose)")
    print("  @gpu.all     -> auto dict/list/string/try")
