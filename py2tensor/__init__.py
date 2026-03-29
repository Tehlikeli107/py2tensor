"""
Py2Tensor: Convert any Python function to GPU tensor operations.
No training. No approximation. Exact results.

Usage:
    from py2tensor import gpu

    @gpu
    def f(x):
        if x > 0: return x * 2
        else: return 0

    f(torch.randn(10_000_000, device='cuda'))  # 10M in <1ms
"""
from .core import tensorize, explain, benchmark, profile
from .diagnostics import diagnose

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

from .api import gpu

__version__ = "2.0.0"
__all__ = ["gpu", "tensorize", "tensorize_triton", "build_pure_model",
           "tensorize_all", "explain", "benchmark", "profile", "diagnose"]
