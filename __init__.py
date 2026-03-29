from .py2tensor import tensorize, explain, benchmark
try:
    from .triton_backend import tensorize_triton
except ImportError:
    tensorize_triton = None

__version__ = "1.0.0"
__all__ = ["tensorize", "tensorize_triton", "explain", "benchmark"]
