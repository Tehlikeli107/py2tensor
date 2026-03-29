# Py2Tensor v1.0

**Convert any Python function to GPU. One decorator. 162x speedup. Exact results.**

```python
from py2tensor import tensorize
import math

@tensorize(backend="auto")
def bisection(lo, hi):
    for i in range(20):
        mid = (lo + hi) / 2
        fmid = mid**3 - 2*mid - 5
        if fmid > 0: hi = mid
        else: lo = mid
    return (lo + hi) / 2

# Solve 10M equations simultaneously
lo = torch.ones(10_000_000, device='cuda')
hi = torch.ones(10_000_000, device='cuda') * 3
roots = bisection(lo, hi)  # 18.3B/s, 162x faster than PyTorch
```

## 3 Backends

| Backend | Best For | Speed |
|---------|----------|-------|
| `@tensorize` | Compatibility, autograd | 3-6B/s |
| `@tensorize(compile=True)` | Simple math | 24-36B/s |
| `@tensorize(backend="triton")` | **Iterative algorithms** | **18-33B/s** |
| `@tensorize(backend="auto")` | **Auto-selects best** | **Best of both** |

## Benchmarks (10M elements, RTX 4070)

| Function | PyTorch | compile | Triton | Triton/PT |
|----------|---------|---------|--------|-----------|
| **Bisection 20iter+if** | 0.1B/s | 0.1B/s | **18.3B/s** | **162.5x** |
| Fixed-point 15iter | 0.4B/s | 0.4B/s | **23.1B/s** | **60.2x** |
| Newton sqrt 10iter | 0.7B/s | 26.9B/s | 20.6B/s | 29.3x |
| Decay 20iter | 1.4B/s | 1.4B/s | **24.2B/s** | 17.2x |
| Multi-input if/else | 3.0B/s | 19.0B/s | **21.5B/s** | 7.2x |
| Damped oscillator | 5.1B/s | **36.4B/s** | 32.8B/s | 6.4x |
| Gaussian PDF | 5.6B/s | 25.3B/s | **27.9B/s** | 5.0x |

**Triton wins 7/11 benchmarks. Average 27.7x over PyTorch ops.**

## Why Triton is Faster

```
PyTorch: for i in range(20): compute(x)
         = 20 kernel launches, 20 memory round-trips

Triton:  ALL 20 iterations in ONE kernel
         = 1 launch, data stays in GPU registers
```

## Features

- **if/else** -> `torch.where` / `tl.where` (nested, multi-variable)
- **for range(N)** -> unrolled in single kernel (up to 64 iter)
- **if/else INSIDE for-loop** -> fused conditional iteration
- **math.sin/cos/exp/log/sqrt/tanh/pi** -> GPU equivalents
- **Ternary** `x if cond else y` -> `where`
- **+=, -=, *=** -> tensor-safe ops
- **Multiple return values** (tuples)
- **Autograd** -> compute gradients, optimize any function
- **NumPy/Pandas input** -> auto-converted to GPU
- **float16** -> 2x extra speed
- **explain(fn)** -> show generated code
- **benchmark(fn, args)** -> auto CPU vs GPU comparison

## Install

```bash
git clone https://github.com/Tehlikeli107/py2tensor.git
cd py2tensor
pip install -e .
```

## Quick Start

```python
from py2tensor import tensorize, explain, benchmark

# Simple: just add decorator
@tensorize
def f(x):
    if x > 0: return math.sin(x)
    else: return 0

# See generated code
explain(f)

# Auto benchmark
benchmark(f, 1.0)

# For iterative algorithms: use auto or triton
@tensorize(backend="auto")
def newton(x):
    g = x / 2
    for i in range(10):
        g = (g + x / g) / 2
    return g
```

## Tests

```bash
python run_all_tests.py  # 52+ tests
python bench_final.py    # Full 3-backend benchmark
```

## License

MIT
