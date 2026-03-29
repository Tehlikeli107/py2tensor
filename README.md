# Py2Tensor

**Convert any Python function to a GPU tensor operation. No training. No approximation. Exact.**

```python
from py2tensor import tensorize, benchmark

@tensorize
def my_func(x):
    if x > 0:
        return x * 2
    else:
        return x + 1

# Works on scalars (CPU)
my_func(5)   # 10

# Works on GPU tensors (10M elements in <2ms)
x = torch.randn(10_000_000, device='cuda')
my_func(x)   # all 10M computed in parallel

# One-line benchmark
benchmark(my_func, 1.0)
# CPU: 10M/s, GPU: 4000M/s, Speedup: 400x
```

## How It Works

Parses the function's AST, transforms each operation to its GPU tensor equivalent:
- `if/else` -> `torch.where`
- `x if cond else y` -> `torch.where`
- `for i in range(N)` -> unrolled N times
- `math.sin/cos/exp/log/sqrt` -> `torch.*`
- `abs/min/max` -> `torch.*`
- `-x` -> `0 - x`
- `x ** n` -> tensor power
- `+= -= *=` -> tensor-safe ops

**No neural network. No training. No approximation.**

## Performance

Tested against hand-written PyTorch on 10M elements — **same speed** (0.97-1.21x ratio):

| Application | GPU Speed | vs Python |
|-------------|-----------|-----------|
| Float16 complex fn | **7.15B/s** | - |
| Gaussian PDF | 5.47B/s | **581x** |
| Damped oscillator | 4.79B/s | **795x** |
| Sigmoid | 4.81B/s | **571x** |
| GELU activation | 2.64B/s | - |
| Newton sqrt (10 iter) | 663M/s | **353x** |
| Mandelbrot (32 iter) | 121M/s | **295x** |
| Black-Scholes 10M | 352M/s | **100x** |
| Bisection root (30 iter) | 71M/s | - |

## Features

### Core
- `@tensorize` — one decorator, function works on both scalars and GPU tensors
- `@tensorize(dtype=torch.float16)` — float16 for 2x extra speed
- `@tensorize(lookup_tables={"table": [values]})` — precomputed tables
- `@tensorize(fallback=True)` — graceful CPU fallback on errors

### API
- `explain(fn)` — show the generated GPU code
- `benchmark(fn, *args)` — auto CPU vs GPU speed comparison

### Supported Python
- Arithmetic, comparisons, power
- `if/else` (nested, different-variable branches)
- Ternary expressions (`x if cond else y`)
- `for range(N)` (unrolled, max 64 iterations)
- `+= -= *= /=`
- `math.sin/cos/tan/exp/log/sqrt/tanh/pi/e`
- `abs/min/max/sum/round`
- `and/or` (bitwise tensor)
- Numpy array input (auto-conversion)
- Multiple inputs `f(x, y, z)`

### Works With
- NumPy arrays (auto-converted)
- PyTorch tensors (native)
- Python scalars (passthrough to original)

## Real-World Demos

Included: Black-Scholes option pricing, Mandelbrot set, Monte Carlo Pi,
Newton's method, bisection root finding, PID controller, Taylor series,
GELU activation, damped oscillator, credit scoring, dynamic pricing.

## Install

```bash
git clone https://github.com/Tehlikeli107/py2tensor.git
cd py2tensor
pip install -e .
```

## Tests

```bash
python run_all_tests.py  # 42 tests, <30s
```

## License

MIT
