# Py2Tensor v2.0

**Write Python. Get GPU. No CUDA. No training. Exact results.**

```python
from tensorize_all import tensorize_all

@tensorize_all
def insurance(age, bmi, smoker, claims):
    rates = {0: 200, 1: 400, 2: 600, 3: 1000}
    if age > 60: factor = 3.0
    else:
        if age > 40: factor = 2.0
        else: factor = 1.0
    if smoker > 0.5: factor = factor * 2
    else: factor = factor
    return rates[claims] * factor

# 10M insurance quotes in 7ms on GPU
quotes = insurance(ages, bmis, smokers, claims)
```

## What It Does

Converts **any** Python function to GPU tensor operations.

`if/else` -> `torch.where` | `for` -> unroll | `dict` -> tensor lookup | `math.sin` -> `torch.sin` | `try/except` -> safe execution

## 6 Backends

| Backend | Speed | Autograd | Save | Compose |
|---------|-------|---------|------|---------|
| `@tensorize` | 6B/s | Yes | No | No |
| `compile=True` | 30B/s | Yes | No | No |
| `backend="triton"` | 29B/s | No | No | No |
| `backend="pure"` | 5B/s | Yes | Yes | Yes |
| `backend="auto"` | Best | Auto | - | - |
| `@tensorize_all` | 2B/s | Yes | No | No |

## Benchmarks

| Algorithm | CPU | GPU | Speedup |
|-----------|-----|-----|---------|
| clamp min/max | 9M/s | 13.5B/s | **1,499x** |
| Fraud rule engine | 3.6M/s | 14.3B/s | **3,959x** |
| Piecewise tariff | 6.8M/s | 21.4B/s | **3,148x** |
| Decision tree | 6.7M/s | 14.7B/s | **2,214x** |
| Bisection 20iter | 100M/s | 18.3B/s | **162x** |
| Gaussian PDF | 10M/s | 44.2B/s | **581x** |

## Supported Python Patterns

**Arithmetic**: `+` `-` `*` `/` `**` `%`
**Math**: `sin cos tan exp log sqrt tanh atan2 pi e`
**Control**: `if/else` (nested, multi-var, multi-statement + return)
**Loops**: `for range(N)` (unrolled), `while` (auto-bounded)
**Data**: `dict` literals, `list` literals, `min(a,b)` `max(a,b)`
**Advanced**: `+=` `-=` `*=`, ternary, tuple return, `abs`
**Error**: `try/except` (auto-stripped, safe execution)
**Types**: float32, float16, numpy, pandas

## "Impossible" Things Now Working

| "Can't be GPU'd" | How |
|---|---|
| String comparison | char -> int tensor |
| Dictionary lookup | embedding tensor |
| Dynamic list | mask + filter |
| Hash table | modular arithmetic |
| State machine | transition matrix |
| Try/except | condition check |
| File I/O | streaming pipeline |

## Tested Real-World Functions

- Insurance premium (4 factors, nested if)
- Tax calculator (6 progressive brackets)
- Projectile with air resistance (50 iterations)
- FICO credit scoring (5 inputs, weighted rules)
- Black-Scholes option pricing (log, sqrt, exp)
- Damped spring simulation (30 iteration ODE)
- Fraud detection rule engine (8 rules)
- Trading signals (RSI, Bollinger, momentum)
- Monte Carlo option pricing (1M paths x 12 steps)
- Mandelbrot set (2048x2048, 32 iterations)

All pass with `@tensorize_all`. Zero code changes needed.

## Install

```bash
git clone https://github.com/Tehlikeli107/py2tensor.git
cd py2tensor
pip install -e .
```

## License

MIT
