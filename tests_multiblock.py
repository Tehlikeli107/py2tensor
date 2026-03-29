"""
Multi-statement if/else blocks + complex patterns
"""
import torch
import math
import time
import sys
sys.path.insert(0, r"C:\Users\salih\Desktop\py2tensor")
from py2tensor import tensorize, explain, benchmark

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
passed = 0
failed = 0

def check(name, expected, got, tol=1e-2):
    global passed, failed
    exp = torch.tensor(expected, dtype=torch.float32) if isinstance(expected, list) else expected
    got = got.cpu().float() if isinstance(got, torch.Tensor) else torch.tensor(got, dtype=torch.float32)
    match = torch.allclose(exp.reshape(-1), got.reshape(-1), atol=tol, rtol=1e-2)
    if match: passed += 1
    else:
        failed += 1
        print(f"  Exp: {exp.reshape(-1)[:5]}")
        print(f"  Got: {got.reshape(-1)[:5]}")
    print(f"  [{'PASS' if match else 'FAIL'}] {name}")

# ================================================================
print("\n=== MULTI-STATEMENT IF/ELSE ===")

@tensorize
def swap_if_needed(a, b):
    if a > b:
        x = b
        y = a
    else:
        x = a
        y = b
    return x + y * 10

pairs = [(1,2), (5,3), (0,0), (-1,1), (10, -5)]
cpu_out = [swap_if_needed(a,b) for a,b in pairs]
a_g = torch.tensor([p[0] for p in pairs], dtype=torch.float32, device=device)
b_g = torch.tensor([p[1] for p in pairs], dtype=torch.float32, device=device)
gpu_out = swap_if_needed(a_g, b_g)
check("multi-assign if/else (swap)", [float(v) for v in cpu_out], gpu_out)
explain(swap_if_needed)

# ================================================================
print("\n=== TRIPLE ASSIGNMENT ===")

@tensorize
def color_classify(x):
    if x > 200:
        r = 255
        g = 0
        b = 0
    else:
        r = 0
        g = 255
        b = 0
    return r * 65536 + g * 256 + b

vals = [50, 100, 200, 250, 300]
cpu_out = [color_classify(v) for v in vals]
gpu_out = color_classify(torch.tensor(vals, dtype=torch.float32, device=device))
check("triple-assign (RGB)", [float(v) for v in cpu_out], gpu_out)

# ================================================================
print("\n=== MIXED: SOME VARS ONLY IN ONE BRANCH ===")

@tensorize
def partial_assign(x):
    y = x * 2
    if x > 0:
        y = x * 3
        bonus = 10
    else:
        y = x * 0.5
        bonus = 0
    return y + bonus

vals = [-5, -1, 0, 1, 5]
cpu_out = [partial_assign(v) for v in vals]
gpu_out = partial_assign(torch.tensor(vals, dtype=torch.float32, device=device))
check("partial multi-assign", [float(v) for v in cpu_out], gpu_out)

# ================================================================
print("\n=== ORIGINAL RSI (now should work) ===")

@tensorize
def rsi_full(price, prev_price):
    change = price - prev_price
    if change > 0:
        gain = change
        loss = 0
    else:
        gain = 0
        loss = -change
    ratio = gain / (loss + 0.001)
    if ratio > 2.0:
        return 1
    else:
        if ratio < 0.5:
            return -1
        else:
            return 0

pairs = [(105,100), (95,100), (100,100), (110,100), (90,100)]
cpu_out = [rsi_full(p, pp) for p, pp in pairs]
p_g = torch.tensor([p[0] for p in pairs], dtype=torch.float32, device=device)
pp_g = torch.tensor([p[1] for p in pairs], dtype=torch.float32, device=device)
gpu_out = rsi_full(p_g, pp_g)
check("RSI full (gain+loss in if/else)", [float(v) for v in cpu_out], gpu_out)
explain(rsi_full)

# ================================================================
print("\n=== NESTED IF WITH MULTI-ASSIGN ===")

@tensorize
def nested_multi(x):
    if x > 10:
        a = x
        b = 100
    else:
        if x > 0:
            a = x * 2
            b = 50
        else:
            a = 0
            b = 0
    return a + b

vals = [-5, 0, 5, 10, 15]
cpu_out = [nested_multi(v) for v in vals]
gpu_out = nested_multi(torch.tensor(vals, dtype=torch.float32, device=device))
check("nested if multi-assign", [float(v) for v in cpu_out], gpu_out)

# ================================================================
print("\n=== BENCHMARK: Full RSI on 10M ===")
benchmark(rsi_full, 105.0, 100.0, n=10_000_000)

# ================================================================
print(f"\n{'='*50}")
print(f"RESULTS: {passed} passed, {failed} failed")
print(f"{'='*50}")
