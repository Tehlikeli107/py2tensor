"""
Demo: explain() and benchmark() API
"""
import sys
sys.path.insert(0, r"C:\Users\salih\Desktop\py2tensor")
from py2tensor import tensorize, explain, benchmark
import math

@tensorize
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

@tensorize
def scoring(x):
    if x > 100:
        return 100 + (x - 100) * 0.5
    else:
        if x > 0:
            return x
        else:
            return 0

@tensorize
def damped(t):
    return math.exp(-0.5 * t) * math.sin(2.0 * math.pi * 5.0 * t)

# Show generated code
print("=== EXPLAIN ===\n")
explain(sigmoid)
print()
explain(scoring)
print()

# Auto benchmark
print("\n=== BENCHMARK ===\n")
benchmark(sigmoid, 1.0)
print()
benchmark(scoring, 50.0)
print()
benchmark(damped, 0.5)
