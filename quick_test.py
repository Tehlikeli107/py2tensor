"""Quick health check for py2tensor package."""
import sys
print(f"Python: {sys.version}")

# Test 1: Import
try:
    from py2tensor import gpu, tensorize, explain, benchmark, profile
    print("[OK] Core imports work")
except Exception as e:
    print(f"[FAIL] Core import: {e}")

# Test 2: Simple function
try:
    import torch
    @gpu
    def double(x):
        return x * 2

    x = torch.randn(1000, device='cuda')
    result = double(x)
    assert result.shape == x.shape
    assert torch.allclose(result, x * 2, atol=1e-5)
    print("[OK] Simple @gpu works")
except Exception as e:
    print(f"[FAIL] Simple @gpu: {e}")

# Test 3: Branching
try:
    @gpu
    def branch(x):
        if x > 0:
            return x * 2
        else:
            return x + 1

    x = torch.randn(10000, device='cuda')
    result = branch(x)
    expected = torch.where(x > 0, x * 2, x + 1)
    assert torch.allclose(result, expected, atol=1e-5)
    print("[OK] Branching @gpu works")
except Exception as e:
    print(f"[FAIL] Branching: {e}")

# Test 4: Explain
try:
    @gpu
    def f(x):
        if x > 0:
            return x ** 2
        else:
            return -x

    info = explain(f)
    print(f"[OK] explain() works: {type(info)}")
except Exception as e:
    print(f"[FAIL] explain: {e}")

# Test 5: Benchmark
try:
    @gpu
    def g(x):
        return x * 3 + 1

    result = benchmark(g, size=100000)
    print(f"[OK] benchmark() works")
except Exception as e:
    print(f"[FAIL] benchmark: {e}")

# Test 6: Triton backend
try:
    from py2tensor import tensorize_triton
    if tensorize_triton:
        print("[OK] Triton backend available")
    else:
        print("[SKIP] Triton backend not available (import failed gracefully)")
except Exception as e:
    print(f"[FAIL] Triton: {e}")

# Test 7: Pure model backend
try:
    from py2tensor import build_pure_model
    if build_pure_model:
        print("[OK] Pure model backend available")
    else:
        print("[SKIP] Pure model not available")
except Exception as e:
    print(f"[FAIL] Pure model: {e}")

# Test 8: sklearn_to_gpu (needed for GPU SHAP)
try:
    sys.path.insert(0, '.')
    from sklearn_to_gpu import convert_rf
    print("[OK] sklearn_to_gpu importable from root")
except Exception as e:
    print(f"[FAIL] sklearn_to_gpu: {e}")

# Test 9: pip installable?
try:
    import py2tensor
    print(f"[OK] py2tensor version: {py2tensor.__version__}")
except Exception as e:
    print(f"[FAIL] package: {e}")

print("\n--- SUMMARY ---")
print("If all [OK]: ready for pip install + announcement")
print("If any [FAIL]: fix those before announcing")
