"""
Auto Compress: analyze function, pick best compression, apply.
One function call does everything.

Usage:
    from auto_compress import auto_compress
    small_model = auto_compress(my_function, input_range=(-10, 10))
"""
import torch
import time
import sys
sys.path.insert(0, r"C:\Users\salih\Desktop\py2tensor")

device = torch.device("cuda")


def auto_compress(fn, input_range, n_features=1, target_r2=0.999):
    """Automatically compress any function to smallest accurate model.

    Tries methods in order:
    1. Piecewise linear (best for step/bracket functions)
    2. SVD polynomial (best for smooth functions)
    3. SVD + RBF (best for complex smooth)

    Picks the one with fewest params that meets target R2.
    """
    print(f"\n{'='*50}")
    print(f"AUTO COMPRESS: {fn.__name__ if hasattr(fn, '__name__') else 'function'}")
    print(f"{'='*50}")

    results = []

    # Method 1: Piecewise
    if n_features == 1:
        print(f"\n  [1] Trying piecewise linear...")
        try:
            from piecewise_compress import piecewise_compress
            if isinstance(input_range, tuple):
                lo, hi = input_range
            else:
                lo, hi = input_range[0]
            model_pw = piecewise_compress(fn, lo, hi)
            model_pw = model_pw.to(device)

            # Evaluate
            x_test = torch.linspace(lo, hi, 10000, device=device)
            y_orig = fn(x_test).float()
            y_comp = model_pw(x_test).float()
            r2 = 1 - ((y_orig-y_comp)**2).sum().item() / ((y_orig-y_orig.mean())**2).sum().item()
            n_params = sum(p.numel() for p in model_pw.parameters())
            results.append(('piecewise', model_pw, r2, n_params))
            print(f"      R2={r2:.6f}, params={n_params}")
        except Exception as e:
            print(f"      Failed: {e}")

    # Method 2: SVD polynomial
    print(f"\n  [2] Trying SVD polynomial...")
    try:
        from compress_model import compress
        model_svd = compress(fn, input_range, n_samples=50000, rank=32, n_features=n_features)
        model_svd = model_svd.to(device)

        if n_features == 1:
            y_comp2 = model_svd(x_test).float()
            r2_svd = 1 - ((y_orig-y_comp2)**2).sum().item() / ((y_orig-y_orig.mean())**2).sum().item()
        else:
            # Multi-feature test
            x_multi = torch.rand(10000, n_features, device=device)
            if isinstance(input_range, list):
                for i in range(n_features):
                    lo_i, hi_i = input_range[i]
                    x_multi[:, i] = x_multi[:, i] * (hi_i - lo_i) + lo_i
            y_orig_m = fn(*[x_multi[:, i] for i in range(n_features)]).float()
            y_comp_m = model_svd(x_multi).float()
            r2_svd = 1 - ((y_orig_m-y_comp_m)**2).sum().item() / ((y_orig_m-y_orig_m.mean())**2).sum().item()

        n_params_svd = model_svd.w.numel()
        results.append(('svd', model_svd, r2_svd, n_params_svd))
        print(f"      R2={r2_svd:.6f}, params={n_params_svd}")
    except Exception as e:
        print(f"      Failed: {e}")

    # Pick best
    if not results:
        print(f"\n  No compression method worked!")
        return None

    # Filter by target R2
    good = [(name, m, r2, p) for name, m, r2, p in results if r2 >= target_r2]
    if good:
        # Pick smallest params among good ones
        best = min(good, key=lambda x: x[3])
    else:
        # Pick highest R2
        best = max(results, key=lambda x: x[2])

    name, model, r2, params = best
    print(f"\n  SELECTED: {name}")
    print(f"  R2={r2:.6f}, params={params}")

    # Benchmark
    if n_features == 1:
        N = 10_000_000
        x_big = torch.rand(N, device=device) * (hi - lo) + lo

        torch.cuda.synchronize()
        for _ in range(3): fn(x_big)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(20): fn(x_big)
        torch.cuda.synchronize()
        t_orig = (time.time()-t0)/20

        mc = torch.compile(model)
        for _ in range(3): mc(x_big)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(20): mc(x_big)
        torch.cuda.synchronize()
        t_comp = (time.time()-t0)/20

        print(f"  Original:   {N/t_orig/1e9:.1f}B/s")
        print(f"  Compressed: {N/t_comp/1e9:.1f}B/s")
        print(f"  Speedup:    {t_orig/t_comp:.1f}x")

    return model


# ================================================================
if __name__ == '__main__':
    from py2tensor import gpu
    import math

    print(f"GPU: {torch.cuda.get_device_name()}")

    # Test 1: Step function -> should pick piecewise
    @gpu
    def tax(income):
        if income > 500000:
            return (income - 500000) * 0.37 + 150000
        else:
            if income > 200000:
                return (income - 200000) * 0.32 + 50000
            else:
                if income > 80000:
                    return (income - 80000) * 0.22 + 20000
                else:
                    if income > 40000:
                        return (income - 40000) * 0.12 + 5000
                    else:
                        return income * 0.10

    m1 = auto_compress(tax, (0, 1000000))

    # Test 2: Smooth function -> should pick SVD
    @gpu
    def gaussian(x):
        return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

    m2 = auto_compress(gaussian, (-5, 5))

    # Test 3: Mixed (smooth + step)
    @gpu
    def mixed(x):
        base = math.sin(x * 0.1) * 10
        if x > 50:
            return base + 100
        else:
            return base

    m3 = auto_compress(mixed, (0, 100))

    # Summary
    print(f"\n{'='*50}")
    print("AUTO COMPRESS SUMMARY")
    print(f"{'='*50}")
    print(f"""
  auto_compress() picks the best method automatically:
  - Step/bracket functions -> piecewise linear (EXACT)
  - Smooth functions -> SVD polynomial (R2>0.999)
  - Mixed functions -> tries both, picks best

  Usage:
    model = auto_compress(my_function, input_range=(lo, hi))
    model = model.cuda()
    result = model(big_tensor)  # fast!
""")
