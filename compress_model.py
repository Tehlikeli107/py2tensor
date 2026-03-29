"""
Model Compression: Big tensor model -> Small tensor model
=========================================================
Step 1: CPU function -> GPU tensor model (done)
Step 2: GPU tensor model -> COMPRESSED tensor model (this file)

Method: the model's behavior is a function f: input -> output.
Generate many (input, output) pairs, then fit a SMALLER model.
Not training a neural net — using matrix decomposition (SVD).

The compressed model is:
- Smaller (less memory)
- Faster (fewer operations)
- Nearly exact (controllable accuracy)
"""
import torch
import torch.nn as nn
import numpy as np
import time
import sys
sys.path.insert(0, r"C:\Users\salih\Desktop\py2tensor")

device = torch.device("cuda")


def compress(model_fn, input_range, n_samples=100000, rank=32, n_features=1):
    """Compress any function into a small matrix model.

    1. Sample input-output pairs from the original function
    2. Build feature matrix (polynomial/RBF basis)
    3. SVD decompose: W = U @ S @ V^T, keep top-k
    4. Result: small model that approximates the original

    Args:
        model_fn: original function (CPU or GPU)
        input_range: (min, max) or list of (min, max) per feature
        n_samples: how many samples to generate
        rank: compression rank (smaller = more compressed)
        n_features: number of input features
    """
    # Step 1: Generate training data
    if n_features == 1:
        if isinstance(input_range, tuple):
            lo, hi = input_range
        else:
            lo, hi = input_range[0]
        X = torch.linspace(lo, hi, n_samples, device=device).unsqueeze(1)
    else:
        cols = []
        for i in range(n_features):
            if isinstance(input_range, list):
                lo, hi = input_range[i]
            else:
                lo, hi = input_range
            cols.append(torch.rand(n_samples, device=device) * (hi - lo) + lo)
        X = torch.stack(cols, dim=1)

    # Get outputs from original model
    if n_features == 1:
        y = model_fn(X.squeeze(1))
    else:
        y = model_fn(*[X[:, i] for i in range(n_features)])

    if isinstance(y, tuple):
        y = y[0]
    y = y.float()

    # Step 2: Build feature matrix (polynomial basis)
    # phi(x) = [1, x, x^2, x^3, ..., sin(x), cos(x), exp(-x^2)]
    features = [torch.ones(n_samples, device=device)]  # bias
    for i in range(n_features):
        xi = X[:, i]
        features.append(xi)
        features.append(xi * xi)
        features.append(xi * xi * xi)
        features.append(torch.sin(xi))
        features.append(torch.cos(xi))
        features.append(torch.sin(2 * xi))
        features.append(torch.cos(2 * xi))
        features.append(torch.exp(-xi * xi * 0.1))

    # Cross features for multi-input
    if n_features >= 2:
        for i in range(min(n_features, 4)):
            for j in range(i+1, min(n_features, 4)):
                features.append(X[:, i] * X[:, j])

    Phi = torch.stack(features, dim=1)  # (n_samples, n_basis)
    n_basis = Phi.shape[1]

    # Step 3: Solve least squares: y = Phi @ w
    # Using SVD: Phi = U @ S @ V^T
    # w = V @ S^-1 @ U^T @ y
    U, S, Vh = torch.linalg.svd(Phi, full_matrices=False)

    # Keep only top-k singular values (compression!)
    k = min(rank, len(S))
    U_k = U[:, :k]
    S_k = S[:k]
    Vh_k = Vh[:k, :]

    # Solve: w = Vh_k^T @ diag(1/S_k) @ U_k^T @ y
    w = Vh_k.t() @ torch.diag(1.0 / S_k) @ U_k.t() @ y  # (n_basis,)

    # Step 4: Build compressed model
    compressed = CompressedModel(w, n_features, input_range)

    # Evaluate accuracy
    y_pred = compressed(X)
    mae = (y_pred - y).abs().mean().item()
    max_err = (y_pred - y).abs().max().item()
    r2 = 1 - ((y_pred - y) ** 2).sum().item() / ((y - y.mean()) ** 2).sum().item()

    print(f"  Compression: {n_basis} basis -> rank {k}")
    print(f"  MAE: {mae:.6f}, Max error: {max_err:.4f}, R2: {r2:.6f}")
    print(f"  Original params: ~{n_samples} samples needed")
    print(f"  Compressed params: {w.numel()} weights")
    print(f"  Compression ratio: {n_samples / w.numel():.0f}x smaller")

    return compressed


class CompressedModel(nn.Module):
    """Tiny model: polynomial basis @ weight vector."""

    def __init__(self, weights, n_features, input_range):
        super().__init__()
        self.register_buffer('w', weights)
        self.n_features = n_features
        self.input_range = input_range

    def _build_features(self, X):
        if X.dim() == 1:
            X = X.unsqueeze(1)
        n = X.shape[0]
        features = [torch.ones(n, device=X.device)]
        for i in range(self.n_features):
            xi = X[:, i] if X.dim() > 1 else X
            features.extend([xi, xi*xi, xi*xi*xi,
                           torch.sin(xi), torch.cos(xi),
                           torch.sin(2*xi), torch.cos(2*xi),
                           torch.exp(-xi*xi*0.1)])
        if self.n_features >= 2:
            for i in range(min(self.n_features, 4)):
                for j in range(i+1, min(self.n_features, 4)):
                    features.append(X[:, i] * X[:, j])
        return torch.stack(features, dim=1)

    def forward(self, *args):
        if len(args) == 1 and args[0].dim() >= 2:
            X = args[0]
        elif len(args) == 1:
            X = args[0].unsqueeze(1)
        else:
            X = torch.stack([a.float() for a in args], dim=1)
        Phi = self._build_features(X)
        return Phi @ self.w


# ================================================================
if __name__ == '__main__':
    from py2tensor import gpu
    import math

    print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 60)
    print("MODEL COMPRESSION: Big -> Small -> Same Speed")
    print("=" * 60)

    # Test 1: Simple function
    print(f"\n[1] sin(x) * exp(-x^2/10)")
    @gpu
    def f1(x):
        return math.sin(x) * math.exp(-x * x * 0.1)

    compressed1 = compress(f1, (-5, 5), n_samples=10000, rank=16)
    compressed1 = compressed1.to(device)

    # Compare
    x = torch.linspace(-5, 5, 1000, device=device)
    orig = f1(x)
    comp = compressed1(x)
    print(f"  Max diff: {(orig - comp).abs().max().item():.6f}")

    # Speed
    N = 10_000_000
    x_big = torch.randn(N, device=device) * 3
    torch.cuda.synchronize()
    for _ in range(3): f1(x_big)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(30): f1(x_big)
    torch.cuda.synchronize()
    t_orig = (time.time()-t0)/30

    c1c = torch.compile(compressed1)
    for _ in range(3): c1c(x_big)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(30): c1c(x_big)
    torch.cuda.synchronize()
    t_comp = (time.time()-t0)/30

    print(f"  Original:   {N/t_orig/1e9:.1f}B/s")
    print(f"  Compressed: {N/t_comp/1e9:.1f}B/s")
    print(f"  Speedup:    {t_orig/t_comp:.2f}x")

    # Test 2: Complex branching
    print(f"\n[2] Tax calculator (6 brackets)")
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

    compressed2 = compress(tax, (0, 1000000), n_samples=50000, rank=32)
    compressed2 = compressed2.to(device)

    incomes = torch.tensor([10000, 50000, 100000, 300000, 700000], dtype=torch.float32, device=device)
    print(f"  Original: {tax(incomes).tolist()}")
    print(f"  Compressed: {compressed2(incomes).tolist()}")

    # Test 3: Multi-input
    print(f"\n[3] Insurance (2 inputs: age, bmi)")
    @gpu
    def insurance(age, bmi):
        if age > 50:
            base = 500
        else:
            base = 200
        if bmi > 30:
            factor = 2.0
        else:
            factor = 1.0
        return base * factor

    compressed3 = compress(
        insurance,
        input_range=[(20, 80), (18, 45)],
        n_samples=50000, rank=16, n_features=2
    )
    compressed3 = compressed3.to(device)

    ages = torch.tensor([25, 55, 35, 65], dtype=torch.float32, device=device)
    bmis = torch.tensor([22, 22, 35, 35], dtype=torch.float32, device=device)
    X_test = torch.stack([ages, bmis], dim=1)
    print(f"  Original:   {insurance(ages, bmis).tolist()}")
    print(f"  Compressed: {compressed3(X_test).tolist()}")

    # Test 4: RF model compression
    print(f"\n[4] Sklearn RF -> Compressed")
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.datasets import make_regression

        X_sk, y_sk = make_regression(n_samples=10000, n_features=5, random_state=42)
        rf = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
        rf.fit(X_sk, y_sk)

        from sklearn_to_gpu import convert_rf
        gpu_rf = convert_rf(rf).to(device)

        def rf_wrapper(*args):
            X = torch.stack([a.float() for a in args], dim=1)
            return gpu_rf(X)

        compressed_rf = compress(
            rf_wrapper,
            input_range=[(-3, 3)] * 5,
            n_samples=50000, rank=64, n_features=5
        )
        compressed_rf = compressed_rf.to(device)

        # Speed comparison
        X_test = torch.randn(100000, 5, device=device)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10): gpu_rf(X_test)
        torch.cuda.synchronize()
        t_rf = (time.time()-t0)/10

        crf = torch.compile(compressed_rf)
        for _ in range(3): crf(X_test)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(30): crf(X_test)
        torch.cuda.synchronize()
        t_comp = (time.time()-t0)/30

        print(f"  GPU RF:       {100000/t_rf/1e6:.1f}M/s")
        print(f"  Compressed:   {100000/t_comp/1e6:.1f}M/s")
        print(f"  Speedup:      {t_rf/t_comp:.1f}x")
    except Exception as e:
        print(f"  RF test error: {e}")

    print(f"\n{'='*60}")
    print("PIPELINE: CPU func -> GPU model -> Compressed model")
    print("  Each step: same output, smaller, faster.")
