"""
Smart Compression: Auto-detect breakpoints + adaptive basis
============================================================
Step 1: Sample function densely
Step 2: Find kinks (where derivative jumps) via second derivative
Step 3: Place sigmoid centers AT the kinks
Step 4: SVD fit with kink-aware basis
Result: even piecewise/step functions compressed accurately
"""
import torch
import torch.nn as nn
import numpy as np
import time
import sys
sys.path.insert(0, r"C:\Users\salih\Desktop\py2tensor")

device = torch.device("cuda")


def detect_kinks(fn, lo, hi, n_probe=10000):
    """Find breakpoints where function has discontinuities in derivative."""
    x = torch.linspace(lo, hi, n_probe, device=device)
    y = fn(x)
    # Second derivative approximation
    dy = torch.diff(y)
    ddy = torch.diff(dy)
    # Kinks = where |ddy| is large relative to mean
    threshold = ddy.abs().mean() * 5
    kink_mask = ddy.abs() > threshold
    kink_positions = x[1:-1][kink_mask]
    return kink_positions


def smart_compress(fn, input_range, n_samples=50000, rank=32, n_features=1):
    """Compress with auto-detected breakpoints."""

    if isinstance(input_range, tuple):
        lo, hi = input_range
    else:
        lo, hi = input_range[0]

    # Step 1: Detect kinks
    if n_features == 1:
        kinks = detect_kinks(fn, lo, hi)
        n_kinks = len(kinks)
        print(f"  Detected {n_kinks} breakpoints: {kinks.cpu().tolist()[:10]}")
    else:
        kinks = torch.tensor([], device=device)
        n_kinks = 0

    # Step 2: Generate samples (extra density near kinks)
    if n_features == 1:
        # Uniform samples
        X_uniform = torch.rand(n_samples // 2, device=device) * (hi - lo) + lo
        # Extra samples near kinks
        if n_kinks > 0:
            X_kink = []
            per_kink = max(1, n_samples // (2 * n_kinks))
            for k in kinks:
                spread = (hi - lo) * 0.01  # 1% of range around each kink
                X_kink.append(k + torch.randn(per_kink, device=device) * spread)
            X_kink = torch.cat(X_kink)
            X = torch.cat([X_uniform, X_kink]).clamp(lo, hi)
        else:
            X = X_uniform
        X = X.unsqueeze(1)
    else:
        cols = []
        for i in range(n_features):
            if isinstance(input_range, list):
                lo_i, hi_i = input_range[i]
            else:
                lo_i, hi_i = lo, hi
            cols.append(torch.rand(n_samples, device=device) * (hi_i - lo_i) + lo_i)
        X = torch.stack(cols, dim=1)

    # Get outputs
    if n_features == 1:
        y = fn(X.squeeze(1)).float()
    else:
        y = fn(*[X[:, i] for i in range(n_features)]).float()

    n_actual = X.shape[0]

    # Step 3: Build adaptive basis
    features = [torch.ones(n_actual, device=device)]
    for i in range(n_features):
        xi = X[:, i]
        # Polynomial
        features.extend([xi, xi*xi, xi*xi*xi])
        # Trig
        features.extend([torch.sin(xi), torch.cos(xi)])
        # Standard sigmoid grid
        if isinstance(input_range, list):
            lo_i, hi_i = input_range[i]
        else:
            lo_i, hi_i = lo, hi
        n_grid = min(16, rank)
        grid = torch.linspace(lo_i, hi_i, n_grid, device=device)
        width = (hi_i - lo_i) / n_grid
        for c in grid:
            features.append(torch.sigmoid((xi - c) / (width * 0.1 + 1e-8)))

        # KINK-AWARE sigmoids (high precision at breakpoints)
        if n_features == 1 and n_kinks > 0:
            for k in kinks[:32]:  # max 32 kink centers
                # Narrow sigmoid at exact kink position
                features.append(torch.sigmoid((xi - k) / (width * 0.005 + 1)))
                # Slightly wider
                features.append(torch.sigmoid((xi - k) / (width * 0.01 + 1e-8)))

    # Cross features
    if n_features >= 2:
        for i in range(min(n_features, 4)):
            for j in range(i+1, min(n_features, 4)):
                features.append(X[:, i] * X[:, j])

    Phi = torch.stack(features, dim=1)
    n_basis = Phi.shape[1]
    print(f"  Basis: {n_basis} features ({n_kinks} kink-aware), samples: {n_actual}")

    # Step 4: Ridge regression
    lam = 1e-4
    try:
        PtP = Phi.t() @ Phi + lam * torch.eye(n_basis, device=device)
        Pty = Phi.t() @ y
        w = torch.linalg.solve(PtP, Pty)
        print(f"  Ridge solve OK, w shape: {w.shape}")
    except Exception as e:
        print(f"  Ridge failed: {e}, falling back to SVD")
        U, S, Vh = torch.linalg.svd(Phi, full_matrices=False)
        k = min(rank, len(S))
        w = Vh[:k].t() @ torch.diag(1.0 / (S[:k] + 1e-8)) @ U[:, :k].t() @ y

    # Build model — w has n_basis elements now (no rank truncation with ridge)
    model = SmartCompressedModel(w, n_features, input_range, kinks.cpu(), n_grid)
    model = model.to(device)

    # Evaluate
    print(f"  w shape: {w.shape}, model basis check...")
    test_phi = model._build_features(X[:10])
    print(f"  Phi shape: {test_phi.shape}, w: {w.shape}")
    y_pred = model(X)
    mae = (y_pred - y).abs().mean().item()
    max_err = (y_pred - y).abs().max().item()
    r2 = 1 - ((y_pred - y)**2).sum().item() / ((y - y.mean())**2).sum().item()

    print(f"  MAE: {mae:.2f}, Max error: {max_err:.2f}, R2: {r2:.6f}")
    print(f"  Weights: {w.numel()}")

    return model, r2


class SmartCompressedModel(nn.Module):
    def __init__(self, weights, n_features, input_range, kinks, n_grid):
        super().__init__()
        self.register_buffer('w', weights)
        self.register_buffer('kinks', kinks.float())
        self.n_features = n_features
        self.input_range = input_range
        self.n_grid = n_grid

    def _build_features(self, X):
        if X.dim() == 1: X = X.unsqueeze(1)
        n = X.shape[0]
        features = [torch.ones(n, device=X.device)]
        for i in range(self.n_features):
            xi = X[:, i]
            features.extend([xi, xi*xi, xi*xi*xi, torch.sin(xi), torch.cos(xi)])
            if isinstance(self.input_range, list):
                lo, hi = self.input_range[i]
            else:
                lo, hi = self.input_range
            grid = torch.linspace(lo, hi, self.n_grid, device=X.device)
            width = (hi - lo) / self.n_grid
            for c in grid:
                features.append(torch.sigmoid((xi - c) / (width * 0.1 + 1e-8)))
            if self.n_features == 1 and len(self.kinks) > 0:
                for k in self.kinks[:32]:
                    k = k.to(X.device)
                    features.append(torch.sigmoid((xi - k) / (width * 0.005 + 1)))
                    features.append(torch.sigmoid((xi - k) / (width * 0.01 + 1e-8)))
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
        return self._build_features(X) @ self.w


# ================================================================
if __name__ == '__main__':
    from py2tensor import gpu
    import math

    print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 60)
    print("SMART COMPRESSION: Auto-detect breakpoints")
    print("=" * 60)

    # Tax calculator (6 brackets — THE HARD CASE)
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

    print(f"\n[1] Tax calculator (6 brackets)")
    model, r2 = smart_compress(tax, (0, 1000000), n_samples=100000, rank=64)

    # Test specific values
    test_incomes = torch.tensor([10000, 25000, 50000, 100000, 300000, 700000],
                                dtype=torch.float32, device=device)
    orig = tax(test_incomes)
    comp = model(test_incomes)
    print(f"  Original:   {[f'{v:.0f}' for v in orig.tolist()]}")
    print(f"  Compressed: {[f'{v:.0f}' for v in comp.tolist()]}")
    print(f"  Error:      {[f'{abs(a-b):.0f}' for a,b in zip(orig.tolist(), comp.tolist())]}")

    # Benchmark
    N = 10_000_000
    x_big = torch.rand(N, device=device) * 1000000
    torch.cuda.synchronize()
    for _ in range(3): tax(x_big)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20): tax(x_big)
    torch.cuda.synchronize()
    t_orig = (time.time()-t0)/20

    mc = torch.compile(model)
    for _ in range(3): mc(x_big)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20): mc(x_big)
    torch.cuda.synchronize()
    t_comp = (time.time()-t0)/20

    print(f"\n  Original:   {N/t_orig/1e9:.1f}B/s")
    print(f"  Compressed: {N/t_comp/1e9:.1f}B/s")
    print(f"  Speedup:    {t_orig/t_comp:.1f}x")

    # Credit scoring (step function)
    @gpu
    def credit(score):
        if score > 800:
            return 5
        else:
            if score > 700:
                return 4
            else:
                if score > 600:
                    return 3
                else:
                    if score > 500:
                        return 2
                    else:
                        return 1

    print(f"\n[2] Credit scoring (5 tiers)")
    model2, r2_2 = smart_compress(credit, (300, 850), n_samples=50000, rank=32)

    test_scores = torch.tensor([450, 550, 650, 750, 820], dtype=torch.float32, device=device)
    print(f"  Original:   {credit(test_scores).tolist()}")
    print(f"  Compressed: {[f'{v:.1f}' for v in model2(test_scores).tolist()]}")

    print(f"\n{'='*60}")
    print("Kink detection = automatic breakpoint discovery")
    print("Sigmoid basis at kinks = exact step function capture")
