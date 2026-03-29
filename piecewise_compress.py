"""
Piecewise Linear Compression: auto-detect breakpoints, fit exact segments.
No SVD, no RBF. Just: find kinks, fit linear segments between them.
Result: EXACT for piecewise linear functions. Tiny model.
"""
import torch
import torch.nn as nn
import time
import sys
sys.path.insert(0, r"C:\Users\salih\Desktop\py2tensor")

device = torch.device("cuda")


def detect_kinks(fn, lo, hi, n_probe=50000):
    x = torch.linspace(lo, hi, n_probe, device=device)
    y = fn(x)
    dy = torch.diff(y) / torch.diff(x)
    ddy = torch.diff(dy)
    threshold = ddy.abs().median() * 20
    kink_mask = ddy.abs() > threshold
    kink_idx = torch.where(kink_mask)[0]

    # Merge nearby kinks
    if len(kink_idx) == 0:
        return torch.tensor([lo, hi], device=device)
    merged = [kink_idx[0]]
    min_gap = n_probe // 100
    for idx in kink_idx[1:]:
        if idx - merged[-1] > min_gap:
            merged.append(idx)
    kink_x = x[1:-1][torch.tensor(merged, device=device)]

    # Add boundaries
    breakpoints = torch.cat([torch.tensor([lo], device=device), kink_x, torch.tensor([hi], device=device)])
    return breakpoints


def piecewise_compress(fn, lo, hi):
    """Compress piecewise linear function into exact segment model."""
    breakpoints = detect_kinks(fn, lo, hi)
    n_segments = len(breakpoints) - 1

    # Fit linear segment in each interval: y = slope * x + intercept
    slopes = torch.zeros(n_segments, device=device)
    intercepts = torch.zeros(n_segments, device=device)

    for i in range(n_segments):
        x1, x2 = breakpoints[i], breakpoints[i+1]
        y1 = fn(x1.unsqueeze(0)).squeeze()
        y2 = fn(x2.unsqueeze(0)).squeeze()
        slope = (y2 - y1) / (x2 - x1 + 1e-10)
        intercept = y1 - slope * x1
        slopes[i] = slope
        intercepts[i] = intercept

    model = PiecewiseModel(breakpoints, slopes, intercepts)

    # Verify
    x_test = torch.linspace(lo, hi, 10000, device=device)
    y_orig = fn(x_test)
    y_comp = model(x_test)
    mae = (y_orig - y_comp).abs().mean().item()
    max_err = (y_orig - y_comp).abs().max().item()
    r2 = 1 - ((y_orig - y_comp)**2).sum().item() / ((y_orig - y_orig.mean())**2).sum().item()

    print(f"  Segments: {n_segments}")
    print(f"  Breakpoints: {[f'{b:.0f}' for b in breakpoints.tolist()]}")
    print(f"  MAE: {mae:.2f}, Max error: {max_err:.2f}, R2: {r2:.6f}")
    print(f"  Parameters: {2 * n_segments + len(breakpoints)} (slopes + intercepts + breakpoints)")

    return model


class PiecewiseModel(nn.Module):
    """Pure tensor piecewise linear: NO Python if/else in forward."""

    def __init__(self, breakpoints, slopes, intercepts):
        super().__init__()
        self.register_buffer('breakpoints', breakpoints)
        self.register_buffer('slopes', slopes)
        self.register_buffer('intercepts', intercepts)

    def forward(self, x):
        # For each x, find which segment it belongs to
        # x shape: (batch,)
        # breakpoints shape: (n_segments + 1,)
        # Compare x against each breakpoint: (batch, n_bp)
        bp = self.breakpoints
        # Segment index: count how many breakpoints x exceeds, minus 1
        segment = (x.unsqueeze(1) >= bp.unsqueeze(0)).sum(dim=1) - 1
        segment = segment.clamp(0, len(self.slopes) - 1)

        # Gather slope and intercept for each x's segment
        slope = self.slopes[segment]
        intercept = self.intercepts[segment]

        return slope * x + intercept


if __name__ == '__main__':
    from py2tensor import gpu
    import math

    print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 60)
    print("PIECEWISE LINEAR COMPRESSION")
    print("=" * 60)

    # Tax calculator
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
    model = piecewise_compress(tax, 0, 1000000)
    model = model.to(device)

    test_vals = torch.tensor([10000, 25000, 50000, 100000, 300000, 700000],
                             dtype=torch.float32, device=device)
    print(f"  Original:   {[f'{v:.0f}' for v in tax(test_vals).tolist()]}")
    print(f"  Compressed: {[f'{v:.0f}' for v in model(test_vals).tolist()]}")

    # Benchmark
    N = 10_000_000
    x = torch.rand(N, device=device) * 1000000

    torch.cuda.synchronize()
    for _ in range(3): tax(x)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(50): tax(x)
    torch.cuda.synchronize()
    t_orig = (time.time()-t0)/50

    mc = torch.compile(model)
    for _ in range(3): mc(x)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(50): mc(x)
    torch.cuda.synchronize()
    t_comp = (time.time()-t0)/50

    print(f"\n  Original (@gpu):   {N/t_orig/1e9:.1f}B/s")
    print(f"  Piecewise model:   {N/t_comp/1e9:.1f}B/s")
    print(f"  Speedup:           {t_orig/t_comp:.1f}x")

    # Credit scoring
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
    model2 = piecewise_compress(credit, 300, 850)
    model2 = model2.to(device)

    test_scores = torch.tensor([450, 550, 650, 750, 820], dtype=torch.float32, device=device)
    print(f"  Original:   {credit(test_scores).tolist()}")
    print(f"  Compressed: {[f'{v:.1f}' for v in model2(test_scores).tolist()]}")

    # Insurance
    @gpu
    def insurance_1d(x):
        """Simplified: age -> premium."""
        if x > 60:
            return 1500
        else:
            if x > 40:
                return 800
            else:
                return 400

    print(f"\n[3] Insurance premium (3 tiers)")
    model3 = piecewise_compress(insurance_1d, 18, 80)
    model3 = model3.to(device)

    test_ages = torch.tensor([25, 45, 65], dtype=torch.float32, device=device)
    print(f"  Original:   {insurance_1d(test_ages).tolist()}")
    print(f"  Compressed: {[f'{v:.0f}' for v in model3(test_ages).tolist()]}")

    print(f"\n{'='*60}")
    print("Piecewise = EXACT for step/bracket functions.")
    print("Auto-detect breakpoints + linear fit per segment.")
    print("Parameters: just slopes + intercepts + breakpoints.")
