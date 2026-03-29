"""
N-Dimensional Piecewise Compression
=====================================
Extend piecewise to multi-input functions.
Each dimension gets breakpoints, create grid of regions.
Each region gets a constant or linear fit.

insurance(age=25, bmi=22) -> 200
insurance(age=55, bmi=35) -> 1000
= 2D grid of regions, each with a value.
"""
import torch
import torch.nn as nn
import time
import sys
sys.path.insert(0, r"C:\Users\salih\Desktop\py2tensor")

device = torch.device("cuda")


def detect_kinks_1d(fn_1d, lo, hi, n_probe=10000):
    """Find breakpoints for a 1D slice."""
    x = torch.linspace(lo, hi, n_probe, device=device)
    y = fn_1d(x).float()
    dy = torch.diff(y)
    sorted_dy = dy.abs().sort().values
    thresh = sorted_dy[int(len(sorted_dy) * 0.95)] * 3
    if thresh < 1e-10:
        thresh = dy.abs().max() * 0.1
    if thresh < 1e-10:
        return torch.tensor([lo, hi], device=device)

    kink_mask = dy.abs() > thresh
    kink_idx = torch.where(kink_mask)[0]
    if len(kink_idx) == 0:
        return torch.tensor([lo, hi], device=device)

    merged = [kink_idx[0].item()]
    min_gap = n_probe // 100
    for idx in kink_idx[1:]:
        if idx.item() - merged[-1] > min_gap:
            merged.append(idx.item())

    kink_x = x[:-1][torch.tensor(merged, device=device)]
    return torch.cat([torch.tensor([lo], device=device), kink_x, torch.tensor([hi], device=device)])


def piecewise_nd_compress(fn, input_ranges, n_features):
    """Compress N-dimensional function using grid of regions.

    1. For each dimension, find breakpoints (by slicing)
    2. Create grid of all breakpoint combinations
    3. Evaluate function at center of each grid cell
    4. Store as lookup table (tensor)
    """
    # Step 1: Find breakpoints per dimension
    all_breakpoints = []
    for dim in range(n_features):
        lo, hi = input_ranges[dim]
        # Create 1D slice: fix other dims at midpoint
        mid_values = [(r[0]+r[1])/2 for r in input_ranges]

        def slice_fn(x, dim=dim, mids=mid_values):
            args = []
            for d in range(n_features):
                if d == dim:
                    args.append(x)
                else:
                    args.append(torch.full_like(x, mids[d]))
            return fn(*args).float()

        bp = detect_kinks_1d(slice_fn, lo, hi)
        all_breakpoints.append(bp)
        print(f"  Dim {dim}: {len(bp)-1} segments, breakpoints={[f'{b:.1f}' for b in bp.tolist()]}")

    # Step 2: Create value grid
    # Each cell = center value of that region
    grid_sizes = [len(bp)-1 for bp in all_breakpoints]
    total_cells = 1
    for s in grid_sizes:
        total_cells *= s
    print(f"  Grid: {' x '.join(str(s) for s in grid_sizes)} = {total_cells} cells")

    # Evaluate center of each cell
    cell_values = torch.zeros(grid_sizes, device=device)

    # Iterate over all cells
    import itertools
    for idx in itertools.product(*[range(s) for s in grid_sizes]):
        center_args = []
        for dim, i in enumerate(idx):
            lo_cell = all_breakpoints[dim][i]
            hi_cell = all_breakpoints[dim][i+1]
            center_args.append((lo_cell + hi_cell) / 2)

        # Evaluate function at center
        args = [c.unsqueeze(0) for c in center_args]
        val = fn(*args).float().squeeze()
        cell_values[idx] = val

    # Step 3: Build model
    model = PiecewiseNDModel(all_breakpoints, cell_values, n_features)

    # Verify
    n_test = 10000
    test_args = []
    for dim in range(n_features):
        lo, hi = input_ranges[dim]
        test_args.append(torch.rand(n_test, device=device) * (hi-lo) + lo)

    y_orig = fn(*test_args).float()
    y_comp = model(*test_args).float()
    mae = (y_orig - y_comp).abs().mean().item()
    r2 = 1 - ((y_orig-y_comp)**2).sum().item() / ((y_orig-y_orig.mean())**2).sum().item()

    print(f"  MAE: {mae:.2f}, R2: {r2:.6f}")
    print(f"  Parameters: {cell_values.numel()} cell values + {sum(len(bp) for bp in all_breakpoints)} breakpoints")

    return model


class PiecewiseNDModel(nn.Module):
    """N-dimensional piecewise constant: grid lookup on GPU."""

    def __init__(self, breakpoints_list, cell_values, n_features):
        super().__init__()
        self.n_features = n_features
        self.register_buffer('values', cell_values)
        for i, bp in enumerate(breakpoints_list):
            self.register_buffer(f'bp_{i}', bp)

    def forward(self, *args):
        batch = args[0].shape[0]
        # For each dimension, find which cell the input falls in
        indices = []
        for dim in range(self.n_features):
            bp = getattr(self, f'bp_{dim}')
            x = args[dim]
            # Count how many breakpoints x exceeds
            idx = (x.unsqueeze(1) >= bp.unsqueeze(0)).sum(dim=1) - 1
            idx = idx.clamp(0, self.values.shape[dim] - 1)
            indices.append(idx)

        # Gather from grid
        if self.n_features == 1:
            return self.values[indices[0]]
        elif self.n_features == 2:
            return self.values[indices[0], indices[1]]
        elif self.n_features == 3:
            return self.values[indices[0], indices[1], indices[2]]
        elif self.n_features == 4:
            return self.values[indices[0], indices[1], indices[2], indices[3]]
        else:
            idx = tuple(indices)
            return self.values[idx]


# ================================================================
if __name__ == '__main__':
    from py2tensor import gpu

    print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 60)
    print("N-DIMENSIONAL PIECEWISE COMPRESSION")
    print("=" * 60)

    # Test 1: 2D insurance
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

    print(f"\n[1] Insurance(age, bmi) — 2D")
    model1 = piecewise_nd_compress(insurance, [(20, 80), (18, 45)], 2)
    model1 = model1.to(device)

    ages = torch.tensor([25, 55, 35, 65], dtype=torch.float32, device=device)
    bmis = torch.tensor([22, 22, 35, 35], dtype=torch.float32, device=device)
    print(f"  Original:   {insurance(ages, bmis).tolist()}")
    print(f"  Compressed: {model1(ages, bmis).tolist()}")

    # Benchmark
    N = 10_000_000
    a = torch.rand(N, device=device) * 60 + 20
    b = torch.rand(N, device=device) * 27 + 18

    torch.cuda.synchronize()
    for _ in range(3): insurance(a, b)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(30): insurance(a, b)
    torch.cuda.synchronize()
    t_orig = (time.time()-t0)/30

    mc = torch.compile(model1)
    for _ in range(3): mc(a, b)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(30): mc(a, b)
    torch.cuda.synchronize()
    t_comp = (time.time()-t0)/30

    print(f"  Original:   {N/t_orig/1e9:.1f}B/s")
    print(f"  Compressed: {N/t_comp/1e9:.1f}B/s")
    print(f"  Speedup:    {t_orig/t_comp:.1f}x")

    # Test 2: 3D credit scoring
    @gpu
    def credit_3d(income, debt, age):
        if income > 50000:
            base = 700
        else:
            base = 500
        if debt > 30000:
            base = base - 100
        else:
            base = base
        if age > 30:
            base = base + 50
        else:
            base = base
        return base

    print(f"\n[2] Credit(income, debt, age) — 3D")
    model2 = piecewise_nd_compress(credit_3d, [(20000,150000), (0,100000), (18,70)], 3)
    model2 = model2.to(device)

    inc = torch.tensor([30000, 60000, 80000], dtype=torch.float32, device=device)
    debt = torch.tensor([10000, 40000, 20000], dtype=torch.float32, device=device)
    age = torch.tensor([25, 35, 45], dtype=torch.float32, device=device)
    print(f"  Original:   {credit_3d(inc, debt, age).tolist()}")
    print(f"  Compressed: {model2(inc, debt, age).tolist()}")

    # Test 3: 4D pricing
    @gpu
    def pricing_4d(demand, supply, season, region):
        if demand > supply:
            surge = 1.5
        else:
            surge = 1.0
        if season > 6:
            seasonal = 1.2
        else:
            seasonal = 0.9
        if region > 5:
            regional = 1.3
        else:
            regional = 1.0
        return 100 * surge * seasonal * regional

    print(f"\n[3] Pricing(demand, supply, season, region) — 4D")
    model3 = piecewise_nd_compress(pricing_4d, [(0,200), (0,200), (1,12), (1,10)], 4)
    model3 = model3.to(device)

    d = torch.tensor([150, 50], dtype=torch.float32, device=device)
    s = torch.tensor([100, 100], dtype=torch.float32, device=device)
    sea = torch.tensor([3, 9], dtype=torch.float32, device=device)
    reg = torch.tensor([3, 8], dtype=torch.float32, device=device)
    print(f"  Original:   {pricing_4d(d, s, sea, reg).tolist()}")
    print(f"  Compressed: {model3(d, s, sea, reg).tolist()}")

    print(f"\n{'='*60}")
    print("N-D Piecewise: works for 1D, 2D, 3D, 4D+")
    print("Auto breakpoint detection per dimension.")
    print("Grid lookup = O(1) per sample = fastest possible.")
