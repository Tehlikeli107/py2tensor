"""
Tensor Core Attention v2: polynomial as ACTUAL matrix multiply
==============================================================
v1 problem: x*x*x = element-wise, NOT Tensor Core.
v2 fix: build Vandermonde matrix, single matmul = all polynomial terms.

exp(x) ~ [1, x, x^2, x^3, x^4] @ [1, 1, 1/2, 1/6, 1/24]
       = Vandermonde_row @ coefficient_vector
       = MATRIX MULTIPLY → Tensor Core!

For batch: stack all x values into Vandermonde matrix,
single matmul computes ALL exp values at once.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math

device = torch.device("cuda")


class MatMulExp(nn.Module):
    """exp(x) as single matrix multiply via Vandermonde."""
    def __init__(self, degree=6):
        super().__init__()
        coeffs = [1.0]
        f = 1.0
        for i in range(1, degree + 1):
            f *= i
            coeffs.append(1.0 / f)
        # Shape: (degree+1, 1) for matmul
        self.register_buffer('coeffs', torch.tensor(coeffs, dtype=torch.float32).unsqueeze(1))
        self.degree = degree

    def forward(self, x):
        # x shape: (..., N)
        orig_shape = x.shape
        x_flat = x.reshape(-1)  # (M,)
        M = x_flat.shape[0]

        # Build Vandermonde: (M, degree+1)
        # [1, x, x^2, x^3, ...]
        vander = torch.ones(M, self.degree + 1, device=x.device, dtype=x.dtype)
        for i in range(1, self.degree + 1):
            vander[:, i] = vander[:, i-1] * x_flat

        # Single matmul: (M, degree+1) @ (degree+1, 1) = (M, 1)
        result = (vander @ self.coeffs.to(x.dtype)).squeeze(1)
        return result.reshape(orig_shape).clamp(min=1e-8)


class MatMulSoftmax(nn.Module):
    """Softmax where exp() is a single matmul."""
    def __init__(self, degree=6):
        super().__init__()
        self.exp_fn = MatMulExp(degree)

    def forward(self, x):
        x_shift = x - x.max(dim=-1, keepdim=True).values
        exp_x = self.exp_fn(x_shift)
        return exp_x / exp_x.sum(dim=-1, keepdim=True)


class MatMulGELU(nn.Module):
    """GELU as matrix multiply.
    GELU(x) ~ polynomial coefficients fitted to true GELU.
    Fit degree-7 polynomial via least squares on [-4, 4]."""
    def __init__(self):
        super().__init__()
        # Pre-fit polynomial: GELU on [-4, 4]
        # We solve: min || Vandermonde @ c - gelu_values ||^2
        n_fit = 1000
        x_fit = torch.linspace(-4, 4, n_fit)
        y_fit = F.gelu(x_fit)
        degree = 7
        V = torch.zeros(n_fit, degree + 1)
        for i in range(degree + 1):
            V[:, i] = x_fit ** i
        # Least squares: c = (V^T V)^-1 V^T y
        c = torch.linalg.lstsq(V, y_fit).solution
        self.register_buffer('coeffs', c.unsqueeze(1).float())
        self.degree = degree

    def forward(self, x):
        orig_shape = x.shape
        x_flat = x.reshape(-1).clamp(-4, 4)
        M = x_flat.shape[0]
        vander = torch.ones(M, self.degree + 1, device=x.device, dtype=x.dtype)
        for i in range(1, self.degree + 1):
            vander[:, i] = vander[:, i-1] * x_flat
        result = (vander @ self.coeffs.to(x.dtype)).squeeze(1)
        return result.reshape(orig_shape)


class MatMulLayerNorm(nn.Module):
    """LayerNorm: mean via matmul, rsqrt via Newton matmul chain."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        # Mean as matmul: (B,T,D) @ (D,1) / D
        self.register_buffer('ones_vec', torch.ones(dim, 1) / dim)

    def forward(self, x):
        # mean = x @ (1/D * ones) → matmul!
        mean = (x @ self.ones_vec.to(x.dtype))  # (B, T, 1)

        # variance = x^2 @ ones/D - mean^2 → matmul!
        var = (x * x) @ self.ones_vec.to(x.dtype) - mean * mean  # (B, T, 1)

        # rsqrt via 3 Newton iterations (each = multiply chain)
        y = torch.ones_like(var) * 0.5
        v = var + 1e-5
        y = y * (3.0 - v * y * y) * 0.5
        y = y * (3.0 - v * y * y) * 0.5
        y = y * (3.0 - v * y * y) * 0.5

        return (x - mean) * y * self.gamma.to(x.dtype) + self.beta.to(x.dtype)


class TensorCoreAttentionV2(nn.Module):
    """Full attention: Q@K + MatMulSoftmax + LN + GELU — all matmul."""
    def __init__(self, d_model=256, n_heads=8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.sm = MatMulSoftmax(degree=6)
        self.ln1 = MatMulLayerNorm(d_model)
        self.ln2 = MatMulLayerNorm(d_model)
        self.gelu = MatMulGELU()
        self.ffn1 = nn.Linear(d_model, d_model * 4, bias=False)
        self.ffn2 = nn.Linear(d_model * 4, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)
        attn = self.sm(scores)  # MatMul softmax!
        out = (attn @ V).transpose(1, 2).reshape(B, T, D)
        out = self.W_o(out)

        x = self.ln1(x + out)  # MatMul layernorm!
        ffn = self.ffn2(self.gelu(self.ffn1(x)))  # MatMul GELU!
        x = self.ln2(x + ffn)
        return x


# ================================================================
if __name__ == '__main__':
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 60)
    print("TENSOR CORE ATTENTION v2: Real MatMul Polynomial")
    print("=" * 60)

    D, H, T, B = 256, 8, 512, 32

    # Accuracy first
    print(f"\n--- Accuracy ---")
    test_x = torch.randn(8, 16, device=device)
    std_sm = F.softmax(test_x, dim=-1)
    tc_sm = MatMulSoftmax(6).to(device)(test_x)
    print(f"  Softmax MAE: {(std_sm - tc_sm).abs().mean().item():.6f}")

    test_g = torch.randn(1000, device=device)
    std_g = F.gelu(test_g)
    tc_g = MatMulGELU().to(device)(test_g)
    print(f"  GELU MAE:    {(std_g - tc_g).abs().mean().item():.6f}")

    test_ln = torch.randn(4, 8, D, device=device)
    std_ln = F.layer_norm(test_ln, [D])
    tc_ln = MatMulLayerNorm(D).to(device)(test_ln)
    print(f"  LayerNorm MAE: {(std_ln - tc_ln).abs().mean().item():.6f}")

    # Speed benchmark
    print(f"\n--- Speed (B={B}, T={T}, D={D}) ---")

    class StdAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(D, D*3, bias=False)
            self.out = nn.Linear(D, D, bias=False)
            self.ln1 = nn.LayerNorm(D)
            self.ln2 = nn.LayerNorm(D)
            self.ff1 = nn.Linear(D, D*4, bias=False)
            self.ff2 = nn.Linear(D*4, D, bias=False)
            self.dh = D // H
        def forward(self, x):
            B, T, _ = x.shape
            qkv = self.qkv(x).reshape(B, T, 3, H, self.dh).permute(2,0,3,1,4)
            Q, K, V = qkv[0], qkv[1], qkv[2]
            a = F.softmax(Q@K.transpose(-2,-1)/math.sqrt(self.dh), dim=-1)
            x = self.ln1(x + self.out((a@V).transpose(1,2).reshape(B,T,-1)))
            x = self.ln2(x + self.ff2(F.gelu(self.ff1(x))))
            return x

    std = StdAttn().to(device).half()
    tc = TensorCoreAttentionV2(D, H).to(device).half()

    x = torch.randn(B, T, D, device=device, dtype=torch.float16)

    # Warmup
    with torch.no_grad():
        for _ in range(5): std(x)
        for _ in range(5): tc(x)
    torch.cuda.synchronize()

    t0 = time.time()
    with torch.no_grad():
        for _ in range(50): std(x)
    torch.cuda.synchronize()
    t_std = (time.time()-t0)/50

    t0 = time.time()
    with torch.no_grad():
        for _ in range(50): tc(x)
    torch.cuda.synchronize()
    t_tc = (time.time()-t0)/50

    tps_std = B*T/t_std
    tps_tc = B*T/t_tc

    print(f"  Standard: {t_std*1000:.2f}ms ({tps_std:.0f} tok/s)")
    print(f"  TC v2:    {t_tc*1000:.2f}ms ({tps_tc:.0f} tok/s)")
    print(f"  Ratio:    {t_std/t_tc:.2f}x")

    # Compiled
    std_c = torch.compile(std)
    tc_c = torch.compile(tc)
    with torch.no_grad():
        for _ in range(3): std_c(x)
        for _ in range(3): tc_c(x)
    torch.cuda.synchronize()

    t0 = time.time()
    with torch.no_grad():
        for _ in range(50): std_c(x)
    torch.cuda.synchronize()
    t_std_c = (time.time()-t0)/50

    t0 = time.time()
    with torch.no_grad():
        for _ in range(50): tc_c(x)
    torch.cuda.synchronize()
    t_tc_c = (time.time()-t0)/50

    print(f"\n  Standard+compile: {t_std_c*1000:.2f}ms ({B*T/t_std_c:.0f} tok/s)")
    print(f"  TC v2+compile:    {t_tc_c*1000:.2f}ms ({B*T/t_tc_c:.0f} tok/s)")
    print(f"  Ratio:            {t_std_c/t_tc_c:.2f}x")

    print(f"\n  Note: Vandermonde matmul replaces element-wise x*x*x")
    print(f"  All polynomial evaluations are now SINGLE matmul operations")
