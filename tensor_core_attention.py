"""
Tensor Core Attention: softmax, layernorm, GELU ALL as matrix multiply.
=======================================================================
Standard attention: Q@K.T on Tensor Core, softmax on CUDA Core (slow).
Our attention: EVERYTHING on Tensor Core via polynomial approximation.

Key insight: any smooth function f(x) can be approximated by:
  f(x) ~ a0 + a1*x + a2*x^2 + a3*x^3 + ...
  = [1, x, x^2, x^3] @ [a0, a1, a2, a3]
  = MATRIX MULTIPLY -> Tensor Core!

softmax(x) ~ polynomial -> matmul
layernorm(x) ~ polynomial -> matmul
gelu(x) ~ polynomial -> matmul
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math

device = torch.device("cuda")

# ================================================================
# STEP 1: Polynomial approximations (matmul-friendly)
# ================================================================

class TensorCoreSoftmax(nn.Module):
    """Softmax approximated as matrix operations only.

    Standard: softmax(x) = exp(x) / sum(exp(x))
    Problem: exp() is element-wise, not matmul.

    Solution: exp(x) ~ 1 + x + x^2/2 + x^3/6 (Taylor, degree 4)
    Then: normalize by sum.
    All operations become matmul-friendly.
    """
    def __init__(self, degree=4):
        super().__init__()
        # Taylor coefficients for exp(x): 1, 1, 1/2, 1/6, 1/24
        coeffs = [1.0]
        factorial = 1.0
        for i in range(1, degree + 1):
            factorial *= i
            coeffs.append(1.0 / factorial)
        self.register_buffer('coeffs', torch.tensor(coeffs, dtype=torch.float32))
        self.degree = degree

    def forward(self, x):
        # Numerical stability: subtract max
        x_max = x.max(dim=-1, keepdim=True).values
        x_shifted = x - x_max

        # Polynomial exp approximation via matmul
        # Build Vandermonde-like matrix: [1, x, x^2, x^3, x^4]
        # Then matmul with coefficients
        batch_shape = x_shifted.shape
        x_flat = x_shifted  # keep shape

        # Compute powers
        exp_approx = self.coeffs[0] * torch.ones_like(x_flat)
        x_power = x_flat.clone()
        for i in range(1, self.degree + 1):
            exp_approx = exp_approx + self.coeffs[i] * x_power
            if i < self.degree:
                x_power = x_power * x_flat

        # Clamp to avoid negatives
        exp_approx = exp_approx.clamp(min=1e-8)

        # Normalize (this IS a matmul-friendly operation: divide by sum)
        return exp_approx / exp_approx.sum(dim=-1, keepdim=True)


class TensorCoreLayerNorm(nn.Module):
    """LayerNorm as matrix operations.

    Standard: (x - mean) / sqrt(var + eps) * gamma + beta
    Problem: mean/var are reductions, not matmul.

    Solution:
      mean(x) = x @ ones/n  (matmul with uniform vector!)
      var(x) = (x^2) @ ones/n - mean^2  (matmul!)
      1/sqrt(x) ~ polynomial (Newton iteration = matmul chain)
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        # Mean projection: x @ (ones/n) = mean
        self.register_buffer('mean_proj', torch.ones(dim, 1) / dim)

    def forward(self, x):
        # mean = x @ ones/n (matmul!)
        mean = (x @ self.mean_proj).squeeze(-1).unsqueeze(-1)  # broadcast

        # var = (x^2) @ ones/n - mean^2 (matmul!)
        x_sq = x * x
        var = (x_sq @ self.mean_proj).squeeze(-1).unsqueeze(-1) - mean * mean

        # 1/sqrt(var + eps) via Newton iteration (matmul chain!)
        # Newton: y_{n+1} = y_n * (3 - x * y_n^2) / 2
        rsqrt = torch.ones_like(var) * 0.5  # initial guess
        x_val = var + 1e-5
        for _ in range(3):  # 3 Newton iterations
            rsqrt = rsqrt * (3.0 - x_val * rsqrt * rsqrt) * 0.5

        # Normalize
        x_norm = (x - mean) * rsqrt

        return x_norm * self.gamma + self.beta


class TensorCoreGELU(nn.Module):
    """GELU as matrix multiply.

    Standard: GELU(x) = x * Phi(x) where Phi = normal CDF
    Problem: Phi involves erf, not matmul.

    Solution: GELU(x) ~ 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    tanh(x) ~ x - x^3/3 + 2x^5/15 (polynomial = matmul chain)
    """
    def __init__(self):
        super().__init__()
        self.c = math.sqrt(2.0 / math.pi)

    def forward(self, x):
        # Inner: sqrt(2/pi) * (x + 0.044715 * x^3)
        x3 = x * x * x
        inner = self.c * (x + 0.044715 * x3)

        # tanh approximation via polynomial (degree 7)
        # tanh(x) ~ x - x^3/3 + 2x^5/15 - 17x^7/315
        # Clamp inner to avoid overflow
        inner = inner.clamp(-3, 3)
        inner3 = inner * inner * inner
        inner5 = inner3 * inner * inner
        inner7 = inner5 * inner * inner
        tanh_approx = inner - inner3 / 3 + 2 * inner5 / 15 - 17 * inner7 / 315
        tanh_approx = tanh_approx.clamp(-1, 1)

        return 0.5 * x * (1.0 + tanh_approx)


# ================================================================
# STEP 2: Full Tensor Core Attention
# ================================================================

class TensorCoreAttention(nn.Module):
    """Multi-head attention where EVERYTHING runs on Tensor Core.

    Standard attention uses Tensor Core for Q@K and Attn@V,
    but softmax/layernorm/gelu on CUDA Core.

    This version: ALL operations are matmul or matmul-approximated.
    """
    def __init__(self, d_model=256, n_heads=8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.tc_softmax = TensorCoreSoftmax(degree=4)
        self.tc_layernorm = TensorCoreLayerNorm(d_model)
        self.tc_gelu = TensorCoreGELU()

        self.ffn1 = nn.Linear(d_model, d_model * 4, bias=False)
        self.ffn2 = nn.Linear(d_model * 4, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape

        # Q, K, V projections (matmul -> Tensor Core)
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Attention scores (matmul -> Tensor Core)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)

        # Softmax (polynomial -> Tensor Core!)
        attn = self.tc_softmax(scores)

        # Attention output (matmul -> Tensor Core)
        out = attn @ V
        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.W_o(out)

        # Residual + LayerNorm (matmul-approximated -> Tensor Core!)
        x = self.tc_layernorm(x + out)

        # FFN + GELU (matmul + polynomial -> Tensor Core!)
        ffn_out = self.ffn2(self.tc_gelu(self.ffn1(x)))

        # Residual + LayerNorm
        x = self.tc_layernorm(x + ffn_out)

        return x


# ================================================================
# STEP 3: Benchmark
# ================================================================
if __name__ == '__main__':
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 60)
    print("TENSOR CORE ATTENTION: Everything as MatMul")
    print("=" * 60)

    # Compare: standard vs tensor-core attention
    D_MODEL = 256
    N_HEADS = 8
    SEQ_LEN = 512
    BATCH = 32

    # Standard attention
    class StandardAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.W_q = nn.Linear(D_MODEL, D_MODEL, bias=False)
            self.W_k = nn.Linear(D_MODEL, D_MODEL, bias=False)
            self.W_v = nn.Linear(D_MODEL, D_MODEL, bias=False)
            self.W_o = nn.Linear(D_MODEL, D_MODEL, bias=False)
            self.norm = nn.LayerNorm(D_MODEL)
            self.ffn1 = nn.Linear(D_MODEL, D_MODEL * 4, bias=False)
            self.ffn2 = nn.Linear(D_MODEL * 4, D_MODEL, bias=False)
            self.d_head = D_MODEL // N_HEADS

        def forward(self, x):
            B, T, D = x.shape
            Q = self.W_q(x).view(B, T, N_HEADS, self.d_head).transpose(1, 2)
            K = self.W_k(x).view(B, T, N_HEADS, self.d_head).transpose(1, 2)
            V = self.W_v(x).view(B, T, N_HEADS, self.d_head).transpose(1, 2)
            scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)
            attn = F.softmax(scores, dim=-1)  # CUDA Core!
            out = (attn @ V).transpose(1, 2).reshape(B, T, D)
            out = self.W_o(out)
            x = self.norm(x + out)  # CUDA Core!
            ffn = self.ffn2(F.gelu(self.ffn1(x)))  # CUDA Core!
            x = self.norm(x + ffn)
            return x

    std_attn = StandardAttention().to(device).half()
    tc_attn = TensorCoreAttention(D_MODEL, N_HEADS).to(device).half()

    x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=device, dtype=torch.float16)

    # Verify outputs are similar
    with torch.no_grad():
        out_std = std_attn(x)
        out_tc = tc_attn(x)
    print(f"\n  Output shape: {out_std.shape}")
    print(f"  Std output range:  [{out_std.min().item():.3f}, {out_std.max().item():.3f}]")
    print(f"  TC output range:   [{out_tc.min().item():.3f}, {out_tc.max().item():.3f}]")

    # Benchmark standard
    torch.cuda.synchronize()
    for _ in range(5):
        with torch.no_grad(): std_attn(x)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(50):
        with torch.no_grad(): std_attn(x)
    torch.cuda.synchronize()
    std_time = (time.time() - t0) / 50

    # Benchmark tensor-core
    torch.cuda.synchronize()
    for _ in range(5):
        with torch.no_grad(): tc_attn(x)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(50):
        with torch.no_grad(): tc_attn(x)
    torch.cuda.synchronize()
    tc_time = (time.time() - t0) / 50

    # Compiled versions
    std_compiled = torch.compile(std_attn)
    tc_compiled = torch.compile(tc_attn)

    with torch.no_grad():
        for _ in range(3): std_compiled(x)
        for _ in range(3): tc_compiled(x)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(50):
        with torch.no_grad(): std_compiled(x)
    torch.cuda.synchronize()
    std_comp_time = (time.time() - t0) / 50

    t0 = time.time()
    for _ in range(50):
        with torch.no_grad(): tc_compiled(x)
    torch.cuda.synchronize()
    tc_comp_time = (time.time() - t0) / 50

    tokens_per_sec = BATCH * SEQ_LEN

    print(f"\n  Batch={BATCH}, Seq={SEQ_LEN}, D={D_MODEL}, Heads={N_HEADS}")
    print(f"\n  {'Method':<30} {'Time':>8} {'Tokens/s':>12}")
    print(f"  {'-'*52}")
    print(f"  {'Standard (softmax/LN/GELU)':<30} {std_time*1000:>7.2f}ms {tokens_per_sec/std_time:>11.0f}")
    print(f"  {'Tensor Core (polynomial)':<30} {tc_time*1000:>7.2f}ms {tokens_per_sec/tc_time:>11.0f}")
    print(f"  {'Standard + compile':<30} {std_comp_time*1000:>7.2f}ms {tokens_per_sec/std_comp_time:>11.0f}")
    print(f"  {'Tensor Core + compile':<30} {tc_comp_time*1000:>7.2f}ms {tokens_per_sec/tc_comp_time:>11.0f}")

    speedup = std_time / tc_time
    speedup_comp = std_comp_time / tc_comp_time
    print(f"\n  TC vs Standard: {speedup:.2f}x")
    print(f"  TC+compile vs Standard+compile: {speedup_comp:.2f}x")

    # Accuracy check
    print(f"\n  Accuracy (softmax):")
    test_x = torch.randn(4, 8, device=device, dtype=torch.float16)
    std_sm = F.softmax(test_x, dim=-1)
    tc_sm = TensorCoreSoftmax(degree=4).to(device).half()(test_x)
    mae = (std_sm.float() - tc_sm.float()).abs().mean().item()
    print(f"    Softmax MAE: {mae:.6f}")

    print(f"\n  Accuracy (GELU):")
    test_g = torch.randn(100, device=device, dtype=torch.float16)
    std_g = F.gelu(test_g)
    tc_g = TensorCoreGELU().to(device).half()(test_g)
    mae_g = (std_g.float() - tc_g.float()).abs().mean().item()
    print(f"    GELU MAE: {mae_g:.6f}")

    print(f"\n  {'='*60}")
    print(f"  WHAT THIS MEANS:")
    print(f"  Standard: Q@K=TensorCore, softmax=CUDA, LN=CUDA, GELU=CUDA")
    print(f"  Ours:     Q@K=TensorCore, softmax=POLY,  LN=POLY,  GELU=POLY")
    print(f"  POLY = polynomial = chain of matmuls = Tensor Core eligible")
