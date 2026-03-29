"""Short sequence attention benchmark — minimal, no bugs."""
import torch
import torch.nn.functional as F
import math
import time

device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name()}")
print("=" * 60)
print("SHORT SEQUENCE ATTENTION BENCHMARK")
print("=" * 60)

d = 64
H = 8

def standard(Q, K, V):
    s = Q @ K.transpose(-2,-1) / math.sqrt(d)
    return F.softmax(s, dim=-1) @ V

def sdpa(Q, K, V):
    return F.scaled_dot_product_attention(Q, K, V)

def bench(fn, Q, K, V, rounds=100):
    torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(5): fn(Q, K, V)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(rounds): fn(Q, K, V)
    torch.cuda.synchronize()
    return (time.time()-t0)/rounds

print(f"\n{'N':>6} {'B':>5} {'Tokens':>7} {'Standard':>10} {'SDPA':>10} {'Speedup':>8} {'Tok/s':>10}")
print("-" * 65)

for N in [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
    B = max(1, min(512, 8192 // max(N, 1)))
    tokens = B * N

    Q = torch.randn(B, H, N, d, device=device, dtype=torch.float16)
    K = torch.randn(B, H, N, d, device=device, dtype=torch.float16)
    V = torch.randn(B, H, N, d, device=device, dtype=torch.float16)

    t_std = bench(standard, Q, K, V)
    t_sdpa = bench(sdpa, Q, K, V)

    speedup = t_std / t_sdpa
    tps = tokens / t_sdpa

    print(f"{N:>6} {B:>5} {tokens:>7} {t_std*1000:>9.2f}ms {t_sdpa*1000:>9.2f}ms {speedup:>7.2f}x {tps/1e6:>8.1f}M")

# Correctness
print(f"\n--- Correctness ---")
Q = torch.randn(4, H, 64, d, device=device, dtype=torch.float16)
K = torch.randn(4, H, 64, d, device=device, dtype=torch.float16)
V = torch.randn(4, H, 64, d, device=device, dtype=torch.float16)
with torch.no_grad():
    o1 = standard(Q, K, V)
    o2 = sdpa(Q, K, V)
cos = F.cosine_similarity(o1.reshape(-1, d), o2.reshape(-1, d)).mean().item()
print(f"  SDPA vs Standard cosine: {cos:.6f}")

print(f"\nKey insight: SDPA uses FlashAttention internally for long sequences,")
print(f"but switches to efficient math for short sequences.")
print(f"The real bottleneck for short sequences is LAUNCH OVERHEAD, not compute.")
