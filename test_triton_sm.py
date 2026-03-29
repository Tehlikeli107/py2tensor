import torch, triton, triton.language as tl, time

device = torch.device("cuda")

@triton.jit
def _softmax_kernel(inp, out, stride, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(inp + row * stride + offs, mask=mask, other=-float('inf'))
    x_max = tl.max(x, axis=0)
    x = x - x_max
    # Horner exp: degree 6
    e = 1.0/720.0
    e = e * x + 1.0/120.0
    e = e * x + 1.0/24.0
    e = e * x + 1.0/6.0
    e = e * x + 0.5
    e = e * x + 1.0
    e = e * x + 1.0
    e = tl.where(e > 0, e, 1e-8)
    s = tl.sum(e, axis=0)
    tl.store(out + row * stride + offs, e / s, mask=mask)

def triton_softmax(x):
    sh = x.shape
    x2 = x.reshape(-1, sh[-1]).contiguous()
    o = torch.empty_like(x2)
    N = x2.shape[1]
    BL = triton.next_power_of_2(N)
    _softmax_kernel[(x2.shape[0],)](x2, o, x2.stride(0), N, BL)
    return o.reshape(sh)

# Test
print("Testing Triton softmax...")
x = torch.randn(8, 16, device=device)
try:
    t = triton_softmax(x)
    s = torch.nn.functional.softmax(x, dim=-1)
    mae = (t - s).abs().mean().item()
    print(f"  MAE: {mae:.6f}")

    # Big
    x_big = torch.randn(32, 8, 512, 512, device=device)
    torch.cuda.synchronize()
    for _ in range(3): triton_softmax(x_big)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100): triton_softmax(x_big)
    torch.cuda.synchronize()
    t_tri = (time.time()-t0)/100

    for _ in range(3): torch.nn.functional.softmax(x_big, dim=-1)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100): torch.nn.functional.softmax(x_big, dim=-1)
    torch.cuda.synchronize()
    t_std = (time.time()-t0)/100

    print(f"  PyTorch: {t_std*1000:.2f}ms")
    print(f"  Triton:  {t_tri*1000:.2f}ms")
    print(f"  Speedup: {t_std/t_tri:.2f}x")
except Exception as e:
    import traceback
    traceback.print_exc()

# GELU
@triton.jit
def _gelu_kernel(inp, out, N: tl.constexpr, BL: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BL + tl.arange(0, BL)
    mask = offs < N
    x = tl.load(inp + offs, mask=mask, other=0.0)
    x3 = x * x * x
    inner = 0.7978845608 * (x + 0.044715 * x3)
    isq = inner * inner
    th = inner * (27.0 + isq) / (27.0 + 9.0 * isq)
    th = tl.where(inner > 3.0, 1.0, th)
    th = tl.where(inner < -3.0, -1.0, th)
    tl.store(out + offs, 0.5 * x * (1.0 + th), mask=mask)

def triton_gelu(x):
    o = torch.empty_like(x)
    N = x.numel()
    _gelu_kernel[(triton.cdiv(N, 1024),)](x.reshape(-1), o.reshape(-1), N, 1024)
    return o

print("\nTesting GELU...")
g = torch.randn(32, 512, 1024, device=device)
tg = triton_gelu(g)
sg = torch.nn.functional.gelu(g)
print(f"  GELU MAE: {(tg-sg).abs().mean().item():.6f}")

torch.cuda.synchronize()
for _ in range(3): triton_gelu(g)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(100): triton_gelu(g)
torch.cuda.synchronize()
tg_t = (time.time()-t0)/100

for _ in range(3): torch.nn.functional.gelu(g)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(100): torch.nn.functional.gelu(g)
torch.cuda.synchronize()
sg_t = (time.time()-t0)/100

print(f"  PyTorch: {sg_t*1000:.2f}ms | Triton: {tg_t*1000:.2f}ms | Speedup: {sg_t/tg_t:.2f}x")
