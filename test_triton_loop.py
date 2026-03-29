import sys, torch, math, time
sys.path.insert(0, r"C:\Users\salih\Desktop\py2tensor")
from triton_backend import tensorize_triton

print("Creating Newton Triton kernel...", flush=True)
try:
    @tensorize_triton
    def newton_tr(x):
        g = x / 2
        for i in range(10):
            g = (g + x / g) / 2
        return g

    print(f"Generated Triton code:", flush=True)
    print(newton_tr._triton_source, flush=True)

    x = torch.rand(100, device='cuda') * 100 + 0.1
    out = newton_tr(x)
    expected = torch.sqrt(x)
    print(f"Output: {out[:5]}", flush=True)
    print(f"Expected: {expected[:5]}", flush=True)
    print(f"Match: {torch.allclose(out, expected, atol=1e-3)}", flush=True)

    # Benchmark
    N = 10_000_000
    x = torch.rand(N, device='cuda') * 1000 + 0.1
    torch.cuda.synchronize()
    for _ in range(3): newton_tr(x)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(30): newton_tr(x)
    torch.cuda.synchronize()
    t = (time.time() - t0) / 30
    print(f"Newton Triton: {N/t/1e9:.1f}B/s ({t*1000:.2f}ms)", flush=True)

except Exception as e:
    import traceback
    traceback.print_exc()
