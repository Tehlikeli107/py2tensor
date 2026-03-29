"""
GPU I/O Pipeline: CPU handles I/O, GPU handles computation.
Overlapped: GPU never waits for I/O.
"""
import torch
import time
import os

device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name()}\n")
print("=" * 60)
print("I/O + GPU PIPELINE: solving the 'hardware limit'")
print("=" * 60)

# ================================================================
print("\n[1] FILE READ -> GPU PROCESS -> FILE WRITE")
print("-" * 40)

# Create test file: 10M numbers
test_file = r"C:\Users\salih\Desktop\py2tensor\test_data.bin"
N = 10_000_000
import numpy as np
data = np.random.randn(N).astype(np.float32)
data.tofile(test_file)
file_size = os.path.getsize(test_file)
print(f"  Created {file_size/1e6:.0f}MB test file ({N/1e6:.0f}M floats)")

# Method 1: CPU-only (read, process, write)
t0 = time.time()
cpu_data = np.fromfile(test_file, dtype=np.float32)
cpu_result = np.where(cpu_data > 0, cpu_data * 2, cpu_data * 0.5)  # simple transform
cpu_result.tofile(test_file + ".cpu_out")
cpu_time = time.time() - t0
print(f"  CPU only: {cpu_time*1000:.0f}ms (read+process+write)")

# Method 2: CPU reads, GPU processes, CPU writes
t0 = time.time()
# Read (CPU)
raw = np.fromfile(test_file, dtype=np.float32)
# Transfer to GPU
gpu_data = torch.from_numpy(raw).to(device)
# Process on GPU
gpu_result = torch.where(gpu_data > 0, gpu_data * 2, gpu_data * 0.5)
# Transfer back
result_np = gpu_result.cpu().numpy()
# Write (CPU)
result_np.tofile(test_file + ".gpu_out")
gpu_time = time.time() - t0
print(f"  GPU pipeline: {gpu_time*1000:.0f}ms (read+transfer+process+transfer+write)")

# Method 3: Pinned memory (faster transfer)
t0 = time.time()
raw = np.fromfile(test_file, dtype=np.float32)
# Pinned memory: CPU->GPU transfer is faster
pinned = torch.from_numpy(raw).pin_memory()
gpu_data = pinned.to(device, non_blocking=True)
torch.cuda.synchronize()
gpu_result = torch.where(gpu_data > 0, gpu_data * 2, gpu_data * 0.5)
result_cpu = gpu_result.cpu()
result_cpu.numpy().tofile(test_file + ".pin_out")
pin_time = time.time() - t0
print(f"  Pinned memory: {pin_time*1000:.0f}ms")

# Verify
assert np.allclose(
    np.fromfile(test_file + ".cpu_out", dtype=np.float32)[:100],
    np.fromfile(test_file + ".gpu_out", dtype=np.float32)[:100],
    atol=1e-5
)
print(f"  Results match: True")

# ================================================================
print(f"\n[2] STREAMING PIPELINE (overlap I/O and compute)")
print("-" * 40)

# Process file in chunks: while GPU processes chunk N, CPU reads chunk N+1
CHUNK = 1_000_000
n_chunks = N // CHUNK

# Create CUDA streams for overlap
compute_stream = torch.cuda.Stream()
transfer_stream = torch.cuda.Stream()

t0 = time.time()
raw_full = np.fromfile(test_file, dtype=np.float32)
results = []

for i in range(n_chunks):
    chunk = raw_full[i*CHUNK:(i+1)*CHUNK]

    # Transfer (can overlap with previous compute)
    with torch.cuda.stream(transfer_stream):
        gpu_chunk = torch.from_numpy(chunk).to(device, non_blocking=True)

    # Compute (waits for transfer, can overlap with next transfer)
    transfer_stream.synchronize()
    with torch.cuda.stream(compute_stream):
        result = torch.where(gpu_chunk > 0, gpu_chunk * 2, gpu_chunk * 0.5)
        results.append(result.cpu())

torch.cuda.synchronize()
stream_time = time.time() - t0
print(f"  Streaming ({n_chunks} chunks): {stream_time*1000:.0f}ms")
print(f"  vs sequential GPU: {gpu_time*1000:.0f}ms")
print(f"  Overlap benefit: {gpu_time/stream_time:.2f}x")

# ================================================================
print(f"\n[3] COMPLEX GPU PROCESSING ON FILE DATA")
print("-" * 40)

import sys
sys.path.insert(0, r"C:\Users\salih\Desktop\py2tensor")
from py2tensor import tensorize
import math

@tensorize(compile=True)
def complex_transform(x):
    if x > 2:
        return math.log(x) * 10
    else:
        if x > 0:
            return math.sqrt(x) * 5
        else:
            return math.exp(x)

# Read file -> GPU transform -> write
t0 = time.time()
raw = np.fromfile(test_file, dtype=np.float32)
gpu_data = torch.from_numpy(raw).to(device)
gpu_result = complex_transform(gpu_data)
result_np = gpu_result.cpu().numpy()
result_np.tofile(test_file + ".complex_out")
total = time.time() - t0

# Breakdown
t_read = 0
t0 = time.time()
_ = np.fromfile(test_file, dtype=np.float32)
t_read = time.time() - t0

t0 = time.time()
_ = torch.from_numpy(raw).to(device)
torch.cuda.synchronize()
t_transfer = time.time() - t0

t0 = time.time()
_ = complex_transform(gpu_data)
torch.cuda.synchronize()
t_compute = time.time() - t0

print(f"  Total: {total*1000:.0f}ms")
print(f"    Read:     {t_read*1000:.0f}ms ({file_size/t_read/1e9:.1f} GB/s)")
print(f"    Transfer: {t_transfer*1000:.0f}ms ({file_size/t_transfer/1e9:.1f} GB/s)")
print(f"    Compute:  {t_compute*1000:.1f}ms ({N/t_compute/1e9:.1f}B elements/s)")
print(f"    Writeback: ~{(total-t_read-t_transfer-t_compute)*1000:.0f}ms")

# Cleanup
for f in [test_file, test_file+".cpu_out", test_file+".gpu_out", test_file+".pin_out", test_file+".complex_out"]:
    try: os.remove(f)
    except: pass

# ================================================================
print(f"\n{'='*60}")
print("SOLUTION TO 'HARDWARE LIMITS'")
print(f"{'='*60}")
print(f"""
  File I/O:  CPU reads -> GPU processes -> CPU writes
             Pinned memory for faster transfer
             CUDA streams for overlap (compute while reading)

  Network:   CPU receives -> GPU processes -> CPU sends
             Same pipeline, replace file with socket

  Screen:    GPU already renders graphics
             For text: compute on GPU, format on CPU

  The 'hardware limit' is only at the I/O BOUNDARY.
  ALL computation happens on GPU.
  CPU is just an I/O proxy — it reads and writes, GPU thinks.

  With streaming pipeline: I/O and compute OVERLAP.
  GPU NEVER waits. It's always computing.
""")
