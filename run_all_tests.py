"""Run all Py2Tensor tests and produce summary report."""
import subprocess
import sys
import time

test_files = [
    "tests.py",
    "tests_advanced.py",
    "tests_math.py",
    "tests_structures.py",
    "tests_ternary.py",
    "tests_numpy_dtype.py",
    "tests_advanced2.py",
    "tests_multiblock.py",
    "test_multistatement.py",
    "test_realworld.py",
    "test_while_minmax.py",
    "test_profile.py",
]

print("=" * 60)
print("PY2TENSOR — FULL TEST SUITE")
print("=" * 60)

total_pass = 0
total_fail = 0
t0 = time.time()

for tf in test_files:
    print(f"\n--- {tf} ---")
    result = subprocess.run(
        [sys.executable, "-u", tf],
        capture_output=True, text=True, timeout=120,
        cwd=r"C:\Users\salih\Desktop\py2tensor"
    )
    output = result.stdout + result.stderr

    # Count passes and fails
    for line in output.split("\n"):
        if "[PASS]" in line:
            total_pass += 1
        elif "[FAIL]" in line:
            total_fail += 1
        if "SPEEDUP" in line or "RESULTS" in line or "M/s" in line:
            print(f"  {line.strip()}")

elapsed = time.time() - t0
print(f"\n{'=' * 60}")
print(f"GRAND TOTAL: {total_pass} passed, {total_fail} failed ({elapsed:.0f}s)")
print(f"{'=' * 60}")
