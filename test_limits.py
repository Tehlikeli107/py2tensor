"""
What CAN and CAN'T be converted to GPU?
Test every common Python pattern.
"""
import torch, math, sys
sys.path.insert(0, r"C:\Users\salih\Desktop\py2tensor")
from pure_model import build_pure_model
from py2tensor import tensorize

device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name()}")
print("=" * 60)
print("WHAT WORKS AND WHAT DOESN'T")
print("=" * 60)

can = 0
cant = 0

def test(name, fn_builder, test_in, test_expected=None, use_pure=True):
    global can, cant
    try:
        if use_pure:
            model = build_pure_model(fn_builder)
            model = model.to(device)
            model = torch.compile(model)
        else:
            model = fn_builder

        if isinstance(test_in, tuple):
            out = model(*[torch.tensor([v], dtype=torch.float32, device=device) for v in test_in])
        else:
            out = model(torch.tensor([test_in], dtype=torch.float32, device=device))

        val = out.item() if out.numel() == 1 else out[0].item()
        if test_expected is not None:
            ok = abs(val - test_expected) < 1.0
        else:
            ok = True
        can += 1
        print(f"  [CAN]  {name:<40} = {val:.2f}")
    except Exception as e:
        cant += 1
        err = str(e)[:50]
        print(f"  [CANT] {name:<40} : {err}")

# === ARITHMETIC ===
print("\n--- Arithmetic ---")
def f1(x): return x * x + 2 * x + 1
test("x^2 + 2x + 1", f1, 3.0, 16.0)

def f2(x): return x ** 3 - x ** 2
test("x^3 - x^2", f2, 3.0, 18.0)

def f3(x): return (x + 1) * (x - 1)
test("(x+1)*(x-1)", f3, 5.0, 24.0)

# === MATH FUNCTIONS ===
print("\n--- Math Functions ---")
def f4(x): return math.sin(x) * math.cos(x)
test("sin(x)*cos(x)", f4, 1.0)

def f5(x): return math.exp(-x * x)
test("exp(-x^2)", f5, 1.0)

def f6(x): return math.log(x + 1)
test("log(x+1)", f6, math.e - 1, 1.0)

def f7(x): return math.sqrt(x)
test("sqrt(x)", f7, 9.0, 3.0)

def f8(x): return math.tanh(x)
test("tanh(x)", f8, 0.0, 0.0)

# === IF/ELSE ===
print("\n--- If/Else ---")
def f9(x):
    if x > 0: return x
    else: return 0
test("relu", f9, -5.0, 0.0)

def f10(x):
    if x > 0: return x
    else: return -x
test("abs via if/else", f10, -3.0, 3.0)

def f11(x):
    if x > 100: return 100
    else:
        if x < 0: return 0
        else: return x
test("clamp(0,100)", f11, 50.0, 50.0)

# === MULTI-VARIABLE IF ===
print("\n--- Multi-variable if/else ---")
def f12(x):
    if x > 0:
        a = x * 2
        b = x + 10
    else:
        a = 0
        b = -x
    return a + b
test("multi-assign if", f12, 5.0, 25.0)

# === FOR LOOP ===
print("\n--- For Loops ---")
def f13(x):
    g = x / 2
    for i in range(10):
        g = (g + x / g) / 2
    return g
test("Newton sqrt 10 iter", f13, 9.0, 3.0)

def f14(x):
    v = x
    for i in range(20):
        v = v * 0.9
    return v
test("decay 0.9^20", f14, 100.0, 100 * 0.9**20)

# === AUGMENTED ASSIGN ===
print("\n--- Augmented Assign ---")
def f15(x):
    r = x
    r += 10
    r *= 2
    return r
test("x += 10; x *= 2", f15, 5.0, 30.0)

# === TERNARY ===
print("\n--- Ternary ---")
@tensorize
def f16(x):
    return x * 2 if x > 0 else x + 1
test("ternary", f16, 5.0, 10.0, use_pure=False)

# === MULTI-INPUT ===
print("\n--- Multi-Input ---")
def f17(x, y):
    return x * x + y * y
test("x^2 + y^2", f17, (3.0, 4.0), 25.0)

def f18(x, y, z):
    if x > y:
        return x + z
    else:
        return y + z
test("3-input if/else", f18, (5.0, 3.0, 10.0), 15.0)

# === TUPLE RETURN ===
print("\n--- Tuple Return ---")
@tensorize
def f19(x, y):
    return math.sqrt(x*x + y*y), math.atan2(y, x)
try:
    r, t = f19(torch.tensor([3.0], device=device), torch.tensor([4.0], device=device))
    can += 1
    print(f"  [CAN]  {'tuple return (r, theta)':<40} = ({r.item():.2f}, {t.item():.2f})")
except Exception as e:
    cant += 1
    print(f"  [CANT] {'tuple return':<40} : {str(e)[:50]}")

# === THINGS THAT PROBABLY CAN'T WORK ===
print("\n--- Probably Can't Work ---")

# String operations
def f20(x):
    return len("hello") + x
test("string len + x", f20, 1.0)

# Recursion
def f21(x):
    if x < 1: return 1
    else: return x * f21(x - 1)
test("recursion (factorial)", f21, 5.0)

# Variable-length loop
def f22(x):
    count = 0
    while x > 1:
        x = x / 2
        count = count + 1
    return count
test("while with data-dependent termination", f22, 16.0)

# Dictionary lookup
def f23(x):
    table = {1: 10, 2: 20, 3: 30}
    return table.get(int(x), 0)
test("dict lookup", f23, 2.0)

# List operations
def f24(x):
    arr = [1, 2, 3, 4, 5]
    return arr[int(x)]
test("list index", f24, 2.0)

# Try/except
def f25(x):
    try:
        return 1 / x
    except:
        return 0
test("try/except", f25, 2.0)

# Print (side effect)
def f26(x):
    return x * 2
test("simple (print skipped)", f26, 5.0, 10.0)

# Nested function call
def f27(x):
    return abs(x) + max(x, 0)
test("abs + max", f27, -3.0)

# === SUMMARY ===
print(f"\n{'='*60}")
print(f"RESULTS: {can} CAN, {cant} CAN'T")
print(f"{'='*60}")
print(f"""
YAPILABILIR ({can}):
  - Aritmetik (+, -, *, /, **, %)
  - Math fonksiyonlar (sin, cos, exp, log, sqrt, tanh)
  - If/else (nested, multi-variable)
  - For loop (sabit range, unrolled)
  - Augmented assign (+=, *=)
  - Ternary (x if cond else y)
  - Multi-input (f(x, y, z))
  - Tuple return
  - Multi-assign if/else

YAPILAMAZ ({cant}):
  - Recursion (stack gerektirir)
  - Data-dependent while (ne zaman duracagi bilinmez)
  - Dictionary lookup (hash table GPU'da yok)
  - List indexing with variable (dinamik boyut)
  - String operations (GPU string desteklemez)
  - Try/except (exception handling GPU'da yok)
  - Side effects (print, file I/O)
""")
