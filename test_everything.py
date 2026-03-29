"""
ULTIMATE TEST: Can @tensorize_all handle REAL complex code?
Not toy examples — actual business logic, scientific formulas, game logic.
"""
import torch, math, time, sys
sys.path.insert(0, r"C:\Users\salih\Desktop\py2tensor")
from tensorize_all import tensorize_all

device = torch.device("cuda")
N = 10_000_000
print(f"GPU: {torch.cuda.get_device_name()}")
print("=" * 60)
print("ULTIMATE: Real-world complex code on GPU")
print("=" * 60)

passed = failed = 0
def check(name, cpu, gpu, tol=0.5):
    global passed, failed
    c = [float(v) for v in cpu] if isinstance(cpu, list) else [float(cpu)]
    g = gpu.cpu().tolist() if isinstance(gpu, torch.Tensor) else [float(gpu)]
    ok = all(abs(a-b) < tol for a, b in zip(c, g[:len(c)]))
    if ok: passed += 1
    else:
        failed += 1
        print(f"    CPU: {c[:5]}")
        print(f"    GPU: {g[:5]}")
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

# ================================================================
print("\n--- [1] Insurance Premium Calculator ---")

@tensorize_all
def insurance(age, bmi, smoker, claims):
    base = 200
    # Age factor
    if age > 60:
        age_f = 3.0
    else:
        if age > 40:
            age_f = 2.0
        else:
            age_f = 1.0
    # BMI factor
    if bmi > 35:
        bmi_f = 2.5
    else:
        if bmi > 30:
            bmi_f = 1.8
        else:
            if bmi > 25:
                bmi_f = 1.3
            else:
                bmi_f = 1.0
    # Smoker
    if smoker > 0.5:
        smoke_f = 2.0
    else:
        smoke_f = 1.0
    # Claims discount/surcharge
    if claims > 3:
        claim_f = 1.5
    else:
        if claims > 0:
            claim_f = 1.1
        else:
            claim_f = 0.9
    return base * age_f * bmi_f * smoke_f * claim_f

vals = [(25,22,0,0), (45,28,1,2), (65,35,0,5), (30,24,0,0)]
cpu = [insurance._original(*v) for v in vals]
gpu = insurance(*[torch.tensor([v[i] for v in vals], dtype=torch.float32, device=device) for i in range(4)])
check("Insurance 4-factor", cpu, gpu)

# ================================================================
print("\n--- [2] Tax Calculator (progressive brackets) ---")

@tensorize_all
def tax(income):
    if income > 500000:
        t = (income - 500000) * 0.37 + 150689
    else:
        if income > 200000:
            t = (income - 200000) * 0.32 + 54689
        else:
            if income > 80000:
                t = (income - 80000) * 0.22 + 28289
            else:
                if income > 40000:
                    t = (income - 40000) * 0.12 + 4689
                else:
                    if income > 10000:
                        t = (income - 10000) * 0.10 + 1000
                    else:
                        t = income * 0.10
    return t

incomes = [5000, 25000, 60000, 150000, 300000, 800000]
cpu = [tax._original(v) for v in incomes]
gpu = tax(torch.tensor(incomes, dtype=torch.float32, device=device))
check("Tax 6 brackets", cpu, gpu)

# ================================================================
print("\n--- [3] Projectile with Air Resistance (20 iter) ---")

@tensorize_all
def projectile_range(v0, angle_deg):
    g = 9.81
    drag = 0.01
    dt = 0.1
    angle = angle_deg * 3.14159 / 180
    vx = v0 * math.cos(angle)
    vy = v0 * math.sin(angle)
    x = 0
    y = 0
    for i in range(50):
        speed = math.sqrt(vx * vx + vy * vy)
        ax = -drag * speed * vx
        ay = -g - drag * speed * vy
        vx = vx + ax * dt
        vy = vy + ay * dt
        x = x + vx * dt
        y = y + vy * dt
    return x

v0s = [50, 100, 200]
angles = [30, 45, 60]
cpu = [projectile_range._original(v, a) for v, a in zip(v0s, angles)]
gpu = projectile_range(
    torch.tensor(v0s, dtype=torch.float32, device=device),
    torch.tensor(angles, dtype=torch.float32, device=device)
)
check("Projectile 50 iter + drag", cpu, gpu, tol=5)

# ================================================================
print("\n--- [4] Credit Scoring (weighted rules) ---")

@tensorize_all
def credit(payment_history, utilization, length, new_accounts, mix):
    # FICO-like scoring
    score = 300
    # Payment history (35%)
    score = score + payment_history * 3.5
    # Utilization (30%)
    if utilization < 10:
        score = score + 100
    else:
        if utilization < 30:
            score = score + 70
        else:
            if utilization < 50:
                score = score + 40
            else:
                score = score + 10
    # Credit length (15%)
    score = score + length * 1.5
    # New accounts (10%)
    if new_accounts > 5:
        score = score - 30
    else:
        score = score + 0
    # Credit mix (10%)
    score = score + mix * 10
    # Clamp
    if score > 850:
        return 850
    else:
        if score < 300:
            return 300
        else:
            return score

vals = [(95, 15, 20, 1, 3), (60, 45, 5, 8, 1), (100, 5, 30, 0, 5)]
cpu = [credit._original(*v) for v in vals]
gpu = credit(*[torch.tensor([v[i] for v in vals], dtype=torch.float32, device=device) for i in range(5)])
check("FICO credit score", cpu, gpu)

# ================================================================
print("\n--- [5] Black-Scholes Option Pricing ---")

@tensorize_all
def bs_call(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T) + 0.0001)
    d2 = d1 - sigma * math.sqrt(T)
    # Approximate normal CDF via sigmoid
    nd1 = 1.0 / (1.0 + math.exp(-1.7 * d1))
    nd2 = 1.0 / (1.0 + math.exp(-1.7 * d2))
    call = S * nd1 - K * math.exp(-r * T) * nd2
    if call < 0:
        return 0
    else:
        return call

params = [(100,100,1,0.05,0.2), (100,110,0.5,0.05,0.3), (50,45,2,0.03,0.25)]
cpu = [bs_call._original(*p) for p in params]
gpu = bs_call(*[torch.tensor([p[i] for p in params], dtype=torch.float32, device=device) for i in range(5)])
check("Black-Scholes call", cpu, gpu, tol=2)

# ================================================================
print("\n--- [6] Damped Spring Simulation (30 iter) ---")

@tensorize_all
def spring(x0, v0):
    k = 10.0
    m = 1.0
    damping = 0.5
    dt = 0.01
    x = x0
    v = v0
    for i in range(30):
        a = (-k * x - damping * v) / m
        v = v + a * dt
        x = x + v * dt
    return x

x0s = [1, 0, 2, -1]
v0s = [0, 5, -3, 2]
cpu = [spring._original(x, v) for x, v in zip(x0s, v0s)]
gpu = spring(
    torch.tensor(x0s, dtype=torch.float32, device=device),
    torch.tensor(v0s, dtype=torch.float32, device=device)
)
check("Damped spring 30 iter", cpu, gpu)

# ================================================================
print(f"\n--- MEGA BENCHMARK ---")
benchmarks = [
    ("Insurance 4-factor", insurance, [torch.rand(N,device=device)*40+20, torch.rand(N,device=device)*20+18, (torch.rand(N,device=device)>0.7).float(), torch.randint(0,5,(N,),device=device).float()]),
    ("Tax 6-bracket", tax, [torch.rand(N,device=device)*1000000]),
    ("Credit FICO", credit, [torch.rand(N,device=device)*100, torch.rand(N,device=device)*100, torch.rand(N,device=device)*30, torch.rand(N,device=device)*10, torch.rand(N,device=device)*5]),
]

for name, fn, args in benchmarks:
    torch.cuda.synchronize()
    for _ in range(3): fn(*args)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20): fn(*args)
    torch.cuda.synchronize()
    t = (time.time()-t0)/20
    print(f"  {name:<25} {N/t/1e9:>5.1f}B/s  {t*1000:.1f}ms")

# ================================================================
print(f"\n{'='*60}")
print(f"RESULTS: {passed} passed, {failed} failed")
print(f"{'='*60}")
print(f"""
  6 real-world functions, ALL on GPU via @tensorize_all:
  - Insurance (4 inputs, 4 nested if/else factors)
  - Tax (6 progressive brackets)
  - Projectile (50 iterations with air resistance)
  - Credit scoring (5 inputs, FICO-like weighted rules)
  - Black-Scholes (log, sqrt, exp, sigmoid, clamp)
  - Damped spring (30 iterations, coupled ODE)

  User changes: ZERO. Just add @tensorize_all.
""")
