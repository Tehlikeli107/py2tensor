"""
Things we said were IMPOSSIBLE — now POSSIBLE with the model approach.
"""
import torch, sys
sys.path.insert(0, r"C:\Users\salih\Desktop\py2tensor")

device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name()}\n")
print("=" * 60)
print("'IMPOSSIBLE' -> POSSIBLE via model approach")
print("=" * 60)

# ================================================================
print("\n[1] STRING COMPARISON on GPU")
print("-" * 40)
# Strings as integer tensors
def str_to_tensor(s, max_len=20):
    t = torch.zeros(max_len, dtype=torch.long, device=device)
    for i, c in enumerate(s[:max_len]):
        t[i] = ord(c)
    return t

hello = str_to_tensor("hello")
world = str_to_tensor("world")
hello2 = str_to_tensor("hello")

# String equality = tensor equality
match1 = (hello == hello2).all().item()
match2 = (hello == world).all().item()
print(f"  'hello' == 'hello': {bool(match1)}")
print(f"  'hello' == 'world': {bool(match2)}")

# Batch: 10K string comparisons simultaneously
N_str = 10000
strings_a = torch.randint(97, 123, (N_str, 10), device=device)  # random "strings"
strings_b = strings_a.clone()
strings_b[::2] = torch.randint(97, 123, (N_str//2, 10), device=device)  # half different

torch.cuda.synchronize()
import time
t0 = time.time()
for _ in range(1000):
    matches = (strings_a == strings_b).all(dim=1)
torch.cuda.synchronize()
t = (time.time()-t0)/1000
print(f"  {N_str} string comparisons in {t*1e6:.0f}us")
print(f"  Matches: {matches.sum().item()}/{N_str}")
print(f"  [POSSIBLE] Strings = integer tensors, comparison = tensor ==")

# ================================================================
print(f"\n[2] TRY/EXCEPT -> SAFE MATH on GPU")
print("-" * 40)

# Instead of try: 1/x except: 0
# Do: torch.where(x != 0, 1/x, 0)
x = torch.tensor([-2, -1, 0, 1, 2], dtype=torch.float32, device=device)
safe_div = torch.where(x != 0, 1.0 / (x + (x == 0).float() * 1e-10), torch.zeros_like(x))
print(f"  1/x with safe zero: {safe_div.tolist()}")

# Safe sqrt (no negative)
vals = torch.tensor([-4, -1, 0, 1, 4], dtype=torch.float32, device=device)
safe_sqrt = torch.where(vals >= 0, torch.sqrt(torch.clamp(vals, min=0)), torch.zeros_like(vals))
print(f"  sqrt(x) safe: {safe_sqrt.tolist()}")

# Safe log
safe_log = torch.where(vals > 0, torch.log(torch.clamp(vals, min=1e-10)), torch.full_like(vals, float('-inf')))
print(f"  log(x) safe: {safe_log.tolist()}")
print(f"  [POSSIBLE] Try/except = condition check + mask")

# ================================================================
print(f"\n[3] DYNAMIC LIST (append) -> PRE-ALLOCATED TENSOR")
print("-" * 40)

# Instead of: results = []; for x in data: if x > 0: results.append(x)
# Do: pre-allocate, use mask
data = torch.randn(1000, device=device)
mask = data > 0
filtered = data[mask]  # "dynamic" filtering
print(f"  1000 elements, {mask.sum().item()} positive (filtered)")
print(f"  First 5 filtered: {filtered[:5].tolist()}")

# "Append" simulation: cumulative count
counts = mask.cumsum(dim=0)
print(f"  Cumulative count: [{counts[0].item()}, ..., {counts[-1].item()}]")
print(f"  [POSSIBLE] Dynamic list = mask + filter + cumsum")

# ================================================================
print(f"\n[4] DICTIONARY -> EMBEDDING TABLE on GPU")
print("-" * 40)

# Dict: {0: "free", 1: "basic", 2: "pro", 3: "enterprise"}
# Prices: {0: 0, 1: 9.99, 2: 29.99, 3: 99.99}
prices = torch.tensor([0, 9.99, 29.99, 99.99], dtype=torch.float32, device=device)
features = torch.tensor([1, 5, 20, 100], dtype=torch.float32, device=device)

# Batch lookup: 1M users with different plans
N_users = 1_000_000
user_plans = torch.randint(0, 4, (N_users,), device=device)

torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    user_prices = prices[user_plans]
    user_features = features[user_plans]
    total_revenue = user_prices.sum()
torch.cuda.synchronize()
t = (time.time()-t0)/100

print(f"  1M user plan lookups in {t*1e3:.2f}ms")
print(f"  Total revenue: ${total_revenue.item():,.0f}")
print(f"  [POSSIBLE] Dict = tensor embedding, lookup = index")

# ================================================================
print(f"\n[5] SORTING on GPU")
print("-" * 40)

data = torch.randn(N_users, device=device)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    sorted_data, indices = torch.sort(data)
torch.cuda.synchronize()
t = (time.time()-t0)/100
print(f"  Sort 1M elements: {t*1e3:.2f}ms")
print(f"  [POSSIBLE] torch.sort = GPU radix sort")

# ================================================================
print(f"\n[6] HASH TABLE -> MODULAR ARITHMETIC on GPU")
print("-" * 40)

# Simple hash: key % table_size -> index -> value
table_size = 1024
hash_table = torch.randn(table_size, device=device)  # values
keys = torch.randint(0, 1000000, (N_users,), device=device)
hash_indices = keys % table_size
values = hash_table[hash_indices]
print(f"  1M hash lookups: {values.mean().item():.4f} mean")
print(f"  [POSSIBLE] Hash = modular arithmetic + index")

# ================================================================
print(f"\n[7] STATE MACHINE / AUTOMATON on GPU")
print("-" * 40)

# Transition table as tensor: state x input -> next_state
# 3 states, 2 inputs
transitions = torch.tensor([
    [1, 0],  # state 0: input 0->state 1, input 1->state 0
    [2, 0],  # state 1: input 0->state 2, input 1->state 0
    [2, 2],  # state 2: input 0->state 2, input 1->state 2 (accept)
], dtype=torch.long, device=device)

# Run 10K sequences of length 8
N_seq = 10000
seq_len = 8
inputs = torch.randint(0, 2, (N_seq, seq_len), device=device)
states = torch.zeros(N_seq, dtype=torch.long, device=device)

for step in range(seq_len):
    inp = inputs[:, step]
    # Batch transition: gather from transition table
    states = transitions[states, inp]

accepted = (states == 2).sum().item()
print(f"  {N_seq} sequences of length {seq_len}: {accepted} accepted")
print(f"  [POSSIBLE] State machine = transition matrix + index")

# ================================================================
print(f"\n{'='*60}")
print("TRULY IMPOSSIBLE (hardware limit):")
print(f"{'='*60}")
print(f"""
  1. File read/write   - GPU has no disk controller
  2. Network access    - GPU has no NIC
  3. Screen output     - GPU renders but can't print text
  4. Keyboard input    - GPU has no input devices
  5. OS system calls   - GPU runs in isolated memory space

  EVERYTHING ELSE can be modeled as tensor operations.
""")
