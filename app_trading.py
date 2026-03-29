"""
Py2Tensor Trading App: GPU-accelerated trading signals
======================================================
Realistic trading indicator computations on 10M+ ticks.
"""
import torch, math, time, sys
sys.path.insert(0, r"C:\Users\salih\Desktop\py2tensor")
from py2tensor import tensorize, benchmark
from triton_backend import tensorize_triton

device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name()}")
print("=" * 60)
print("GPU TRADING SIGNALS")
print("=" * 60)

N = 10_000_000

# === Indicators ===
@tensorize(compile=True)
def ema_step(price, prev_ema, alpha):
    return alpha * price + (1 - alpha) * prev_ema

@tensorize(compile=True)
def bollinger_signal(price, mean, std):
    upper = mean + 2 * std
    lower = mean - 2 * std
    if price > upper:
        return -1
    else:
        if price < lower:
            return 1
        else:
            return 0

@tensorize(compile=True)
def momentum(price, prev_price):
    change = (price - prev_price) / (prev_price + 0.001)
    if change > 0.02:
        return 1
    else:
        if change < -0.02:
            return -1
        else:
            return 0

@tensorize(compile=True)
def position_size(signal, volatility, capital):
    risk = capital * 0.02
    size = risk / (volatility + 0.001)
    upper = capital * 0.1
    lower = -capital * 0.1
    raw = size * signal
    if raw > upper:
        return upper
    else:
        if raw < lower:
            return lower
        else:
            return raw

@tensorize(compile=True)
def kelly_fraction(win_rate, avg_win, avg_loss):
    """Kelly criterion for optimal bet sizing."""
    if avg_loss > 0:
        b = avg_win / (avg_loss + 0.001)
        kelly = win_rate - (1 - win_rate) / (b + 0.001)
        if kelly > 0:
            return kelly * 0.5
        else:
            return 0
    else:
        return 0

# === Generate market data ===
print(f"\nGenerating {N/1e6:.0f}M market ticks...")
prices = torch.cumsum(torch.randn(N, device=device) * 0.5, dim=0) + 100
prev_prices = torch.cat([prices[:1], prices[:-1]])
means = prices.clone()  # simplified
stds = torch.full((N,), 5.0, device=device)
capitals = torch.full((N,), 100000.0, device=device)

# === Run pipeline ===
print("Computing trading signals...")
torch.cuda.synchronize()
t0 = time.time()

# Signals
boll = bollinger_signal(prices, means, stds)
mom = momentum(prices, prev_prices)
combined = boll + mom

# Position sizing
vol = stds
sizes = position_size(combined, vol, capitals)

# Kelly
win_rates = torch.full((N,), 0.55, device=device)
avg_wins = torch.full((N,), 2.0, device=device)
avg_losses = torch.full((N,), 1.0, device=device)
kellys = kelly_fraction(win_rates, avg_wins, avg_losses)

torch.cuda.synchronize()
total_time = time.time() - t0

# Stats
buys = (combined > 0).sum().item()
sells = (combined < 0).sum().item()
holds = (combined == 0).sum().item()

print(f"\n  {N/1e6:.0f}M ticks processed in {total_time*1000:.0f}ms")
print(f"  Rate: {N/total_time/1e6:.0f}M ticks/sec")
print(f"  Signals: Buy={buys} ({100*buys/N:.1f}%), Sell={sells} ({100*sells/N:.1f}%), Hold={holds} ({100*holds/N:.1f}%)")
print(f"  Mean position size: ${sizes.mean().item():.0f}")
print(f"  Kelly fraction: {kellys.mean().item():.4f}")

# === Individual benchmarks ===
print(f"\nIndividual indicator benchmarks:")
benchmark(bollinger_signal, 100.0, 100.0, 5.0)
print()
benchmark(momentum, 101.0, 100.0)
print()
benchmark(kelly_fraction, 0.55, 2.0, 1.0)

# === Backtest simulation ===
print(f"\n{'='*60}")
print("BACKTEST: 1M ticks, full pipeline")
print("=" * 60)

N_bt = 1_000_000
bt_prices = torch.cumsum(torch.randn(N_bt, device=device) * 0.3, dim=0) + 100
bt_prev = torch.cat([bt_prices[:1], bt_prices[:-1]])

torch.cuda.synchronize()
t0 = time.time()

# Full pipeline
bt_mom = momentum(bt_prices, bt_prev)
bt_boll = bollinger_signal(bt_prices, bt_prices.mean(), bt_prices.std())
bt_signal = bt_mom + bt_boll
bt_returns = (bt_prices - bt_prev) / bt_prev * bt_signal

torch.cuda.synchronize()
bt_time = time.time() - t0

total_return = bt_returns.sum().item() * 100
sharpe = bt_returns.mean().item() / (bt_returns.std().item() + 1e-8) * (252 ** 0.5)

print(f"  1M ticks backtest in {bt_time*1000:.0f}ms")
print(f"  Total return: {total_return:.2f}%")
print(f"  Sharpe ratio: {sharpe:.2f}")
print(f"  Rate: {N_bt/bt_time/1e6:.0f}M ticks/sec")

print(f"\n  All computed with @tensorize. No CUDA code written.")
