import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Pocket Option V5 Backtester", page_icon="📊", layout="wide")

REQUIRED = ["timestamp", "market", "open", "high", "low", "close"]

def sample_data(n=20000, markets=("EURUSD", "GBPUSD", "USDJPY"), seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    for m_i, market in enumerate(markets):
        t = pd.date_range(start + pd.Timedelta(hours=m_i*2), periods=n, freq="1s")
        base = 1.08 + m_i * 0.25
        returns = rng.normal(0, 0.00006, n)
        # Add mild changing drift so the sample is not perfectly random.
        drift = 0.000012 * np.sin(np.arange(n) / 900)
        close = base * np.exp(np.cumsum(returns + drift))
        close = close / close[0] * base
        op = np.r_[close[0], close[:-1]]
        wiggle = np.abs(rng.normal(0, 0.000025, n))
        high = np.maximum(op, close) + wiggle
        low = np.minimum(op, close) - wiggle
        rows.append(pd.DataFrame({
            "timestamp": t,
            "market": market,
            "open": op, "high": high, "low": low, "close": close
        }))
    return pd.concat(rows, ignore_index=True)

def normalize(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    for c in ["open","high","low","close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["market"] = df["market"].astype(str)
    if "payout" in df:
        df["payout"] = pd.to_numeric(df["payout"], errors="coerce")
    else:
        df["payout"] = 0.80
    df = df.dropna(subset=REQUIRED).sort_values(["market","timestamp"]).reset_index(drop=True)
    return df

def backtest(df, fast=9, slow=21, expiry=60, stake=10, starting_balance=1000, payout=0.80, cooldown=0):
    trades = []
    for market, g in df.groupby("market", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True).copy()
        g["ema_fast"] = g["close"].ewm(span=fast, adjust=False).mean()
        g["ema_slow"] = g["close"].ewm(span=slow, adjust=False).mean()
        prev_fast = g["ema_fast"].shift(1)
        prev_slow = g["ema_slow"].shift(1)
        up = (g["ema_fast"] > g["ema_slow"]) & (prev_fast <= prev_slow)
        down = (g["ema_fast"] < g["ema_slow"]) & (prev_fast >= prev_slow)
        signal_idx = np.flatnonzero((up | down).fillna(False).to_numpy())
        last_end = -1
        for i in signal_idx:
            if i <= last_end or i + expiry >= len(g):
                continue
            direction = "UP" if bool(up.iloc[i]) else "DOWN"
            entry = float(g.loc[i, "close"])
            exit_i = i + expiry
            exit_price = float(g.loc[exit_i, "close"])
            win = exit_price > entry if direction == "UP" else exit_price < entry
            p = float(g.loc[i, "payout"]) if np.isfinite(g.loc[i, "payout"]) else payout
            if p > 1: p /= 100
            profit = stake * p if win else -stake
            trades.append({
                "timestamp": g.loc[i, "timestamp"], "market": market,
                "direction": direction, "entry": entry, "exit": exit_price,
                "expiry_seconds": expiry, "stake": stake, "payout": p,
                "result": "WIN" if win else "LOSS", "profit": profit
            })
            last_end = exit_i + cooldown
    t = pd.DataFrame(trades)
    if t.empty:
        return t
    t = t.sort_values("timestamp").reset_index(drop=True)
    t["balance"] = starting_balance + t["profit"].cumsum()
    t["hour"] = t["timestamp"].dt.hour
    t["date"] = t["timestamp"].dt.date.astype(str)
    return t

def max_losing_streak(results):
    best = cur = 0
    for x in results:
        if x == "LOSS":
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best

st.title("📊 Pocket Option V5 — Cloud Backtester")
st.caption("Mobile-friendly historical backtesting. This version analyzes CSV data; it does not place live trades.")

with st.sidebar:
    st.header("⚙️ Strategy")
    fast = st.number_input("Fast EMA", 2, 200, 9)
    slow = st.number_input("Slow EMA", 3, 400, 21)
    expiry = st.number_input("Expiry (seconds)", 1, 3600, 60)
    stake = st.number_input("Stake per trade", 0.01, 100000.0, 10.0, step=1.0)
    starting = st.number_input("Starting balance", 0.0, 10000000.0, 1000.0, step=100.0)
    payout_default = st.number_input("Default payout", 0.01, 2.0, 0.80, step=0.01)
    cooldown = st.number_input("Cooldown after trade (seconds)", 0, 3600, 0)
    st.divider()
    st.write("**CSV columns:** timestamp, market, open, high, low, close; optional payout.")

st.subheader("1. Load historical data")
uploaded = st.file_uploader("Upload CSV", type=["csv"])

col1, col2 = st.columns(2)
with col1:
    if st.button("🧪 Generate sample 1-second CSV", use_container_width=True):
        s = sample_data()
        st.session_state["data"] = s
        st.success(f"Generated {len(s):,} sample rows.")
with col2:
    if st.button("🗑️ Clear data", use_container_width=True):
        st.session_state.pop("data", None)
        st.session_state.pop("results", None)
        st.rerun()

if uploaded:
    try:
        st.session_state["data"] = normalize(pd.read_csv(uploaded))
        st.success(f"Loaded {len(st.session_state['data']):,} rows.")
    except Exception as e:
        st.error(str(e))

df = st.session_state.get("data")
if df is None:
    st.info("Upload a CSV or generate the sample data to begin.")
    st.stop()

st.write(f"**Rows:** {len(df):,}  |  **Markets:** {df['market'].nunique()}  |  **From:** {df['timestamp'].min()}  |  **To:** {df['timestamp'].max()}")
st.dataframe(df.head(10), use_container_width=True)

st.subheader("2. Run V5 backtest")
if st.button("▶️ Run Backtest", type="primary", use_container_width=True):
    if fast >= slow:
        st.error("Fast EMA must be smaller than Slow EMA.")
    else:
        st.session_state["results"] = backtest(df, fast, slow, expiry, stake, starting, payout_default, cooldown)

t = st.session_state.get("results")
if t is None:
    st.stop()
if t.empty:
    st.warning("No trades were generated. Try different EMA settings or more data.")
    st.stop()

wins = int((t.result == "WIN").sum())
losses = int((t.result == "LOSS").sum())
total = len(t)
winrate = wins / total * 100
net = float(t.profit.sum())
final_balance = float(t.balance.iloc[-1])
avg = float(t.profit.mean())
streak = max_losing_streak(t.result.tolist())
peak = t.balance.cummax()
drawdown = t.balance - peak
max_dd = float(drawdown.min())

st.subheader("3. Results")
metrics = st.columns(7)
metrics[0].metric("Trades", f"{total:,}")
metrics[1].metric("Wins", f"{wins:,}")
metrics[2].metric("Losses", f"{losses:,}")
metrics[3].metric("Win rate", f"{winrate:.2f}%")
metrics[4].metric("Net P/L", f"{net:+,.2f}")
metrics[5].metric("Max losing streak", f"{streak}")
metrics[6].metric("Max drawdown", f"{max_dd:,.2f}")

fig = px.line(t, x="timestamp", y="balance", title="Balance / Equity Curve")
st.plotly_chart(fig, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    hourly = t.groupby("hour").agg(trades=("profit","size"), net_pnl=("profit","sum"), win_rate=("result", lambda x: (x=="WIN").mean()*100)).reset_index()
    st.subheader("Hourly performance")
    st.dataframe(hourly.round(2), use_container_width=True)
    st.plotly_chart(px.bar(hourly, x="hour", y="net_pnl", title="Net P/L by Hour"), use_container_width=True)
with c2:
    market = t.groupby("market").agg(trades=("profit","size"), net_pnl=("profit","sum"), win_rate=("result", lambda x: (x=="WIN").mean()*100)).reset_index()
    st.subheader("Market performance")
    st.dataframe(market.round(2), use_container_width=True)
    st.plotly_chart(px.bar(market, x="market", y="net_pnl", title="Net P/L by Market"), use_container_width=True)

st.subheader("Trade history")
st.dataframe(t, use_container_width=True)
csv = t.to_csv(index=False).encode()
st.download_button("⬇️ Download trade results CSV", csv, "v5_trade_results.csv", "text/csv", use_container_width=True)

st.caption("Backtests depend on the supplied historical data, assumptions, payout, execution model and strategy rules. They do not guarantee future results.")
