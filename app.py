import streamlit as st
import os
from twelvedata import TDClient
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Pocket Option Analysis Bot", page_icon="📊", layout="wide")

st.title("📊 Pocket Option — Analysis Dashboard")
st.caption("Analysis/demo only • No broker connection • No automatic trading or real-money execution")

st.warning(
    "This dashboard does not connect to Pocket Option and cannot read its live chart. "
    "Use a CSV containing historical candles, or the clearly labelled sample data. "
    "Signals are educational indicators, not guaranteed trade instructions."
)

def make_sample_data(rows=300, seconds=5):
    end = datetime.utcnow()
    times = [end - timedelta(seconds=seconds * (rows - 1 - i)) for i in range(rows)]
    rng = np.random.default_rng(7)
    changes = rng.normal(0, 0.00035, rows)
    close = 1.085 + np.cumsum(changes)
    open_ = np.r_[close[0], close[:-1]]
    spread = rng.uniform(0.00005, 0.00025, rows)
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    return pd.DataFrame({"time": times, "open": open_, "high": high, "low": low, "close": close})

def clean_data(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    aliases = {"timestamp": "time", "date": "time", "datetime": "time",
               "o": "open", "h": "high", "l": "low", "c": "close"}
    df = df.rename(columns=aliases)
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("CSV must contain open, high, low and close columns.")
    if "time" not in df.columns:
        df["time"] = np.arange(len(df))
    else:
        parsed = pd.to_datetime(df["time"], errors="coerce", utc=True)
        df["time"] = parsed.where(parsed.notna(), df["time"].astype(str))
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required).reset_index(drop=True)
    if len(df) < 30:
        raise ValueError("Please provide at least 30 valid candle rows.")
    return df

with st.sidebar:
    st.header("Settings")
    fast = st.number_input("Fast EMA", min_value=2, max_value=50, value=9)
    slow = st.number_input("Slow EMA", min_value=3, max_value=200, value=21)
    rsi_period = st.number_input("RSI period", min_value=2, max_value=50, value=14)
    st.caption("Signals are generated only from the uploaded or sample candle data.")

source = st.radio("Choose data source", ["Sample demo data", "Upload candle CSV"], horizontal=True)
if source == "Upload candle CSV":
    uploaded = st.file_uploader("Upload CSV candles", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV with columns: time (optional), open, high, low, close.")
        st.stop()
    try:
        raw = clean_data(pd.read_csv(uploaded))
        source_label = "Uploaded CSV"
    except Exception as exc:
        st.error(str(exc))
        st.stop()
else:
    raw = make_sample_data()
    source_label = "SAMPLE DATA — NOT LIVE MARKET DATA"

if fast >= slow:
    st.error("Fast EMA must be smaller than Slow EMA.")
    st.stop()

df = raw.copy()
df["ema_fast"] = df["close"].ewm(span=int(fast), adjust=False).mean()
df["ema_slow"] = df["close"].ewm(span=int(slow), adjust=False).mean()

delta = df["close"].diff()
gain = delta.clip(lower=0).ewm(alpha=1 / int(rsi_period), adjust=False).mean()
loss = (-delta.clip(upper=0)).ewm(alpha=1 / int(rsi_period), adjust=False).mean()
rs = gain / loss.replace(0, np.nan)
df["rsi"] = (100 - (100 / (1 + rs))).fillna(50)

df["signal"] = "WAIT"
cross_up = (df["ema_fast"] > df["ema_slow"]) & (df["ema_fast"].shift(1) <= df["ema_slow"].shift(1))
cross_down = (df["ema_fast"] < df["ema_slow"]) & (df["ema_fast"].shift(1) >= df["ema_slow"].shift(1))
df.loc[cross_up & (df["rsi"] < 70), "signal"] = "CALL"
df.loc[cross_down & (df["rsi"] > 30), "signal"] = "PUT"

last = df.iloc[-1]
previous = df.iloc[-2]
latest_signal = last["signal"]
signal_color = "green" if latest_signal == "CALL" else ("red" if latest_signal == "PUT" else "gray")

st.subheader(source_label)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest close", f'{last["close"]:.5f}')
c2.metric("Fast EMA", f'{last["ema_fast"]:.5f}')
c3.metric("Slow EMA", f'{last["ema_slow"]:.5f}')
c4.metric("RSI", f'{last["rsi"]:.1f}')

st.markdown(f"### Latest signal: :{signal_color}[{latest_signal}]")
st.caption("A signal appears only when the fast EMA crosses the slow EMA and RSI passes the simple filter. WAIT means no new crossover on the latest candle.")

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
    name="Candles"
))
fig.add_trace(go.Scatter(x=df["time"], y=df["ema_fast"], name=f"EMA {fast}", line=dict(width=1.5)))
fig.add_trace(go.Scatter(x=df["time"], y=df["ema_slow"], name=f"EMA {slow}", line=dict(width=1.5)))
calls = df[df["signal"] == "CALL"]
puts = df[df["signal"] == "PUT"]
fig.add_trace(go.Scatter(x=calls["time"], y=calls["low"], mode="markers", name="CALL",
                         marker=dict(symbol="triangle-up", size=11)))
fig.add_trace(go.Scatter(x=puts["time"], y=puts["high"], mode="markers", name="PUT",
                         marker=dict(symbol="triangle-down", size=11)))
fig.update_layout(height=560, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Signal history")
history = df[df["signal"].isin(["CALL", "PUT"])][["time", "close", "signal", "ema_fast", "ema_slow", "rsi"]].tail(30)
if history.empty:
    st.info("No EMA crossover signals found in this data set.")
else:
    st.dataframe(history.iloc[::-1], use_container_width=True, hide_index=True)

st.subheader("Simple historical signal count")
counts = df["signal"].value_counts()
m1, m2, m3 = st.columns(3)
m1.metric("CALL signals", int(counts.get("CALL", 0)))
m2.metric("PUT signals", int(counts.get("PUT", 0)))
m3.metric("WAIT candles", int(counts.get("WAIT", 0)))

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download analysed candles (CSV)", data=csv, file_name="pocket_option_analysis.csv", mime="text/csv")

st.caption(
    "Important: Sample data is synthetic. This app does not provide verified 5-second Pocket Option prices, "
    "does not calculate a reliable win rate, and does not place trades. Test any strategy independently on a demo account."
)
