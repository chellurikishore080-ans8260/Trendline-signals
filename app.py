# -*- coding: utf-8 -*-
"""Kishore Trend-Pullback Scanner + Backtest (Binance Spot USDT)."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Kishore Trend-Pullback Scanner", page_icon="📈", layout="wide")

HOSTS = [
    "https://data-api.binance.vision",
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
]
HEADERS = {"User-Agent": "Mozilla/5.0"}
STABLES = {"USDC", "FDUSD", "TUSD", "USDP", "DAI", "EUR", "AEUR", "USDE", "XUSD", "BUSD", "USD1", "UST", "USDS", "PYUSD", "RLUSD", "BFUSD", "EURI", "GBP", "TRY", "BRL"}
FALLBACK = ["BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "ADA", "AVAX", "LINK", "DOT", "LTC", "BCH", "TRX", "NEAR", "SUI", "APT", "ARB", "OP", "ATOM", "FIL", "INJ", "SEI", "TIA", "AAVE", "UNI", "ETC", "HBAR", "ICP", "RUNE", "PEPE", "SHIB", "TON", "WLD", "FET", "RENDER", "TAO", "ENA", "JUP", "ONDO", "PENDLE"]
TIMEFRAMES = ["15m", "30m", "1h", "4h", "1d"]


def _get(path, params=None):
    """Try Binance market-data mirror first, then alternate public API hosts."""
    failures = []
    for host in HOSTS:
        try:
            response = requests.get(host + path, params=params or {}, headers=HEADERS, timeout=8)
            if response.status_code == 200:
                return response.json()
            failures.append(f"{host}: HTTP {response.status_code}")
        except requests.RequestException as exc:
            failures.append(f"{host}: {type(exc).__name__}")
    raise RuntimeError("All Binance public data hosts failed: " + "; ".join(failures))


@st.cache_data(ttl=900, show_spinner=False)
def get_universe(n):
    """Rank live Binance Spot USDT markets by 24h quote volume; use a liquid fallback list if blocked."""
    try:
        data = _get("/api/v3/ticker/24hr")
        ranked = []
        for ticker in data:
            symbol = str(ticker.get("symbol", ""))
            if not symbol.endswith("USDT"):
                continue
            base = symbol[:-4]
            if base in STABLES or base.endswith(("UP", "DOWN", "BULL", "BEAR")):
                continue
            try:
                ranked.append((float(ticker.get("quoteVolume", 0)), symbol))
            except (TypeError, ValueError):
                continue
        ranked.sort(reverse=True)
        if ranked:
            return [symbol for _, symbol in ranked[:n]], "Live 24h quote-volume ranking"
    except Exception as exc:
        return [coin + "USDT" for coin in FALLBACK[:n]], f"Ticker endpoint unavailable; using fallback symbols ({exc})"
    return [coin + "USDT" for coin in FALLBACK[:n]], "Ticker data empty; using fallback symbols"


@st.cache_data(ttl=300, show_spinner=False)
def fetch_klines(symbol, interval, total):
    """Page backward through Binance's 1000-candle limit; discard the unfinished candle."""
    rows, end_time = [], None
    while len(rows) < total:
        limit = min(1000, total - len(rows))
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if end_time is not None:
            params["endTime"] = end_time
        batch = _get("/api/v3/klines", params)
        if not batch:
            break
        rows = batch + rows
        next_end = int(batch[0][0]) - 1
        if next_end >= (end_time if end_time is not None else next_end + 1) or len(batch) < limit:
            # A short response means this symbol has no older data available.
            if len(batch) < limit:
                break
        end_time = next_end
        if len(rows) >= total:
            break
    if not rows:
        raise RuntimeError("No candles returned (symbol may not be an available Binance Spot USDT pair).")
    frame = pd.DataFrame(rows).iloc[:, :7].copy()
    frame.columns = ["time", "open", "high", "low", "close", "volume", "close_time"]
    frame["time"] = pd.to_numeric(frame["time"], errors="coerce")
    frame["close_time"] = pd.to_numeric(frame["close_time"], errors="coerce")
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna().drop_duplicates("time").sort_values("time")
    frame = frame[frame["close_time"] < int(time.time() * 1000)]
    frame = frame.iloc[-total:].reset_index(drop=True)
    if len(frame) < 50:
        raise RuntimeError(f"Only {len(frame)} closed candles available; need at least 50.")
    frame["time"] = pd.to_datetime(frame["time"].astype("int64"), unit="ms", utc=True)
    return frame


def load_many(symbols, timeframe, candle_count, progress=None):
    results, errors = {}, {}
    max_workers = min(8, max(1, len(symbols)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_klines, symbol, timeframe, candle_count): symbol for symbol in symbols}
        for done, future in enumerate(as_completed(futures), 1):
            symbol = futures[future]
            try:
                results[symbol] = future.result()
            except Exception as exc:
                errors[symbol] = str(exc)
            if progress:
                progress(done / len(futures))
    return results, errors


def prepare(frame):
    d = frame.copy()
    close = d["close"]
    d["ema20"] = close.ewm(span=20, adjust=False).mean()
    d["ema50"] = close.ewm(span=50, adjust=False).mean()
    d["ema200"] = close.ewm(span=200, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    d["rsi"] = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).fillna(50)
    prev_close = close.shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - prev_close).abs(), (d["low"] - prev_close).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    return d


def make_signals(d, zone_period=20, use_structure=False):
    """Closed-bar EMA trend + pullback/rejection + RSI setup, no future candle data."""
    c, o, h, low, atr = d["close"], d["open"], d["high"], d["low"], d["atr"]
    ema_zone = d["ema20"] if zone_period == 20 else d["ema50"]
    candle_range = (h - low).replace(0, np.nan)
    long_trend = (c > d["ema200"]) & (d["ema50"] > d["ema200"]) & (d["ema50"] > d["ema50"].shift(5))
    short_trend = (c < d["ema200"]) & (d["ema50"] < d["ema200"]) & (d["ema50"] < d["ema50"].shift(5))
    touch = (low <= ema_zone + 0.3 * atr) & (h >= ema_zone - 0.3 * atr)
    bull_rejection = (c > o) & (c > ema_zone) & ((c - low) / candle_range >= 0.55)
    bear_rejection = (c < o) & (c < ema_zone) & ((h - c) / candle_range >= 0.55)
    long_signal = long_trend & touch & bull_rejection & d["rsi"].between(40, 65)
    short_signal = short_trend & touch & bear_rejection & d["rsi"].between(35, 60)
    if use_structure:
        # Confirm pivot only after three bars, preventing look-ahead.
        span = 3
        lows = d["low"].to_numpy()
        highs = d["high"].to_numpy()
        hl = np.zeros(len(d), dtype=bool)
        lh = np.zeros(len(d), dtype=bool)
        last_low = previous_low = last_high = previous_high = np.nan
        for i in range(len(d)):
            pivot = i - span
            if pivot >= span and pivot + span < len(d):
                if lows[pivot] == np.min(lows[pivot - span:pivot + span + 1]):
                    previous_low, last_low = last_low, lows[pivot]
                if highs[pivot] == np.max(highs[pivot - span:pivot + span + 1]):
                    previous_high, last_high = last_high, highs[pivot]
            hl[i] = np.isfinite(last_low) and np.isfinite(previous_low) and last_low > previous_low
            lh[i] = np.isfinite(last_high) and np.isfinite(previous_high) and last_high < previous_high
        long_signal &= pd.Series(hl, index=d.index)
        short_signal &= pd.Series(lh, index=d.index)
    out = d.copy()
    out["long_signal"] = long_signal.fillna(False)
    out["short_signal"] = short_signal.fillna(False)
    return out


def backtest(frame, symbol, timeframe, notional, fee_pct, stop_atr, reward_risk, max_bars, zone_period, use_structure):
    d = make_signals(prepare(frame), zone_period, use_structure)
    trades = []
    i = 0
    fee_rate = fee_pct / 100
    while i < len(d) - 1:
        row = d.iloc[i]
        side = 1 if row["long_signal"] else -1 if row["short_signal"] else 0
        if side == 0 or not np.isfinite(row["atr"]) or row["atr"] <= 0:
            i += 1
            continue
        entry_i = i + 1
        entry = float(d.iloc[entry_i]["open"])
        risk = float(row["atr"]) * stop_atr
        stop = entry - side * risk
        target = entry + side * risk * reward_risk
        exit_i, exit_price, reason = None, None, "Time exit"
        last_i = min(len(d) - 1, entry_i + max_bars - 1)
        for j in range(entry_i, last_i + 1):
            bar = d.iloc[j]
            hit_stop = float(bar["low"]) <= stop if side == 1 else float(bar["high"]) >= stop
            hit_target = float(bar["high"]) >= target if side == 1 else float(bar["low"]) <= target
            # Conservative if both are touched within the same OHLC candle: assume stop first.
            if hit_stop:
                exit_i, exit_price, reason = j, stop, "Stop loss"
                break
            if hit_target:
                exit_i, exit_price, reason = j, target, "Take profit"
                break
        if exit_i is None:
            exit_i, exit_price = last_i, float(d.iloc[last_i]["close"])
        entry_time, exit_time = d.iloc[entry_i]["time"], d.iloc[exit_i]["time"]
        qty = notional / entry
        gross = (float(exit_price) - entry) * qty * side
        fees = fee_rate * (notional + qty * float(exit_price))
        net = gross - fees
        trades.append({"Symbol": symbol, "Timeframe": timeframe, "Side": "LONG" if side == 1 else "SHORT",
                       "Entry time (UTC)": entry_time, "Entry": entry, "Exit time (UTC)": exit_time,
                       "Exit": float(exit_price), "Exit reason": reason, "Notional ($)": notional,
                       "Gross PnL ($)": gross, "Fees ($)": fees, "Net PnL ($)": net,
                       "Return (%)": net / notional * 100})
        i = exit_i + 1
    return trades


def render_results(trades):
    if not trades:
        st.warning("No trades met all selected conditions. Try a longer candle history, another timeframe, or turn off the optional structure filter.")
        return
    result = pd.DataFrame(trades).sort_values(["Exit time (UTC)", "Symbol"]).reset_index(drop=True)
    pnl = result["Net PnL ($)"]
    gains, losses = pnl[pnl > 0], pnl[pnl < 0]
    equity = pnl.cumsum()
    drawdown = equity.cummax() - equity
    cols = st.columns(5)
    cols[0].metric("Trades", f"{len(result):,}")
    cols[1].metric("Win rate", f"{(pnl > 0).mean() * 100:.1f}%")
    cols[2].metric("Net P&L", f"${pnl.sum():,.2f}")
    cols[3].metric("Profit factor", f"{gains.sum() / abs(losses.sum()):.2f}" if losses.sum() < 0 else "∞")
    cols[4].metric("Max drawdown", f"${drawdown.max():,.2f}")
    st.subheader("Cumulative realized P&L")
    st.line_chart(result.set_index("Exit time (UTC)")["Net PnL ($)"].cumsum().rename("P&L ($)"))
    st.subheader("Per-symbol results")
    st.dataframe(result.groupby("Symbol").agg(Trades=("Net PnL ($)", "size"), Net_PnL=("Net PnL ($)", "sum"), Win_rate_pct=("Net PnL ($)", lambda x: (x > 0).mean() * 100)).sort_values("Net_PnL", ascending=False), use_container_width=True)
    st.subheader("Trade log")
    st.dataframe(result.sort_values("Entry time (UTC)", ascending=False), use_container_width=True)
    st.download_button("Download trades CSV", result.to_csv(index=False), "trend_pullback_trades.csv", "text/csv")


st.title("📈 Kishore Trend-Pullback Scanner + Backtest")
st.caption("Binance Spot USDT candles • Binance Vision/API host fallback • closed candles only • no live orders")
with st.sidebar:
    st.header("Data")
    timeframe = st.selectbox("Timeframe", TIMEFRAMES, index=2)
    candle_count = st.select_slider("Historical candles per symbol", options=[500, 1000, 2000, 5000, 10000], value=2000)
    coin_count = st.slider("USDT symbols by 24h volume", min_value=5, max_value=40, value=20, step=5)
    st.header("Trend-pullback rules")
    zone = st.selectbox("Pullback EMA zone", [20, 50], index=0)
    use_structure = st.checkbox("Require confirmed higher-low / lower-high", value=False)
    stop_atr = st.number_input("Stop distance (ATR multiplier)", min_value=0.5, max_value=5.0, value=1.5, step=0.25)
    reward_risk = st.number_input("Target risk/reward", min_value=0.5, max_value=5.0, value=2.0, step=0.25)
    max_bars = st.number_input("Maximum holding candles", min_value=5, max_value=500, value=48, step=1)
    notional = st.number_input("USD notional per trade", min_value=10.0, value=1000.0, step=100.0)
    fee_pct = st.number_input("Fee per side (%)", min_value=0.0, value=0.1, step=0.01, format="%.3f")
    include_shorts = st.checkbox("Include shorts (research only; not ordinary spot)", value=False)

st.markdown("**Long:** close above EMA200, EMA50 above and rising; pullback touches EMA zone; bullish rejection candle; RSI 40–65. **Short:** mirrored downtrend, bearish rejection, RSI 35–60. Entry is next candle open; stop is ATR-based, target uses selected R:R, and timeout closes at candle close. When stop and target both fall inside one candle, backtest conservatively counts stop first.")
if st.button("Load Binance data and run backtest", type="primary"):
    symbols, universe_note = get_universe(coin_count)
    st.caption(universe_note)
    bar = st.progress(0.0, text=f"Loading {len(symbols)} {timeframe} Binance Spot series...")
    data, errors = load_many(symbols, timeframe, int(candle_count), lambda fraction: bar.progress(fraction, text=f"Loaded {fraction:.0%}"))
    bar.empty()
    valid = {symbol: frame for symbol, frame in data.items() if not frame.empty}
    st.session_state["trend_data_errors"] = errors
    st.session_state["trend_loaded"] = {symbol: len(frame) for symbol, frame in valid.items()}
    all_trades = []
    for symbol, frame in valid.items():
        all_trades.extend(backtest(frame, symbol, timeframe, notional, fee_pct, stop_atr, reward_risk, int(max_bars), zone, use_structure))
    if not include_shorts:
        all_trades = [trade for trade in all_trades if trade["Side"] == "LONG"]
    st.session_state["trend_trades"] = all_trades
    st.session_state["trend_run_settings"] = f"{timeframe} · {candle_count:,} candles requested · ${notional:,.0f} per trade"

if "trend_trades" in st.session_state:
    st.caption("Last run: " + st.session_state["trend_run_settings"])
    loaded = st.session_state.get("trend_loaded", {})
    if loaded:
        st.write("**Candles loaded:** " + " · ".join(f"{symbol}: {count:,}" for symbol, count in loaded.items()))
    errors = st.session_state.get("trend_data_errors", {})
    if errors:
        with st.expander(f"Symbols skipped ({len(errors)}) — data/API errors"):
            st.write("\n".join(f"{symbol}: {message}" for symbol, message in errors.items()))
    render_results(st.session_state["trend_trades"])
