# -*- coding: utf-8 -*-
"""Kishore Trend-Pullback Scanner + Backtest — EMA/price-action only, no RSI."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Kishore Trend-Pullback Backtest", page_icon="📈", layout="wide")

HOSTS = [
    "https://data-api.binance.vision", "https://api.binance.com",
    "https://api1.binance.com", "https://api2.binance.com",
    "https://api3.binance.com", "https://api4.binance.com",
]
HEADERS = {"User-Agent": "Mozilla/5.0"}
STABLES = {"USDC", "FDUSD", "TUSD", "USDP", "DAI", "EUR", "AEUR", "USDE", "XUSD", "BUSD", "USD1", "UST", "USDS", "PYUSD", "RLUSD", "BFUSD", "EURI", "GBP", "TRY", "BRL"}
FALLBACK = ["BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "ADA", "AVAX", "LINK", "DOT", "LTC", "BCH", "TRX", "NEAR", "SUI", "APT", "ARB", "OP", "ATOM", "FIL", "INJ", "SEI", "TIA", "AAVE", "UNI", "ETC", "HBAR", "ICP", "RUNE", "PEPE", "SHIB", "TON", "WLD", "FET", "RENDER", "TAO", "ENA", "JUP", "ONDO", "PENDLE"]
TIMEFRAMES = ["15m", "30m", "1h", "4h", "1d"]


def _get(path, params=None):
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
    try:
        tickers = _get("/api/v3/ticker/24hr")
        ranked = []
        for item in tickers:
            symbol = str(item.get("symbol", ""))
            if not symbol.endswith("USDT"):
                continue
            base = symbol[:-4]
            if base in STABLES or base.endswith(("UP", "DOWN", "BULL", "BEAR")):
                continue
            try:
                ranked.append((float(item.get("quoteVolume", 0)), symbol))
            except (TypeError, ValueError):
                continue
        ranked.sort(reverse=True)
        if ranked:
            return [symbol for _, symbol in ranked[:n]], "Current Binance Spot USDT pairs ranked by 24h quote volume"
    except Exception as exc:
        return [coin + "USDT" for coin in FALLBACK[:n]], f"Ticker endpoint unavailable; fallback symbols used: {exc}"
    return [coin + "USDT" for coin in FALLBACK[:n]], "Ticker data empty; fallback symbols used"


@st.cache_data(ttl=300, show_spinner=False)
def fetch_klines(symbol, interval, total):
    """Fetch recent spot candles backwards in pages; use only completed candles."""
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
        end_time = int(batch[0][0]) - 1
        if len(batch) < limit:
            break
    if not rows:
        raise RuntimeError("No candles returned; pair may not be a listed Binance Spot USDT market.")
    frame = pd.DataFrame(rows).iloc[:, :7].copy()
    frame.columns = ["time", "open", "high", "low", "close", "volume", "close_time"]
    for col in ["time", "open", "high", "low", "close", "volume", "close_time"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna().drop_duplicates("time").sort_values("time")
    frame = frame[frame["close_time"] < int(time.time() * 1000)].iloc[-total:].reset_index(drop=True)
    if len(frame) < 50:
        raise RuntimeError(f"Only {len(frame)} completed candles returned; need at least 50.")
    frame["time"] = pd.to_datetime(frame["time"].astype("int64"), unit="ms", utc=True)
    return frame


def load_many(symbols, timeframe, candle_count, progress=None):
    data, errors = {}, {}
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(symbols)))) as pool:
        futures = {pool.submit(fetch_klines, symbol, timeframe, candle_count): symbol for symbol in symbols}
        for done, future in enumerate(as_completed(futures), 1):
            symbol = futures[future]
            try:
                data[symbol] = future.result()
            except Exception as exc:
                errors[symbol] = str(exc)
            if progress:
                progress(done / len(futures))
    return data, errors


def prepare(frame):
    d = frame.copy()
    c = d["close"]
    d["ema20"] = c.ewm(span=20, adjust=False).mean()
    d["ema50"] = c.ewm(span=50, adjust=False).mean()
    d["ema200"] = c.ewm(span=200, adjust=False).mean()
    prev_close = c.shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - prev_close).abs(), (d["low"] - prev_close).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    return d


def make_signals(d, zone_period=20, use_structure=False):
    """Trend-pullback signals based on EMA alignment, price action and ATR only. No RSI."""
    c, o, h, low, atr = d["close"], d["open"], d["high"], d["low"], d["atr"]
    ema_zone = d["ema20"] if zone_period == 20 else d["ema50"]
    candle_range = (h - low).replace(0, np.nan)
    long_trend = (c > d["ema200"]) & (d["ema50"] > d["ema200"]) & (d["ema50"] > d["ema50"].shift(5))
    short_trend = (c < d["ema200"]) & (d["ema50"] < d["ema200"]) & (d["ema50"] < d["ema50"].shift(5))
    pullback = (low <= ema_zone + 0.3 * atr) & (h >= ema_zone - 0.3 * atr)
    bullish_rejection = (c > o) & (c > ema_zone) & ((c - low) / candle_range >= 0.55)
    bearish_rejection = (c < o) & (c < ema_zone) & ((h - c) / candle_range >= 0.55)
    long_signal = long_trend & pullback & bullish_rejection
    short_signal = short_trend & pullback & bearish_rejection
    if use_structure:
        # A pivot is used only after three later candles confirm it (no look-ahead).
        span = 3
        lows, highs = d["low"].to_numpy(), d["high"].to_numpy()
        higher_low = np.zeros(len(d), dtype=bool)
        lower_high = np.zeros(len(d), dtype=bool)
        last_low = prev_low = last_high = prev_high = np.nan
        for i in range(len(d)):
            p = i - span
            if p >= span:
                if lows[p] == np.min(lows[p - span:p + span + 1]):
                    prev_low, last_low = last_low, lows[p]
                if highs[p] == np.max(highs[p - span:p + span + 1]):
                    prev_high, last_high = last_high, highs[p]
            higher_low[i] = np.isfinite(last_low) and np.isfinite(prev_low) and last_low > prev_low
            lower_high[i] = np.isfinite(last_high) and np.isfinite(prev_high) and last_high < prev_high
        long_signal &= pd.Series(higher_low, index=d.index)
        short_signal &= pd.Series(lower_high, index=d.index)
    out = d.copy()
    out["long_signal"] = long_signal.fillna(False)
    out["short_signal"] = short_signal.fillna(False)
    return out


def backtest(frame, symbol, timeframe, notional, fee_pct, stop_atr, reward_risk, max_bars, zone_period, use_structure):
    d = make_signals(prepare(frame), zone_period, use_structure)
    trades, i = [], 0
    fee_rate = fee_pct / 100
    while i < len(d) - 1:
        signal = d.iloc[i]
        side = 1 if signal["long_signal"] else -1 if signal["short_signal"] else 0
        if side == 0 or not np.isfinite(signal["atr"]) or signal["atr"] <= 0:
            i += 1
            continue
        entry_i = i + 1  # signal candle close -> following candle open
        entry = float(d.iloc[entry_i]["open"])
        risk = float(signal["atr"]) * stop_atr
        stop = entry - side * risk
        target = entry + side * risk * reward_risk
        last_i = min(len(d) - 1, entry_i + int(max_bars) - 1)
        exit_i, exit_price, reason = None, None, "Time exit"
        for j in range(entry_i, last_i + 1):
            bar = d.iloc[j]
            hit_stop = float(bar["low"]) <= stop if side == 1 else float(bar["high"]) >= stop
            hit_target = float(bar["high"]) >= target if side == 1 else float(bar["low"]) <= target
            if hit_stop:  # conservative assumption if stop and target both touch in one candle
                exit_i, exit_price, reason = j, stop, "Stop loss"
                break
            if hit_target:
                exit_i, exit_price, reason = j, target, "Take profit"
                break
        if exit_i is None:
            exit_i, exit_price = last_i, float(d.iloc[last_i]["close"])
        qty = notional / entry
        gross = (float(exit_price) - entry) * qty * side
        fees = fee_rate * (notional + qty * float(exit_price))
        trades.append({"Symbol": symbol, "Timeframe": timeframe, "Side": "LONG" if side == 1 else "SHORT",
                       "Entry time (UTC)": d.iloc[entry_i]["time"], "Entry": entry,
                       "Exit time (UTC)": d.iloc[exit_i]["time"], "Exit": float(exit_price),
                       "Exit reason": reason, "Notional ($)": notional, "Gross PnL ($)": gross,
                       "Fees ($)": fees, "Net PnL ($)": gross - fees,
                       "Return (%)": (gross - fees) / notional * 100})
        i = exit_i + 1
    signal_counts = {"Long signals": int(d["long_signal"].sum()), "Short signals": int(d["short_signal"].sum())}
    return trades, signal_counts


def show_results(trades):
    if not trades:
        st.warning("No trades were generated. Check loaded candle counts and signal counts above; try a longer history or a different timeframe.")
        return
    result = pd.DataFrame(trades).sort_values(["Exit time (UTC)", "Symbol"]).reset_index(drop=True)
    pnl = result["Net PnL ($)"]
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    equity = pnl.cumsum()
    cols = st.columns(5)
    cols[0].metric("Trades", f"{len(result):,}")
    cols[1].metric("Win rate", f"{(pnl > 0).mean() * 100:.1f}%")
    cols[2].metric("Net P&L", f"${pnl.sum():,.2f}")
    cols[3].metric("Profit factor", f"{wins.sum() / abs(losses.sum()):.2f}" if losses.sum() < 0 else "∞")
    cols[4].metric("Max drawdown", f"${(equity.cummax() - equity).max():,.2f}")
    st.subheader("Cumulative realized P&L")
    st.line_chart(result.set_index("Exit time (UTC)")["Net PnL ($)"].cumsum().rename("P&L ($)"))
    st.subheader("Per-symbol results")
    summary = result.groupby("Symbol").agg(Trades=("Net PnL ($)", "size"), Net_PnL=("Net PnL ($)", "sum"), Win_rate_pct=("Net PnL ($)", lambda x: (x > 0).mean() * 100)).sort_values("Net_PnL", ascending=False)
    st.dataframe(summary, use_container_width=True)
    st.subheader("Trade log")
    st.dataframe(result.sort_values("Entry time (UTC)", ascending=False), use_container_width=True)
    st.download_button("Download trades CSV", result.to_csv(index=False), "trend_pullback_trades.csv", "text/csv")


st.title("📈 Trend-Pullback Scanner + Backtest")
st.caption("Binance Spot USDT candles · EMA + price action + ATR only · NO RSI · no live orders")
st.markdown("**Long:** close above EMA200, EMA50 above EMA200 and rising; price pulls back to EMA zone; bullish rejection candle. **Short:** mirrored downtrend and bearish rejection. No RSI condition or RSI indicator is used. Entry is next candle open; exit is ATR stop, risk/reward target, or time limit.")
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

if st.button("Load Binance data and run backtest", type="primary"):
    symbols, universe_note = get_universe(coin_count)
    st.caption(universe_note)
    progress = st.progress(0.0, text=f"Loading {len(symbols)} {timeframe} Binance Spot series...")
    frames, errors = load_many(symbols, timeframe, int(candle_count), lambda fraction: progress.progress(fraction, text=f"Data loaded {fraction:.0%}"))
    progress.empty()
    valid = {symbol: frame for symbol, frame in frames.items() if not frame.empty}
    all_trades, diagnostics = [], []
    for symbol, frame in valid.items():
        pair_trades, counts = backtest(frame, symbol, timeframe, notional, fee_pct, stop_atr, reward_risk, max_bars, zone, use_structure)
        diagnostics.append({"Symbol": symbol, "Candles": len(frame), **counts, "Trades": len(pair_trades)})
        all_trades.extend(pair_trades)
    if not include_shorts:
        all_trades = [trade for trade in all_trades if trade["Side"] == "LONG"]
    st.session_state["trend_trades"] = all_trades
    st.session_state["trend_errors"] = errors
    st.session_state["trend_diagnostics"] = diagnostics
    st.session_state["trend_run"] = f"{timeframe} · {candle_count:,} requested candles/symbol · ${notional:,.0f} per trade"

if "trend_trades" in st.session_state:
    st.caption("Last run: " + st.session_state["trend_run"])
    if st.session_state["trend_diagnostics"]:
        st.subheader("Data and signal diagnostics")
        st.dataframe(pd.DataFrame(st.session_state["trend_diagnostics"]), use_container_width=True)
    errors = st.session_state["trend_errors"]
    if errors:
        with st.expander(f"Symbols skipped ({len(errors)}) — data/API errors"):
            st.write("\n".join(f"{symbol}: {message}" for symbol, message in errors.items()))
    show_results(st.session_state["trend_trades"])
