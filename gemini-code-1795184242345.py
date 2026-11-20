# -*- coding: utf-8 -*-
"""
Kishore Trend-Pullback Scanner + Walk-Forward Backtest
Author: Kishore

Logic (non-repainting, every signal uses only CLOSED candles, entry = NEXT candle open):
  LONG : close > EMA200, EMA50 > EMA200 and rising  (uptrend only)
         price pulls back into the EMA20/EMA50 zone (+-0.3 ATR) and a bullish
         rejection candle closes back above the EMA, RSI 40-65
         optional: confirmed Higher-Low market structure
  SHORT: exact mirror (downtrend, rejection candle, RSI 35-60, Lower-High structure)
  SL = k x ATR, TP = RR x risk, time-stop after N candles. Win = real net PnL > 0 (fees included).

Walk-forward: first 60% of candles = TRAIN (best setup chosen per coin / timeframe / side),
last 40% = TEST (never seen while choosing). Coins are marked "Qualified" only when the
TEST win rate >= your minimum, the TEST net PnL is positive and there are enough test trades.
"""
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ----------------------------------------------------------------------------------------
# DATA  (Binance Spot Vision first, other hosts as fallback)
# ----------------------------------------------------------------------------------------
HOSTS = [
    "https://data-api.binance.vision",
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
]
HEADERS = {"User-Agent": "Mozilla/5.0"}
STABLES = {"USDC", "FDUSD", "TUSD", "USDP", "DAI", "EUR", "AEUR", "USDE", "XUSD", "BUSD",
           "USD1", "UST", "USDS", "PYUSD", "RLUSD", "BFUSD", "EURI", "GBP", "TRY", "BRL"}
FALLBACK = ["BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "ADA", "AVAX", "LINK", "DOT", "LTC",
            "BCH", "TRX", "NEAR", "SUI", "APT", "ARB", "OP", "ATOM", "FIL", "INJ", "SEI",
            "TIA", "AAVE", "UNI", "ETC", "HBAR", "ICP", "RUNE", "PEPE", "SHIB", "TON",
            "WLD", "FET", "RENDER", "TAO", "ENA", "JUP", "ONDO", "PENDLE"]
TF_MS = {"15m": 900_000, "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
IST = "Asia/Kolkata"


def _get(path, params):
    last = None
    for h in HOSTS:
        try:
            r = requests.get(h + path, params=params, headers=HEADERS, timeout=5)
            if r.status_code == 200:
                return r.json()
            last = f"{h} -> HTTP {r.status_code}"
        except Exception as e:  # noqa
            last = f"{h} -> {type(e).__name__}: {str(e)[:45]}"
    raise RuntimeError(f"All Binance hosts failed ({last})")


_KCACHE = {}


def fetch_klines(symbol, interval, total, ttl=300):
    key = (symbol, interval, total)
    hit = _KCACHE.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    rows, end = [], None
    while len(rows) < total:
        p = {"symbol": symbol, "interval": interval, "limit": min(1000, total - len(rows))}
        if end:
            p["endTime"] = end
        data = _get("/api/v3/klines", p)
        if not data:
            break
        rows = data + rows
        end = data[0][0] - 1
        if len(data) < p["limit"]:
            break
    if not rows:
        raise RuntimeError("no klines")
    df = pd.DataFrame(rows).iloc[:, :7]
    df.columns = ["time", "open", "high", "low", "close", "volume", "close_time"]
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["time"] = df["time"].astype("int64")
    df["close_time"] = df["close_time"].astype("int64")
    df = df.drop_duplicates("time").sort_values("time")
    now_ms = int(time.time() * 1000)
    df = df[df["close_time"] < now_ms]  # closed candles only
    df = df.iloc[-total:].reset_index(drop=True)
    _KCACHE[key] = (time.time(), df)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def get_universe(n):
    try:
        data = _get("/api/v3/ticker/24hr", {})
        rows = []
        for d in data:
            s = d["symbol"]
            if not s.endswith("USDT"):
                continue
            base = s[:-4]
            if base in STABLES or base.endswith(("UP", "DOWN", "BULL", "BEAR")):
                continue
            rows.append((float(d["quoteVolume"]), s))
        rows.sort(reverse=True)
        out = [s for _, s in rows[:n]]
        if out:
            return out
    except Exception:
        pass
    return [b + "USDT" for b in FALLBACK[:n]]


# ----------------------------------------------------------------------------------------
# INDICATORS
# ----------------------------------------------------------------------------------------
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def atr(df, n=14):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def structure(low, high, s=3):
    """Confirmed (non-repainting) swing pivots. Pivot at bar i is only known at bar i+s."""
    n = len(low)
    ll, hh = pd.Series(low), pd.Series(high)
    pl = (ll == ll.rolling(2 * s + 1, center=True).min()).to_numpy()
    ph = (hh == hh.rolling(2 * s + 1, center=True).max()).to_numpy()
    out = {}
    for name, flags, vals in (("L", pl, low), ("H", ph, high)):
        last = np.full(n, np.nan)
        prev = np.full(n, np.nan)
        cur = pr = np.nan
        for i in range(n):
            j = i - s
            if j >= 0 and flags[j]:
                pr, cur = cur, vals[j]
            last[i], prev[i] = cur, pr
        out[name] = (last, prev)
    higher_low = out["L"][0] > out["L"][1]
    lower_high = out["H"][0] < out["H"][1]
    return higher_low, lower_high


def prepare(df):
    d = df.copy()
    d["ema20"] = ema(d["close"], 20)
    d["ema50"] = ema(d["close"], 50)
    d["ema200"] = ema(d["close"], 200)
    d["rsi"] = rsi(d["close"])
    d["atr"] = atr(d)
    hl, lh = structure(d["low"].to_numpy(), d["high"].to_numpy())
    d["hl"], d["lh"] = hl, lh
    return d


def build_signal(d, side, zone, use_struct):
    c, o, h, l, a = d["close"], d["open"], d["high"], d["low"], d["atr"]
    ez = d["ema20"] if zone == 20 else d["ema50"]
    rng = (h - l).replace(0, np.nan)
    if side == "long":
        trend = (c > d["ema200"]) & (d["ema50"] > d["ema200"]) & (d["ema50"] > d["ema50"].shift(5))
        touch = l.rolling(2).min() <= ez + 0.3 * a
        candle = (c > o) & (c > ez) & ((c - l) / rng