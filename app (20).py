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
            r = requests.get(h + path, params=params, timeout=12)
            if r.status_code == 200:
                return r.json()
            last = f"{h} -> HTTP {r.status_code}"
        except Exception as e:  # noqa
            last = f"{h} -> {str(e)[:60]}"
    raise RuntimeError(f"All Binance hosts failed ({last})")


_KCACHE = {}


def fetch_klines(symbol, interval, total, ttl=300):
    key = (symbol, interval, total)
    hit = _KCACHE.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    rows, end = [], None
    while len(rows) < total:
        p = {"symbol": symbol, "interval": interval, "limit": 1000}
        if end:
            p["endTime"] = end
        data = _get("/api/v3/klines", p)
        if not data:
            break
        rows = data + rows
        end = data[0][0] - 1
        if len(data) < 1000:
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
        candle = (c > o) & (c > ez) & ((c - l) / rng >= 0.6)
        rs = (d["rsi"] >= 40) & (d["rsi"] <= 65)
        sig = trend & touch & candle & rs
        if use_struct:
            sig &= d["hl"]
    else:
        trend = (c < d["ema200"]) & (d["ema50"] < d["ema200"]) & (d["ema50"] < d["ema50"].shift(5))
        touch = h.rolling(2).max() >= ez - 0.3 * a
        candle = (c < o) & (c < ez) & ((h - c) / rng >= 0.6)
        rs = (d["rsi"] >= 35) & (d["rsi"] <= 60)
        sig = trend & touch & candle & rs
        if use_struct:
            sig &= d["lh"]
    sig = sig.fillna(False).to_numpy().copy()
    sig[:210] = False
    return sig


# ----------------------------------------------------------------------------------------
# BACKTEST ENGINE
# ----------------------------------------------------------------------------------------
def simulate(o, h, l, c, a, sig, side, sl_atr, rr, fee, max_hold):
    """Entry = next candle open. SL & TP in same candle -> SL first (conservative)."""
    n = len(o)
    trades = []
    free = 0
    d = 1 if side == "long" else -1
    for s in np.flatnonzero(sig):
        if s < free or s + 1 >= n:
            continue
        entry = o[s + 1]
        risk = a[s] * sl_atr
        if not risk > 0:
            continue
        sl = entry - d * risk
        tp = entry + d * rr * risk
        end = min(n - 1, s + max_hold)
        xi, xp, why = end, c[end], "TIME"
        hit = False
        for j in range(s + 1, end + 1):
            if d == 1:
                if l[j] <= sl:
                    xi, xp, why, hit = j, sl, "SL", True
                    break
                if h[j] >= tp:
                    xi, xp, why, hit = j, tp, "TP", True
                    break
            else:
                if h[j] >= sl:
                    xi, xp, why, hit = j, sl, "SL", True
                    break
                if l[j] <= tp:
                    xi, xp, why, hit = j, tp, "TP", True
                    break
        if not hit and s + max_hold > n - 1:
            continue  # still open -> not counted
        net = d * (xp - entry) / entry - fee
        trades.append((s + 1, xi, entry, xp, sl, tp, net, why))
        free = xi + 1
    return trades


def stats(nets):
    nets = np.asarray(nets, float)
    n = len(nets)
    if n == 0:
        return dict(n=0, wr=np.nan, mean=np.nan, total=0.0, t=-9)
    wr = float((nets > 0).mean() * 100)
    sd = nets.std(ddof=1) if n > 1 else 0
    t = nets.mean() / (sd + 1e-9) * np.sqrt(n)
    return dict(n=n, wr=wr, mean=float(nets.mean()), total=float(nets.sum()), t=float(t))


SLS = (1.0, 1.5, 2.0)
RRS = (0.8, 1.0, 1.5, 2.0)
MIN_TRAIN = 12
MIN_TEST = 8
MIN_TRAIN_T = 2.0   # train edge must be statistically meaningful (48 setups are tried per coin)
MIN_TEST_T = 1.0    # test edge must also be clearly positive


def trades_df(trades, d, symbol, tf, side, phase, capital):
    if not trades:
        return pd.DataFrame()
    t = pd.DataFrame(trades, columns=["ei", "xi", "Entry", "Exit", "SL", "TP", "net", "Reason"])
    times = pd.to_datetime(d["time"].to_numpy(), unit="ms", utc=True).tz_convert(IST)
    t["Entry time"] = times[t["ei"].to_numpy()].strftime("%d-%b %H:%M")
    t["Exit time"] = times[t["xi"].to_numpy()].strftime("%d-%b %H:%M")
    t["Coin"], t["TF"], t["Side"], t["Phase"] = symbol[:-4], tf, side.upper(), phase
    t["Result"] = np.where(t["net"] > 0, "WIN", "LOSS")
    t["Net %"] = (t["net"] * 100).round(2)
    t["PnL"] = (t["net"] * capital).round(0)
    return t[["Coin", "TF", "Side", "Phase", "Entry time", "Exit time", "Entry", "Exit", "SL", "TP",
              "Result", "Reason", "Net %", "PnL"]]


@st.cache_data(ttl=600, show_spinner=False)
def analyze(symbol, tf, candles, side, target, fee, max_hold, capital):
    df = fetch_klines(symbol, tf, candles)
    if len(df) < 700:
        return None
    d = prepare(df)
    n = len(d)
    split = int(n * 0.6)
    o, h, l, c, a = (d[k].to_numpy(float) for k in ("open", "high", "low", "close", "atr"))
    sigs = {(z, s): build_signal(d, side, z, s) for z in (20, 50) for s in (False, True)}

    cands = []
    for (z, s), sig in sigs.items():
        for sl in SLS:
            for rr in RRS:
                tr = simulate(o[:split], h[:split], l[:split], c[:split], a[:split], sig[:split],
                              side, sl, rr, fee, max_hold)
                st_ = stats([x[6] for x in tr])
                if st_["n"] >= MIN_TRAIN and st_["mean"] > 0 and st_["t"] >= MIN_TRAIN_T:
                    cands.append((st_["wr"] >= target, st_["t"], z, s, sl, rr, st_))
    row = dict(Coin=symbol[:-4], TF=tf, Side=side.upper(), symbol=symbol)
    if not cands:
        row.update(Status="❌ No edge", Setup="-", trainN=0, trainWR=np.nan, testN=0, testWR=np.nan,
                   testNet=np.nan, testPnL=0.0, live=None, trades=pd.DataFrame(), qualified=False)
        return row
    _, _, z, s, sl, rr, trn = max(cands, key=lambda x: (x[0], x[1]))
    sig = sigs[(z, s)]
    sig_te = sig.copy()
    sig_te[:split] = False
    tr_train = simulate(o[:split], h[:split], l[:split], c[:split], a[:split], sig[:split], side, sl, rr, fee, max_hold)
    tr_test = simulate(o, h, l, c, a, sig_te, side, sl, rr, fee, max_hold)
    te = stats([x[6] for x in tr_test])
    qualified = te["n"] >= MIN_TEST and te["wr"] >= target and te["total"] > 0 and te["t"] >= MIN_TEST_T
    if qualified:
        status = "✅ Qualified"
    elif te["n"] < MIN_TEST:
        status = "⚠️ Too few test trades"
    elif te["wr"] < target:
        status = "⚠️ Test WR below target"
    elif te["total"] <= 0:
        status = "⚠️ Test PnL negative"
    else:
        status = "⚠️ Test edge too weak"
    # live signal on the last CLOSED candle
    live = None
    if sig[-1]:
        dirn = 1 if side == "long" else -1
        risk = a[-1] * sl
        live = dict(time=pd.to_datetime(d["time"].iloc[-1], unit="ms", utc=True).tz_convert(IST).strftime("%d-%b %H:%M"),
                    entry=c[-1], sl=c[-1] - dirn * risk, tp=c[-1] + dirn * rr * risk)
    both = pd.concat([trades_df(tr_train, d, symbol, tf, side, "TRAIN", capital),
                      trades_df(tr_test, d, symbol, tf, side, "TEST", capital)], ignore_index=True)
    row.update(Status=status,
               Setup=f"EMA{z} pullback | {'HL/LH structure' if s else 'no structure'} | SL {sl}xATR | RR 1:{rr}",
               trainN=trn["n"], trainWR=trn["wr"], testN=te["n"], testWR=te["wr"],
               testNet=te["total"] * 100, testPnL=te["total"] * capital, live=live,
               trades=both, qualified=bool(qualified), rr=rr)
    return row


# ----------------------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Kishore Trend-Pullback Scanner", page_icon="📈", layout="wide")
    st.title("📈 Kishore Trend-Pullback Scanner + Backtest")
    st.caption("Binance Spot (Vision) • closed candles only • entry next candle open • win = real net PnL after fees")

    with st.expander("⚙️ Settings", expanded=False):
        c1, c2 = st.columns(2)
        tfs = c1.multiselect("Timeframes", list(TF_MS), default=["15m", "30m", "1h", "4h"])
        n_coins = c2.slider("Top coins (by volume)", 10, 80, 50, 5)
        candles = c1.selectbox("Candles per coin", [3000, 5000, 8000], index=1)
        side_opt = c2.selectbox("Side", ["Both", "Long", "Short"])
        target = c1.slider("Minimum win rate %", 50, 70, 60)
        fee_pct = c2.number_input("Fees round-trip %", 0.0, 1.0, 0.10, 0.01)
        capital = c1.number_input("Capital per trade", 1000, 10_000_000, 100_000, 1000)
        max_hold = c2.slider("Max hold (candles)", 12, 200, 48, 4)
        show_all = c1.checkbox("Show all coins (not only qualified)", False)
        if c2.button("🔄 Force fresh data"):
            _KCACHE.clear()
            analyze.clear()

    if not tfs:
        st.warning("Select at least one timeframe.")
        return
    sides = ["long", "short"] if side_opt == "Both" else [side_opt.lower()]
    fee = fee_pct / 100.0
    symbols = get_universe(n_coins)

    pairs = [(s, tf) for s in symbols for tf in tfs]
    bar = st.progress(0.0, text="Downloading Binance candles…")
    failed = []

    def _pre(p):
        try:
            fetch_klines(p[0], p[1], candles)
        except Exception as e:  # noqa
            failed.append((p, str(e)[:80]))

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(_pre, pairs))
    if len(failed) == len(pairs):
        bar.empty()
        st.error("Binance data not reachable from this server: " + failed[0][1])
        return

    rows = []
    jobs = [(s, tf, sd) for s in symbols for tf in tfs for sd in sides]
    for i, (s, tf, sd) in enumerate(jobs):
        bar.progress((i + 1) / len(jobs), text=f"Backtesting {s} {tf} {sd}…")
        try:
            r = analyze(s, tf, candles, sd, target, fee, max_hold, capital)
        except Exception:
            r = None
        if r:
            rows.append(r)
    bar.empty()
    if not rows:
        st.error("No results. Try again in a minute.")
        return

    qual = [r for r in rows if r["qualified"]]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Coins scanned", len({r["Coin"] for r in rows}))
    k2.metric("Qualified setups", len(qual))
    if qual:
        tn = sum(r["testN"] for r in qual)
        tw = sum(r["testWR"] * r["testN"] / 100 for r in qual)
        k3.metric("Test win rate (qualified)", f"{tw / tn * 100:.1f}%")
        k4.metric("Test PnL (qualified)", f"{sum(r['testPnL'] for r in qual):,.0f}")
    st.caption("⚠️ Test numbers are out-of-sample for the setup choice, but qualifying coins by test result is itself "
               "a filter — expect live results a bit lower. Fewer than ~30 trades = noisy.")

    t1, t2, t3 = st.tabs(["🔔 Live signals", "📊 Coin-wise backtest", "🧾 All trades"])

    with t1:
        live = [r for r in (rows if show_all else qual) if r["live"]]
        if not live:
            st.info("No fresh signal on the last closed candle for the " +
                    ("coins" if show_all else "qualified coins") + " right now.")
        else:
            st.dataframe(pd.DataFrame([{
                "Coin": r["Coin"], "TF": r["TF"], "Side": r["Side"], "Candle (IST)": r["live"]["time"],
                "Entry ≈": round(r["live"]["entry"], 6), "SL": round(r["live"]["sl"], 6),
                "TP": round(r["live"]["tp"], 6), "RR": f"1:{r['rr']}",
                "Test WR %": round(r["testWR"], 1), "Test trades": r["testN"], "Status": r["Status"],
            } for r in live]), hide_index=True, width="stretch")

    with t2:
        order = {"✅ Qualified": 0}
        view = rows if show_all else qual
        if not view:
            st.info("No setup reached the target on test data. Lower the minimum win rate to 50% or add more candles.")
        else:
            df = pd.DataFrame([{
                "Coin": r["Coin"], "TF": r["TF"], "Side": r["Side"], "Status": r["Status"], "Setup": r["Setup"],
                "Train WR %": None if np.isnan(r["trainWR"]) else round(r["trainWR"], 1), "Train N": r["trainN"],
                "Test WR %": None if np.isnan(r["testWR"]) else round(r["testWR"], 1), "Test N": r["testN"],
                "Test Net %": None if np.isnan(r["testNet"]) else round(r["testNet"], 1),
                "Test PnL": round(r["testPnL"]),
            } for r in view])
            df["_o"] = df["Status"].map(lambda x: order.get(x, 1))
            df = df.sort_values(["_o", "Test WR %", "Test PnL"], ascending=[True, False, False]).drop(columns="_o")
            st.dataframe(df, hide_index=True, width="stretch")
            st.markdown("**By coin (all timeframes/sides combined, test period)**")
            agg = {}
            for r in view:
                a_ = agg.setdefault(r["Coin"], [0, 0.0, 0.0])
                if r["testN"]:
                    a_[0] += r["testN"]
                    a_[1] += r["testWR"] * r["testN"] / 100
                    a_[2] += r["testPnL"]
            cdf = pd.DataFrame([{"Coin": k, "Test trades": v[0], "Win rate %": round(v[1] / v[0] * 100, 1),
                                 "PnL": round(v[2])} for k, v in agg.items() if v[0]])
            if not cdf.empty:
                st.dataframe(cdf.sort_values("Win rate %", ascending=False), hide_index=True, width="stretch")

    with t3:
        pick_rows = [r for r in (rows if show_all else qual) if not r["trades"].empty]
        if not pick_rows:
            st.info("No trades to show.")
        else:
            labels = [f"{r['Coin']} {r['TF']} {r['Side']}  ({r['Status']})" for r in pick_rows]
            sel = st.selectbox("Coin / timeframe / side", labels)
            r = pick_rows[labels.index(sel)]
            st.caption(r["Setup"])
            tdf = r["trades"]
            ph = st.radio("Phase", ["TEST", "TRAIN", "Both"], horizontal=True)
            if ph != "Both":
                tdf = tdf[tdf["Phase"] == ph]
            w = int((tdf["Result"] == "WIN").sum())
            st.write(f"Trades: **{len(tdf)}**  |  Wins: **{w}**  |  Win rate: **{(w / len(tdf) * 100 if len(tdf) else 0):.1f}%**"
                     f"  |  PnL: **{tdf['PnL'].sum():,.0f}**")
            st.dataframe(tdf.iloc[::-1], hide_index=True, width="stretch")

    if failed:
        st.caption(f"{len(failed)} coin/timeframe downloads failed and were skipped.")


if __name__ == "__main__":
    main()
