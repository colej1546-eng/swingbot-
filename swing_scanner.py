"""
Swing trade scanner for QQQ, built for 2-5 day holds, with confluence scoring.

Setup:   pip install yfinance pandas requests
Run:     python swing_scanner.py
Best:    run once daily after the close (4:15pm ET), place orders next morning.

STEP 1 - a base setup must appear:
  PULLBACK  (uptrend: price > 50 SMA > 200 SMA) price dips to the 20 EMA,
            RSI cooled off and turning up
  BREAKOUT  (same uptrend) closes above the 20-day high on strong volume
  DIP       price above the 200 SMA and the 2-day RSI is below 10
            (short-term oversold in a long-term uptrend; a well-known
            index ETF mean-reversion setup)

STEP 2 - confluences. Each one that agrees adds a point (max 5). They are
chosen because each adds different, researched information:
  1. Momentum: price is higher than it was ~6 months ago (time-series momentum)
  2. Trend strength: ADX above 20
  3. Calm market: VIX below 25 (stops get hit far more in panicky markets)
  4. Not overextended: RSI below 70
  5. Near highs: price within 5% of its 52-week high

An alert is only sent when score >= MIN_CONFLUENCE.
Evidence for these ideas is mostly from older data and edges shrink over time.
This is a screening tool, not advice. Backtest it and paper trade first.
"""

import os

import pandas as pd
import yfinance as yf
import requests

# --- Settings -------------------------------------------------------------
DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK", "")  # set as a secret in GitHub, or paste the URL here
ACCOUNT_SIZE = 10_000  # used for position sizing
RISK_PER_TRADE = 0.01  # risk 1% of account per trade
ATR_STOP_MULT = 1.5    # stop = entry - 1.5 * ATR
VOL_SURGE = 1.5        # breakout volume vs 20-day average
VIX_MAX = 25           # "calm market" threshold
MIN_CONFLUENCE = 3     # out of 5; raise for fewer, pickier alerts

TICKERS = ["QQQ"]  # add more tickers here any time, e.g. ["QQQ", "SPY"]


def rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def true_range(df):
    return pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift()).abs(),
            (df["Low"] - df["Close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(df, n=14):
    return true_range(df).ewm(alpha=1 / n, adjust=False).mean()


def adx(df, n=14):
    up = df["High"].diff()
    down = -df["Low"].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr_n = atr(df, n)
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr_n
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr_n
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def analyze(ticker, df, vix=None):
    df = df.dropna().copy()
    if len(df) < 210:
        return None

    c = df["Close"]
    df["ema20"] = c.ewm(span=20, adjust=False).mean()
    df["sma50"] = c.rolling(50).mean()
    df["sma200"] = c.rolling(200).mean()
    df["rsi"] = rsi(c)
    df["rsi2"] = rsi(c, 2)
    df["atr"] = atr(df)
    df["adx"] = adx(df)
    df["vol_avg"] = df["Volume"].rolling(20).mean()
    df["high20"] = df["High"].rolling(20).max().shift(1)
    df["high252"] = df["High"].rolling(252, min_periods=120).max()
    df["mom126"] = c / c.shift(126) - 1

    t, y = df.iloc[-1], df.iloc[-2]

    uptrend = t.Close > t.sma50 > t.sma200
    setup = None
    if uptrend:
        touched_ema = t.Low <= t.ema20 * 1.01 and t.Close >= t.ema20
        if touched_ema and 38 <= t.rsi <= 55 and t.rsi > y.rsi:
            setup = "PULLBACK"
        elif t.Close > t.high20 and t.Volume >= VOL_SURGE * t.vol_avg:
            setup = "BREAKOUT"
    if not setup and t.Close > t.sma200 and t.rsi2 < 10:
        setup = "DIP"
    if not setup:
        return None

    vix_now = None
    if vix is not None and len(vix.dropna()):
        vix_now = float(vix.reindex(df.index).ffill().iloc[-1])
    hits = {
        "momentum": bool(t.mom126 > 0),
        "ADX": bool(t.adx > 20),
        "calm VIX": bool(vix_now is not None and vix_now < VIX_MAX),
        "RSI": bool(t.rsi < 70),
        "near highs": bool(t.Close >= 0.95 * t.high252),
    }
    score = sum(hits.values())
    if score < MIN_CONFLUENCE:
        return None

    entry = float(t.Close)
    stop = entry - ATR_STOP_MULT * float(t.atr)
    risk = entry - stop
    shares = int((ACCOUNT_SIZE * RISK_PER_TRADE) / risk) if risk > 0 else 0

    return {
        "ticker": ticker,
        "setup": setup,
        "score": score,
        "why": ", ".join(k for k, v in hits.items() if v),
        "entry": round(entry, 2),
        "stop": round(stop, 2),
        "target1": round(entry + 1.5 * risk, 2),
        "target2": round(entry + 3.0 * risk, 2),
        "shares": shares,
        "vix": None if vix_now is None else round(vix_now, 1),
    }


def main():
    data = yf.download(
        TICKERS + ["^VIX"], period="2y", interval="1d",
        group_by="ticker", auto_adjust=True, progress=False,
    )
    try:
        vix = data["^VIX"]["Close"]
    except Exception:
        vix = None
        print("Warning: could not load VIX; the calm-market point is skipped.")

    signals = []
    for tk in TICKERS:
        try:
            sig = analyze(tk, data[tk].copy(), vix)
            if sig:
                signals.append(sig)
        except Exception as e:
            print(f"skip {tk}: {e}")

    if not signals:
        msg = "No QQQ swing setup today."
    else:
        lines = ["**Swing setups (hold 2-5 days)**"]
        for s in signals:
            lines.append(
                f"{s['ticker']}: {s['setup']} | score {s['score']}/5 ({s['why']})\n"
                f"Buy near ${s['entry']} | Stop ${s['stop']} | "
                f"Take profit 1 ${s['target1']} | Take profit 2 ${s['target2']} | "
                f"Size {s['shares']} sh | VIX {s['vix']}"
            )
        lines.append("Exit: take profit 2, the stop, or end of day 5.")
        msg = "\n".join(lines)

    print(msg)
    if DISCORD_WEBHOOK:
        requests.post(DISCORD_WEBHOOK, json={"content": msg[:1900]})


if __name__ == "__main__":
    main()
