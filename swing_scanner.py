"""
Swing trade scanner for the Nasdaq-100 (QQQ holdings), built for 2-5 day holds.

Setup:   pip install yfinance pandas requests
Run:     python swing_scanner.py
Best:    run once daily after the close (4:15pm ET), place orders next morning.

It looks for two setups, both only in uptrends:
  1. PULLBACK: price dips to the 20 EMA, RSI cooled off and turning up
  2. BREAKOUT: closes above the 20-day high on strong volume

Each signal gets an entry, an ATR-based stop, and 1.5R / 3R targets.
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

# Nasdaq-100 constituents change; update this list or swap in your own watchlist.
TICKERS = ["QQQ"]


def rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def atr(df, n=14):
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift()).abs(),
            (df["Low"] - df["Close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def analyze(ticker, df):
    df = df.dropna()
    if len(df) < 210:
        return None

    c = df["Close"]
    df["ema20"] = c.ewm(span=20, adjust=False).mean()
    df["sma50"] = c.rolling(50).mean()
    df["sma200"] = c.rolling(200).mean()
    df["rsi"] = rsi(c)
    df["atr"] = atr(df)
    df["vol_avg"] = df["Volume"].rolling(20).mean()
    df["high20"] = df["High"].rolling(20).max().shift(1)

    t, y = df.iloc[-1], df.iloc[-2]

    # Trend filter: only trade longs in uptrends
    uptrend = t.Close > t.sma50 > t.sma200
    if not uptrend:
        return None

    setup = None

    # 1) Pullback to 20 EMA, RSI cooled and ticking up
    touched_ema = t.Low <= t.ema20 * 1.01 and t.Close >= t.ema20
    if touched_ema and 38 <= t.rsi <= 55 and t.rsi > y.rsi:
        setup = "PULLBACK"

    # 2) Breakout above 20-day high on volume
    elif t.Close > t.high20 and t.Volume >= VOL_SURGE * t.vol_avg:
        setup = "BREAKOUT"

    if not setup:
        return None

    entry = float(t.Close)
    stop = entry - ATR_STOP_MULT * float(t.atr)
    risk = entry - stop
    shares = int((ACCOUNT_SIZE * RISK_PER_TRADE) / risk) if risk > 0 else 0

    return {
        "ticker": ticker,
        "setup": setup,
        "entry": round(entry, 2),
        "stop": round(stop, 2),
        "target1": round(entry + 1.5 * risk, 2),
        "target2": round(entry + 3.0 * risk, 2),
        "shares": shares,
        "rsi": round(float(t.rsi), 1),
    }


def main():
    data = yf.download(
        TICKERS, period="1y", interval="1d",
        group_by="ticker", auto_adjust=True, progress=False,
    )

    signals = []
    for tk in TICKERS:
        try:
            sig = analyze(tk, data[tk].copy())
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
                f"{s['ticker']} | {s['setup']} | buy ~{s['entry']} | "
                f"stop {s['stop']} | T1 {s['target1']} | T2 {s['target2']} | "
                f"size {s['shares']} sh | RSI {s['rsi']}"
            )
        lines.append("Exit rule: sell at T2, stop, or end of day 5.")
        msg = "\n".join(lines)

    print(msg)
    if DISCORD_WEBHOOK:
        requests.post(DISCORD_WEBHOOK, json={"content": msg[:1900]})


if __name__ == "__main__":
    main()
