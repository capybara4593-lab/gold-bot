import os, json, requests, pandas as pd
from datetime import datetime, timedelta, timezone

TD_KEY = os.environ["TD_KEY"]
TG_TOKEN = os.environ["TG_TOKEN"]
CHAT_ID = os.environ["CHAT_ID"]

SIG_MODE = "FU"          # "FU", "FU+ATT", "HCS"
RR = 2.0
BUF = 0.0
TRADE_LONG = True
TRADE_SHORT = True
USE_DOJI_FU = False
USE_MA = False
USE_BEARBULL = True
MA_LEN = 9
BODY_RATIO = 0.3

URL = "https://api.twelvedata.com/time_series"

def get_df():
    r = requests.get(URL, timeout=15, params={
        "symbol": "XAU/USD", "interval": "1h", "outputsize": 500,
        "timezone": "UTC", "apikey": TD_KEY}).json()
    if "values" not in r:
        raise Exception(f"Twelve Data xatosi: {r}")
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    rows = []
    for v in reversed(r["values"]):
        t = datetime.strptime(v["datetime"], "%Y-%m-%d %H:%M:%S")
        rows.append((v["datetime"], float(v["open"]), float(v["high"]),
                     float(v["low"]), float(v["close"]),
                     t + timedelta(hours=1) <= now))
    return pd.DataFrame(rows, columns=["time", "Open", "High", "Low", "Close", "complete"])

def compute(df):
    o, h, l, c = df.Open.values, df.High.values, df.Low.values, df.Close.values
    n = len(df)
    sma = df.Close.rolling(MA_LEN).mean().values
    doji = abs(o - c) <= (h - l) * BODY_RATIO
    bT = rT = 0
    bH = bL = rH = rL = None
    longs, shorts = [False] * n, [False] * n
    for i in range(1, n):
        bullBase = c[i] > o[i] and l[i] < l[i-1] and c[i] > c[i-1]
        bearBase = c[i] < o[i] and h[i] > h[i-1] and c[i] < c[i-1]
        fuBull = bullBase and h[i] > h[i-1] and c[i] > h[i-1]
        attBull = ((bullBase and h[i] > h[i-1] and c[i] <= h[i-1]) or
                   (bullBase and h[i] < h[i-1] and c[i] < o[i-1]) or
                   (c[i] < o[i] and l[i] < l[i-1] and c[i] < c[i-1]
                    and h[i] > h[i-1] and c[i] > o[i-1]))
        fuBear = bearBase and l[i] < l[i-1] and c[i] != l[i-1]
        attBear = ((bearBase and l[i] > l[i-1]) or
                   (c[i] > o[i] and h[i] > h[i-1] and o[i] < o[i-1] and l[i] < l[i-1]))
        if USE_MA and c[i] < sma[i]: fuBull = attBull = False
        if USE_MA and c[i] > sma[i]: fuBear = attBear = False
        if USE_DOJI_FU and not doji[i-1]:
            fuBull = attBull = fuBear = attBear = False
        if USE_BEARBULL and c[i] > o[i]: attBear = False
        if USE_BEARBULL and c[i] < o[i]: attBull = False
        if doji[i]: fuBull = fuBear = False

        bullHCS = (((fuBull and bT == 2) or (attBull and bT == 1))
                   and bL is not None and l[i] > bL and l[i] < bH)
        bearHCS = (((fuBear and rT == 2) or (attBear and rT == 1))
                   and rL is not None and h[i] < rH and h[i] > rL)
        if fuBull or attBull:
            bT = 1 if fuBull else 2; bH, bL = h[i], l[i]
        if fuBear or attBear:
            rT = 1 if fuBear else 2; rH, rL = h[i], l[i]

        if SIG_MODE == "FU":
            longs[i], shorts[i] = fuBull, fuBear
        elif SIG_MODE == "FU+ATT":
            longs[i], shorts[i] = fuBull or attBull, fuBear or attBear
        else:
            longs[i], shorts[i] = bullHCS, bearHCS
    return longs, shorts

def send(text):
    print(text)
    requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                  data={"chat_id": CHAT_ID, "text": text}, timeout=10)

try:
    st = json.load(open("state.json"))
except Exception:
    st = None

df = get_df()
cdf = df[df.complete].reset_index(drop=True)
if st is None:
    st = {"pos": None, "last_t": cdf.time.iloc[-1]}
    send("Bot ishga tushdi: XAUUSD H1")
pos, last_t = st["pos"], st["last_t"]

if pos:
    for _, b in df[df.time > pos["t"]].iterrows():
        if pos["side"] == "LONG":
            sl_hit, tp_hit = b.Low <= pos["sl"], b.High >= pos["tp"]
        else:
            sl_hit, tp_hit = b.High >= pos["sl"], b.Low <= pos["tp"]
        if sl_hit or tp_hit:
            why = "STOP" if sl_hit else "TAKE PROFIT"
            px = pos["sl"] if sl_hit else pos["tp"]
            send(f"CHIQISH {pos['side']} ({why}) @ {px:.2f}")
            pos = None
            break

if cdf.time.iloc[-1] != last_t:
    last_t = cdf.time.iloc[-1]
    longs, shorts = compute(cdf)
    i = len(cdf) - 1
    c = cdf.Close.iloc[i]
    if TRADE_LONG and longs[i] and (pos is None or pos["side"] == "SHORT"):
        sl = cdf.Low.iloc[i] - BUF
        risk = c - sl
        if risk > 0:
            if pos: send("CHIQISH SHORT (teskari signal)")
            pos = dict(side="LONG", sl=sl, tp=c + RR * risk, t=last_t)
            send(f"LONG kirish @ {c:.2f}\nSL {sl:.2f}\nTP {pos['tp']:.2f}")
    elif TRADE_SHORT and shorts[i] and (pos is None or pos["side"] == "LONG"):
        sl = cdf.High.iloc[i] + BUF
        risk = sl - c
        if risk > 0:
            if pos: send("CHIQISH LONG (teskari signal)")
            pos = dict(side="SHORT", sl=sl, tp=c - RR * risk, t=last_t)
            send(f"SHORT kirish @ {c:.2f}\nSL {sl:.2f}\nTP {pos['tp']:.2f}")

json.dump({"pos": pos, "last_t": last_t}, open("state.json", "w"))
