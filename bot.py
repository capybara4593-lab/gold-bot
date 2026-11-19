import os, json, requests, pandas as pd
from datetime import datetime, timedelta, timezone

TD_KEY = os.environ["TD_KEY"]
TG_TOKEN = os.environ["TG_TOKEN"]
CHAT_ID = os.environ["CHAT_ID"]

# (nom, manba, symbol, interval)  <- keraksizini o'chirib tashlasa bo'ladi
JOBS = [
    ("XAUUSD", "td", "XAU/USD", "1h"),
    ("XAUUSD", "td", "XAU/USD", "30min"),
    ("XAUUSD", "td", "XAU/USD", "4h"),
    ("XAGUSD", "td", "XAG/USD", "1h"),
    ("XAGUSD", "td", "XAG/USD", "30min"),
    ("XAGUSD", "td", "XAG/USD", "4h"),
]


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

FMT = "%Y-%m-%d %H:%M:%S"
TFM = {"30min": 30, "1h": 60, "4h": 240}
TFL = {"30min": "M30", "1h": "H1", "4h": "H4"}

def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)

def get_td(sym, iv):
    r = requests.get("https://api.twelvedata.com/time_series", timeout=15, params={
        "symbol": sym, "interval": iv, "outputsize": 500,
        "timezone": "UTC", "apikey": TD_KEY}).json()
    if "values" not in r:
        raise Exception(f"Twelve Data xatosi: {r}")
    rows = [(v["datetime"], float(v["open"]), float(v["high"]),
             float(v["low"]), float(v["close"])) for v in reversed(r["values"])]
    return pd.DataFrame(rows, columns=["time", "Open", "High", "Low", "Close"])

def get_yf(sym, iv):
    import yfinance as yf
    base = "30m" if iv == "30min" else "1h"
    d = yf.download(sym, period="55d", interval=base, progress=False, auto_adjust=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d.dropna()
    d.index = d.index.tz_convert("America/New_York")
    if iv == "4h":
        d = d.resample("4h", offset="2h").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    d = d.tail(500)
    d.index = d.index.tz_convert("UTC")
    return pd.DataFrame({"time": d.index.strftime(FMT), "Open": d.Open.values,
                         "High": d.High.values, "Low": d.Low.values,
                         "Close": d.Close.values})

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

def run_job(name, src, sym, iv, S):
    tf = TFM[iv]
    key = f"{sym}|{iv}"
    label = f"{name} {TFL[iv]}"
    st = S.get(key)
    now = utcnow()

    if st is not None:
        if st.get("retry") and now < datetime.strptime(st["retry"], FMT):
            return
        bar_due = now >= datetime.strptime(st["last_t"], FMT) + timedelta(minutes=2 * tf)
        if not st["pos"] and not bar_due:
            return   # kerak bo'lmasa so'rov yubormaymiz (limitni tejash)

    df = get_td(sym, iv) if src == "td" else get_yf(sym, iv)
    df["complete"] = [datetime.strptime(t, FMT) + timedelta(minutes=tf) <= now
                      for t in df.time]
    cdf = df[df.complete].reset_index(drop=True)

    if st is None:
        st = {"pos": None, "last_t": cdf.time.iloc[-1], "retry": None}
        send(f"Kuzatuv boshlandi: {label}")
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
                send(f"{label}: CHIQISH {pos['side']} ({why}) @ {px:.2f}")
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
                if pos: send(f"{label}: CHIQISH SHORT (teskari signal)")
                pos = dict(side="LONG", sl=float(sl), tp=float(c + RR * risk), t=last_t)
                send(f"{label}: LONG kirish @ {c:.2f}\nSL {sl:.2f}\nTP {pos['tp']:.2f}")
        elif TRADE_SHORT and shorts[i] and (pos is None or pos["side"] == "LONG"):
            sl = cdf.High.iloc[i] + BUF
            risk = sl - c
            if risk > 0:
                if pos: send(f"{label}: CHIQISH LONG (teskari signal)")
                pos = dict(side="SHORT", sl=float(sl), tp=float(c - RR * risk), t=last_t)
                send(f"{label}: SHORT kirish @ {c:.2f}\nSL {sl:.2f}\nTP {pos['tp']:.2f}")

    # bozor yopiq bo'lsa (yangi sham yo'q) 30 daqiqa so'rov yubormaymiz
    newest = datetime.strptime(df.time.iloc[-1], FMT)
    retry = None
    if newest + timedelta(minutes=2 * tf) < now:
        retry = (now + timedelta(minutes=30)).strftime(FMT)
    S[key] = {"pos": pos, "last_t": last_t, "retry": retry}

try:
    S = json.load(open("state.json"))
except Exception:
    S = {}
if "pos" in S:                       # eski format (faqat oltin H1)
    S = {"XAU/USD|1h": S}

errs = []
for job in JOBS:
    try:
        run_job(*job, S)
    except Exception as e:
        print(job, "xato:", e)
        errs.append(f"{job[0]} {TFL[job[3]]}: {str(e)[:150]}")

msg = "\n".join(errs)
if msg and msg != S.get("_err"):
    send("XATO:\n" + msg)
S["_err"] = msg

json.dump(S, open("state.json", "w"))
