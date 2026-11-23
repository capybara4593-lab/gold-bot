import os, json, requests, numpy as np, pandas as pd
from datetime import datetime, timedelta, timezone
from numpy.lib.stride_tricks import sliding_window_view

TD_KEY = os.environ["TD_KEY"]
TG_TOKEN = os.environ["TG_TOKEN"]
CHAT_ID = os.environ["CHAT_ID"]

# (nom, manba, symbol, interval[, rejim[, sozlamalar]])
#   rejim bo'lmasa SIG_MODE ishlatiladi.  CRANK sozlamalari: {"htf": 1440} = 1D trend filtri, {"htf": 240} = 4h filtri (minutda)
JOBS = [
    # --- CRANK + trend filtri (gold). 16.7 yillik tarixda tekshirilgan ---
    ("XAUUSD", "td", "XAU/USD", "4h", "CRANK", {"htf": 1440}),    # eng yaxshi tasdiqlangan: PF 1.52 (kam signal)
    ("XAUUSD", "td", "XAU/USD", "1h", "CRANK", {"htf": 1440}),    # PF 1.33, ko'proq signal
    # tajribaviy: faqat 4.2 yillik tarixda tekshirilgan
    ("XAUUSD", "td", "XAU/USD", "30min", "CRANK", {"htf": 240}),

    # --- eski FU ishlari: backtestda foyda bermadi (gold 1h PF 0.88), o'chirib qo'yildi ---
    # ("XAUUSD", "td", "XAU/USD", "1h"),
    # ("XAUUSD", "td", "XAU/USD", "4h"),
    # ("XAGUSD", "yf", "SI=F", "1h"),
    # ("XAGUSD", "yf", "SI=F", "4h"),
]

SIG_MODE = "FU"          # "FU", "FU+ATT", "HCS"  (CRANK ishlariga ta'sir qilmaydi)
RR = 2.0
BUF = 0.0
TRADE_LONG = True
TRADE_SHORT = True
USE_DOJI_FU = False
USE_MA = False
USE_BEARBULL = True
MA_LEN = 9
BODY_RATIO = 0.3

# ---- CRANK [HIGH VOLTAGE] (backtest qilingan sozlamalar, hajm o'chiq) ----
CRANK = dict(fast=11, slow=28, basis=56, norm_len=56, mult=1.6, thr=0.2,
             use_cci=True, cci_len=55, cci_up=120.0, cci_dn=-130.0)
CRANK_HTF_EMA = 50       # yuqori timeframe trend filtri: HTF yopilish > EMA(50) => faqat LONG
CRANK_MAX_LATE = 3       # signal kechiksa ham kirish faqat shu necha sham ichida (undan keyin o'tkazib yuboriladi)
CRANK_MIN_BARS = 500     # indikator isinishi uchun minimal sham soni
CRANK_NOTIFY_SKIPPED = True   # filtr rad etgan signalni ham xabar qilish
CRANK_BARS = {"15min": 1500, "30min": 2000, "1h": 5000, "4h": 1500}   # so'raladigan sham soni (1D filtr uchun 1h da ko'p tarix kerak)
CRANK_MIN_HTF_BARS = 120   # filtr EMA(50) to'g'ri hisoblanishi uchun minimal yuqori timeframe sham soni

FMT = "%Y-%m-%d %H:%M:%S"
TFM = {"15min": 15, "30min": 30, "1h": 60, "4h": 240}
TFL = {"15min": "M15", "30min": "M30", "1h": "H1", "4h": "H4"}

def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)

def get_td(sym, iv, size=500):
    r = requests.get("https://api.twelvedata.com/time_series", timeout=15, params={
        "symbol": sym, "interval": iv, "outputsize": size,
        "timezone": "UTC", "apikey": TD_KEY}).json()
    if "values" not in r:
        raise Exception(f"Twelve Data xatosi: {r}")
    rows = [(v["datetime"], float(v["open"]), float(v["high"]),
             float(v["low"]), float(v["close"])) for v in reversed(r["values"])]
    return pd.DataFrame(rows, columns=["time", "Open", "High", "Low", "Close"])

def get_yf(sym, iv):
    import yfinance as yf
    base = {"15min": "15m", "30min": "30m"}.get(iv, "1h")
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

# ------------------------------------------------------------------ FU / ATT / HCS (o'zgarmagan)
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

# ------------------------------------------------------------------ CRANK
def _ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().values

def _cci(x, n):
    w = sliding_window_view(x, n)
    sma = w.mean(1)
    md = np.abs(w - sma[:, None]).mean(1)
    out = np.full(len(x), np.nan)
    out[n - 1:] = (x[n - 1:] - sma) / (0.015 * md + 1e-12)
    return out

def compute_crank(close, p=CRANK):
    """Har sham uchun trend holati: +1 (LONG) / -1 (SHORT)."""
    c = np.asarray(close, float)
    basis = _ema(c, p["basis"])
    ema_dir = np.where(_ema(c, p["fast"]) > _ema(c, p["slow"]), 1, -1)
    sd = pd.Series(c).rolling(p["norm_len"]).std(ddof=0).values
    norm = (c - basis) / (sd * p["mult"])
    cc = _cci(c, p["cci_len"])
    st = np.zeros(len(c), int)
    st[0] = ema_dir[0]
    for i in range(1, len(c)):
        if ema_dir[i] == 1 and norm[i] > p["thr"] and (not p["use_cci"] or cc[i] > p["cci_up"]):
            st[i] = 1
        elif ema_dir[i] == -1 and norm[i] < -p["thr"] and (not p["use_cci"] or cc[i] < p["cci_dn"]):
            st[i] = -1
        else:
            st[i] = st[i - 1]
    return st

def htf_name(m):
    return "1D" if m == 1440 else f"{m // 60}h"

def htf_dir_asof(cdf, tf, htf_min, n=CRANK_HTF_EMA):
    """Har sham uchun: shu sham YOPILGAN paytda ma'lum bo'lgan oxirgi TUGAGAN yuqori timeframe
    shamining yo'nalishi (+1: yopilish > EMA(n), -1: aks holda). Kelajakka qarash yo'q."""
    t = pd.to_datetime(cdf.time)
    c = pd.Series(cdf.Close.values, index=t)
    h = c.resample(f"{htf_min}min", label="left", closed="left").last().dropna()
    if len(h) < CRANK_MIN_HTF_BARS:
        raise Exception(f"{htf_name(htf_min)} filtr uchun tarix yetarli emas: {len(h)} < {CRANK_MIN_HTF_BARS}")
    ema = h.ewm(span=n, adjust=False).mean()
    known = pd.Series(np.where(h > ema, 1, -1), index=h.index + pd.Timedelta(minutes=htf_min))
    bar_end = t + pd.Timedelta(minutes=tf)
    return known.reindex(bar_end, method="ffill").fillna(0).astype(int).values

def _flips(st):
    return [i for i in range(1, len(st)) if st[i] != st[i - 1]]

def crank_init(label, cdf, tf, htf_min):
    """Birinchi ishga tushish: joriy holatni eslab qoladi (filtr mos bo'lsa pozitsiya sifatida)."""
    c = cdf.Close.values
    st = compute_crank(c)
    fl = _flips(st)
    f = fl[-1] if fl else len(st) - 1
    side = "LONG" if st[-1] == 1 else "SHORT"
    allowed = TRADE_LONG if side == "LONG" else TRADE_SHORT
    txt = f"Kuzatuv boshlandi: {label}\nJoriy CRANK trendi: {side}, signal narxi {c[f]:.2f}"
    pos = None
    if htf_min:
        hd = htf_dir_asof(cdf, tf, htf_min)
        aligned = hd[-1] == st[-1]
        txt += f"\n{htf_name(htf_min)} trend filtri: {'LONG' if hd[-1] == 1 else 'SHORT'} -> " + ("mos, kuzatamiz" if aligned else "mos emas, kutamiz")
    else:
        aligned = True
    if allowed and aligned:
        pos = dict(side=side, entry=float(c[f]), sl=None, tp=None, t=cdf.time.iloc[-1], ft=cdf.time.iloc[f])
    txt += "\nBu eski signal, yangi kirish EMAS. Keyingi yangi signal xabar qilinadi."
    send(txt)
    return pos, cdf.time.iloc[f]

def crank_step(label, cdf, pos, flip_t, tf, htf_min):
    """Yangi sham yopilganda chaqiriladi. Qaytaradi: (pos, flip_t).
    Faqat YANGI signalga (flip_t dan keyingi) harakat qiladi, chiqish faqat teskari signalda."""
    c = cdf.Close.values
    st = compute_crank(c)
    n = len(st)
    fl = _flips(st)
    if not fl:
        return pos, flip_t
    f = fl[-1]
    ft = cdf.time.iloc[f]
    if flip_t is not None and ft <= flip_t:
        return pos, flip_t                      # yangi signal yo'q (takroriy xabar bo'lmaydi)
    new_n = sum(1 for i in fl if cdf.time.iloc[i] > (flip_t or ""))
    target = "LONG" if st[f] == 1 else "SHORT"
    cp = float(c[f])
    late = n - 1 - f
    note = f"\n(kechikkan signal: {late} sham oldin)" if late > 0 else ""
    if new_n > 1:
        note += f"\n(diqqat: {new_n} ta signal o'tkazib yuborilgan, oxirgisi hisobga olindi)"
    if pos is not None and pos["side"] != target:
        pnl = (cp - pos["entry"]) if pos["side"] == "LONG" else (pos["entry"] - cp)
        send(f"{label}: CHIQISH {pos['side']} (teskari signal) @ {cp:.2f}\nNatija: {pnl:+.2f} punkt{note}")
        pos = None
    elif pos is not None:
        return pos, ft                          # holat allaqachon mos
    allowed = TRADE_LONG if target == "LONG" else TRADE_SHORT
    aligned = True
    if htf_min:
        aligned = htf_dir_asof(cdf, tf, htf_min)[f] == st[f]
    if allowed and aligned and late <= CRANK_MAX_LATE:
        pos = dict(side=target, entry=cp, sl=None, tp=None, t=cdf.time.iloc[-1], ft=ft)
        extra = f"\n{htf_name(htf_min)} trend filtri: mos" if htf_min else ""
        send(f"{label}: {target} kirish @ {cp:.2f}\nChiqish: teskari signalda (SL/TP yo'q){extra}{note}")
    elif CRANK_NOTIFY_SKIPPED:
        if not allowed:
            why = "bu yo'nalish o'chirilgan"
        elif not aligned:
            why = f"filtr rad etdi ({htf_name(htf_min)} trend teskari)"
        else:
            why = "juda kech qoldi"
        send(f"{label}: {target} signal o'tkazib yuborildi - {why}{note}")
    return pos, ft

# ------------------------------------------------------------------ umumiy
def send(text):
    print(text)
    requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                  data={"chat_id": CHAT_ID, "text": text}, timeout=10)

def run_job(S, name, src, sym, iv, mode=None, opt=None):
    mode = mode or SIG_MODE
    opt = opt or {}
    htf_min = opt.get("htf")
    tf = TFM[iv]
    key = f"{sym}|{iv}" + ("|CRANK" + (f"|h{htf_min}" if htf_min else "") if mode == "CRANK" else "")
    label = f"{name} {TFL[iv]}" + (" CRANK" + (f"+{htf_name(htf_min)}" if htf_min else "") if mode == "CRANK" else "")
    st = S.get(key)
    now = utcnow()

    if st is not None:
        if st.get("retry") and now < datetime.strptime(st["retry"], FMT):
            return
        bar_due = now >= datetime.strptime(st["last_t"], FMT) + timedelta(minutes=2 * tf)
        # CRANK da SL/TP kuzatilmaydi -> faqat yangi sham yopilganda so'rov
        if (not st["pos"] or mode == "CRANK") and not bar_due:
            return   # kerak bo'lmasa so'rov yubormaymiz (limitni tejash)

    size = CRANK_BARS.get(iv, 1500) if mode == "CRANK" else 500
    df = get_td(sym, iv, size) if src == "td" else get_yf(sym, iv)
    df["complete"] = [datetime.strptime(t, FMT) + timedelta(minutes=tf) <= now
                      for t in df.time]
    cdf = df[df.complete].reset_index(drop=True)

    if mode == "CRANK" and len(cdf) < CRANK_MIN_BARS:
        raise Exception(f"Sham yetarli emas: {len(cdf)} < {CRANK_MIN_BARS}")

    if st is None:
        flip0 = None
        if mode == "CRANK":
            pos0, flip0 = crank_init(label, cdf, tf, htf_min)
        else:
            pos0 = None
            send(f"Kuzatuv boshlandi: {label}")
        st = {"pos": pos0, "last_t": cdf.time.iloc[-1], "retry": None, "flip_t": flip0}
    pos, last_t, flip_t = st["pos"], st["last_t"], st.get("flip_t")

    if pos and pos.get("sl") is not None:
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
        if mode == "CRANK":
            pos, flip_t = crank_step(label, cdf, pos, flip_t, tf, htf_min)
        else:
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
    S[key] = {"pos": pos, "last_t": last_t, "retry": retry, "flip_t": flip_t}

def main():
    try:
        S = json.load(open("state.json"))
    except Exception:
        S = {}
    if "pos" in S:                       # eski format (faqat oltin H1)
        S = {"XAU/USD|1h": S}

    errs = []
    for job in JOBS:
        try:
            run_job(S, *job)
        except Exception as e:
            print(job, "xato:", e)
            errs.append(f"{job[0]} {TFL[job[3]]}: {str(e)[:150]}")

    msg = "\n".join(errs)
    if msg and msg != S.get("_err"):
        send("XATO:\n" + msg)
    S["_err"] = msg

    json.dump(S, open("state.json", "w"))

if __name__ == "__main__":
    main()
