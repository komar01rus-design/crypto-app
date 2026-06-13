# core/engine.py — МОЗГ системы (без единого print, только функции)
import requests, os, time
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# ─── НАСТРОЙКИ ───
BINANCE_URL = "https://api.binance.us/api/v3/klines"
CACHE_DIR = "crypto_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
START = 1000.0
FEE = 0.001

COINS = {
    "BTCUSDT": "BTCUSDT", "ETHUSDT": "ETHUSDT", "BNBUSDT": "BNBUSDT",
    "SOLUSDT": "SOLUSDT", "XRPUSDT": "XRPUSDT", "DOGEUSDT": "DOGEUSDT",
    "ADAUSDT": "ADAUSDT", "AVAXUSDT": "AVAXUSDT", "DOTUSDT": "DOTUSDT",
    "LINKUSDT": "LINKUSDT", "LTCUSDT": "LTCUSDT", "ATOMUSDT": "ATOMUSDT",
    "UNIUSDT": "UNIUSDT", "TRXUSDT": "TRXUSDT", "NEARUSDT": "NEARUSDT",
    "APTUSDT": "APTUSDT", "POLUSDT": "POLUSDT",
}

FEATURES = [
    "Move_lag1", "Move_lag2", "Move_lag3",
    "Move_avg3", "Move_avg7", "Move_avg14", "Move_std7",
    "Range", "Range_avg7", "Vol_change", "Vol_ratio",
    "DayOfWeek", "Dist_MA20", "RSI",
    "Move_accel", "Body_ratio",
    "Range_ratio", "Range_accel", "Week_span",
]


def _new_model():
    return GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42)


# ─── ЗАГРУЗКА ДАННЫХ ───
def _fetch_klines(symbol, interval="1d", start_time=None, limit=1000):
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_time is not None:
        params["startTime"] = int(start_time)
    r = requests.get(BINANCE_URL, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def _raw_to_df(raw):
    df = pd.DataFrame(raw, columns=[
        "OpenTime", "Open", "High", "Low", "Close", "Volume",
        "CloseTime", "QuoteVolume", "Trades",
        "TakerBuyBase", "TakerBuyQuote", "Ignore"])
    df["Date"] = pd.to_datetime(df["OpenTime"], unit="ms")
    for c in ["Open", "High", "Low", "Close", "Volume",
              "QuoteVolume", "Trades", "TakerBuyBase", "TakerBuyQuote"]:
        df[c] = df[c].astype(float)
    df = df.set_index("Date")
    return df[["Open", "High", "Low", "Close", "Volume",
               "QuoteVolume", "Trades", "TakerBuyBase", "TakerBuyQuote"]]


def _download_all(symbol, interval="1d", start_time=0):
    all_raw, st = [], start_time
    while True:
        raw = _fetch_klines(symbol, interval, start_time=st, limit=1000)
        if not raw:
            break
        all_raw.extend(raw)
        if len(raw) < 1000:
            break
        st = raw[-1][0] + 1
        time.sleep(0.25)
    return _raw_to_df(all_raw) if all_raw else pd.DataFrame()


def get_crypto(symbol, interval="1d"):
    cache_file = os.path.join(CACHE_DIR, f"{symbol}_{interval}.csv")
    if os.path.exists(cache_file):
        old = pd.read_csv(cache_file, index_col="Date", parse_dates=True)
        last_ms = int(old.index[-1].timestamp() * 1000)
        new = _download_all(symbol, interval, start_time=last_ms + 1)
        df = (pd.concat([old, new])
              .pipe(lambda x: x[~x.index.duplicated(keep="last")])
              .sort_index()) if not new.empty else old
    else:
        df = _download_all(symbol, interval, start_time=0)
    today = pd.Timestamp.utcnow().normalize().tz_localize(None)
    df = df[df.index < today]
    df.to_csv(cache_file)
    return df


# ─── ПРИЗНАКИ ───
def prepare_volatility(ticker):
    symbol = COINS.get(ticker, ticker)
    df = get_crypto(symbol)
    if df is None or len(df) < 300:
        return None
    df = df.copy()
    df["Move"] = (df["Close"] / df["Close"].shift(1) - 1).abs() * 100
    df["Move_lag1"], df["Move_lag2"], df["Move_lag3"] = \
        df["Move"].shift(1), df["Move"].shift(2), df["Move"].shift(3)
    df["Move_avg3"] = df["Move"].rolling(3).mean()
    df["Move_avg7"] = df["Move"].rolling(7).mean()
    df["Move_avg14"] = df["Move"].rolling(14).mean()
    df["Move_std7"] = df["Move"].rolling(7).std()
    df["Range"] = (df["High"] - df["Low"]) / df["Close"] * 100
    df["Range_avg7"] = df["Range"].rolling(7).mean()
    df["Range_ratio"] = df["Range"] / (df["Range_avg7"] + 1e-9)
    df["Range_accel"] = df["Range"].rolling(3).mean() - df["Range_avg7"]
    df["Week_span"] = (df["Close"].rolling(7).max() /
                       df["Close"].rolling(7).min() - 1) * 100
    df["Vol_change"] = df["Volume"].pct_change()
    df["Vol_avg7"] = df["Volume"].rolling(7).mean()
    df["Vol_ratio"] = df["Volume"] / df["Vol_avg7"]
    df["DayOfWeek"] = df.index.dayofweek
    ma20 = df["Close"].rolling(20).mean()
    df["Dist_MA20"] = (df["Close"] / ma20 - 1) * 100
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - 100 / (1 + gain / (loss + 1e-9))
    df["Move_accel"] = df["Move_avg3"] - df["Move_avg7"]
    body = (df["Close"] - df["Open"]).abs()
    rng = (df["High"] - df["Low"]) + 1e-9
    df["Body_ratio"] = body / rng
    median_move = df["Move"].median()
    df["Target"] = (df["Move"].shift(-1) > median_move).astype(int)
    return df.dropna()


# ─── АНАЛИЗ (возвращают данные, НЕ печатают!) ───
def walk_forward(ticker, n_chunks=5):
    d = prepare_volatility(ticker)
    if d is None:
        return None
    X, y = d[FEATURES].values, d["Target"].values
    chunk = len(X) // (n_chunks + 1)
    aucs = []
    for i in range(1, n_chunks + 1):
        tr_end, te_end = chunk * i, chunk * (i + 1)
        X_tr, y_tr = X[:tr_end], y[:tr_end]
        X_te, y_te = X[tr_end:te_end], y[tr_end:te_end]
        if len(X_te) < 20 or len(np.unique(y_te)) < 2:
            continue
        sc = StandardScaler()
        X_tr, X_te = sc.fit_transform(X_tr), sc.transform(X_te)
        m = _new_model(); m.fit(X_tr, y_tr)
        a = roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])
        aucs.append(a)
    if not aucs:
        return None
    return {"chunks": aucs, "mean": float(np.mean(aucs)),
            "std": float(np.std(aucs))}


def simulate(ticker):
    d = prepare_volatility(ticker)
    if d is None:
        return None
    X, y = d[FEATURES].values, d["Target"].values
    ret_next = d["Close"].pct_change().shift(-1).values
    split = int(len(X) * 0.8)
    sc = StandardScaler()
    X_tr = sc.fit_transform(X[:split])
    X_te = sc.transform(X[split:])
    ret_te = ret_next[split:]
    m = _new_model(); m.fit(X_tr, y[:split])
    proba = m.predict_proba(X_te)[:, 1]
    money, trades = START, 0
    for i in range(len(proba) - 1):
        r = ret_te[i]
        if np.isnan(r):
            continue
        if proba[i] > 0.55:
            money *= (1 + r) * (1 - FEE); trades += 1
    bh = START
    for i in range(len(ret_te) - 1):
        r = ret_te[i]
        if not np.isnan(r):
            bh *= (1 + r)
    return {"strategy": money, "buyhold": bh, "trades": trades,
            "days": len(proba)}


def honest_check(ticker, n_random=100):
    d = prepare_volatility(ticker)
    if d is None:
        return None
    X, y = d[FEATURES].values, d["Target"].values
    ret_next = d["Close"].pct_change().shift(-1).values
    split = int(len(X) * 0.8)
    sc = StandardScaler()
    X_tr = sc.fit_transform(X[:split])
    X_te = sc.transform(X[split:])
    ret_te = ret_next[split:]
    m = _new_model(); m.fit(X_tr, y[:split])
    proba = m.predict_proba(X_te)[:, 1]
    enter = proba > 0.55
    n_days = int(enter.sum())
    sig = START
    for i in range(len(proba) - 1):
        r = ret_te[i]
        if enter[i] and not np.isnan(r):
            sig *= (1 + r) * (1 - FEE)
    rng = np.random.default_rng(42)
    valid_idx = np.arange(len(proba) - 1)
    rand_results = []
    for _ in range(n_random):
        pick = rng.choice(valid_idx, size=min(n_days, len(valid_idx)),
                          replace=False)
        mask = np.zeros(len(proba), dtype=bool); mask[pick] = True
        money = START
        for i in range(len(proba) - 1):
            r = ret_te[i]
            if mask[i] and not np.isnan(r):
                money *= (1 + r) * (1 - FEE)
        rand_results.append(money)
    rand_avg = float(np.mean(rand_results))
    smart = sig > rand_avg * 1.05
    return {"signal": sig, "random": rand_avg, "smart": smart}


def predict_tomorrow(ticker):
    d = prepare_volatility(ticker)
    if d is None:
        return None
    X, y = d[FEATURES].values, d["Target"].values
    sc = StandardScaler()
    X_train = sc.fit_transform(X[:-1])
    X_today = sc.transform(X[-1:])
    m = _new_model(); m.fit(X_train, y[:-1])
    prob_storm = float(m.predict_proba(X_today)[0, 1])
    return {"prob_storm": prob_storm, "date": str(d.index[-1].date())}
