# main.py
from __future__ import annotations

import datetime as dt
import math
import os
import socket
import time
from decimal import Decimal, ROUND_HALF_UP, ROUND_FLOOR, ROUND_CEILING
from typing import Dict, Any, List, Optional, Tuple

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# =====================================================
# TIMEZONE (Render/Server UTC -> TR)
# =====================================================
TR_TZ = ZoneInfo(os.getenv("APP_TZ", "Europe/Istanbul"))

# =====================================================
# DEFAULT PARAMS (same spirit as your notebook)
# =====================================================

BASE_TICKERS_DEFAULT = [
    "AKBNK","ALARK","ARCLK","ASELS","BIMAS","EKGYO","ENKAI","EREGL",
    "FROTO","GARAN","HALKB","ISCTR","KCHOL","KRDMD","PETKM","PGSUS",
    "SAHOL","SISE","TAVHL","TCELL","THYAO","TOASO","TTKOM","TUPRS",
    "VAKBN","YKBNK"
]

BENCH_TICKER_DEFAULT = "XU100.IS"
REL_LOOKBACK_DAYS_DEFAULT = 20

PERIOD_DEFAULT = "12mo"
INTERVAL_DEFAULT = "1d"

WINDOW_Z_DEFAULT = 40
RSI_PERIOD_DEFAULT = 14
ADX_PERIOD_DEFAULT = 14

STO_K_DEFAULT = 10
STO_SMOOTH_K_DEFAULT = 6
STO_D_DEFAULT = 3

SMA_50_DEFAULT = 50
SMA_200_DEFAULT = 200

BB_LEN_DEFAULT = 22
BB_STD_DEFAULT = 2

ADX_TREND_STRONG_DEFAULT = 20
ADX_TREND_VERY_DEFAULT = 25
ADX_FLAT_MAX_DEFAULT = 22
RSI_FLOOR_DEFAULT = 30
ADX_SLOPE_DAYS_DEFAULT = 10

W_FLAT_TREND_DEFAULT = 30
W_DI_SUPPORT_DEFAULT = 25
W_STO_RECOVERY_DEFAULT = 25
W_RSI_SUPPORT_DEFAULT = 10
W_PRICE_SUPPORT_DEFAULT = 10

ZS_PENALTY_Z1_DEFAULT = 1.5
ZS_PENALTY_Z2_DEFAULT = 2.0
ZS_PENALTY_Z3_DEFAULT = 2.5
ZS_PENALTY_STEP_DEFAULT = 5

BS_SIGMA_DEFAULT = 0.35
TARGET_PUT_DELTA_DEFAULT = -0.30
RISK_FREE_DEFAULT = 0.40
DIV_YIELD_DEFAULT = 0.00

EXPIRY_DAY_OF_MONTH_DEFAULT = 30
ROLLOVER_DAY_DEFAULT = 15
STRIKE_ROUND_MODE_DEFAULT = "nearest"  # nearest|floor|ceil

# =====================================================
# Runtime knobs (Render'da stabilite için)
# =====================================================
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "600"))  # 10 min default
YF_TIMEOUT = int(os.getenv("YF_TIMEOUT", "25"))                # seconds
YF_RETRIES = int(os.getenv("YF_RETRIES", "3"))                 # retry count
YF_SLEEP_BASE = float(os.getenv("YF_RETRY_SLEEP", "1.2"))      # seconds

# Chunk + hard timeout
YF_CHUNK_SIZE = int(os.getenv("YF_CHUNK_SIZE", "8"))           # 8-10 iyi
SCAN_HARD_LIMIT_SEC = int(os.getenv("SCAN_HARD_LIMIT_SEC", "90"))

# yfinance bazen kendi timeout'unu tam uygulamayabiliyor; socket timeout ekliyoruz
socket.setdefaulttimeout(YF_TIMEOUT)

# =====================================================
# Simple in-memory cache
# =====================================================
_cache: Dict[str, Tuple[float, Any]] = {}


def _cache_get(key: str):
    now = time.time()
    item = _cache.get(key)
    if not item:
        return None
    ts, val = item
    if (now - ts) > CACHE_TTL_SECONDS:
        _cache.pop(key, None)
        return None
    return val


def _cache_set(key: str, val: Any):
    _cache[key] = (time.time(), val)


# =====================================================
# Helpers (ported from your notebook)
# =====================================================

def zscore_excel(close: pd.Series, window: int = 40) -> pd.Series:
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    return (close - sma) / std


def zscore_logprice(close: pd.Series, window: int = 40) -> pd.Series:
    lp = np.log(close.replace(0, np.nan))
    sma = lp.rolling(window).mean()
    std = lp.rolling(window).std()
    return (lp - sma) / std


def zscore_logreturn(close: pd.Series, window: int = 40) -> pd.Series:
    lr = np.log(close / close.shift(1))
    sma = lr.rolling(window).mean()
    std = lr.rolling(window).std()
    return (lr - sma) / std


def rsi_tv(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def adx_wilder(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14):
    up_move   = high.diff()
    down_move = -low.diff()

    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm  = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low  - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/length, adjust=False).mean()

    plus_di  = 100 * (plus_dm.ewm(alpha=1/length, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/length, adjust=False).mean() / atr)

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    adx = dx.ewm(alpha=1/length, adjust=False).mean()

    return adx, plus_di, minus_di


def stochastic_kd(high: pd.Series, low: pd.Series, close: pd.Series,
                  k_len: int = 10, smooth_k: int = 6, d_len: int = 3):
    lowest_low   = low.rolling(k_len).min()
    highest_high = high.rolling(k_len).max()
    raw_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    k = raw_k.rolling(smooth_k).mean()
    d = k.rolling(d_len).mean()
    return k, d


def slope_last(series: pd.Series, days: int = 10) -> float:
    s = series.dropna()
    if len(s) < days + 2:
        return np.nan
    return float((s.iloc[-1] - s.iloc[-1-days]) / days)


def clamp(x, lo=0, hi=100):
    try:
        return float(max(lo, min(hi, x)))
    except Exception:
        return np.nan


def zscore_penalty(z: float,
                   z1: float, z2: float, z3: float,
                   step: float) -> float:
    pen = 0.0
    if np.isnan(z):
        return 0.0
    if z > z1:
        pen -= step
    if z > z2:
        pen -= step
    if z > z3:
        pen -= step
    return pen


def bollinger_ema(close: pd.Series, length: int = 22, mult: float = 2.0):
    mid = close.ewm(span=length, adjust=False).mean()
    std = close.rolling(length).std()
    upper = mid + mult * std
    lower = mid - mult * std
    return mid, upper, lower


# ----- expiry rule -----
def month_last_day(year: int, month: int) -> int:
    if month == 12:
        nxt = dt.date(year + 1, 1, 1)
    else:
        nxt = dt.date(year, month + 1, 1)
    return (nxt - dt.timedelta(days=1)).day


def add_months(year: int, month: int, add: int):
    m = month + add
    y = year + (m - 1) // 12
    m2 = (m - 1) % 12 + 1
    return y, m2


def expiry_by_rollover_rule(asof_date: dt.date,
                            rollover_day: int = 15,
                            expiry_dom: int = 30) -> dt.date:
    y, m = asof_date.year, asof_date.month
    if asof_date.day > rollover_day:
        y, m = add_months(y, m, 1)
    dom = min(expiry_dom, month_last_day(y, m))
    return dt.date(y, m, dom)


# ----- BIST strike tick + rounding -----
def bist_pay_option_strike_tick(K: float) -> float:
    if K < 0.01:
        return 0.01
    if K <= 0.99:
        return 0.02
    if K <= 2.49:
        return 0.05
    if K <= 4.99:
        return 0.10
    if K <= 9.99:
        return 0.20
    if K <= 24.99:
        return 0.50
    if K <= 49.99:
        return 1.00
    if K <= 99.99:
        return 2.00
    if K <= 249.99:
        return 5.00
    if K <= 499.99:
        return 10.00
    if K <= 999.99:
        return 20.00
    return 50.00


def round_to_tick(x: float, tick: float, mode: str = "nearest") -> float:
    xd = Decimal(str(x))
    td = Decimal(str(tick))
    if td <= 0:
        return float(xd)

    q = xd / td
    if mode == "floor":
        q_ = q.to_integral_value(rounding=ROUND_FLOOR)
    elif mode == "ceil":
        q_ = q.to_integral_value(rounding=ROUND_CEILING)
    else:
        q_ = q.to_integral_value(rounding=ROUND_HALF_UP)

    out = (q_ * td).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return float(out)


def round_strike_to_bist_steps(K_raw: float, mode: str = "nearest") -> float:
    if np.isnan(K_raw) or K_raw <= 0:
        return np.nan
    tick = bist_pay_option_strike_tick(K_raw)
    K_mkt = round_to_tick(K_raw, tick, mode=mode)
    tick2 = bist_pay_option_strike_tick(K_mkt)
    if tick2 != tick:
        K_mkt = round_to_tick(K_mkt, tick2, mode=mode)
    return K_mkt


# ----- Black–Scholes -----
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_put_price(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return np.nan
    d1 = (math.log(S / K) + (r - q + 0.5*sigma*sigma)*T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r*T) * norm_cdf(-d2) - S * math.exp(-q*T) * norm_cdf(-d1)


def bs_put_delta(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return np.nan
    d1 = (math.log(S / K) + (r - q + 0.5*sigma*sigma)*T) / (sigma * math.sqrt(T))
    return -math.exp(-q*T) * norm_cdf(-d1)


def bs_put_strike_from_delta_solve(
    S: float, T: float, r: float, q: float, sigma: float,
    target_put_delta: float, tol: float = 1e-7, max_iter: int = 80
) -> float:
    if S <= 0 or T <= 0 or sigma <= 0:
        return np.nan
    if target_put_delta >= 0:
        return np.nan

    lo = S * 0.20
    hi = S * 2.50

    for _ in range(12):
        dlo = bs_put_delta(S, lo, T, r, q, sigma)
        dhi = bs_put_delta(S, hi, T, r, q, sigma)
        if np.isnan(dlo) or np.isnan(dhi):
            return np.nan
        if (dlo > target_put_delta) and (dhi < target_put_delta):
            break
        if dhi > target_put_delta:
            hi *= 1.35
        else:
            lo *= 0.75

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        dmid = bs_put_delta(S, mid, T, r, q, sigma)
        if np.isnan(dmid):
            return np.nan
        if abs(dmid - target_put_delta) < tol:
            return mid
        if dmid < target_put_delta:
            hi = mid
        else:
            lo = mid

    return 0.5 * (lo + hi)


# ----- RelPerf vs benchmark -----
def compute_rel_perf_vs_bench(stock_close: pd.Series, bench_close: pd.Series, lookback: int = 20):
    if stock_close is None or bench_close is None:
        return np.nan
    s = stock_close.dropna()
    b = bench_close.dropna()
    if s.empty or b.empty:
        return np.nan

    idx = s.index.intersection(b.index)
    if len(idx) < lookback + 2:
        return np.nan

    s2 = s.loc[idx]
    b2 = b.loc[idx]

    s_ret = (float(s2.iloc[-1]) / float(s2.iloc[-1 - lookback]) - 1.0)
    b_ret = (float(b2.iloc[-1]) / float(b2.iloc[-1 - lookback]) - 1.0)

    return (s_ret - b_ret) * 100.0


# =====================================================
# Data access helpers
# =====================================================
def _get_series(panel: pd.DataFrame, field: str, sym: str) -> pd.Series:
    if panel is None or panel.empty:
        return pd.Series(dtype=float)
    if isinstance(panel.columns, pd.MultiIndex):
        # yfinance output often: (PriceField, Ticker)
        if field in panel.columns.levels[0]:
            sub = panel[field]
            if isinstance(sub, pd.DataFrame) and sym in sub.columns:
                return sub[sym].dropna()
        return pd.Series(dtype=float)
    if field in panel.columns:
        return panel[field].dropna()
    return pd.Series(dtype=float)


def _chunk_list(items: List[str], size: int) -> List[List[str]]:
    return [items[i:i+size] for i in range(0, len(items), size)]


def _download_yf(all_tickers: List[str], period: str, interval: str) -> pd.DataFrame:
    """
    Render için daha stabil yfinance:
    - socket timeout (global)
    - chunk download (8 ticker)
    - retries + backoff
    - partial success: bir chunk bozulsa bile diğerlerini birleştir
    - scan hard timeout
    """
    t0 = time.time()

    tickers = [t.strip() for t in all_tickers if t and t.strip()]
    tickers = list(dict.fromkeys(tickers))  # unique, keep order

    print(
        f"[YF] start | tickers={len(tickers)} | period={period} | interval={interval} "
        f"| chunk={YF_CHUNK_SIZE} | timeout={YF_TIMEOUT}s"
    )

    chunks = _chunk_list(tickers, YF_CHUNK_SIZE)
    panels: List[pd.DataFrame] = []
    warnings_local: List[str] = []

    for ci, ch in enumerate(chunks, start=1):
        if (time.time() - t0) > SCAN_HARD_LIMIT_SEC:
            raise RuntimeError(
                f"scan hard timeout: exceeded {SCAN_HARD_LIMIT_SEC}s before finishing downloads"
            )

        last_err: Optional[Exception] = None
        ok = False

        for attempt in range(1, YF_RETRIES + 1):
            if (time.time() - t0) > SCAN_HARD_LIMIT_SEC:
                raise RuntimeError(
                    f"scan hard timeout: exceeded {SCAN_HARD_LIMIT_SEC}s during retries"
                )

            try:
                print(f"[YF] chunk {ci}/{len(chunks)} attempt {attempt}/{YF_RETRIES} | {ch}")
                p = yf.download(
                    tickers=ch,
                    period=period,
                    interval=interval,
                    auto_adjust=False,
                    progress=False,
                    group_by="column",
                    threads=False,
                    timeout=YF_TIMEOUT,
                )

                if p is None or p.empty:
                    raise RuntimeError("yfinance returned empty dataframe")

                panels.append(p)
                ok = True
                print(f"[YF] chunk {ci} OK | shape={p.shape} | elapsed={time.time()-t0:.1f}s")
                break

            except Exception as e:
                last_err = e
                sleep_s = YF_SLEEP_BASE * attempt
                print(f"[YF] chunk {ci} FAIL | {type(e).__name__}: {e} | sleep={sleep_s:.1f}s")
                time.sleep(sleep_s)

        if not ok:
            warnings_local.append(
                f"Chunk {ci} failed: {ch} | last_err={type(last_err).__name__}: {last_err}"
            )

    if not panels:
        raise RuntimeError("yfinance download failed: no chunks succeeded")

    panel = pd.concat(panels, axis=1)
    panel = panel.loc[:, ~panel.columns.duplicated()]

    if panel is None or panel.empty:
        raise RuntimeError("yfinance returned empty dataframe after concat")

    if warnings_local:
        print("[YF] WARN:", " || ".join(warnings_local))

    print(f"[YF] DONE | final_shape={panel.shape} | total_elapsed={time.time()-t0:.1f}s")
    return panel


def _run_scan(
    base_tickers: List[str],
    bench_ticker: str,
    rel_lookback_days: int,
    period: str,
    interval: str,
    window_z: int,
    rsi_period: int,
    adx_period: int,
    sto_k: int,
    sto_smooth_k: int,
    sto_d: int,
    sma50: int,
    sma200: int,
    bb_len: int,
    bb_std: float,
    adx_trend_strong: float,
    adx_trend_very: float,
    adx_flat_max: float,
    rsi_floor: float,
    adx_slope_days: int,
    w_flat_trend: float,
    w_di_support: float,
    w_sto_recovery: float,
    w_rsi_support: float,
    w_price_support: float,
    z1: float, z2: float, z3: float, z_step: float,
    bs_sigma: float,
    target_put_delta: float,
    risk_free: float,
    div_yield: float,
    expiry_dom: int,
    rollover_day: int,
    strike_round_mode: str,
) -> Dict[str, Any]:

    tickers = [t.strip().upper() for t in base_tickers if t and t.strip()]
    yahoo_tickers = [t + ".IS" if "." not in t else t for t in tickers]
    all_tickers = yahoo_tickers + [bench_ticker]

    panel = _download_yf(all_tickers=all_tickers, period=period, interval=interval)
    print(f"[SCAN] panel rows={len(panel)} cols={len(panel.columns)}")

    bench_close = _get_series(panel, "Close", bench_ticker)

    rows: List[Dict[str, Any]] = []
    warnings: List[str] = []

    if bench_close.empty:
        warnings.append(f"Benchmark verisi gelmedi: {bench_ticker} (RelPerf NaN kalır)")

    # TZ-aware timestamps
    run_dt_tr = dt.datetime.now(TR_TZ)
    run_dt_utc = dt.datetime.now(dt.timezone.utc)
    run_ts = run_dt_tr.isoformat(timespec="seconds")      # 2026-03-05T13:27:10+03:00
    run_ts_utc = run_dt_utc.isoformat(timespec="seconds")  # 2026-03-05T10:27:10+00:00

    for base, sym in zip(tickers, yahoo_tickers):
        close = _get_series(panel, "Close", sym)
        high  = _get_series(panel, "High",  sym)
        low   = _get_series(panel, "Low",   sym)

        if close.empty or high.empty or low.empty:
            warnings.append(f"OHLC verisi yok: {sym}")
            continue

        min_len = max(
            window_z + 5, rsi_period + 5, adx_period + 5,
            sto_k + sto_smooth_k + sto_d + 10, adx_slope_days + 15,
            bb_len + 5
        )
        if len(close) < min_len:
            warnings.append(f"Yetersiz veri: {sym} | len={len(close)}")
            continue

        data_asof_ts = close.index[-1]
        data_asof = pd.Timestamp(data_asof_ts).date()
        price = float(close.iloc[-1])

        sma50_v  = float(close.rolling(sma50).mean().iloc[-1]) if len(close) >= sma50 else np.nan
        sma200_v = float(close.rolling(sma200).mean().iloc[-1]) if len(close) >= sma200 else np.nan

        z_series = zscore_excel(close, window=window_z)
        z_last = float(z_series.iloc[-1])

        z_lp = zscore_logprice(close, window=window_z)
        z_lp_last = float(z_lp.iloc[-1])

        z_lr = zscore_logreturn(close, window=window_z)
        z_lr_last = float(z_lr.iloc[-1])

        emaN = close.ewm(span=window_z, adjust=False).mean()
        ema_last = float(emaN.iloc[-1])
        ema_diff = (price - ema_last) / ema_last
        ema_diff_pct = ema_diff * 100.0

        rsi_s = rsi_tv(close, length=rsi_period)
        rsi_last = float(rsi_s.iloc[-1])

        adx_s, dip_s, din_s = adx_wilder(high, low, close, length=adx_period)
        adx_last = float(adx_s.iloc[-1])
        dip_last = float(dip_s.iloc[-1])
        din_last = float(din_s.iloc[-1])
        adx_slp = slope_last(adx_s, days=adx_slope_days)

        sto_k_s, sto_d_s = stochastic_kd(high, low, close, k_len=sto_k, smooth_k=sto_smooth_k, d_len=sto_d)
        k_last = float(sto_k_s.iloc[-1])
        d_last = float(sto_d_s.iloc[-1])

        cross_up = False
        if sto_k_s.notna().sum() > 3 and sto_d_s.notna().sum() > 3:
            k1, k0 = sto_k_s.iloc[-1], sto_k_s.iloc[-2]
            d1, d0 = sto_d_s.iloc[-1], sto_d_s.iloc[-2]
            cross_up = (k1 > d1) and (k0 <= d0)

        bb_mid, bb_up, bb_low = bollinger_ema(close, length=bb_len, mult=bb_std)
        bb_mid_last = float(bb_mid.iloc[-1])
        bb_up_last = float(bb_up.iloc[-1])
        bb_low_last = float(bb_low.iloc[-1])
        denom = (bb_up_last - bb_low_last)
        bb_score = np.nan if (np.isnan(denom) or denom == 0) else ((price - bb_low_last) / denom) * 100.0

        rel_perf_xu100 = compute_rel_perf_vs_bench(close, bench_close, lookback=rel_lookback_days)

        setup1 = (
            (adx_last <= adx_flat_max) and
            (dip_last >= din_last) and
            (rsi_last >= rsi_floor)
        )

        setup2 = (
            (k_last <= 35) and
            (cross_up is True) and
            (rsi_last >= rsi_floor) and
            (adx_last <= adx_trend_very) and
            (np.isnan(adx_slp) or adx_slp <= 0.8)
        )

        risk_veto = (
            ((adx_last >= adx_trend_strong) and (din_last > dip_last)) or
            ((adx_last >= adx_trend_very) and (din_last >= dip_last))
        )

        put_candidate = (not risk_veto) and (setup1 or setup2)

        score_raw = 0.0
        score_raw += w_flat_trend * (1 - min(adx_last, 40) / 40)
        score_raw += w_di_support * (1 if dip_last >= din_last else 0)
        score_raw += w_sto_recovery * (1 if cross_up else (0.5 if k_last > d_last else 0))
        score_raw += w_rsi_support * (1 if rsi_last >= rsi_floor else 0)
        score_raw += w_price_support * (1 if ema_diff >= 0 else 0)

        z_pen = zscore_penalty(z_last, z1, z2, z3, z_step)
        bb_pen = -10.0 if (not np.isnan(bb_score) and bb_score >= 70) else 0.0

        score = clamp(score_raw + z_pen + bb_pen, 0, 100)

        expiry_date = expiry_by_rollover_rule(data_asof, rollover_day=rollover_day, expiry_dom=expiry_dom)
        dte_days = int((expiry_date - data_asof).days)
        T_years = max(dte_days, 1) / 365.0

        K_raw = bs_put_strike_from_delta_solve(
            S=price, T=T_years, r=risk_free, q=div_yield, sigma=bs_sigma,
            target_put_delta=target_put_delta
        )
        K_30 = round_strike_to_bist_steps(K_raw, mode=strike_round_mode)

        bs_prem = bs_put_price(S=price, K=K_30, T=T_years, r=risk_free, q=div_yield, sigma=bs_sigma)
        bs_prem_pct = (bs_prem / price) * 100.0 if (price > 0 and not np.isnan(bs_prem)) else np.nan

        reasons = []
        if risk_veto:
            reasons.append("RİSK VETO: Güçlü aşağı trend (ADX yüksek & DI- baskın)")
        else:
            if setup1:
                reasons.append("Setup-1: Trend zayıf/yatay (ADX düşük), DI+ ≥ DI-")
            if setup2:
                reasons.append("Setup-2: Stoch bullish cross ile toparlanma teyidi")
            if rsi_last < rsi_floor:
                reasons.append("RSI düşük (freefall riski)")
        if z_pen < 0:
            reasons.append(f"ZScore Penalty: Z={z_last:.2f} -> {z_pen:.0f} puan")
        if bb_pen < 0:
            reasons.append(f"BB Penalty: BB_Skor={bb_score:.2f} -> {bb_pen:.0f} puan")
        reason_text = " | ".join(reasons) if reasons else "—"

        rows.append({
            "Hisse": base,
            "Data_AsOf": data_asof.isoformat(),
            "Run_Timestamp": run_ts,          # TZ-aware
            "Run_Timestamp_UTC": run_ts_utc,  # debug
            "Price": price,

            "RelPerf_XU100_%": float(rel_perf_xu100) if not np.isnan(rel_perf_xu100) else None,

            "SMA50": sma50_v if not np.isnan(sma50_v) else None,
            "SMA200": sma200_v if not np.isnan(sma200_v) else None,

            "BB_Mid(EMA22)": bb_mid_last if not np.isnan(bb_mid_last) else None,
            "BB_Upper": bb_up_last if not np.isnan(bb_up_last) else None,
            "BB_Lower": bb_low_last if not np.isnan(bb_low_last) else None,
            "BB_Skor": float(bb_score) if not np.isnan(bb_score) else None,

            "ZScore": z_last if not np.isnan(z_last) else None,
            "LogPrice_ZScore": z_lp_last if not np.isnan(z_lp_last) else None,
            "LogReturn_ZScore": z_lr_last if not np.isnan(z_lr_last) else None,

            "EMA_Diff": ema_diff if not np.isnan(ema_diff) else None,
            "EMA40 Fark (%)": ema_diff_pct if not np.isnan(ema_diff_pct) else None,
            "RSI": rsi_last if not np.isnan(rsi_last) else None,

            "ADX": adx_last if not np.isnan(adx_last) else None,
            "DI+": dip_last if not np.isnan(dip_last) else None,
            "DI-": din_last if not np.isnan(din_last) else None,
            "ADX_Slope": adx_slp if not np.isnan(adx_slp) else None,

            "StochK": k_last if not np.isnan(k_last) else None,
            "StochD": d_last if not np.isnan(d_last) else None,
            "StochCrossUp": bool(cross_up),

            "Risk_Veto": bool(risk_veto),
            "Setup1": bool(setup1),
            "Setup2": bool(setup2),
            "Put_Aday": bool(put_candidate),
            "Skor": score if not np.isnan(score) else None,

            "Expiry_30th": expiry_date.isoformat(),
            "DTE_Days": int(dte_days),

            "BS_Sigma": float(bs_sigma),
            "BS_TargetPutDelta": float(target_put_delta),
            "BS_RiskFree": float(risk_free),
            "BS_DivYield": float(div_yield),

            "BS_Strike_RAW": float(K_raw) if not np.isnan(K_raw) else None,
            "Strike_Tick": float(bist_pay_option_strike_tick(K_30)) if not np.isnan(K_30) else None,
            "Strike_Round_Mode": strike_round_mode,

            "BS_Strike_D30P_30th": float(K_30) if not np.isnan(K_30) else None,
            "BS_Premium_D30P_30th": float(bs_prem) if not np.isnan(bs_prem) else None,
            "BS_Premium_%Spot": float(bs_prem_pct) if not np.isnan(bs_prem_pct) else None,

            "Gerekce": reason_text
        })

    rows_sorted = sorted(
        rows,
        key=lambda r: (r.get("Skor") is not None, r.get("Skor", -1)),
        reverse=True
    )

    return {
        "run_ts": run_ts,              # TR timezone-aware
        "run_ts_utc": run_ts_utc,      # debug
        "tz": str(TR_TZ),
        "data_asof": rows_sorted[0]["Data_AsOf"] if rows_sorted else None,
        "params": {
            "bench_ticker": bench_ticker,
            "rel_lookback_days": rel_lookback_days,
            "period": period,
            "interval": interval,
            "window_z": window_z,
            "rsi_period": rsi_period,
            "adx_period": adx_period,
            "stoch": {"k": sto_k, "smooth_k": sto_smooth_k, "d": sto_d},
            "bb": {"len": bb_len, "std": bb_std},
            "expiry": {"rollover_day": rollover_day, "expiry_dom": expiry_dom},
            "bs": {"sigma": bs_sigma, "target_put_delta": target_put_delta, "r": risk_free, "q": div_yield},
            "strike_round_mode": strike_round_mode,
            "cache_ttl_seconds": CACHE_TTL_SECONDS,
            "yf_timeout": YF_TIMEOUT,
            "yf_retries": YF_RETRIES,
            "yf_chunk_size": YF_CHUNK_SIZE,
            "scan_hard_limit_sec": SCAN_HARD_LIMIT_SEC
        },
        "warnings": warnings,
        "rows": rows_sorted
    }


# =====================================================
# FastAPI app
# =====================================================
app = FastAPI(title="BIST Put Screener API", version="1.2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "ok": True,
        "service": "bist-put-screener",
        "ts": dt.datetime.now(TR_TZ).isoformat(timespec="seconds"),
        "tz": str(TR_TZ),
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "ts": dt.datetime.now(TR_TZ).isoformat(timespec="seconds"),
        "tz": str(TR_TZ),
    }


@app.get("/scan")
def scan(
    tickers: Optional[str] = Query(
        default=None,
        description="Virgülle ayrılmış semboller (AKBNK,ASELS). Boşsa default evren."
    ),
    period: str = Query(default=PERIOD_DEFAULT),
    interval: str = Query(default=INTERVAL_DEFAULT),
):
    try:
        base_tickers = BASE_TICKERS_DEFAULT if not tickers else [
            t.strip().upper() for t in tickers.split(",") if t.strip()
        ]
        if not base_tickers:
            raise HTTPException(status_code=400, detail="tickers list is empty")

        # cache key includes period/interval + tickers
        cache_key = f"scan|{','.join(base_tickers)}|{period}|{interval}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return {"cached": True, **cached}

        res = _run_scan(
            base_tickers=base_tickers,
            bench_ticker=BENCH_TICKER_DEFAULT,
            rel_lookback_days=REL_LOOKBACK_DAYS_DEFAULT,
            period=period,
            interval=interval,

            window_z=WINDOW_Z_DEFAULT,
            rsi_period=RSI_PERIOD_DEFAULT,
            adx_period=ADX_PERIOD_DEFAULT,

            sto_k=STO_K_DEFAULT,
            sto_smooth_k=STO_SMOOTH_K_DEFAULT,
            sto_d=STO_D_DEFAULT,

            sma50=SMA_50_DEFAULT,
            sma200=SMA_200_DEFAULT,

            bb_len=BB_LEN_DEFAULT,
            bb_std=float(BB_STD_DEFAULT),

            adx_trend_strong=float(ADX_TREND_STRONG_DEFAULT),
            adx_trend_very=float(ADX_TREND_VERY_DEFAULT),
            adx_flat_max=float(ADX_FLAT_MAX_DEFAULT),
            rsi_floor=float(RSI_FLOOR_DEFAULT),
            adx_slope_days=int(ADX_SLOPE_DAYS_DEFAULT),

            w_flat_trend=float(W_FLAT_TREND_DEFAULT),
            w_di_support=float(W_DI_SUPPORT_DEFAULT),
            w_sto_recovery=float(W_STO_RECOVERY_DEFAULT),
            w_rsi_support=float(W_RSI_SUPPORT_DEFAULT),
            w_price_support=float(W_PRICE_SUPPORT_DEFAULT),

            z1=float(ZS_PENALTY_Z1_DEFAULT),
            z2=float(ZS_PENALTY_Z2_DEFAULT),
            z3=float(ZS_PENALTY_Z3_DEFAULT),
            z_step=float(ZS_PENALTY_STEP_DEFAULT),

            bs_sigma=float(BS_SIGMA_DEFAULT),
            target_put_delta=float(TARGET_PUT_DELTA_DEFAULT),
            risk_free=float(RISK_FREE_DEFAULT),
            div_yield=float(DIV_YIELD_DEFAULT),

            expiry_dom=int(EXPIRY_DAY_OF_MONTH_DEFAULT),
            rollover_day=int(ROLLOVER_DAY_DEFAULT),
            strike_round_mode=str(STRIKE_ROUND_MODE_DEFAULT),
        )

        _cache_set(cache_key, res)
        return {"cached": False, **res}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"scan failed: {type(e).__name__}: {e}")
