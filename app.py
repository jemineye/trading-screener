"""
==============================================================
  TRADING SCREENER — Web App (Flask)
  v5 — Finviz locally, Manual Search via yfinance everywhere
==============================================================

SETUP (run once):
  pip install finviz pandas yfinance pytz flask flask-cors

RUN:
  python app.py → http://localhost:5000
  Railway      → Manual Search tab only
==============================================================
"""

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import pandas as pd
import yfinance as yf
from datetime import datetime, time as dtime
import pytz
import os
import warnings
warnings.filterwarnings("ignore")

try:
    from finviz.screener import Screener
    HAS_FINVIZ = True
except ImportError:
    HAS_FINVIZ = False

app = Flask(__name__)
CORS(app)

# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════

MIN_VOLUME  = 1_000_000
EMA_SHORT   = 5
EMA_LONG    = 9
ET          = pytz.timezone("America/New_York")
MARKET_OPEN = dtime(9, 30)

# Detect if running on Railway (cloud) or locally
# Railway sets RAILWAY_PROJECT_ID; local machines don't
IS_CLOUD = bool(
    os.environ.get("RAILWAY_PROJECT_ID") or
    os.environ.get("RAILWAY_ENVIRONMENT") or
    os.environ.get("RAILWAY_SERVICE_ID")
)

DAY_TRADE_FILTERS = [
    "cap_smallunder",
    "sh_curvol_o5000",
    "sh_float_u100",
    "sh_price_u50",
    "sh_relvol_o2",
    "ta_perf_1wup",
]

SWING_FILTERS = [
    "cap_midover",
    "sh_avgvol_o2000",
    "sh_curvol_o5000",
    "ta_beta_o1",
    "ta_sma200_pa",
]

SQUEEZE_FILTERS = [
    "cap_smallunder",
    "sh_curvol_o5000",
    "sh_float_u20",
    "sh_short_o20",
    "sh_relvol_o2",
    "sh_price_u50",
]


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def get_session():
    now_et = datetime.now(ET).time()
    if now_et < MARKET_OPEN:
        return "premarket"
    elif now_et <= dtime(16, 0):
        return "open"
    else:
        return "afterhours"


def normalize_columns(df):
    if df.empty:
        return df
    rename_map = {
        "ticker": "Ticker", "company": "Company", "sector": "Sector",
        "industry": "Industry", "price": "Price", "change": "Change",
        "volume": "Volume", "rel volume": "Rel Volume",
        "rel_volume": "Rel Volume", "relvol": "Rel Volume",
        "relative volume": "Rel Volume", "rel. volume": "Rel Volume",
        "short float": "Short Float", "float": "Float",
        "beta": "Beta", "sma200": "SMA200",
    }
    df = df.copy()
    df.columns = [rename_map.get(c.lower().strip(), c) for c in df.columns]
    return df


def fetch_screener(filters, label):
    if not HAS_FINVIZ:
        return pd.DataFrame()
    try:
        print(f"  Fetching {label} from Finviz...")
        screen_ov = Screener(filters=filters, table="Overview", order="-volume")
        data_ov   = list(screen_ov)
        if not data_ov:
            print(f"  → {label}: no results")
            return pd.DataFrame()

        raw_df = pd.DataFrame(data_ov)

        # Finviz 2.0.0 bug: values are shifted right by 1 position regardless
        # of whether Price appears numeric. Always remap columns.
        if "No." in raw_df.columns and not raw_df.empty:
            # Check if Ticker column has suspiciously short values (1-2 chars)
            # while Company column has longer values that look like tickers
            avg_ticker_len = raw_df["Ticker"].astype(str).str.len().mean()
            avg_company_len = raw_df["Company"].astype(str).str.len().mean()
            if avg_ticker_len < 2.5 and avg_company_len <= 5:
                # Values are shifted — Company col has the real tickers
                raw_df = raw_df.rename(columns={
                    "Ticker":     "_drop",
                    "Company":    "Ticker",
                    "Sector":     "Company",
                    "Industry":   "Sector",
                    "Country":    "Industry",
                    "Market Cap": "Country",
                    "P/E":        "Market Cap",
                    "Price":      "P/E",
                    "Change":     "Price",
                    "Volume":     "Change",
                })
                raw_df["Volume"] = None

        df_ov = normalize_columns(raw_df.drop(columns=["No.", "_drop"], errors="ignore"))
        df_ov = df_ov[df_ov["Ticker"].astype(str).str.match(r'^[A-Z]{1,5}(\.[A-Z])?$')].copy()
        print(f"  → {label}: {len(df_ov)} tickers")

        # Calculate Rel Volume and actual Volume via yfinance in parallel
        print(f"  Calculating rel volume for {len(df_ov)} tickers...")

        def get_vol_data(ticker):
            try:
                hist = yf.Ticker(ticker).history(period="1mo", interval="1d")
                if hist.empty or len(hist) < 5:
                    return None, None
                avg_vol = float(hist["Volume"].iloc[:-1].mean())
                cur_vol = float(hist["Volume"].iloc[-1])
                relvol  = round(cur_vol / avg_vol, 1) if avg_vol > 0 else None
                return relvol, int(cur_vol)
            except:
                return None, None

        import concurrent.futures
        tickers = df_ov["Ticker"].tolist()
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
            vol_results = list(ex.map(get_vol_data, tickers))

        rvols   = [r[0] for r in vol_results]
        volumes = [r[1] for r in vol_results]
        df_ov["Rel Volume"] = rvols
        df_ov["yf_volume"]  = volumes

        # Apply 1M volume gate using yfinance volume
        before = len(df_ov)
        df_ov = df_ov[df_ov["yf_volume"].apply(lambda v: v is not None and v >= MIN_VOLUME)].copy()
        print(f"  → {before} → {len(df_ov)} after {MIN_VOLUME:,} volume gate")
        print(f"  → {sum(1 for r in rvols if r)} rel vol values calculated")

        # Use yfinance volume as display volume
        df_ov["Volume"] = df_ov["yf_volume"]
        return df_ov

        # Apply 1M volume gate using yfinance volume
        before = len(df_ov)
        df_ov = df_ov[df_ov["yf_volume"].apply(lambda v: v is not None and v >= MIN_VOLUME)].copy()
        print(f"  → {before} tickers → {len(df_ov)} after {MIN_VOLUME:,} volume gate")
        print(f"  → {sum(1 for r in rvols if r)} rel vol values calculated")

        # Use yfinance volume as the display volume
        df_ov["Volume"] = df_ov["yf_volume"]
        return df_ov

    except Exception as e:
        print(f"  Error fetching {label}: {e}")
        import traceback; traceback.print_exc()
        return pd.DataFrame()


def apply_volume_gate(df):
    if df.empty:
        return df
    if "Ticker" in df.columns:
        df = df[df["Ticker"].astype(str).str.match(r'^[A-Z]{1,5}(\.[A-Z])?$')].copy()
    if "Rel Volume" in df.columns:
        df = df.copy()
        df["Rel Volume"] = pd.to_numeric(df["Rel Volume"], errors="coerce")
        df = df.sort_values("Rel Volume", ascending=False)
        df["Rel Volume"] = df["Rel Volume"].apply(
            lambda x: str(round(x, 1)) if pd.notna(x) else "—"
        )
    return df


def safe_float(val):
    try:
        return float(str(val).replace(",", "").replace("$", "").replace("%", ""))
    except:
        return 0.0


def get_news(ticker):
    """Fetch recent news headlines via yfinance."""
    news = []
    try:
        t = yf.Ticker(ticker)
        items = t.news or []
        for item in items[:6]:
            try:
                # yfinance news structure varies by version
                content = item.get("content", item)
                title   = content.get("title") or item.get("title", "")
                url     = ""
                # try nested canonical url
                click_url = content.get("clickThroughUrl") or {}
                if isinstance(click_url, dict):
                    url = click_url.get("url", "")
                if not url:
                    url = content.get("canonicalUrl", {}).get("url", "") if isinstance(content.get("canonicalUrl"), dict) else ""
                if not url:
                    url = item.get("link", "") or item.get("url", "")

                # publisher
                provider = content.get("provider") or {}
                source = provider.get("displayName", "") if isinstance(provider, dict) else ""
                if not source:
                    source = item.get("publisher", "") or item.get("source", "Unknown")

                # timestamp
                pub_time = content.get("pubDate") or item.get("providerPublishTime", 0)
                if isinstance(pub_time, (int, float)) and pub_time > 0:
                    dt = datetime.fromtimestamp(pub_time, tz=ET)
                    ts = dt.strftime("%b %d  %I:%M %p")
                elif isinstance(pub_time, str):
                    ts = pub_time[:16]
                else:
                    ts = ""

                if title:
                    news.append({"title": title, "url": url, "source": source, "time": ts})
            except:
                continue
    except:
        pass
    return news


def get_extended_hours_price(ticker, prev_close):
    """
    Fetch the latest pre-market or after-hours price via yfinance.
    Returns current extended hours price, change $, change %, volume,
    and whether it's a gap up or down.
    """
    result = {
        "ext_price": None,
        "ext_change": None,
        "ext_change_pct": None,
        "ext_volume": None,
        "ext_session": None,   # 'premarket', 'afterhours', or None
        "gap_pct": None,       # vs prior close
    }
    try:
        now_et = datetime.now(ET).time()
        t = yf.Ticker(ticker)

        # Pull 1-min bars with pre/post market
        df = t.history(period="1d", interval="1m", prepost=True)
        if df.empty:
            return result

        df.index = df.index.tz_convert(ET)

        # Determine session window
        if now_et < dtime(9, 30):
            session_data = df.between_time("04:00", "09:29")
            result["ext_session"] = "premarket"
        elif now_et >= dtime(16, 0):
            session_data = df.between_time("16:00", "20:00")
            result["ext_session"] = "afterhours"
        else:
            # Regular market hours — show intraday last price vs prev close gap
            session_data = df.between_time("09:30", "16:00")
            result["ext_session"] = "open"

        if session_data.empty:
            return result

        ext_price  = round(float(session_data["Close"].iloc[-1]), 2)
        ext_vol    = int(session_data["Volume"].sum())
        change     = round(ext_price - prev_close, 2)
        change_pct = round((change / prev_close) * 100, 2) if prev_close else 0

        result["ext_price"]      = ext_price
        result["ext_change"]     = change
        result["ext_change_pct"] = change_pct
        result["ext_volume"]     = ext_vol
        result["gap_pct"]        = change_pct  # gap vs prior close

    except Exception as e:
        print(f"Extended hours error {ticker}: {e}")

    return result




def get_tv_technicals_batch(tickers, interval="1d"):
    """
    Fetch TradingView technical analysis for a batch of tickers at once.
    Returns dict keyed by ticker with EMA5, EMA9, SMA20/50/200, VWAP, RSI, price, volume.
    Falls back gracefully to yfinance if tradingview-ta not installed.
    interval: "1m", "5m", "1d" etc.
    """
    results = {}
    try:
        from tradingview_ta import get_multiple_analysis, Interval
        interval_map = {
            "1m": Interval.INTERVAL_1_MINUTE,
            "5m": Interval.INTERVAL_5_MINUTES,
            "15m": Interval.INTERVAL_15_MINUTES,
            "1h": Interval.INTERVAL_1_HOUR,
            "1d": Interval.INTERVAL_1_DAY,
            "1W": Interval.INTERVAL_1_WEEK,
        }
        tv_interval = interval_map.get(interval, Interval.INTERVAL_1_DAY)

        # Build symbol list — TradingView needs exchange prefix
        # We try NASDAQ first, fall back to NYSE for unknowns
        symbols = [f"NASDAQ:{t}" for t in tickers]
        analysis = get_multiple_analysis(
            screener="america",
            interval=tv_interval,
            symbols=symbols
        )

        for ticker in tickers:
            sym = f"NASDAQ:{ticker}"
            a   = analysis.get(sym)
            if not a:
                # Try NYSE
                try:
                    from tradingview_ta import TA_Handler
                    handler = TA_Handler(
                        symbol=ticker, screener="america",
                        exchange="NYSE", interval=tv_interval
                    )
                    a = handler.get_analysis()
                except:
                    results[ticker] = None
                    continue

            ind = a.indicators
            results[ticker] = {
                "price":    round(float(ind.get("close", 0)), 2),
                "open":     round(float(ind.get("open", 0)), 2),
                "high":     round(float(ind.get("high", 0)), 2),
                "low":      round(float(ind.get("low", 0)), 2),
                "volume":   int(ind.get("volume", 0)),
                "ema5":     round(float(ind.get("EMA5",  ind.get("close", 0))), 2),
                "ema9":     round(float(ind.get("EMA9",  ind.get("close", 0))), 2),
                "sma20":    round(float(ind.get("SMA20", 0)), 2),
                "sma50":    round(float(ind.get("SMA50", 0)), 2),
                "sma200":   round(float(ind.get("SMA200", 0)), 2),
                "vwap":     round(float(ind.get("VWAP", 0)), 2),
                "rsi":      round(float(ind.get("RSI", 0)), 1),
                "atr":      round(float(ind.get("ATR", 0)), 2),
                "change":   round(float(ind.get("change", 0)), 2),
                "ema_ok":   bool(ind.get("EMA5", 0) > ind.get("EMA9", 0)),
                "source":   "tradingview",
            }

    except ImportError:
        print("  tradingview-ta not installed — falling back to yfinance")
        # Graceful fallback: use yfinance for each ticker
        import concurrent.futures
        def yf_snapshot(ticker):
            try:
                hist = yf.Ticker(ticker).history(period="6mo", interval="1d")
                if hist.empty or len(hist) < 10:
                    return ticker, None
                closes = hist["Close"]
                ema5 = closes.ewm(span=5, adjust=False).mean()
                ema9 = closes.ewm(span=9, adjust=False).mean()
                prev = hist.iloc[-2]
                vwap = round(float((prev["High"] + prev["Low"] + prev["Close"]) / 3), 2)
                return ticker, {
                    "price":   round(float(closes.iloc[-1]), 2),
                    "high":    round(float(hist["High"].iloc[-1]), 2),
                    "low":     round(float(hist["Low"].iloc[-1]), 2),
                    "volume":  int(hist["Volume"].iloc[-1]),
                    "ema5":    round(float(ema5.iloc[-1]), 2),
                    "ema9":    round(float(ema9.iloc[-1]), 2),
                    "sma20":   round(float(closes.rolling(20).mean().iloc[-1]), 2),
                    "sma50":   round(float(closes.rolling(50).mean().iloc[-1]), 2),
                    "sma200":  round(float(closes.rolling(200).mean().iloc[-1]), 2),
                    "vwap":    vwap,
                    "rsi":     None,
                    "atr":     None,
                    "change":  round(float((closes.iloc[-1] - closes.iloc[-2]) / closes.iloc[-2] * 100), 2),
                    "ema_ok":  bool(ema5.iloc[-1] > ema9.iloc[-1]),
                    "source":  "yfinance",
                }
            except:
                return ticker, None

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
            for ticker, data in ex.map(lambda t: yf_snapshot(t), tickers):
                results[ticker] = data

    except Exception as e:
        print(f"  TradingView batch error: {e}")

    return results


def get_premarket_levels(ticker):
    """Fetch pre-market OHLC using yfinance 1-min bars."""
    result = {"pm_high": None, "pm_low": None, "pm_open": None, "pm_last": None}
    try:
        df = yf.Ticker(ticker).history(period="1d", interval="1m", prepost=True)
        if df.empty:
            return result
        df.index = df.index.tz_convert(ET)
        pm = df.between_time("04:00", "09:29")
        if len(pm) < 3:
            return result
        result["pm_high"]  = round(float(pm["High"].max()), 2)
        result["pm_low"]   = round(float(pm["Low"].min()), 2)
        result["pm_open"]  = round(float(pm["Open"].iloc[0]), 2)
        result["pm_last"]  = round(float(pm["Close"].iloc[-1]), 2)
    except:
        pass
    return result



    result = {
        "ema5": None, "ema9": None, "ema_ok": False,
        "key_high": None, "key_low": None, "key_close": None,
        "vwap_proxy": None,
        "pm_high": None, "pm_low": None, "pm_open": None, "pm_last": None,
        "level_mode": session, "error": None,
    }
    try:
        hist = yf.Ticker(ticker).history(period="6mo", interval="1d")
        if hist.empty or len(hist) < 10:
            result["error"] = "insufficient history"
            return result

        closes = hist["Close"]
        ema5 = closes.ewm(span=EMA_SHORT, adjust=False).mean()
        ema9 = closes.ewm(span=EMA_LONG,  adjust=False).mean()
        result["ema5"]   = round(float(ema5.iloc[-1]), 2)
        result["ema9"]   = round(float(ema9.iloc[-1]), 2)
        result["ema_ok"] = bool(ema5.iloc[-1] > ema9.iloc[-1])

        prev = hist.iloc[-2] if len(hist) >= 2 else hist.iloc[-1]
        result["key_close"]  = round(float(prev["Close"]), 2)
        result["vwap_proxy"] = round(float((prev["High"] + prev["Low"] + prev["Close"]) / 3), 2)

        if session == "premarket":
            result["key_high"] = round(float(prev["High"]), 2)
            result["key_low"]  = round(float(prev["Low"]), 2)
        else:
            pm = get_premarket_levels(ticker)
            result.update(pm)
            result["key_high"] = pm["pm_high"] or round(float(prev["High"]), 2)
            result["key_low"]  = pm["pm_low"]  or round(float(prev["Low"]),  2)
    except Exception as e:
        result["error"] = str(e)
    return result


def get_day_trade_technicals(ticker, session):
    """yfinance-based day trade technicals — used by manual search."""
    result = {
        "ema5": None, "ema9": None, "ema_ok": False,
        "key_high": None, "key_low": None, "key_close": None,
        "vwap_proxy": None,
        "pm_high": None, "pm_low": None, "pm_open": None, "pm_last": None,
        "level_mode": session, "error": None,
    }
    try:
        hist = yf.Ticker(ticker).history(period="6mo", interval="1d")
        if hist.empty or len(hist) < 10:
            result["error"] = "insufficient history"
            return result
        closes = hist["Close"]
        ema5 = closes.ewm(span=EMA_SHORT, adjust=False).mean()
        ema9 = closes.ewm(span=EMA_LONG,  adjust=False).mean()
        result["ema5"]   = round(float(ema5.iloc[-1]), 2)
        result["ema9"]   = round(float(ema9.iloc[-1]), 2)
        result["ema_ok"] = bool(ema5.iloc[-1] > ema9.iloc[-1])
        prev = hist.iloc[-2] if len(hist) >= 2 else hist.iloc[-1]
        result["key_close"]  = round(float(prev["Close"]), 2)
        result["vwap_proxy"] = round(float((prev["High"] + prev["Low"] + prev["Close"]) / 3), 2)
        if session == "premarket":
            result["key_high"] = round(float(prev["High"]), 2)
            result["key_low"]  = round(float(prev["Low"]), 2)
        else:
            pm = get_premarket_levels(ticker)
            result.update(pm)
            result["key_high"] = pm["pm_high"] or round(float(prev["High"]), 2)
            result["key_low"]  = pm["pm_low"]  or round(float(prev["Low"]),  2)
    except Exception as e:
        result["error"] = str(e)
    return result


def get_swing_technicals(ticker):
    result = {
        "ema5": None, "ema9": None, "ema_ok": False,
        "sma20": None, "sma50": None, "sma200": None,
        "sma200_rising": False,
        "week52_high": None, "week52_low": None,
        "swing_high_20d": None, "swing_low_20d": None,
        "support_label": None, "support_val": None,
        "current_price": None, "error": None,
    }
    try:
        hist = yf.Ticker(ticker).history(period="2y", interval="1d")
        if hist.empty or len(hist) < 200:
            result["error"] = "insufficient history"
            return result

        closes = hist["Close"]
        highs  = hist["High"]
        lows   = hist["Low"]
        price  = float(closes.iloc[-1])
        result["current_price"] = round(price, 2)

        ema5 = closes.ewm(span=5, adjust=False).mean()
        ema9 = closes.ewm(span=9, adjust=False).mean()
        result["ema5"]   = round(float(ema5.iloc[-1]), 2)
        result["ema9"]   = round(float(ema9.iloc[-1]), 2)
        result["ema_ok"] = bool(ema5.iloc[-1] > ema9.iloc[-1])

        sma20  = closes.rolling(20).mean()
        sma50  = closes.rolling(50).mean()
        sma200 = closes.rolling(200).mean()
        result["sma20"]  = round(float(sma20.iloc[-1]), 2)
        result["sma50"]  = round(float(sma50.iloc[-1]), 2)
        result["sma200"] = round(float(sma200.iloc[-1]), 2)
        result["sma200_rising"] = bool(float(sma200.iloc[-1]) > float(sma200.iloc[-10]))

        lookback = min(252, len(closes))
        result["week52_high"]    = round(float(highs.iloc[-lookback:].max()), 2)
        result["week52_low"]     = round(float(lows.iloc[-lookback:].min()), 2)
        result["swing_high_20d"] = round(float(highs.iloc[-20:].max()), 2)
        result["swing_low_20d"]  = round(float(lows.iloc[-20:].min()), 2)

        candidates = {"SMA20": float(sma20.iloc[-1]), "SMA50": float(sma50.iloc[-1]), "SMA200": float(sma200.iloc[-1])}
        below = {k: v for k, v in candidates.items() if v < price}
        if below:
            best = max(below, key=lambda k: below[k])
            result["support_label"] = best
            result["support_val"]   = round(below[best], 2)
        else:
            result["support_label"] = "Swing Low"
            result["support_val"]   = result["swing_low_20d"]
    except Exception as e:
        result["error"] = str(e)
    return result


def build_day_card(row, session):
    ticker  = str(row.get("Ticker", ""))
    company = str(row.get("Company", ""))
    sector  = str(row.get("Sector", ""))
    industry = str(row.get("Industry", ""))
    price   = safe_float(row.get("Price", 0))
    change  = str(row.get("Change", "—"))
    relvol  = str(row.get("Rel Volume") or row.get("Relative Volume") or row.get("RelVolume") or "—").replace("x", "").strip()
    volume  = str(row.get("Volume", "—"))
    atr     = safe_float(row.get("ATR", 0)) or None
    sma50   = safe_float(row.get("SMA50", 0)) or None
    sma200  = safe_float(row.get("SMA200", 0)) or None

    tech = get_day_trade_technicals(ticker, session)
    news = get_news(ticker)

    # Extended hours price (pre-market or AH movement not shown by Finviz)
    prev_close = tech.get("key_close") or safe_float(row.get("Price", 0))
    ext = get_extended_hours_price(ticker, prev_close)

    ema5     = tech["ema5"] or price
    ema9     = tech["ema9"] or price * 0.98
    vwap     = tech["vwap_proxy"] or price
    key_high = tech["key_high"] or price
    key_low  = tech["key_low"]  or price * 0.97

    if session == "premarket":
        entry_low  = round(min(vwap, ema5), 2)
        entry_high = round(max(vwap, ema5), 2)
        level_label = "Prior day"
        entry_ref   = "VWAP"
    else:
        entry_low  = round(min(key_low, ema5), 2)
        entry_high = round(max(key_low, ema5), 2)
        level_label = "Pre-market"
        entry_ref   = "PM Low"

    stop  = round(ema9 * 0.995, 2)
    risk  = entry_low - stop
    t1    = round(entry_low + risk, 2) if risk > 0 else round(entry_low * 1.01, 2)
    t2    = round(key_high * 1.002, 2)
    rr1   = round((t1 - entry_low) / risk, 1) if risk > 0 else 0
    rr2   = round((t2 - entry_low) / risk, 1) if risk > 0 else 0
    pos   = not change.startswith("-") if change not in ("—", "") else True

    return {
        "ticker": ticker, "company": company,
        "sector": sector, "industry": industry,
        "price": price, "change": change, "relvol": relvol,
        "volume": volume, "pos": pos, "squeeze": False,
        "ema_ok": tech["ema_ok"],
        "atr": atr, "sma50": sma50, "sma200": sma200,
        "news": news,
        "levels": {
            "label": level_label,
            "key_high": key_high, "key_low": key_low,
            "prev_close": tech["key_close"], "vwap": vwap,
            "pm_high": tech.get("pm_high"), "pm_low": tech.get("pm_low"),
            "pm_open": tech.get("pm_open"), "pm_last": tech.get("pm_last"),
        },
        "plan": {
            "entry_low": entry_low, "entry_high": entry_high,
            "entry_ref": entry_ref,
            "stop": stop, "t1": t1, "t2": t2, "rr1": rr1, "rr2": rr2,
        },
        "emas": {"ema5": ema5, "ema9": ema9},
        "ext": ext,
    }


def build_swing_card(row):
    ticker   = str(row.get("Ticker", ""))
    company  = str(row.get("Company", ""))
    sector   = str(row.get("Sector", ""))
    industry = str(row.get("Industry", ""))
    price    = safe_float(row.get("Price", 0))
    change   = str(row.get("Change", "—"))
    relvol   = str(row.get("Rel Volume") or row.get("Relative Volume") or row.get("RelVolume") or "—").replace("x", "").strip()
    volume   = str(row.get("Volume", "—"))
    pos      = not change.startswith("-") if change not in ("—", "") else True

    tech = get_swing_technicals(ticker)
    if tech.get("error") or not tech.get("sma200_rising"):
        return None

    news = get_news(ticker)

    support_val   = tech["support_val"] or price * 0.95
    support_label = tech["support_label"] or "SMA50"
    entry = round(support_val * 1.005, 2)
    stop  = round(support_val * 0.97,  2)
    risk  = entry - stop
    t1    = tech["swing_high_20d"] or round(entry + risk * 2, 2)
    t2    = tech["week52_high"]    or round(entry + risk * 3, 2)
    rr1   = round((t1 - entry) / risk, 1) if risk > 0 else 0
    rr2   = round((t2 - entry) / risk, 1) if risk > 0 else 0

    return {
        "ticker": ticker, "company": company,
        "sector": sector, "industry": industry,
        "price": price, "change": change, "relvol": relvol,
        "volume": volume, "pos": pos,
        "ema_ok": tech["ema_ok"],
        "sma200_rising": tech["sma200_rising"],
        "news": news,
        "levels": {
            "week52_high": tech["week52_high"], "week52_low": tech["week52_low"],
            "swing_high_20d": tech["swing_high_20d"], "swing_low_20d": tech["swing_low_20d"],
            "sma20": tech["sma20"], "sma50": tech["sma50"], "sma200": tech["sma200"],
        },
        "plan": {
            "support_label": support_label, "entry": entry,
            "stop": stop, "t1": t1, "t2": t2, "rr1": rr1, "rr2": rr2,
        },
        "emas": {"ema5": tech["ema5"], "ema9": tech["ema9"]},
    }


def build_day_card_tv(row, session, tv=None):
    """
    Build day trade card using TradingView data for live technicals.
    Falls back to yfinance if tv is None.
    """
    ticker   = str(row.get("Ticker", ""))
    company  = str(row.get("Company", ""))
    sector   = str(row.get("Sector", ""))
    industry = str(row.get("Industry", ""))
    change   = str(row.get("Change", "—"))
    relvol   = str(row.get("Rel Volume") or row.get("Relative Volume") or "—").replace("x","").strip()
    atr      = safe_float(row.get("ATR", 0)) or None
    sma50    = safe_float(row.get("SMA50", 0)) or None
    sma200   = safe_float(row.get("SMA200", 0)) or None

    # Use TradingView data if available, else fall back to yfinance
    if tv:
        price  = tv["price"]
        volume = tv["volume"]
        ema5   = tv["ema5"]
        ema9   = tv["ema9"]
        vwap   = tv.get("vwap") or 0
        ema_ok = tv["ema_ok"]
        atr    = atr or tv.get("atr")
        sma50  = sma50 or tv.get("sma50")
        sma200 = sma200 or tv.get("sma200")
        data_source = "TradingView (live)"
    else:
        # yfinance fallback
        tech   = get_day_trade_technicals(ticker, session)
        price  = safe_float(row.get("Price", 0))
        volume = safe_float(row.get("Volume", 0))
        ema5   = tech["ema5"] or price
        ema9   = tech["ema9"] or price * 0.98
        vwap   = tech["vwap_proxy"] or 0
        ema_ok = tech["ema_ok"]
        data_source = "yfinance (delayed)"

    pos = not change.startswith("-") if change not in ("—", "") else True

    # Fetch pre-market levels (always via yfinance — TV doesn't do pre-market)
    pm = {}
    if session != "premarket":
        pm = get_premarket_levels(ticker)

    # Key levels — pull 6 months of history
    hist_6m    = yf.Ticker(ticker).history(period="6mo", interval="1d")
    prev_close = round(float(hist_6m["Close"].iloc[-2]), 2) if len(hist_6m) >= 2 else price
    prev_high  = round(float(hist_6m["High"].iloc[-2]),  2) if len(hist_6m) >= 2 else price
    prev_low   = round(float(hist_6m["Low"].iloc[-2]),   2) if len(hist_6m) >= 2 else price
    vwap_proxy = round((prev_high + prev_low + prev_close) / 3, 2)

    # 6-month range levels
    high_6m      = round(float(hist_6m["High"].max()), 2)
    low_6m       = round(float(hist_6m["Low"].min()),  2)

    # 20-day consolidation range (the "box")
    high_20d     = round(float(hist_6m["High"].iloc[-20:].max()), 2) if len(hist_6m) >= 20 else prev_high
    low_20d      = round(float(hist_6m["Low"].iloc[-20:].min()),  2) if len(hist_6m) >= 20 else prev_low

    # Key swing highs — last 3 local peaks above current price
    highs = hist_6m["High"]
    swing_highs = []
    for i in range(2, len(highs) - 2):
        if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i-2] and \
           highs.iloc[i] > highs.iloc[i+1] and highs.iloc[i] > highs.iloc[i+2]:
            h = round(float(highs.iloc[i]), 2)
            if h > price:
                swing_highs.append(h)
    swing_highs = sorted(set(swing_highs))[:3]  # nearest 3 above price

    if session == "premarket":
        key_high    = prev_high
        key_low     = prev_low
        entry_low   = round(min(vwap_proxy, ema5), 2)
        entry_high  = round(max(vwap_proxy, ema5), 2)
        level_label = "Prior day"
        entry_ref   = "EMA5 / VWAP"
        stop        = round(min(ema9, vwap_proxy) * 0.998, 2)
        stop_ref    = "below EMA9" if ema9 <= vwap_proxy else "below VWAP"
    else:
        key_high    = pm.get("pm_high") or prev_high
        key_low     = pm.get("pm_low")  or prev_low
        entry_low   = round(min(key_low, ema5), 2)
        entry_high  = round(max(key_low, ema5), 2)
        level_label = "Pre-market"
        entry_ref   = "EMA5 / PM Low"
        stop        = round(ema9 * 0.998, 2)
        stop_ref    = "below EMA9"

    risk = entry_low - stop

    # T1 = nearest swing high above entry, fallback to 1:1R
    t1_candidates = [h for h in swing_highs if h > entry_low]
    t1 = round(min(t1_candidates), 2) if t1_candidates else round(entry_low + risk, 2) if risk > 0 else round(entry_low * 1.01, 2)

    # T2 = prior day HOD breakout level
    t2 = round(key_high * 1.002, 2)

    # T3 = 6-month high ONLY if it's higher than T2
    t3 = round(high_6m * 1.002, 2) if high_6m > t2 else None

    rr1 = round((t1 - entry_low) / risk, 1) if risk > 0 else 0
    rr2 = round((t2 - entry_low) / risk, 1) if risk > 0 else 0
    rr3 = round((t3 - entry_low) / risk, 1) if t3 and risk > 0 else None

    news = get_news(ticker)
    ext  = get_extended_hours_price(ticker, prev_close)

    return {
        "ticker": ticker, "company": company,
        "sector": sector, "industry": industry,
        "price": price, "change": change, "relvol": relvol,
        "volume": volume, "pos": pos, "squeeze": False,
        "ema_ok": bool(ema_ok),
        "atr": atr, "sma50": sma50, "sma200": sma200,
        "data_source": data_source,
        "news": news, "ext": ext,
        "levels": {
            "label": level_label,
            "key_high": key_high, "key_low": key_low,
            "prev_close": prev_close, "vwap": vwap_proxy,
            "pm_high": pm.get("pm_high"), "pm_low": pm.get("pm_low"),
            "pm_open": pm.get("pm_open"), "pm_last": pm.get("pm_last"),
            "high_6m": high_6m, "low_6m": low_6m,
            "high_20d": high_20d, "low_20d": low_20d,
            "swing_highs": swing_highs,
        },
        "plan": {
            "entry_low": entry_low, "entry_high": entry_high,
            "entry_ref": entry_ref,
            "stop": stop, "stop_ref": stop_ref,
            "t1": t1, "t2": t2, "t3": t3,
            "rr1": rr1, "rr2": rr2, "rr3": rr3,
        },
        "emas": {"ema5": ema5, "ema9": ema9},
    }


def build_swing_card_tv(row, tv=None, tv_daily=None):
    """
    Build swing card using TradingView data for live technicals.
    tv = intraday TV data, tv_daily = daily TV data for SMA200 slope.
    """
    ticker   = str(row.get("Ticker", ""))
    company  = str(row.get("Company", ""))
    sector   = str(row.get("Sector", ""))
    industry = str(row.get("Industry", ""))
    change   = str(row.get("Change", "—"))
    # Use yfinance-calculated rel volume, not Finviz's broken column
    relvol   = str(row.get("Rel Volume") or "—").replace("x","").strip()
    volume   = row.get("yf_volume") or row.get("Volume") or 0
    pos      = not change.startswith("-") if change not in ("—","") else True

    # Get full swing technicals from yfinance (52wk, swing highs/lows need history)
    tech = get_swing_technicals(ticker)
    if tech.get("error"):
        return None

    # Override EMAs and SMAs with TradingView if available
    if tv_daily:
        ema5   = tv_daily.get("ema5")   or tech["ema5"]
        ema9   = tv_daily.get("ema9")   or tech["ema9"]
        sma20  = tv_daily.get("sma20")  or tech["sma20"]
        sma50  = tv_daily.get("sma50")  or tech["sma50"]
        sma200 = tv_daily.get("sma200") or tech["sma200"]
        price  = tv_daily.get("price")  or tech["current_price"]
        ema_ok = bool(tv_daily.get("ema_ok", tech["ema_ok"]))
        data_source = "TradingView (live)"
    else:
        ema5   = tech["ema5"]
        ema9   = tech["ema9"]
        sma20  = tech["sma20"]
        sma50  = tech["sma50"]
        sma200 = tech["sma200"]
        price  = tech["current_price"]
        ema_ok = tech["ema_ok"]
        data_source = "yfinance (delayed)"

    # SMA200 rising check
    sma200_rising = tech["sma200_rising"]
    if not sma200_rising:
        return None  # skip non-rising 200MA

    support_val   = tech["support_val"] or price * 0.95
    support_label = tech["support_label"] or "SMA50"
    entry = round(support_val * 1.005, 2)
    stop  = round(support_val * 0.97,  2)
    risk  = entry - stop
    t1    = tech["swing_high_20d"] or round(entry + risk * 2, 2)
    t2    = tech["week52_high"]    or round(entry + risk * 3, 2)
    rr1   = round((t1 - entry) / risk, 1) if risk > 0 else 0
    rr2   = round((t2 - entry) / risk, 1) if risk > 0 else 0

    news = get_news(ticker)

    return {
        "ticker": ticker, "company": company,
        "sector": sector, "industry": industry,
        "price": price, "change": change, "relvol": relvol,
        "volume": volume, "pos": pos,
        "ema_ok": bool(ema_ok),
        "sma200_rising": bool(sma200_rising),
        "data_source": data_source,
        "news": news,
        "levels": {
            "week52_high": tech["week52_high"], "week52_low": tech["week52_low"],
            "swing_high_20d": tech["swing_high_20d"], "swing_low_20d": tech["swing_low_20d"],
            "sma20": sma20, "sma50": sma50, "sma200": sma200,
        },
        "plan": {
            "support_label": support_label, "entry": entry,
            "stop": stop, "t1": t1, "t2": t2, "rr1": rr1, "rr2": rr2,
        },
        "emas": {"ema5": ema5, "ema9": ema9},
    }


# ══════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")



# Caches Finviz results for 10 minutes to avoid redundant fetches
import time as _time
_cache = {}
_CACHE_TTL = 600  # 10 minutes

def cached_fetch(filters, label):
    key = label
    now = _time.time()
    if key in _cache and (now - _cache[key]["ts"]) < _CACHE_TTL:
        print(f"  → {label}: using cached results ({len(_cache[key]['df'])} tickers)")
        return _cache[key]["df"]
    df = fetch_screener(filters, label)
    _cache[key] = {"df": df, "ts": now}
    return df




def build_manual_card(ticker, mode, session):
    """
    Build a full analysis card for a manually entered ticker.
    mode: 'day' or 'swing'
    Pulls company info, price, change, volume from yfinance directly.
    """
    ticker = ticker.upper().strip()
    try:
        t    = yf.Ticker(ticker)
        info = t.info or {}
        hist = t.history(period="5d", interval="1d")
        if hist.empty:
            return {"ticker": ticker, "error": f"No data found for {ticker}"}

        price      = round(float(hist["Close"].iloc[-1]), 2)
        prev_close = round(float(hist["Close"].iloc[-2]), 2) if len(hist) >= 2 else price
        change_val = round(price - prev_close, 2)
        change_pct = round((change_val / prev_close) * 100, 2) if prev_close else 0
        change_str = ('+' if change_val >= 0 else '') + f"{change_pct:.2f}%"
        volume     = int(hist["Volume"].iloc[-1])
        company    = info.get("longName") or info.get("shortName") or ticker
        sector     = info.get("sector", "")
        industry   = info.get("industry", "")
        pos        = change_val >= 0

        # avg volume for rel vol calc
        avg_vol  = int(hist["Volume"].mean()) if len(hist) > 1 else volume
        relvol   = f"{round(volume / avg_vol, 1)}" if avg_vol > 0 else "—"

        news = get_news(ticker)
        ext  = get_extended_hours_price(ticker, prev_close)

        if mode == "swing":
            tech = get_swing_technicals(ticker)
            if tech.get("error"):
                return {"ticker": ticker, "error": tech["error"]}

            support_val   = tech["support_val"] or price * 0.95
            support_label = tech["support_label"] or "SMA50"
            entry = round(support_val * 1.005, 2)
            stop  = round(support_val * 0.97,  2)
            risk  = entry - stop
            t1    = tech["swing_high_20d"] or round(entry + risk * 2, 2)
            t2    = tech["week52_high"]    or round(entry + risk * 3, 2)
            rr1   = round((t1 - entry) / risk, 1) if risk > 0 else 0
            rr2   = round((t2 - entry) / risk, 1) if risk > 0 else 0

            return {
                "ticker": ticker, "company": company,
                "sector": sector, "industry": industry,
                "price": price, "change": change_str,
                "relvol": relvol, "volume": volume, "pos": pos,
                "mode": "swing",
                "ema_ok": tech["ema_ok"],
                "sma200_rising": bool(tech["sma200_rising"]),
                "news": news, "ext": ext,
                "levels": {
                    "week52_high": tech["week52_high"], "week52_low": tech["week52_low"],
                    "swing_high_20d": tech["swing_high_20d"], "swing_low_20d": tech["swing_low_20d"],
                    "sma20": tech["sma20"], "sma50": tech["sma50"], "sma200": tech["sma200"],
                },
                "plan": {
                    "support_label": support_label, "entry": entry,
                    "stop": stop, "t1": t1, "t2": t2, "rr1": rr1, "rr2": rr2,
                },
                "emas": {"ema5": tech["ema5"], "ema9": tech["ema9"]},
            }

        else:  # day trade
            tech      = get_day_trade_technicals(ticker, session)
            ema5      = tech["ema5"] or price
            ema9      = tech["ema9"] or price * 0.98
            vwap      = tech["vwap_proxy"] or price
            key_high  = tech["key_high"] or price
            key_low   = tech["key_low"]  or price * 0.97

            if session == "premarket":
                entry_low   = round(min(vwap, ema5), 2)
                entry_high  = round(max(vwap, ema5), 2)
                level_label = "Prior day"
                entry_ref   = "VWAP"
            else:
                entry_low   = round(min(key_low, ema5), 2)
                entry_high  = round(max(key_low, ema5), 2)
                level_label = "Pre-market"
                entry_ref   = "PM Low"

            stop  = round(ema9 * 0.995, 2)
            risk  = entry_low - stop
            t1    = round(entry_low + risk, 2) if risk > 0 else round(entry_low * 1.01, 2)
            t2    = round(key_high * 1.002, 2)
            rr1   = round((t1 - entry_low) / risk, 1) if risk > 0 else 0
            rr2   = round((t2 - entry_low) / risk, 1) if risk > 0 else 0

            return {
                "ticker": ticker, "company": company,
                "sector": sector, "industry": industry,
                "price": price, "change": change_str,
                "relvol": relvol, "volume": volume, "pos": pos,
                "mode": "day", "squeeze": False,
                "ema_ok": tech["ema_ok"],
                "news": news, "ext": ext,
                "levels": {
                    "label": level_label,
                    "key_high": key_high, "key_low": key_low,
                    "prev_close": tech["key_close"], "vwap": vwap,
                    "pm_high": tech.get("pm_high"), "pm_low": tech.get("pm_low"),
                    "pm_open": tech.get("pm_open"), "pm_last": tech.get("pm_last"),
                },
                "plan": {
                    "entry_low": entry_low, "entry_high": entry_high,
                    "entry_ref": entry_ref,
                    "stop": stop, "t1": t1, "t2": t2, "rr1": rr1, "rr2": rr2,
                },
                "emas": {"ema5": ema5, "ema9": ema9},
            }

    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


def cloud_blocked():
    return jsonify({
        "cloud":   True,
        "message": "The main screener uses Finviz and only works when running locally on your PC. Use the 🔍 Manual Search tab to analyze specific tickers here.",
        "session": get_session(),
        "time":    datetime.now(ET).strftime("%I:%M:%S %p ET"),
        "cards": [], "sectors": [],
    })


def build_day_sq_response(filters, label, session, is_squeeze=False, exclude_tickers=None, raw_df=None):
    """Shared logic for day trades and squeezes."""
    raw = apply_volume_gate(raw_df if raw_df is not None else fetch_screener(filters, label))
    if raw.empty:
        return [], []

    exclude = set(exclude_tickers or [])
    tickers = [t for t in raw["Ticker"].tolist() if t not in exclude]

    tv_interval = "1d" if session == "premarket" else "1m"
    print(f"  Fetching TradingView data for {len(tickers)} {label} tickers ({tv_interval})...")
    tv_data = get_tv_technicals_batch(tickers, interval=tv_interval)

    cards = []
    for _, row in raw.iterrows():
        try:
            ticker = str(row.get("Ticker", ""))
            if ticker in exclude:
                continue
            tv   = tv_data.get(ticker)
            card = build_day_card_tv(row, session, tv)
            if card:
                if is_squeeze:
                    card["squeeze"]     = True
                    card["short_float"] = str(row.get("Short Float", "—"))
                    card["float_size"]  = str(row.get("Float", "—"))
                cards.append(card)
        except Exception as e:
            print(f"  !! {label} card error {row.get('Ticker')}: {e}")

    sectors = sorted(set(c["sector"] for c in cards if c.get("sector")))
    return cards, sectors


@app.route("/api/run/day")
def run_day_trades():
    if IS_CLOUD: return cloud_blocked()
    session = get_session()
    now_et  = datetime.now(ET).strftime("%I:%M:%S %p ET")
    cards, sectors = build_day_sq_response(
        DAY_TRADE_FILTERS, "Day Trades", session,
        raw_df=cached_fetch(DAY_TRADE_FILTERS, "Day Trades")
    )
    return jsonify({"cloud": False, "session": session, "time": now_et,
                    "cards": cards, "sectors": sectors})


@app.route("/api/run/squeeze")
def run_squeezes():
    if IS_CLOUD: return cloud_blocked()
    session = get_session()
    now_et  = datetime.now(ET).strftime("%I:%M:%S %p ET")
    # Reuse cached day trades for exclusion — no extra Finviz call
    day_raw     = cached_fetch(DAY_TRADE_FILTERS, "Day Trades")
    day_tickers = day_raw["Ticker"].tolist() if not day_raw.empty else []
    cards, sectors = build_day_sq_response(
        SQUEEZE_FILTERS, "Squeezes", session,
        is_squeeze=True, exclude_tickers=day_tickers,
        raw_df=cached_fetch(SQUEEZE_FILTERS, "Squeezes")
    )
    return jsonify({"cloud": False, "session": session, "time": now_et,
                    "cards": cards, "sectors": sectors})


@app.route("/api/run/swing")
def run_swings():
    if IS_CLOUD: return cloud_blocked()
    session = get_session()
    now_et  = datetime.now(ET).strftime("%I:%M:%S %p ET")
    raw = apply_volume_gate(cached_fetch(SWING_FILTERS, "Swings"))
    if raw.empty:
        return jsonify({"cloud": False, "session": session, "time": now_et,
                        "cards": [], "sectors": []})
    tickers = raw["Ticker"].tolist()
    print(f"  Fetching TradingView daily data for {len(tickers)} swing tickers...")
    tv_daily = get_tv_technicals_batch(tickers, interval="1d")
    cards = []
    for _, row in raw.iterrows():
        try:
            ticker = str(row.get("Ticker", ""))
            tv_day = tv_daily.get(ticker)
            card   = build_swing_card_tv(row, None, tv_day)
            if card:
                cards.append(card)
        except Exception as e:
            print(f"Swing card error {row.get('Ticker')}: {e}")
    sectors = sorted(set(c["sector"] for c in cards if c.get("sector")))
    return jsonify({"cloud": False, "session": session, "time": now_et,
                    "cards": cards, "sectors": sectors})


@app.route("/api/run")
def run_screener():
    """Run All — fetches all three screeners in parallel."""
    if IS_CLOUD: return cloud_blocked()
    session = get_session()
    now_et  = datetime.now(ET).strftime("%I:%M:%S %p ET")

    import concurrent.futures as cf

    def fetch_day():
        raw = cached_fetch(DAY_TRADE_FILTERS, "Day Trades")
        cards, sectors = build_day_sq_response(
            DAY_TRADE_FILTERS, "Day Trades", session,
            raw_df=apply_volume_gate(raw)
        )
        return cards, sectors

    def fetch_squeeze():
        day_raw     = cached_fetch(DAY_TRADE_FILTERS, "Day Trades")
        day_tickers = day_raw["Ticker"].tolist() if not day_raw.empty else []
        cards, sectors = build_day_sq_response(
            SQUEEZE_FILTERS, "Squeezes", session,
            is_squeeze=True, exclude_tickers=day_tickers,
            raw_df=apply_volume_gate(cached_fetch(SQUEEZE_FILTERS, "Squeezes"))
        )
        return cards, sectors

    def fetch_swing():
        raw = apply_volume_gate(cached_fetch(SWING_FILTERS, "Swings"))
        if raw.empty:
            return [], []
        tickers = raw["Ticker"].tolist()
        print(f"  Fetching TradingView daily data for {len(tickers)} swing tickers...")
        tv_daily = get_tv_technicals_batch(tickers, interval="1d")
        cards = []
        for _, row in raw.iterrows():
            try:
                card = build_swing_card_tv(row, None, tv_daily.get(str(row.get("Ticker", ""))))
                if card:
                    cards.append(card)
            except Exception as e:
                print(f"  Swing card error {row.get('Ticker')}: {e}")
        sectors = sorted(set(c["sector"] for c in cards if c.get("sector")))
        return cards, sectors

    print("  Running all screeners in parallel...")
    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        f_day   = ex.submit(fetch_day)
        f_sq    = ex.submit(fetch_squeeze)
        f_swing = ex.submit(fetch_swing)
        day_cards,   day_sectors   = f_day.result()
        sq_cards,    sq_sectors    = f_sq.result()
        swing_cards, swing_sectors = f_swing.result()

    all_sectors = sorted(set(day_sectors + sq_sectors + swing_sectors))

    return jsonify({
        "cloud":   False,
        "session": session,
        "time":    now_et,
        "day":     day_cards,
        "squeeze": sq_cards,
        "swing":   swing_cards,
        "sectors": all_sectors,
    })


@app.route("/api/cache/clear")
def clear_cache():
    _cache.clear()
    return jsonify({"cleared": True, "message": "Cache cleared — next run will fetch fresh data"})


@app.route("/api/environment")
def environment():
    """Let the frontend know if it's running on cloud or locally."""
    return jsonify({"cloud": IS_CLOUD})



@app.route("/api/analyze", methods=["POST"])
def analyze_tickers():
    """Analyze a list of manually entered tickers."""
    from flask import request
    data    = request.get_json()
    tickers = [t.strip().upper() for t in data.get("tickers", []) if t.strip()]
    mode    = data.get("mode", "day")   # 'day' or 'swing'
    session = get_session()

    cards = []
    for ticker in tickers:
        try:
            card = build_manual_card(ticker, mode, session)
            cards.append(card)
        except Exception as e:
            cards.append({"ticker": ticker, "error": str(e)})

    return jsonify({"cards": cards, "session": session})


if __name__ == "__main__":
    print("\n" + "="*52)
    print("  Trading Screener is running!")
    print("  Open your browser to: http://localhost:5000")
    print("="*52 + "\n")
    app.run(debug=False, port=5000)
