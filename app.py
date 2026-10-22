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
    "ta_sma200_pa",     # Price above 200 SMA
    "ta_sma50_pa",      # Price above 50 SMA
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


FINVIZ_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def finviz_direct(filters, label):
    """Read Finviz screener pages directly. Tries table parse, then ticker links,
    and saves finviz_debug.html if nothing can be read."""
    import requests, time, re
    from io import StringIO
    base = "https://finviz.com/screener.ashx"
    sess = requests.Session()
    sess.headers.update(FINVIZ_HEADERS)
    cols = ["No.", "Ticker", "Company", "Sector", "Industry", "Country",
            "Market Cap", "P/E", "Price", "Change", "Volume"]
    rows, seen = [], set()

    colmap = {"row": "No.", "ticker": "Ticker", "company": "Company", "sector": "Sector",
              "industry": "Industry", "country": "Country", "marketCap": "Market Cap",
              "PE": "P/E", "price": "Price", "change": "Change", "volume": "Volume"}

    def parse_grid(html):
        """Finviz's 2025+ div-grid layout. Returns (records, total_count)."""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            return [], None
        total = None
        tc = soup.select_one(".fv-table_toolbar-count")
        if tc:
            m = re.search(r"/\s*([\d,]+)\s*Total", tc.get_text())
            if m:
                total = int(m.group(1).replace(",", ""))
        recs = []
        for r in soup.select("div.fv-table_row"):
            tk = (r.get("data-row-key") or "").strip().upper()
            if not tk:
                continue
            rec = {"Ticker": tk}
            for c in r.select("[data-column-id]"):
                name = colmap.get(c.get("data-column-id"))
                if name and name != "Ticker":
                    rec[name] = c.get_text(strip=True)
            recs.append(rec)
        return recs, total

    def parse_page(html):
        # Strategy 1: real HTML table with headers
        try:
            for t in pd.read_html(StringIO(html), keep_default_na=False):
                if {"Ticker", "Company", "Price"} <= set(map(str, t.columns)):
                    t = t.astype(str)
                    return t.to_dict("records")
        except Exception:
            pass
        # Strategy 2: ticker links (row cells if available, else ticker only)
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            soup = None
        out = []
        if soup is not None:
            links = soup.select('a.tab-link[href*="quote.ashx?t="]') or \
                    soup.select('a[href*="quote.ashx?t="]')
            for a in links:
                m = re.search(r"quote\.ashx\?t=([A-Za-z0-9.\-]+)", a.get("href", ""))
                if not m:
                    continue
                tk = m.group(1).upper()
                rec = {"Ticker": tk}
                tr = a.find_parent("tr")
                if tr is not None:
                    cells = [c.get_text(strip=True) for c in tr.find_all("td")]
                    if tk in cells:
                        i = cells.index(tk)
                        for j, name in enumerate(cols[1:], start=0):
                            if i + j < len(cells):
                                rec[name] = cells[i + j]
                out.append(rec)
        else:
            for tk in re.findall(r"quote\.ashx\?t=([A-Za-z0-9.\-]+)", html):
                out.append({"Ticker": tk.upper()})
        return out

    next_start, total = 1, None
    for page in range(80):
        url = f"{base}?v=111&f={','.join(filters)}&o=-volume&r={next_start}"
        resp = sess.get(url, timeout=20)
        if resp.status_code != 200:
            print(f"  !! Finviz HTTP {resp.status_code} for {label} (page {page+1})")
            break
        recs, pg_total = parse_grid(resp.text)
        if pg_total is not None:
            total = pg_total
        if not recs:
            if total == 0 or (page == 0 and pg_total == 0):
                print(f"  → {label}: Finviz says 0 matches right now")
                break
            recs = parse_page(resp.text)
        new = [r for r in recs if r.get("Ticker") and r["Ticker"] not in seen]
        if not new:
            if page == 0:
                try:
                    with open("finviz_debug.html", "w", encoding="utf-8") as f:
                        f.write(resp.text)
                except Exception:
                    pass
                print(f"  !! Could not read tickers from Finviz page for {label} (total={total}). Saved finviz_debug.html")
            break
        seen.update(r["Ticker"] for r in new)
        rows.extend(new)
        # advance by absolute row index when available (robust to virtualised grids)
        idxs = [int(str(r.get("No.", "")).replace(",", "")) for r in new if str(r.get("No.", "")).replace(",", "").isdigit()]
        next_start = (max(idxs) + 1) if idxs else (next_start + len(new))
        if total is not None and next_start > total:
            break
        time.sleep(0.4)
    return pd.DataFrame(rows).astype(str) if rows else pd.DataFrame()


def fetch_screener(filters, label):
    if not HAS_FINVIZ:
        return pd.DataFrame()
    try:
        print(f"  Fetching {label} from Finviz...")
        raw_df = finviz_direct(filters, label)
        if raw_df.empty:
            print(f"  Direct scrape empty for {label}; trying finviz library...")
            try:
                data_ov = list(Screener(filters=filters, table="Overview", order="-volume"))
                raw_df = pd.DataFrame(data_ov) if data_ov else pd.DataFrame()
            except Exception as e:
                print(f"  Library also failed: {e}")
        if raw_df.empty:
            print(f"  → {label}: no results")
            return pd.DataFrame()
        print(f"  Raw columns: {list(raw_df.columns)} | first row: {raw_df.iloc[0].to_dict()}")

        import re
        tick_re = re.compile(r'^[A-Z]{1,5}(\.[A-Z])?$')

        def tick_score(col):
            if col not in raw_df.columns:
                return 0.0
            return raw_df[col].astype(str).apply(lambda v: bool(tick_re.match(v))).mean()

        # Finviz 2.0.0 sometimes shifts values one column right.
        # Detect by which column actually holds ticker-looking values.
        if tick_score("Company") > tick_score("Ticker"):
            print("  Detected shifted Finviz columns - remapping")
            raw_df = raw_df.rename(columns={
                "Ticker": "_drop", "Company": "Ticker", "Sector": "Company",
                "Industry": "Sector", "Country": "Industry", "Market Cap": "Country",
                "P/E": "Market Cap", "Price": "P/E", "Change": "Price", "Volume": "Change",
            })
            raw_df["Volume"] = None

        df_ov = normalize_columns(raw_df.drop(columns=["No.", "_drop"], errors="ignore"))
        df_ov = df_ov[df_ov["Ticker"].astype(str).apply(lambda v: bool(tick_re.match(v)))].copy()
        print(f"  → {label}: {len(df_ov)} tickers")
        if df_ov.empty:
            print("  !! Finviz returned rows but none had valid tickers - column layout may have changed")
            return df_ov

        print(f"  Calculating rel volume for {len(df_ov)} tickers...")

        def get_vol_data(ticker):
            try:
                hist = yf.Ticker(ticker).history(period="1mo", interval="1d")
                if hist.empty or len(hist) < 5:
                    return None, None, None
                avg_vol = float(hist["Volume"].iloc[:-1].mean())
                cur_vol = float(hist["Volume"].iloc[-1])
                relvol  = round(cur_vol / avg_vol, 1) if avg_vol > 0 else None
                last = float(hist["Close"].iloc[-1]); prev = float(hist["Close"].iloc[-2])
                price_chg = (round(last, 2), f"{(last/prev-1)*100:+.2f}%")
                return relvol, int(cur_vol), price_chg
            except Exception:
                return None, None, None

        import concurrent.futures
        tickers = df_ov["Ticker"].tolist()
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
            vol_results = list(ex.map(get_vol_data, tickers))

        df_ov["Rel Volume"] = [r[0] for r in vol_results]
        df_ov["yf_volume"]  = [r[1] for r in vol_results]
        # If Finviz gave only tickers, fill Price/Change/Company/Sector from Yahoo data
        for col, default in (("Company", None), ("Sector", "—"), ("Industry", "—"), ("Country", "—"), ("Market Cap", "—"), ("P/E", "—")):
            if col not in df_ov.columns:
                df_ov[col] = df_ov["Ticker"] if default is None else default
        if "Price" not in df_ov.columns or df_ov["Price"].isna().all():
            df_ov["Price"]  = [str(r[2][0]) if r[2] else "0" for r in vol_results]
        if "Change" not in df_ov.columns or df_ov["Change"].isna().all():
            df_ov["Change"] = [r[2][1] if r[2] else "0%" for r in vol_results]
        n_ok = sum(1 for r in vol_results if r[1] is not None)
        print(f"  → {n_ok}/{len(df_ov)} volume lookups succeeded")

        # Volume gate: only drop tickers KNOWN to be under the minimum.
        # If Yahoo failed/rate-limited for a ticker, keep it (Finviz already
        # filtered on current volume) instead of wiping out the whole list.
        before = len(df_ov)
        df_ov = df_ov[df_ov["yf_volume"].apply(lambda v: v is None or pd.isna(v) or v >= MIN_VOLUME)].copy()
        print(f"  → {before} → {len(df_ov)} after {MIN_VOLUME:,} volume gate")
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
    """Recent headlines via yfinance (handles old and new Yahoo response shapes).
    Each item: title, url, source, time (ET text), age_hours (None if unknown). Newest first."""
    news = []
    try:
        items = yf.Ticker(ticker).news or []
    except Exception as e:
        print(f"  news fetch failed for {ticker}: {type(e).__name__}")
        return news
    now_utc = datetime.now(pytz.utc)
    for item in items[:10]:
        try:
            content = item.get("content", item)
            title   = content.get("title") or item.get("title", "")
            url = ""
            for key in ("clickThroughUrl", "canonicalUrl"):
                obj = content.get(key)
                if isinstance(obj, dict) and obj.get("url"):
                    url = obj["url"]; break
            if not url:
                url = item.get("link", "") or item.get("url", "")
            provider = content.get("provider") or {}
            source = provider.get("displayName", "") if isinstance(provider, dict) else ""
            if not source:
                source = item.get("publisher", "") or item.get("source", "") or "Yahoo Finance"

            dt = None
            pub = content.get("pubDate") or content.get("displayTime") or item.get("providerPublishTime")
            if isinstance(pub, (int, float)) and pub > 0:
                dt = datetime.fromtimestamp(pub, tz=pytz.utc)
            elif isinstance(pub, str) and pub:
                try:
                    dt = datetime.fromisoformat(pub.replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = pytz.utc.localize(dt)
                except Exception:
                    dt = None
            ts, age = "", None
            if dt:
                ts  = dt.astimezone(ET).strftime("%b %d  %I:%M %p ET")
                age = round((now_utc - dt).total_seconds() / 3600.0, 1)
            if title:
                news.append({"title": title, "url": url, "source": source, "time": ts, "age_hours": age})
        except Exception:
            continue
    news.sort(key=lambda n: (n["age_hours"] is None, n["age_hours"] if n["age_hours"] is not None else 0))
    return news[:6]


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
            def _opt(key, nd=2):
                v = ind.get(key)
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    return None
                return round(v, nd) if v > 0 else None
            results[ticker] = {
                "price":    round(float(ind.get("close", 0)), 2),
                "open":     round(float(ind.get("open", 0)), 2),
                "high":     round(float(ind.get("high", 0)), 2),
                "low":      round(float(ind.get("low", 0)), 2),
                "volume":   int(ind.get("volume", 0)),
                # NOTE: TradingView's scanner list has EMA5/EMA10/EMA20 but NO EMA9, VWAP or ATR.
                # Leave those None so callers compute real values instead of a fake one.
                "ema5":     _opt("EMA5"),
                "ema9":     _opt("EMA9"),
                "sma20":    round(float(ind.get("SMA20", 0)), 2),
                "sma50":    round(float(ind.get("SMA50", 0)), 2),
                "sma200":   round(float(ind.get("SMA200", 0)), 2),
                "vwap":     _opt("VWAP"),
                "rsi":      _opt("RSI", 1),
                "atr":      _opt("ATR"),
                "change":   round(float(ind.get("change") or 0), 2),
                "ema_ok":   None,
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



# ══════════════════════════════════════════════════════════════
#  INTRADAY CONTEXT, SUPPORT/FLIP LEVELS, STATUS, RANGE
# ══════════════════════════════════════════════════════════════

def get_intraday_context(ticker):
    """Real intraday VWAP, HOD/LOD and hold-tracking from yfinance 1-min bars."""
    ctx = {"ok": False}
    try:
        df = yf.Ticker(ticker).history(period="2d", interval="1m", prepost=True)
        if df is None or df.empty:
            return ctx
        df.index = df.index.tz_convert(ET)
        last_date = df.index[-1].date()
        day = df[df.index.date == last_date]
        reg = day.between_time("09:30", "15:59")
        pmk = day.between_time("04:00", "09:29")
        base = reg if len(reg) >= 1 else pmk
        if len(base) < 3:
            return ctx
        tp  = (base["High"] + base["Low"] + base["Close"]) / 3
        vol = base["Volume"].astype(float)
        if vol.sum() <= 0:
            return ctx
        cum_vwap = (tp * vol).cumsum() / vol.cumsum()
        last_ts  = day.index[-1]
        age_min  = (datetime.now(ET) - last_ts).total_seconds() / 60.0
        ctx.update({
            "ok": True,
            "vwap": round(float(cum_vwap.iloc[-1]), 4),
            "vwap_src": "live" + (" (regular session)" if len(reg) >= 1 else " (pre-market)"),
            "hod": round(float(base["High"].max()), 4),
            "lod": round(float(base["Low"].min()), 4),
            "last": round(float(day["Close"].iloc[-1]), 4),
            "last_time": last_ts.strftime("%I:%M %p"),
            "age_min": round(age_min, 1),
            "ema5": round(float(df["Close"].ewm(span=EMA_SHORT, adjust=False).mean().iloc[-1]), 4),
            "ema9": round(float(df["Close"].ewm(span=EMA_LONG, adjust=False).mean().iloc[-1]), 4),
            "bars_above_vwap_last10": int((base["Close"].iloc[-10:] > cum_vwap.iloc[-10:]).sum()),
            "_closes": [float(x) for x in day["Close"].values],
            "_lows":   [float(x) for x in day["Low"].values],
        })
    except Exception:
        pass
    return ctx


def level_hold(ctx, level):
    """How many consecutive trailing 1-min closes are above `level`, and whether it was retested."""
    if not ctx.get("ok") or not level:
        return {"held_min": None, "retested": False}
    closes, lows = ctx["_closes"], ctx["_lows"]
    n = 0
    for c in reversed(closes):
        if c > level:
            n += 1
        else:
            break
    recent_lows = lows[-45:]
    touched = any(level * 0.995 <= lo <= level * 1.006 for lo in recent_lows)
    return {"held_min": n, "retested": bool(touched and n >= 2 and closes[-1] > level)}


def cluster_levels(levels, pct=0.03, descending=False):
    """Collapse levels within `pct` of each other, keeping the first seen in sorted order."""
    out = []
    for lv in sorted(set(levels), reverse=descending):
        if not out or abs(lv - out[-1]) / max(out[-1], 1e-9) > pct:
            out.append(lv)
    return out


def compute_atr14(hist):
    try:
        h, l, c = hist["High"], hist["Low"], hist["Close"]
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        v = float(tr.iloc[-14:].mean())
        return round(v, 4) if v > 0 else None
    except Exception:
        return None


def build_support_levels(price, hist_6m, ctx, vwap_live, pm_high, prev_high):
    """Levels BELOW price: flipped resistance (broken swing highs), swing lows, VWAP, PM/prior-day high."""
    cands = []   # (level, kind, priority)
    highs, lows = hist_6m["High"], hist_6m["Low"]
    ph, pl = [], []
    for i in range(2, len(highs) - 2):
        if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i-2] and \
           highs.iloc[i] > highs.iloc[i+1] and highs.iloc[i] > highs.iloc[i+2]:
            ph.append(round(float(highs.iloc[i]), 2))
        if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i-2] and \
           lows.iloc[i] < lows.iloc[i+1] and lows.iloc[i] < lows.iloc[i+2]:
            pl.append(round(float(lows.iloc[i]), 2))
    floor = price * 0.75
    for h in cluster_levels([h for h in ph if floor <= h < price * 0.998], descending=True):
        cands.append((h, "Flipped resistance (old swing high)", 0))
    if pm_high and floor <= pm_high < price * 0.998:
        cands.append((pm_high, "Flipped resistance (pre-market high)", 0))
    if prev_high and floor <= prev_high < price * 0.998:
        cands.append((prev_high, "Flipped resistance (prior-day high)", 0))
    if vwap_live and floor <= vwap_live < price * 0.998:
        cands.append((vwap_live, "VWAP", 1))
    for l in cluster_levels([l for l in pl if floor <= l < price * 0.998], descending=True):
        cands.append((l, "Demand (swing low)", 2))
    cands.sort(key=lambda t: (-t[0], t[2]))
    out = []
    for lv, kind, _p in cands:
        if any(abs(lv - o["level"]) / o["level"] < 0.01 for o in out):
            continue
        hold = level_hold(ctx, lv)
        out.append({
            "level": round(lv, 2), "kind": kind,
            "dist_pct": round((price - lv) / price * 100, 1),
            "held_min": hold["held_min"], "retested": hold["retested"],
            "flipped": kind.startswith("Flipped"),
        })
        if len(out) >= 4:
            break
    return out


def determine_status(price, vwap, vwap_live_ok, ema5, entry_low, entry_high, resistance,
                     supports, atr, age_min):
    """Single live read on where price sits versus the playbook levels."""
    if age_min and age_min > 120:
        stale = f" · last bar {age_min/60:.0f}h ago (market closed?)"
    elif age_min and age_min > 5:
        stale = f" · data ~{age_min:.0f}m old"
    else:
        stale = ""
    if vwap_live_ok and vwap and price < vwap * 0.998:
        return {"code": "below_vwap", "label": "🔻 Below VWAP", "tone": "red",
                "detail": f"Price {price:.2f} under VWAP {vwap:.2f} — demand lost intraday; no long until reclaimed{stale}"}
    if resistance:
        gap = (resistance - price) / price * 100
        if gap <= 1.5:
            return {"code": "testing", "label": "👀 Testing resistance", "tone": "amber",
                    "detail": f"{gap:.1f}% under {resistance:.2f} — needs a clean break and hold{stale}"}
        return {"code": "below_res", "label": "⏳ Below resistance", "tone": "amber",
                "detail": f"{gap:.1f}% under {resistance:.2f} — wait for break and hold{stale}"}
    flip = next((s for s in supports if s["flipped"]), None)
    if flip and flip["retested"] and (flip["held_min"] or 0) >= 2:
        return {"code": "retest_hold", "label": "✅ Retest holding", "tone": "green",
                "detail": f"{flip['kind']} {flip['level']:.2f} retested and held {flip['held_min']}m{stale}"}
    if entry_low and entry_high and entry_low * 0.995 <= price <= entry_high * 1.005:
        return {"code": "in_zone", "label": "🎯 In entry zone", "tone": "green",
                "detail": f"{entry_low:.2f} – {entry_high:.2f} (EMA5 / VWAP){stale}"}
    if entry_high and price > entry_high:
        ext = price - entry_high
        extended = (ext / atr > 1.0) if atr else ((price / entry_high - 1) > 0.04)
        if extended:
            return {"code": "extended", "label": "🚀 Extended", "tone": "red",
                    "detail": f"{(price/entry_high-1)*100:.1f}% above entry zone — wait for pullback to {entry_high:.2f}{stale}"}
        return {"code": "above_zone", "label": "↗ Above zone", "tone": "blue",
                "detail": f"Watching for pullback to {entry_high:.2f}{stale}"}
    return {"code": "below_zone", "label": "🔻 Below zone", "tone": "red",
            "detail": f"Under entry zone {entry_low}{stale}"}


def compute_range(price, atr, hod, lod, resistance, t1, t2):
    """How much room is left versus the stock's typical daily range."""
    out = {"atr": atr, "used_pct": None, "room_t1_pct": None, "room_t1_atr": None,
           "room_t2_pct": None, "verdict": "unknown", "note": ""}
    if hod and lod and atr:
        out["used_pct"] = round((hod - lod) / atr * 100)
    tgt = t1 if (t1 and t1 > price * 1.001) else (t2 if (t2 and t2 > price * 1.001) else None)
    if t2 and t2 > price * 1.001:
        out["room_t2_pct"] = round((t2 / price - 1) * 100, 1)
    if not tgt:
        out["verdict"] = "open"
        out["note"] = "No known overhead target — price above mapped levels"
        return out
    out["room_t1_pct"] = round((tgt / price - 1) * 100, 1)
    if atr:
        out["room_t1_atr"] = round((tgt - price) / atr, 2)
        used = out["used_pct"] if out["used_pct"] is not None else 0
        if out["room_t1_atr"] >= 1.0 and used < 100:
            out["verdict"] = "plenty"
        elif out["room_t1_atr"] >= 0.5 and used < 150:
            out["verdict"] = "limited"
        else:
            out["verdict"] = "tight"
        out["note"] = f"{out['room_t1_pct']}% to T1 = {out['room_t1_atr']}× ATR; day range used {used}% of ATR"
    return out


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

    # Live intraday context (true VWAP, HOD/LOD, hold tracking)
    ctx = get_intraday_context(ticker)
    vwap_live_ok = bool(ctx.get("ok"))
    vwap_val = round(ctx["vwap"], 2) if vwap_live_ok else vwap_proxy
    vwap_src = ctx["vwap_src"] if vwap_live_ok else "proxy (prior-day pivot)"
    atr = atr or compute_atr14(hist_6m)

    # Real EMA5/EMA9 — 1-minute bars during/after the session, daily bars in pre-market
    if session != "premarket" and vwap_live_ok and ctx.get("ema5") and ctx.get("ema9"):
        ema5, ema9, ema_src = round(ctx["ema5"], 2), round(ctx["ema9"], 2), "1-min"
    else:
        _c = hist_6m["Close"]
        ema5 = round(float(_c.ewm(span=EMA_SHORT, adjust=False).mean().iloc[-1]), 2)
        ema9 = round(float(_c.ewm(span=EMA_LONG,  adjust=False).mean().iloc[-1]), 2)
        ema_src = "daily"
    ema_ok = bool(ema5 > ema9)

    # 6-month range levels
    high_6m      = round(float(hist_6m["High"].max()), 2)
    low_6m       = round(float(hist_6m["Low"].min()),  2)

    # 20-day consolidation range (the "box")
    high_20d     = round(float(hist_6m["High"].iloc[-20:].max()), 2) if len(hist_6m) >= 20 else prev_high
    low_20d      = round(float(hist_6m["Low"].iloc[-20:].min()),  2) if len(hist_6m) >= 20 else prev_low

    # Key swing highs — cluster levels within 3% of each other, keep only distinct ones
    highs = hist_6m["High"]
    raw_swing_highs = []
    for i in range(2, len(highs) - 2):
        if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i-2] and \
           highs.iloc[i] > highs.iloc[i+1] and highs.iloc[i] > highs.iloc[i+2]:
            h = round(float(highs.iloc[i]), 2)
            if h > price:
                raw_swing_highs.append(h)

    # Cluster swing highs within 3% of each other — keep the highest of each cluster
    raw_swing_highs = sorted(set(raw_swing_highs))
    swing_highs = []
    for h in raw_swing_highs:
        if not swing_highs or (h - swing_highs[-1]) / swing_highs[-1] > 0.03:
            swing_highs.append(h)
    swing_highs = swing_highs[:3]  # keep nearest 3 distinct levels

    if session == "premarket":
        key_high    = prev_high
        key_low     = prev_low
        level_label = "Prior day"
    else:
        key_high    = pm.get("pm_high") or prev_high
        key_low     = pm.get("pm_low")  or prev_low
        level_label = "Pre-market"
    # Playbook entry: pullback to VWAP or EMA5 (whichever band is lower→higher)
    entry_low   = round(min(vwap_val, ema5), 2)
    entry_high  = round(max(vwap_val, ema5), 2)
    entry_ref   = "EMA5 / VWAP" if vwap_live_ok else "EMA5 / VWAP (proxy)"
    stop        = round(min(ema9, vwap_val) * 0.998, 2)
    stop_ref    = "below EMA9 / VWAP"

    risk = entry_low - stop

    # Overhead resistance: nearest swing high price has not cleared yet
    nearest_swing   = swing_highs[0] if swing_highs else None
    resistance      = None
    resistance_ref  = None
    if nearest_swing and price < nearest_swing:
        resistance     = nearest_swing
        resistance_ref = "overhead resistance — needs to clear before entry"

    # Targets: every mapped level AHEAD of price (or ahead of resistance), sorted ascending
    anchor  = resistance if resistance else price
    cand_t  = [(h, "swing high") for h in swing_highs if h > anchor]
    cand_t += [(round(key_high * 1.002, 2), f"{level_label} high")]
    cand_t += [(round(high_6m * 1.002, 2), "6-month high")]
    # first target must be meaningfully ahead (>=3%) so T1 is never "basically the entry"
    min_first = anchor * 1.03
    cand_t = sorted([(v, r) for v, r in cand_t if v > anchor * 1.003], key=lambda x: x[0])
    targets = []
    for v, r in cand_t:
        if not targets:
            if v < min_first:
                continue
        elif (v - targets[-1][0]) / targets[-1][0] < 0.02:
            continue
        targets.append((v, r))
    if not targets:
        proj = round(anchor * 1.01 + risk * 2, 2) if risk > 0 else round(anchor * 1.10, 2)
        targets = [(proj, "2R projection")]
    targets = targets[:3]
    t1, t1_ref = targets[0]
    t2, t2_ref = targets[1] if len(targets) > 1 else (None, None)
    t3, t3_ref = targets[2] if len(targets) > 2 else (None, None)

    rr1 = round((t1 - entry_low) / risk, 1) if risk > 0 else 0
    rr2 = round((t2 - entry_low) / risk, 1) if t2 and risk > 0 else None
    rr3 = round((t3 - entry_low) / risk, 1) if t3 and risk > 0 else None

    supports = build_support_levels(price, hist_6m, ctx, vwap_val if vwap_live_ok else None,
                                    pm.get("pm_high"), prev_high)
    status = determine_status(price, vwap_val, vwap_live_ok, ema5, entry_low, entry_high,
                              resistance, supports, atr, ctx.get("age_min"))
    range_info = compute_range(price, atr, ctx.get("hod"), ctx.get("lod"), resistance, t1, t2)

    news = get_news(ticker)
    ext  = get_extended_hours_price(ticker, prev_close)

    return {
        "ticker": ticker, "company": company,
        "sector": sector, "industry": industry,
        "price": price, "change": change, "relvol": relvol,
        "volume": volume, "pos": pos, "squeeze": False,
        "status": status, "range": range_info, "supports": supports,
        "intraday": {k: v for k, v in ctx.items() if not k.startswith("_")},
        "ema_ok": bool(ema_ok),
        "atr": atr, "sma50": sma50, "sma200": sma200,
        "data_source": data_source,
        "news": news, "ext": ext,
        "levels": {
            "label": level_label,
            "key_high": key_high, "key_low": key_low,
            "prev_close": prev_close, "vwap": vwap_val, "vwap_src": vwap_src,
            "hod": ctx.get("hod"), "lod": ctx.get("lod"),
            "pm_high": pm.get("pm_high"), "pm_low": pm.get("pm_low"),
            "pm_open": pm.get("pm_open"), "pm_last": pm.get("pm_last"),
            "high_6m": high_6m, "low_6m": low_6m,
            "high_20d": high_20d, "low_20d": low_20d,
            "swing_highs": swing_highs,
        },
        "plan": {
            "entry_low": entry_low if not resistance else None,
            "entry_high": entry_high if not resistance else None,
            "entry_ref": entry_ref,
            "stop": stop,
            "stop_ref": stop_ref,
            "resistance": resistance, "resistance_ref": resistance_ref,
            "breakout_level": resistance,
            "t1": t1, "t2": t2, "t3": t3,
            "t1_ref": t1_ref, "t2_ref": t2_ref, "t3_ref": t3_ref,
            "rr1": rr1, "rr2": rr2, "rr3": rr3,
        },
        "emas": {"ema5": ema5, "ema9": ema9, "src": ema_src},
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
        ema5   = tech["ema5"]      # TradingView has no EMA9, so use our own daily EMAs
        ema9   = tech["ema9"]
        sma20  = tv_daily.get("sma20")  or tech["sma20"]
        sma50  = tv_daily.get("sma50")  or tech["sma50"]
        sma200 = tv_daily.get("sma200") or tech["sma200"]
        price  = tv_daily.get("price")  or tech["current_price"]
        ema_ok = bool(tech["ema_ok"])
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

    # Additional quality filters
    if not tech["ema_ok"]:
        return None  # EMA5 must be above EMA9 on daily

    if (tech["current_price"] or 0) < 5:
        return None  # minimum price $5 for swings

    # Extension check — how far is price from SMA20?
    sma20_val = sma20 or tech["sma20"] or 0
    rsi_val   = tv_daily.get("rsi") if tv_daily else None

    if sma20_val and price:
        pct_above_sma20 = ((price - sma20_val) / sma20_val) * 100
    else:
        pct_above_sma20 = 0

    # Determine extension status
    if (pct_above_sma20 > 15 and rsi_val and rsi_val > 70) or pct_above_sma20 > 20:
        extension = "extended"      # 🔴 avoid
    elif pct_above_sma20 > 10 or (rsi_val and rsi_val > 65):
        extension = "caution"       # 🟡 slightly extended
    else:
        extension = "healthy"       # 🟢 good entry zone

    extension_detail = f"{pct_above_sma20:+.1f}% vs SMA20" + (f" | RSI {rsi_val:.0f}" if rsi_val else "")

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
        "extension": extension,
        "extension_detail": extension_detail,
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
_CACHE_TTL = 600  # swings: 10 minutes
_CACHE_TTL_FAST = 120  # day trades / squeezes: 2 minutes (momentum moves fast)

def cached_fetch(filters, label):
    key = label
    now = _time.time()
    ttl = _CACHE_TTL if label == "Swings" else _CACHE_TTL_FAST
    if key in _cache and (now - _cache[key]["ts"]) < ttl:
        print(f"  → {label}: using cached results ({len(_cache[key]['df'])} tickers)")
        return _cache[key]["df"]
    df = fetch_screener(filters, label)
    if df is not None and not df.empty:
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

    log_snapshots(cards, session, label)
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



# ══════════════════════════════════════════════════════════════
#  SNAPSHOT LOG + REVIEW  (did the levels actually work?)
# ══════════════════════════════════════════════════════════════
import json as _json
SNAP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots.jsonl")
_last_logged = {}

def log_snapshots(cards, session, label):
    try:
        now = datetime.now(ET)
        lines = []
        for c in cards:
            tk = c.get("ticker")
            if not tk or (_time.time() - _last_logged.get(tk, 0)) < 300:
                continue
            _last_logged[tk] = _time.time()
            plan, lvl, st = c.get("plan", {}), c.get("levels", {}), c.get("status", {})
            lines.append(_json.dumps({
                "ts": now.isoformat(), "date": now.strftime("%Y-%m-%d"), "session": session,
                "tab": label, "ticker": tk, "price": c.get("price"),
                "status": st.get("code"), "vwap": lvl.get("vwap"),
                "entry_low": plan.get("entry_low"), "entry_high": plan.get("entry_high"),
                "resistance": plan.get("resistance"), "t1": plan.get("t1"), "t2": plan.get("t2"),
                "t3": plan.get("t3"), "supports": [s["level"] for s in c.get("supports", [])],
                "range_verdict": (c.get("range") or {}).get("verdict"),
            }))
        if lines:
            with open(SNAP_FILE, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
    except Exception as e:
        print(f"  snapshot log error: {e}")


def _read_snapshots():
    if not os.path.exists(SNAP_FILE):
        return []
    out = []
    with open(SNAP_FILE, encoding="utf-8") as f:
        for ln in f:
            try:
                out.append(_json.loads(ln))
            except Exception:
                pass
    return out


@app.route("/api/snapshots/dates")
def snapshot_dates():
    snaps = _read_snapshots()
    dates = sorted({s["date"] for s in snaps}, reverse=True)
    return jsonify({"dates": dates, "count": len(snaps)})


@app.route("/api/snapshots/review")
def snapshot_review():
    """For each logged snapshot on a date, see what price did afterward."""
    date = request.args.get("date")
    snaps = [s for s in _read_snapshots() if s["date"] == date]
    if not snaps:
        return jsonify({"rows": [], "summary": []})
    hist_cache, rows = {}, []
    for s in snaps:
        tk = s["ticker"]
        try:
            if tk not in hist_cache:
                h = yf.Ticker(tk).history(period="7d", interval="1m", prepost=True)
                h.index = h.index.tz_convert(ET)
                hist_cache[tk] = h
            h = hist_cache[tk]
            ts = datetime.fromisoformat(s["ts"])
            after = h[h.index > ts]
            if after.empty:
                rows.append({**s, "outcome": "no later data yet"}); continue
            # window = rest of that session day if snapshot was before 4pm, else the next trading day
            first_day = after.index[0].date()
            win = after[after.index.date == first_day]
            price = float(s["price"])
            hi, lo = float(win["High"].max()), float(win["Low"].min())
            r = {**s, "window_date": str(first_day),
                 "mfe_pct": round((hi / price - 1) * 100, 1),
                 "mae_pct": round((lo / price - 1) * 100, 1),
                 "hit_t1": bool(s.get("t1") and hi >= s["t1"]),
                 "hit_t2": bool(s.get("t2") and hi >= s["t2"]),
                 "touched_entry": bool(s.get("entry_high") and lo <= s["entry_high"]),
                 "cleared_resistance": bool(s.get("resistance") and hi > s["resistance"]),
                 "lost_vwap_support": bool(s.get("vwap") and lo < s["vwap"] * 0.995)}
            rows.append(r)
        except Exception as e:
            rows.append({**s, "outcome": f"error: {e}"})
    # summary by status
    summ = {}
    for r in rows:
        if "mfe_pct" not in r:
            continue
        g = summ.setdefault(r.get("status") or "?", {"status": r.get("status") or "?", "n": 0, "t1": 0, "t2": 0, "mfe": 0.0, "mae": 0.0})
        g["n"] += 1; g["t1"] += r["hit_t1"]; g["t2"] += r["hit_t2"]
        g["mfe"] += r["mfe_pct"]; g["mae"] += r["mae_pct"]
    summary = [{"status": g["status"], "n": g["n"],
                "t1_rate": round(g["t1"] / g["n"] * 100), "t2_rate": round(g["t2"] / g["n"] * 100),
                "avg_mfe": round(g["mfe"] / g["n"], 1), "avg_mae": round(g["mae"] / g["n"], 1)}
               for g in summ.values()]
    return jsonify({"rows": rows, "summary": summary})

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
    app.run(debug=False, host="0.0.0.0", port=5000)
