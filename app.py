"""
==============================================================
  TRADING SCREENER — Web App (Flask)
  v3 — Sector filter + News feed
==============================================================

SETUP (run once):
  pip install finviz pandas yfinance pytz flask flask-cors

RUN:
  python app.py

Then open your browser to: http://localhost:5000
==============================================================
"""

from flask import Flask, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import yfinance as yf
from datetime import datetime, time as dtime
import pytz
import warnings
warnings.filterwarnings("ignore")

try:
    from finviz.screener import Screener
    import finviz
    HAS_FINVIZ = True
except ImportError:
    HAS_FINVIZ = False

app = Flask(__name__)
CORS(app)

# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════

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

MIN_VOLUME  = 1_000_000
EMA_SHORT   = 5
EMA_LONG    = 9
ET          = pytz.timezone("America/New_York")
MARKET_OPEN = dtime(9, 30)


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
        "short float": "Short Float", "float": "Float",
    }
    df = df.copy()
    df.columns = [rename_map.get(c.lower().strip(), c) for c in df.columns]
    return df


def fetch_screener(filters, label):
    if not HAS_FINVIZ:
        return pd.DataFrame()
    try:
        screen = Screener(filters=filters, table="Overview", order="-relativevolume")
        data = list(screen)
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df = normalize_columns(df)
        return df
    except Exception as e:
        print(f"Error fetching {label}: {e}")
        return pd.DataFrame()


def apply_volume_gate(df):
    if df.empty:
        return df
    if "Volume" in df.columns:
        df = df.copy()
        df["Volume"] = pd.to_numeric(
            df["Volume"].astype(str).str.replace(",", ""), errors="coerce"
        )
        return df[df["Volume"] >= MIN_VOLUME]
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


    result = {"pm_high": None, "pm_low": None, "pm_open": None, "pm_last": None}
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="1d", interval="1m", prepost=True)
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


def get_day_trade_technicals(ticker, session):
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
    relvol  = str(row.get("Rel Volume", "—"))
    volume  = str(row.get("Volume", "—"))

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
    pos   = change.startswith("+") if change != "—" else True

    return {
        "ticker": ticker, "company": company,
        "sector": sector, "industry": industry,
        "price": price, "change": change, "relvol": relvol,
        "volume": volume, "pos": pos, "squeeze": False,
        "ema_ok": tech["ema_ok"],
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
    relvol   = str(row.get("Rel Volume", "—"))
    volume   = str(row.get("Volume", "—"))
    pos      = change.startswith("+") if change != "—" else True

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
#  ROUTES
# ══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/run")
def run_screener():
    session = get_session()
    now_et  = datetime.now(ET).strftime("%I:%M:%S %p ET")

    day_raw   = apply_volume_gate(fetch_screener(DAY_TRADE_FILTERS, "Day Trades"))
    sq_raw    = apply_volume_gate(fetch_screener(SQUEEZE_FILTERS,   "Squeezes"))
    swing_raw = apply_volume_gate(fetch_screener(SWING_FILTERS,     "Swings"))

    day_tickers = set(day_raw["Ticker"].tolist()) if not day_raw.empty else set()

    day_cards = []
    for _, row in day_raw.iterrows():
        try:
            day_cards.append(build_day_card(row, session))
        except Exception as e:
            print(f"Day card error {row.get('Ticker')}: {e}")

    sq_cards = []
    if not sq_raw.empty:
        for _, row in sq_raw.iterrows():
            ticker = str(row.get("Ticker", ""))
            if ticker in day_tickers:
                continue
            try:
                card = build_day_card(row, session)
                card["squeeze"]     = True
                card["short_float"] = str(row.get("Short Float", "—"))
                card["float_size"]  = str(row.get("Float", "—"))
                sq_cards.append(card)
            except Exception as e:
                print(f"Squeeze card error {ticker}: {e}")

    swing_cards = []
    if not swing_raw.empty:
        for _, row in swing_raw.iterrows():
            try:
                card = build_swing_card(row)
                if card:
                    swing_cards.append(card)
            except Exception as e:
                print(f"Swing card error {row.get('Ticker')}: {e}")

    # Collect all unique sectors across all results for the dropdown
    all_sectors = sorted(set(
        c["sector"] for cards in [day_cards, sq_cards, swing_cards]
        for c in cards if c.get("sector")
    ))

    return jsonify({
        "session":  session,
        "time":     now_et,
        "day":      day_cards,
        "squeeze":  sq_cards,
        "swing":    swing_cards,
        "sectors":  all_sectors,
    })


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
