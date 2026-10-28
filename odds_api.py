"""odds_api.py - real player-prop lines from The Odds API (PrizePicks)."""

from __future__ import annotations

import io
import os
import sqlite3
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

from nba_api.stats.static import teams as _teams_static

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "propengine.sqlite")
KEY_PATH = os.path.join(HERE, "apikey.txt")

BASE = "https://api.the-odds-api.com/v4/sports/basketball_nba"
BOOK = "prizepicks"
SLATE_DATE = "10/23/2026"
NET_TIMEOUT = 20
CACHE_MIN = 30
DEFAULT_PRICE = -110
ODDS_COLS = ["date", "player", "market", "line", "over_price", "under_price"]

MARKET_KEYS = {"PTS": "player_points", "REB": "player_rebounds", "AST": "player_assists",
               "FG3M": "player_threes", "PRA": "player_points_rebounds_assists",
               "STL": "player_steals", "BLK": "player_blocks"}

_last_remaining = None


def _cache_get(k, max_age=3600):
    try:
        con = sqlite3.connect(DB_PATH)
        row = con.execute("SELECT ts, payload FROM cache WHERE k=?", (k,)).fetchone()
        con.close()
    except sqlite3.Error:
        return None
    if not row:
        return None
    ts, payload = row
    if time.time() - float(ts) > max_age:
        return None
    try:
        return pd.read_csv(io.StringIO(payload))
    except Exception:
        return None


def _cache_put(k, df):
    try:
        con = sqlite3.connect(DB_PATH)
        con.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, ts REAL, payload TEXT)")
        con.execute("REPLACE INTO cache (k, ts, payload) VALUES (?,?,?)",
                    (k, time.time(), df.to_csv(index=False)))
        con.commit()
        con.close()
    except Exception:
        pass


def clear_odds_cache():
    try:
        con = sqlite3.connect(DB_PATH)
        con.execute("DELETE FROM cache WHERE k LIKE 'realodds:%'")
        con.commit()
        con.close()
    except Exception:
        pass


def _api_key():
    k = os.environ.get("ODDS_API_KEY")
    if k:
        return k.strip()
    if os.path.exists(KEY_PATH):
        try:
            with open(KEY_PATH) as f:
                return f.read().strip()
        except OSError:
            return None
    return None


def has_api_key():
    return bool(_api_key())


def odds_credits_left():
    return _last_remaining


def _track(r):
    global _last_remaining
    try:
        _last_remaining = int(r.headers.get("x-requests-remaining"))
    except (TypeError, ValueError):
        pass


def _to_date(s):
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(s), fmt)
        except ValueError:
            continue
    return datetime.now()


def _coerce(df):
    out = df.copy()
    for c in ODDS_COLS:
        if c not in out.columns:
            out[c] = None
    out = out[ODDS_COLS]
    out["date"] = out["date"].fillna("").astype(str)
    out["player"] = out["player"].fillna("").astype(str)
    out["market"] = out["market"].fillna("").astype(str).str.upper()
    out["line"] = pd.to_numeric(out["line"], errors="coerce")
    out["over_price"] = pd.to_numeric(out["over_price"], errors="coerce").fillna(DEFAULT_PRICE)
    out["under_price"] = pd.to_numeric(out["under_price"], errors="coerce").fillna(DEFAULT_PRICE)
    return out.reset_index(drop=True)


def _team_abbr(full_name):
    n = str(full_name or "").strip().lower()
    if not n:
        return None
    for t in _teams_static.get_teams():
        if n in (t["full_name"].lower(), t["nickname"].lower(), t["city"].lower()):
            return t["abbreviation"]
    return None


def _events_on(date_str):
    try:
        r = requests.get(BASE + "/events", params={"apiKey": _api_key()}, timeout=NET_TIMEOUT)
    except Exception as e:
        print("  events fetch failed:", e)
        return []
    if r.status_code != 200:
        print("  events status:", r.status_code, r.text[:200])
        return []
    _track(r)
    target = _to_date(date_str).strftime("%Y-%m-%d")
    out = []
    for ev in r.json():
        ct = ev.get("commence_time")
        if not ct:
            continue
        try:
            local = datetime.fromisoformat(ct.replace("Z", "+00:00")) + timedelta(hours=-4)
        except ValueError:
            continue
        if local.strftime("%Y-%m-%d") != target:
            continue
        out.append(ev)
    return out


def fetch_real_odds(date_str=None, markets=None, teams=None, book=BOOK):
    date_str = date_str or SLATE_DATE
    markets = markets or list(MARKET_KEYS)
    if not _api_key():
        return None

    ck = "realodds:" + str(date_str) + ":" + ",".join(sorted(markets)) + ":" + book + ":" + \
         ",".join(sorted(str(t) for t in (teams or [])))
    cached = _cache_get(ck, max_age=CACHE_MIN * 60)
    if cached is not None and not cached.empty:
        return _coerce(cached)

    mk = [MARKET_KEYS[m] for m in markets if m in MARKET_KEYS]
    if not mk:
        return None

    events = _events_on(date_str)
    if teams:
        want = {str(t) for t in teams}
        events = [e for e in events
                  if _team_abbr(e.get("home_team")) in want or _team_abbr(e.get("away_team")) in want]
    if not events:
        print("  no events matched for", date_str)
        return pd.DataFrame(columns=ODDS_COLS)

    rev = {v: k for k, v in MARKET_KEYS.items()}
    raw = []
    for ev in events:
        params = {"apiKey": _api_key(), "bookmakers": book, "markets": ",".join(mk),
                  "oddsFormat": "american", "dateFormat": "iso"}
        try:
            r = requests.get(BASE + "/events/" + ev["id"] + "/odds", params=params,
                             timeout=NET_TIMEOUT)
        except Exception as e:
            print("  event odds failed:", e)
            continue
        _track(r)
        if r.status_code != 200:
            print("  event odds status:", r.status_code, r.text[:160])
            continue
        try:
            data = r.json()
        except Exception:
            continue
        for bk in data.get("bookmakers", []):
            for m in bk.get("markets", []):
                market = rev.get(m.get("key"))
                if not market:
                    continue
                for o in m.get("outcomes", []):
                    raw.append({"player": str(o.get("description") or ""), "market": market,
                                "side": str(o.get("name") or "").lower(),
                                "line": o.get("point"), "price": o.get("price")})
    if not raw:
        return pd.DataFrame(columns=ODDS_COLS)

    d = pd.DataFrame(raw)
    d["line"] = pd.to_numeric(d["line"], errors="coerce")
    d["price"] = pd.to_numeric(d["price"], errors="coerce")
    d = d.dropna(subset=["line", "price"])
    if d.empty:
        return pd.DataFrame(columns=ODDS_COLS)

    rows = []
    for (pl, mkt), grp in d.groupby(["player", "market"]):
        line = float(grp.groupby("line").size().sort_values(ascending=False).index[0])
        at = grp[grp["line"] == line]
        ov = at[at["side"] == "over"]["price"]
        un = at[at["side"] == "under"]["price"]
        if ov.empty or un.empty:
            continue
        rows.append({"date": date_str, "player": pl, "market": mkt, "line": line,
                     "over_price": float(ov.max()), "under_price": float(un.max())})
    if not rows:
        return pd.DataFrame(columns=ODDS_COLS)
    out = _coerce(pd.DataFrame(rows))
    _cache_put(ck, out)
    return out