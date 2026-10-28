# prizepicks.py - real PrizePicks lines from their public projections feed.
# Free: no API key, no credits, no account.

from __future__ import annotations

import io
import os
import sqlite3
import time
from datetime import date as _date
from datetime import datetime

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "propengine.sqlite")

URL = "https://partner-api.prizepicks.com/projections"
LEAGUE = 9
PAGE_SIZE = 500
MAX_PAGES = 10
NET_TIMEOUT = 30
CACHE_MIN = 10
RAW_TTL = 90
_RAW = {"ts": 0.0, "projs": [], "players": {}}

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

ODDS_COLS = ["date", "player", "market", "line", "over_price", "under_price"]
DEFAULT_PRICE = -110

USE_VARIANTS = False

MARKETS = ['REC_YDS', 'REC', 'TGT', 'RUSH_YDS', 'CAR', 'PASS_YDS', 'ATT', 'RR_YDS']

STAT_MAP = {'Receiving Yards': 'REC_YDS', 'Receptions': 'REC', 'Rec Targets': 'TGT', 'Rush Yards': 'RUSH_YDS', 'Rush Attempts': 'CAR', 'Pass Yards': 'PASS_YDS', 'Pass Attempts': 'ATT', 'Rush+Rec Yds': 'RR_YDS'}

VAR_PRICE = {"standard": -110, "goblin": -135, "demon": 120}


# --- storage ---------------------------------------------------------------
def _con():
    con = sqlite3.connect(DB_PATH)
    con.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, ts REAL, payload TEXT)")
    return con


def _cache_get(k, max_age):
    try:
        con = _con()
        row = con.execute("SELECT ts, payload FROM cache WHERE k=?", (k,)).fetchone()
        con.close()
    except sqlite3.Error:
        return None
    if not row:
        return None
    if time.time() - float(row[0]) > max_age:
        return None
    try:
        return pd.read_csv(io.StringIO(row[1]))
    except Exception:
        return None


def _cache_put(k, df):
    try:
        con = _con()
        con.execute("REPLACE INTO cache (k, ts, payload) VALUES (?,?,?)",
                    (k, time.time(), df.to_csv(index=False)))
        con.commit()
        con.close()
    except Exception:
        pass


def clear_odds_cache():
    try:
        con = _con()
        con.execute("DELETE FROM cache WHERE k LIKE 'realodds:ppnfl:%'")
        con.commit()
        con.close()
    except Exception:
        pass


def has_api_key():
    return True


def odds_credits_left():
    return None


# --- helpers ---------------------------------------------------------------
def _today():
    return datetime.now().strftime("%m/%d/%Y")


def _to_date(s):
    if isinstance(s, datetime):
        return s.date()
    if isinstance(s, _date):
        return s
    t = str(s).strip()
    for f in ("%m/%d/%Y", "%Y-%m-%d", "%m-%d-%Y", "%Y/%m/%d", "%m/%d/%y"):
        try:
            return datetime.strptime(t, f).date()
        except ValueError:
            pass
    try:
        return pd.to_datetime(t).date()
    except Exception:
        return datetime.now().date()


def _coerce(df):
    out = df.copy()
    for c in ODDS_COLS:
        if c not in out.columns:
            out[c] = np.nan
    out = out[ODDS_COLS]
    out["date"] = out["date"].fillna("").astype(str)
    out["player"] = out["player"].fillna("").astype(str)
    out["market"] = out["market"].fillna("").astype(str).str.upper()
    out["line"] = pd.to_numeric(out["line"], errors="coerce")
    out["over_price"] = pd.to_numeric(out["over_price"], errors="coerce").fillna(DEFAULT_PRICE)
    out["under_price"] = pd.to_numeric(out["under_price"], errors="coerce").fillna(DEFAULT_PRICE)
    return out.reset_index(drop=True)


# --- fetch -----------------------------------------------------------------
def _pull_raw():
    if _RAW["projs"] and (time.time() - _RAW["ts"]) < RAW_TTL:
        return _RAW["projs"], _RAW["players"]
    projs = {}
    players = {}
    for page in range(1, MAX_PAGES + 1):
        r = None
        for _try in range(3):
            try:
                r = requests.get(URL,
                                 params={'league_id': LEAGUE, 'per_page': PAGE_SIZE, 'page': page},
                                 headers={'User-Agent': UA, 'Accept': 'application/json'},
                                 timeout=NET_TIMEOUT)
            except Exception as e:
                print('  prizepicks failed:', e)
                r = None
            if r is not None and r.status_code == 200:
                break
            if r is not None:
                print('  prizepicks status:', r.status_code, str(r.text)[:120])
            time.sleep(1.5 * (_try + 1))
        if r is None or r.status_code != 200:
            break
        try:
            blob = r.json()
        except Exception:
            break
        data = blob.get("data") or []
        if not data:
            break
        before = len(projs)
        for it in data:
            projs[it.get("id")] = it
        for it in (blob.get("included") or []):
            if it.get("type") in ("new_player", "player"):
                a = it.get("attributes") or {}
                nm = a.get("name") or a.get("display_name")
                if nm:
                    players[it.get("id")] = {"name": nm, "team": a.get("team") or ""}
        if len(projs) == before:
            break
        meta = blob.get("meta") or {}
        tp = meta.get("total_pages")
        if tp and page >= int(tp):
            break
        if len(data) < PAGE_SIZE:
            break
    if projs:
        _RAW.update(ts=time.time(), projs=list(projs.values()), players=players)
    return list(projs.values()), players


def diag():
    projs, players = _pull_raw()
    print("  projections:", len(projs), " players(included):", len(players))
    if projs:
        a = projs[0].get("attributes") or {}
        print("  sample attr keys:", sorted(a.keys()))
        print("  sample stat:", a.get("stat_type"), "| line:", a.get("line_score"),
              "| type:", a.get("odds_type"), "| start:", a.get("start_time"))
    if players:
        print("  sample player:", list(players.items())[0])
    return {"projections": len(projs), "players": len(players)}


def probe(date_str=None):
    date_str = str(date_str) if date_str else _today()
    target = _to_date(date_str).strftime("%Y-%m-%d")
    projs, players = _pull_raw()
    n = 0
    names = set()
    for p in projs:
        a = p.get("attributes") or {}
        if str(a.get("start_time") or "")[:10] != target:
            continue
        n += 1
        pid = ((p.get("relationships") or {}).get("new_player") or {}).get("data") or {}
        nm = (players.get(pid.get("id")) or {}).get("name")
        if nm:
            names.add(nm)
    print("  prizepicks probe: %d projections, %d players on %s (free)"
          % (n, len(names), date_str))
    return {"projections": n, "players": len(names)}


def fetch_real_odds(date_str=None, markets=None, teams=None, force=False):
    date_str = str(date_str) if date_str else _today()
    want_mk = [str(m).strip().upper() for m in (markets or MARKETS)]
    want_mk = [m for m in want_mk if m in MARKETS]
    if not want_mk:
        return None

    target = _to_date(date_str).strftime("%Y-%m-%d")
    tkey = ",".join(sorted(str(t).strip().upper() for t in (teams or [])))
    ck = "realodds:ppnfl:" + date_str + ":" + tkey
    if not force:
        cached = _cache_get(ck, CACHE_MIN * 60)
        if cached is not None and not cached.empty:
            print("  prizepicks: cache hit, %d rows" % len(cached))
            return _coerce(cached)

    projs, players = _pull_raw()
    if not projs:
        return None

    want = set(str(t).strip().upper() for t in (teams or []) if str(t).strip())
    rows = []
    for p in projs:
        a = p.get("attributes") or {}
        if a.get("is_live"):
            continue
        if str(a.get("status") or "") not in ("", "pre_game", "post_game"):
            continue
        if not USE_VARIANTS and str(a.get("odds_type") or "standard") != "standard":
            continue
        mk = STAT_MAP.get(str(a.get("stat_type") or "").strip())
        if not mk or mk not in want_mk:
            continue
        if str(a.get("start_time") or "")[:10] != target:
            continue
        team = str(a.get("description") or "").strip().upper()
        if want and team not in want:
            continue
        pid = ((p.get("relationships") or {}).get("new_player") or {}).get("data") or {}
        nm = (players.get(pid.get("id")) or {}).get("name")
        if not nm:
            continue
        price = VAR_PRICE.get(str(a.get("odds_type") or "standard"), DEFAULT_PRICE)
        rows.append({"date": date_str, "player": str(nm).strip(), "market": mk,
                     "line": a.get("line_score"), "over_price": price, "under_price": price})

    if not rows:
        print("  prizepicks: 0 usable rows for", date_str)
        return None

    df = pd.DataFrame(rows)
    df["line"] = pd.to_numeric(df["line"], errors="coerce")
    df = df.dropna(subset=["line"]).drop_duplicates(["player", "market"])
    res = _coerce(df)
    _cache_put(ck, res)
    print("  prizepicks: %d lines on %s (free)" % (len(res), date_str))
    return res