"""
engine.py - NBA player prop model. Odds fetching lives in odds_api.py.
"""

from __future__ import annotations

import io
import os
import sqlite3
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.special import betainc, ndtr

import odds_api

from nba_api.stats.endpoints import (
    LeagueDashTeamStats, LeagueGameFinder, PlayerGameLogs, ScoreboardV2, ScoreboardV3,
)
from nba_api.stats.static import teams as _teams_static

HERE = os.path.dirname(os.path.abspath(__file__))
ODDS_PATH = os.path.join(HERE, "odds.csv")
DB_PATH = os.path.join(HERE, "propengine.sqlite")

# CONFIG -------------------------------------------------------------------
SEASON = "2026-27"
SEASONS = ("2026-27", "2025-26", "2024-25")
SLATE_DATE = "10/23/2026"
NET_TIMEOUT = 12
HALF_LIFE_GAMES = 25
SEASON_GAP_DAYS = 120
SEASON_GAP_GAMES = 70
MINUTES_SD_CAP = 10.0
MIN_MINUTES = 8.0
MIN_GAMES = 3
K_MINUTES = 0.2
BLEND_W = 0.75
MINUTES_HALF_LIFE = 4
MINUTES_SEASON_GAP = 220
LOG_CACHE_HOURS = 6
SLATE_CACHE_MIN = 240
RECENT_DAYS = 21
RECENT_FORM_GAMES = 10
RECENT_FORM_MIN_MINUTES = 10.0
EST_BLEND = 0.5

MARKETS = {"PTS": "points", "REB": "rebounds", "AST": "assists", "FG3M": "threes",
           "PRA": "pts+reb+ast", "STL": "steals", "BLK": "blocks"}
SUGGEST_MARKETS = ["PTS", "REB", "AST", "PRA", "FG3M", "STL", "BLK"]

SHRINK = {'PTS': 17.5, 'REB': 7.5, 'AST': 7.5, 'FG3M': 7.5, 'PRA': 30.0, 'STL': 11.25, 'BLK': 11.25}
SPIKINESS = {"PTS": 0.03, "REB": 0.05, "AST": 0.06, "FG3M": 0.10, "PRA": 0.02, "STL": 0.12, "BLK": 0.12}
SPREAD = {"PTS": 1.3, "REB": 1.3, "AST": 1.3, "FG3M": 1.3, "PRA": 1.3, "STL": 1.1, "BLK": 1.1}
DIST = {m: "normal" for m in MARKETS}
LEVEL = 1.03
CENTER = True
PROB_SHRINK = 0.70
MAX_EXPOSURE = 0.30
MAX_PLAYS = 300
MIN_EDGE = -0.10
KELLY_FRACTION = 0.25
DEFAULT_BANKROLL = 200.0
DEFAULT_PRICE = -110
SUGGEST_PRICE = -110

STAT_COL = {"PTS": "PTS", "REB": "REB", "AST": "AST", "FG3M": "FG3M", "STL": "STL", "BLK": "BLK"}
ODDS_COLS = ["date", "player", "market", "line", "over_price", "under_price"]
_last_slate_error = ""


# helpers ------------------------------------------------------------------
def _norm(s):
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def _find(df, *names):
    lut = {_norm(c): c for c in df.columns}
    for n in names:
        if _norm(n) in lut:
            return lut[_norm(n)]
    return None


def _clean(name):
    return str(name).replace(".", "").replace("'", "").replace("-", " ").strip()


def _parse_min(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    s = str(v).strip()
    if not s:
        return np.nan
    if ":" in s:
        parts = s.split(":")
        try:
            return float(parts[0]) + float(parts[1]) / 60.0
        except (ValueError, IndexError):
            return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _promote_header(df):
    if df is None or df.empty:
        return df
    if all(isinstance(c, (int, np.integer)) for c in df.columns):
        header = [str(x) for x in df.iloc[0].tolist()]
        out = df.iloc[1:].copy()
        out.columns = header
        return out.reset_index(drop=True)
    return df


def _to_date(date_str):
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(date_str), fmt)
        except ValueError:
            continue
    return datetime.now()


def _date_variants(date_str):
    d = _to_date(date_str)
    return list(dict.fromkeys([d.strftime("%Y-%m-%d"), d.strftime("%m/%d/%Y")]))


def _season_for(date_str):
    d = _to_date(date_str)
    y = d.year
    return f"{y}-{str(y + 1)[2:]}" if d.month >= 8 else f"{y - 1}-{str(y)[2:]}"


# cache --------------------------------------------------------------------
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


def clear_cache():
    try:
        con = sqlite3.connect(DB_PATH)
        con.execute("DELETE FROM cache")
        con.commit()
        con.close()
    except Exception:
        pass


# odds delegation ----------------------------------------------------------
def has_api_key():
    return odds_api.has_api_key()


def odds_credits_left():
    return odds_api.odds_credits_left()


def clear_odds_cache():
    odds_api.clear_odds_cache()


def fetch_real_odds(date_str=None, markets=None, teams=None):
    import prizepicks
    return prizepicks.fetch_real_odds(date_str or SLATE_DATE, markets, teams)


def _coerce_odds(df):
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


# schedule -----------------------------------------------------------------
def _rows_from_frames(frames):
    id2ab = {t["id"]: t["abbreviation"] for t in _teams_static.get_teams()}
    for df in frames:
        if df is None or df.empty:
            continue
        df = _promote_header(df)
        gid = _find(df, "GAME_ID")
        if gid is None:
            continue
        h = _find(df, "HOME_TEAM_ID")
        a = _find(df, "AWAY_TEAM_ID", "VISITOR_TEAM_ID")
        hab = _find(df, "HOME_TEAM_ABBREVIATION", "HOME_TEAM_ABBR")
        aab = _find(df, "AWAY_TEAM_ABBREVIATION", "VISITOR_TEAM_ABBREVIATION", "AWAY_TEAM_ABBR")
        rows = []
        for _, r in df.iterrows():
            ht = at = None
            if hab is not None and aab is not None and pd.notna(r[hab]) and pd.notna(r[aab]):
                ht, at = str(r[hab]), str(r[aab])
            elif h is not None and a is not None:
                try:
                    ht = id2ab.get(int(r[h]))
                    at = id2ab.get(int(r[a]))
                except (TypeError, ValueError):
                    ht = at = None
            if not ht or not at or ht == "nan" or at == "nan":
                continue
            rows.append({"game_id": str(r[gid]), "home": ht, "away": at})
        if rows:
            return rows
    return []


def _rows_from_gamefinder(date_str):
    iso = _to_date(date_str).strftime("%Y-%m-%d")
    season = _season_for(date_str)
    for kw in ({"date_from_nullable": iso, "date_to_nullable": iso, "season_nullable": season},
               {"date_from_nullable": iso, "date_to_nullable": iso}):
        try:
            df = LeagueGameFinder(timeout=NET_TIMEOUT, **kw).get_data_frames()[0]
        except Exception as e:
            print("  gamefinder failed:", e)
            continue
        if df is None or df.empty:
            continue
        acc = {}
        for _, r in df.iterrows():
            gid = str(r.get("GAME_ID", ""))
            ab = str(r.get("TEAM_ABBREVIATION", ""))
            mu = str(r.get("MATCHUP", ""))
            if not gid or not ab:
                continue
            acc.setdefault(gid, {"game_id": gid})
            if " vs. " in mu:
                acc[gid]["home"] = ab
            elif " @ " in mu:
                acc[gid]["away"] = ab
        rows = [d for d in acc.values() if "home" in d and "away" in d]
        if rows:
            return rows
    return []


def get_slate(date_str):
    global _last_slate_error
    import requests
    try:
        ymd = _to_date(date_str).strftime('%Y%m%d')
    except Exception:
        ymd = str(date_str)
    url = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=' + ymd
    try:
        r = requests.get(url, timeout=NET_TIMEOUT, headers={'User-Agent': 'Mozilla/5.0'})
        blob = r.json()
    except Exception as e:
        _last_slate_error = 'espn ' + type(e).__name__
        return pd.DataFrame(columns=['game_id', 'home', 'away'])
    rows = []
    for ev in (blob.get('events') or []):
        try:
            comp = (ev.get('competitions') or [{}])[0]
            home = ''
            away = ''
            for c in (comp.get('competitors') or []):
                ab = str((c.get('team') or {}).get('abbreviation') or '').upper()
                if not ab:
                    continue
                if c.get('homeAway') == 'home':
                    home = ab
                elif c.get('homeAway') == 'away':
                    away = ab
            if home and away:
                rows.append({'game_id': str(ev.get('id') or ''), 'home': _EFIX.get(home, home), 'away': _EFIX.get(away, away)})
        except Exception:
            continue
    if not rows:
        _last_slate_error = 'espn no games for ' + str(date_str)
        return pd.DataFrame(columns=['game_id', 'home', 'away'])
    _last_slate_error = ''
    return pd.DataFrame(rows)


def get_slate_cached(date_str, max_age=None):
    max_age = SLATE_CACHE_MIN * 60 if max_age is None else max_age
    key = 'slate:' + str(date_str)
    df = _cache_get(key, max_age=max_age)
    if df is not None and not df.empty:
        return df
    out = get_slate(date_str)
    if out is not None and not out.empty:
        _cache_put(key, out)
        return out
    old = _cache_get(key, max_age=86400 * 90)
    if old is not None and not old.empty:
        print('slate: stale cache for ' + str(date_str))
        return old
    if out is not None:
        _cache_put(key, out)
    return out if out is not None else pd.DataFrame()


def slate_teams(slate):
    if slate.empty:
        return {}
    out = {}
    for _, r in slate.iterrows():
        out[r["home"]] = r["away"]
        out[r["away"]] = r["home"]
    return out


def detect_slate_date(offsets=(0, 1, -1, 2, -2)):
    for off in offsets:
        ds = (datetime.now() + timedelta(days=off)).strftime('%m/%d/%Y')
        if not get_slate_cached(ds).empty:
            return ds
    return datetime.now().strftime('%m/%d/%Y')


def _prep_logs(df):
    d = df.copy()
    d["GAME_DATE"] = pd.to_datetime(d["GAME_DATE"], errors="coerce")
    d["MIN"] = d["MIN"].map(_parse_min)
    for c in ("PTS", "REB", "AST", "FG3M", "STL", "BLK"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d["PRA"] = d["PTS"] + d["REB"] + d["AST"]
    d["PLAYER_ID"] = pd.to_numeric(d["PLAYER_ID"], errors="coerce")
    d = d.dropna(subset=["GAME_DATE", "MIN", "PLAYER_ID"])
    d = d[d["MIN"] > 0]
    d["PLAYER_ID"] = d["PLAYER_ID"].astype(int)
    return d.sort_values(["PLAYER_ID", "GAME_DATE"])


def _fetch_logs_one_season(s):
    for kw in ({"league_id_nullable": "00", "season_nullable": s, "season_type_nullable": "Regular Season"},
               {"season_nullable": s, "season_type_nullable": "Regular Season"}):
        try:
            raw = PlayerGameLogs(timeout=NET_TIMEOUT, **kw).get_data_frames()[0]
            if raw is not None and not raw.empty:
                return raw
        except Exception as e:
            print("  log fetch failed:", s, kw, e)
    return None


def get_logs(seasons=SEASONS):
    key = "logs:" + "|".join(seasons)
    cached = _cache_get(key, max_age=LOG_CACHE_HOURS * 3600)
    if cached is not None and not cached.empty:
        return _prep_logs(cached)
    frames = []
    for s in seasons:
        raw = _fetch_logs_one_season(s)
        if raw is not None:
            frames.append(raw)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["PLAYER_ID", "GAME_ID"])
    _cache_put(key, df)
    return _prep_logs(df)


def _slate_players(logs, teams):
    newest = logs["GAME_DATE"].max()
    cutoff = newest - pd.Timedelta(days=RECENT_DAYS)
    latest = logs.sort_values("GAME_DATE").groupby("PLAYER_ID").tail(1)
    latest = latest[latest["GAME_DATE"] >= cutoff]
    keep = latest[latest["TEAM_ABBREVIATION"].isin(teams.keys())]
    ids = set(keep["PLAYER_ID"])
    names = dict(zip(keep["PLAYER_ID"], keep["PLAYER_NAME"]))
    teammap = dict(zip(keep["PLAYER_ID"], keep["TEAM_ABBREVIATION"]))
    return ids, names, teammap


# opponent factors ---------------------------------------------------------
def get_opp_factors():
    key = "oppfac:" + SEASON
    df = _cache_get(key, max_age=12 * 3600)
    if df is None or df.empty:
        try:
            df = LeagueDashTeamStats(measure_type_detailed_defense="Opponent",
                                     per_mode_detailed="PerGame", season=SEASON,
                                     timeout=NET_TIMEOUT).get_data_frames()[0]
        except Exception as e:
            print("  opponent factor fetch failed:", e)
            return {}
        if df is None or df.empty:
            return {}
        _cache_put(key, df)
    tid = _find(df, "TEAM_ID")
    if tid is None:
        return {}
    id2ab = {t["id"]: t["abbreviation"] for t in _teams_static.get_teams()}
    cols = {m: _find(df, "OPP_" + m) for m in ("PTS", "REB", "AST", "FG3M", "STL", "BLK")}
    out = {}
    for _, r in df.iterrows():
        try:
            ab = id2ab.get(int(r[tid]))
        except (TypeError, ValueError):
            continue
        if not ab:
            continue
        facs = {}
        for m, c in cols.items():
            if c is None:
                continue
            lg = pd.to_numeric(df[c], errors="coerce").mean()
            v = pd.to_numeric(r[c], errors="coerce")
            if lg and lg > 0 and pd.notna(v):
                facs[m] = float(np.clip(v / lg, 0.85, 1.15))
        if facs:
            facs["PRA"] = float(np.mean([facs.get(k, 1.0) for k in ("PTS", "REB", "AST")]))
        out[ab] = facs
    return out


# projection ---------------------------------------------------------------
def _league_priors(logs):
    pri = {}
    tot_min = float(logs["MIN"].sum())
    for m in MARKETS:
        col = "PRA" if m == "PRA" else STAT_COL[m]
        pri[m] = float(logs[col].sum()) / tot_min if tot_min > 0 else 0.0
    return pri, float(logs["MIN"].mean())

def _decay_weights(g, half_life=None, gap=None):
    half_life = HALF_LIFE_GAMES if half_life is None else half_life
    gap = SEASON_GAP_GAMES if gap is None else gap
    n = len(g)
    if n == 0:
        return np.zeros(0)
    d = pd.to_datetime(g["GAME_DATE"], errors="coerce")
    delta = d.diff(-1).dt.days.fillna(0.0)
    bound = (delta > SEASON_GAP_DAYS).astype(int)
    breaks = bound.cumsum().shift(1).fillna(0).to_numpy(dtype=float)
    idx = np.arange(n, dtype=float) + gap * breaks
    return 0.5 ** (idx / half_life)


def _season_split(g):
    d = pd.to_datetime(g["GAME_DATE"], errors="coerce")
    delta = d.diff(-1).dt.days.fillna(0.0)
    bound = (delta > SEASON_GAP_DAYS).to_numpy()
    idx = np.where(bound)[0]
    if len(idx) == 0:
        return len(g)
    return int(idx[0] + 1)


def _mean_minutes(g, prior_min, K=K_MINUTES):
    g = g.sort_values("GAME_DATE", ascending=False).head(40)
    n = len(g)
    if n < MIN_GAMES:
        return None
    w = _decay_weights(g, MINUTES_HALF_LIFE, MINUTES_SEASON_GAP)
    mins = g["MIN"].to_numpy(dtype=float)
    return (float(np.sum(w * mins)) + K * prior_min) / (float(np.sum(w)) + K)


def _recent_form(g, market):
    col = "PRA" if market == "PRA" else STAT_COL[market]
    recent = g[g["MIN"] >= RECENT_FORM_MIN_MINUTES]
    recent = recent.sort_values("GAME_DATE", ascending=False).head(RECENT_FORM_GAMES)
    if len(recent) < 3:
        recent = g.sort_values("GAME_DATE", ascending=False).head(RECENT_FORM_GAMES)
    if len(recent) == 0:
        return None
    return float(recent[col].mean())


def _project_player(g, market, prior_rate, prior_min, opp_fac, min_mult=1.0, blow_mult=1.0):
    g = g.sort_values("GAME_DATE", ascending=False).head(40)
    n = len(g)
    if n < MIN_GAMES:
        return None
    w = _decay_weights(g)
    wm = _decay_weights(g, MINUTES_HALF_LIFE, MINUTES_SEASON_GAP)
    mins = g["MIN"].to_numpy(dtype=float)
    col = "PRA" if market == "PRA" else STAT_COL[market]
    stat = g[col].to_numpy(dtype=float)

    wmin = float(np.sum(w * mins))
    if wmin <= 0:
        return None

    n_cur = _season_split(g)
    K = SHRINK[market] * min(1.0, n_cur / 15.0)
    rate = (float(np.sum(w * stat)) + K * prior_rate) / (wmin + K)

    n_cur = _season_split(g)
    own_prior = float(mins[n_cur:].mean()) if len(mins) - n_cur >= 8 else prior_min
    n_eff = float(np.sum(wm))
    mean_min = (float(np.sum(wm * mins)) + K_MINUTES * own_prior) / (n_eff + K_MINUTES)
    mean_min = min(40.0, mean_min * min_mult)
    if mean_min < MIN_MINUTES:
        return None

    var_m = float(np.average((mins - np.average(mins, weights=wm)) ** 2, weights=wm))
    msd = float(np.sqrt(max(var_m, 1.0) + 49.0 / max(n_eff, 1.0)))
    msd = min(msd, MINUTES_SD_CAP)
    msd = min(MINUTES_SD_CAP, msd * blow_mult)

    fac = 1.0 + 0.5 * (opp_fac - 1.0)
    mu = mean_min * rate * fac * LEVEL
    mu = mu * (mean_min / 22.0) ** -0.216 * _mcorr(market, mean_min)
    mult = fac * LEVEL * (mean_min / 22.0) ** -0.216 * _mcorr(market, mean_min)
    pg = np.where(mins > 0, stat / np.maximum(mins, 1e-9), 0.0)
    wmr = w * mins
    swr = float(np.sum(wmr))
    if swr > 0:
        rbar = float(np.sum(wmr * pg) / swr)
        var_r = float(np.average((pg - rbar) ** 2, weights=wmr))
    else:
        var_r = 0.0
    if not np.isfinite(var_r): var_r = rate / max(mean_min, 1.0)
    var_r = max(var_r, rate / max(mean_min, 1.0))
    msd = mult * float(np.sqrt(rate * rate * msd * msd + mean_min * mean_min * var_r))
    if mu <= 0:
        return None

    avg_l10 = float(stat[:10].mean())
    return {"mu": mu, "rate": rate, "mean_min": mean_min, "msd": msd, "avg_l10": avg_l10, "n": n}

def prob_over(mu, line, market, extra_sd):
    alpha = SPIKINESS.get(market, 0.05)
    var = SPREAD.get(market, 1.0) * (mu + alpha * mu * mu) + extra_sd ** 2
    sd = float(np.sqrt(max(var, 1e-9)))
    L = float(np.floor(line))
    U = float(np.ceil(line))
    a_n = float(1.0 - ndtr((L + 0.5 - mu) / sd))
    b_n = float(ndtr((U - 0.5 - mu) / sd))
    vn = a_n / (a_n + b_n) if (a_n + b_n) > 0 else 0.5
    if DIST.get(market, "normal") == "normal":
        return float(1.0 - ndtr((line - mu) / sd))
    a_eff = max((var - mu) / (mu * mu), 1e-6)
    r = 1.0 / a_eff
    p = 1.0 / (1.0 + a_eff * mu)
    k = float(np.floor(line)) + 1.0
    if k <= 0:
        return 1.0
    a_b = float(betainc(L + 1.0, r, 1.0 - p))
    b_b = float(1.0 - betainc(U, r, 1.0 - p))
    nb = a_b / (a_b + b_b) if (a_b + b_b) > 0 else 0.5
    return float((1.0 - SKEW_W) * vn + SKEW_W * nb)


def american_to_prob(o):
    o = float(o)
    if o == 0:
        return 0.5
    return (-o) / ((-o) + 100.0) if o < 0 else 100.0 / (o + 100.0)


def american_to_dec(o):
    o = float(o)
    return 1.0 + (100.0 / (-o) if o < 0 else o / 100.0)


# odds file ----------------------------------------------------------------
def read_odds():
    if not os.path.exists(ODDS_PATH):
        write_odds(pd.DataFrame(columns=ODDS_COLS))
    try:
        return _coerce_odds(pd.read_csv(ODDS_PATH))
    except Exception:
        return pd.DataFrame(columns=ODDS_COLS)


def write_odds(df):
    out = _coerce_odds(df)
    out.to_csv(ODDS_PATH, index=False)
    return out


# candidates ---------------------------------------------------------------
def _candidates(date_str, markets, teams=None):
    slate = get_slate_cached(date_str)
    if slate.empty:
        return pd.DataFrame()
    teams_map = slate_teams(slate)
    if teams:
        teams_map = {k: v for k, v in teams_map.items() if k in set(teams)}
        if not teams_map:
            return pd.DataFrame()
    logs = get_logs()
    if logs.empty:
        return pd.DataFrame()
    ids, names, teammap = _slate_players(logs, teams_map)
    if not ids:
        return pd.DataFrame()
    priors, prior_min = _league_priors(logs)
    oppfac = get_opp_factors()
    _risk = {}
    try:
        _bb = blowout_board(date_str)
        for _i, _r in _bb.iterrows():
            _pp = str(_r['matchup']).split(' @ ')
            if len(_pp) == 2:
                _risk[(_pp[0], _pp[1])] = str(_r['risk'])
    except Exception:
        pass
    _bm = {'MEDIUM': 1.10, 'HIGH': 1.28, 'SEVERE': 1.50}
    import injuries
    inj = injuries.fetch()
    _lo = {'out', 'injured reserve', 'suspension', 'suspend'}
    _lq = {'day-to-day', 'questionable', 'doubtful'}
    out_names = {n for n, s in inj.items() if str(s).strip().lower() in _lo}
    q_names = {n for n, s in inj.items() if str(s).strip().lower() in _lq}
    _gm = {_p: _g for _p, _g in logs.groupby('PLAYER_ID') if _p in ids}
    _mm = {}
    for _p in _gm:
        _v = float(pd.to_numeric(_gm[_p]['MIN'], errors='coerce').tail(10).mean())
        _mm[_p] = 0.0 if _v != _v else _v
    mult = {}
    for _t in set(teammap.values()):
        _ps = [q for q in _gm if teammap[q] == _t]
        _tot = sum(_mm.get(q, 0.0) for q in _ps)
        _miss = sum(_mm.get(q, 0.0) for q in _ps if names[q] in out_names)
        _sc = (_tot / (_tot - _miss)) if (_tot - _miss) > 1.0 else 1.0
        for q in _ps:
            mult[q] = _sc * (0.85 if names[q] in q_names else 1.0)

    odds = read_odds()
    odds = odds[odds["date"].astype(str) == date_str]
    recs = []
    for pid, g in logs.groupby("PLAYER_ID"):
        if pid not in ids or names[pid] in out_names:
            continue
        team = teammap[pid]
        opp = teams_map[team]
        for m in markets:
            sub = odds[(odds["player"].map(_clean) == _clean(names[pid])) & (odds["market"] == m)]
            if sub.empty or pd.isna(sub.iloc[0]["line"]):
                continue
            o = sub.iloc[0]
            p = _project_player(g, m, priors[m], prior_min, oppfac.get(opp, {}).get(m, 1.0), mult.get(pid, 1.0), _bm.get(_risk.get((team, opp)) or _risk.get((opp, team)), 1.0))
            if p is None:
                continue
            recs.append({"player": names[pid], "team": team, "opp": opp, "market": m,
                         "line": float(o["line"]), "over_price": float(o["over_price"]),
                         "under_price": float(o["under_price"]), "mu_raw": p["mu"],
                         "rate": p["rate"], "msd": p["msd"], "form": _recent_form(g, m),
                         "avg_l10": p["avg_l10"], "mean_min": p["mean_min"], "n_games": p["n"]})
    return pd.DataFrame(recs)


# board --------------------------------------------------------------------
def build_board(date_str=None, markets=None, min_edge=None, max_plays=None, bankroll=None,
                teams=None, center=CENTER, prob_shrink=PROB_SHRINK, max_exposure=MAX_EXPOSURE):
    date_str = date_str or SLATE_DATE
    markets = markets or list(MARKETS)
    min_edge = MIN_EDGE if min_edge is None else float(min_edge)
    max_plays = MAX_PLAYS if max_plays is None else int(max_plays)
    bankroll = DEFAULT_BANKROLL if bankroll is None else float(bankroll)

    d = _candidates(date_str, markets, teams)
    if d.empty:
        return pd.DataFrame()
    if center:
        d["ratio"] = d["line"] / d["mu_raw"]
        med = d.groupby("market")["ratio"].median()
        d["scale"] = d["market"].map(med).fillna(1.0)
    else:
        d["scale"] = 1.0
    d["mu"] = BLEND_W * d["mu_raw"] * d["scale"] + (1.0 - BLEND_W) * d["line"]
    d = d.reset_index(drop=True)
    _g = {}
    for _m, _s in d.groupby("market"):
        _g[str(_m)] = float((_s["mu"] / _s["line"]).median())
    d["mu"] = d["mu"] / d["market"].astype(str).map(_g).astype(float).fillna(1.0)

    rows = []
    for _, r in d.iterrows():
        line, mu = float(r["line"]), float(r["mu"])
        if line <= 0 or mu <= 0:
            continue
        po_raw = prob_over(mu, line, r["market"], float(r["msd"]))
        side = "OVER" if po_raw >= 0.5 else "UNDER"
        our = 0.5 + float(prob_shrink) * abs(po_raw - 0.5)
        price = float(r["over_price"] if side == "OVER" else r["under_price"])
        imp = american_to_prob(price)
        edge = our - imp
        dec = american_to_dec(price)
        ev = our * (dec - 1.0) - (1.0 - our)
        kelly = max(ev / (dec - 1.0), 0.0) * KELLY_FRACTION
        rows.append({"player": r["player"], "team": r["team"], "opp": r["opp"],
                     "market": r["market"], "line": line, "side": side, "proj": round(mu, 2),
                     "form": round(float(r["form"]), 2) if pd.notna(r["form"]) else np.nan,
                     "avg_l10": round(float(r["avg_l10"]), 2),
                     "mean_min": round(float(r["mean_min"]), 1), "n_games": int(r["n_games"]),
                     "edge": edge, "win_pct": our, "our_%": our * 100.0, "our_pct": our,
                     "price": price, "ev": ev, "conviction": abs(our - 0.5),
                     "stake": round(bankroll * kelly, 2), "gap_pct": (mu - line) / line})
    if not rows:
        return pd.DataFrame()
    b = pd.DataFrame(rows)
    b = b[b["edge"] >= min_edge]
    if b.empty:
        return b
    b = b.sort_values("conviction", ascending=False).head(max_plays).reset_index(drop=True)
    cap = float(max_exposure) * bankroll
    total = float(b["stake"].sum())
    if cap > 0 and total > cap:
        b["stake"] = (b["stake"] * (cap / total)).round(2)
        b["scaled"] = True
    else:
        b["scaled"] = False
    b.insert(0, "rank", range(1, len(b) + 1))
    return b


# calibration --------------------------------------------------------------
def calibration(date_str=None, markets=None, teams=None):
    date_str = date_str or SLATE_DATE
    markets = markets or list(MARKETS)
    d = _candidates(date_str, markets, teams)
    if d.empty:
        return pd.DataFrame()
    out = []
    for m, grp in d.groupby("market"):
        ratio = (grp["line"] / grp["mu_raw"]).median()
        mu_adj = grp["mu_raw"] * ratio
        out.append({"market": m, "n": len(grp),
                    "median_line": round(float(grp["line"].median()), 2),
                    "median_form": round(float(grp["form"].median()), 2),
                    "median_proj_raw": round(float(grp["mu_raw"].median()), 2),
                    "scale": round(float(ratio), 3),
                    "median_proj": round(float(mu_adj.median()), 2),
                    "pct_over_before": round(100 * float((grp["mu_raw"] > grp["line"]).mean()), 1),
                    "pct_over_after": round(100 * float((mu_adj > grp["line"]).mean()), 1)})
    return pd.DataFrame(out)


# estimated lines ----------------------------------------------------------
def suggest_lines(date_str=None, markets=None, min_minutes=20.0, max_per_team=8, teams=None,
                  price=SUGGEST_PRICE, blend=EST_BLEND):
    date_str = date_str or SLATE_DATE
    markets = markets or SUGGEST_MARKETS
    slate = get_slate_cached(date_str)
    if slate.empty:
        return pd.DataFrame()
    teams_map = slate_teams(slate)
    if teams:
        teams_map = {k: v for k, v in teams_map.items() if k in set(teams)}
        if not teams_map:
            return pd.DataFrame()
    logs = get_logs()
    if logs.empty:
        return pd.DataFrame()
    ids, names, teammap = _slate_players(logs, teams_map)
    if not ids:
        return pd.DataFrame()
    priors, prior_min = _league_priors(logs)

    shortlist = []
    for pid, g in logs.groupby("PLAYER_ID"):
        if pid not in ids:
            continue
        mm = _mean_minutes(g, prior_min)
        if mm is None or mm < min_minutes:
            continue
        shortlist.append({"pid": pid, "mm": mm, "team": teammap[pid]})
    if not shortlist:
        return pd.DataFrame()
    sl = pd.DataFrame(shortlist).sort_values("mm", ascending=False)
    sl = sl.groupby("team", group_keys=False).head(max_per_team)

    rows = []
    for _, s in sl.iterrows():
        pid = s["pid"]
        g = logs[logs["PLAYER_ID"] == pid]
        for m in markets:
            form = _recent_form(g, m)
            if form is None or form <= 0:
                continue
            p = _project_player(g, m, priors[m], prior_min, 1.0)
            if p is None:
                continue
            raw = (1.0 - float(blend)) * form + float(blend) * p["mu"]
            line = max(0.5, round(raw * 2.0) / 2.0)
            rows.append({"date": date_str, "player": names[pid], "market": m, "line": line,
                         "over_price": price, "under_price": price})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates(subset=["player", "market"])
    return _coerce_odds(df.sort_values(["player", "market"]))


# diagnostics --------------------------------------------------------------
def _abbr(v, _c={}):
    if not _c:
        from nba_api.stats.static import teams as _t
        _c.update({str(x["id"]): x["abbreviation"] for x in _t.get_teams()})
    return _c.get(str(v), str(v).upper())

def _net_ratings(season):
    try:
        d = LeagueDashTeamStats(season=season, measure_type_detailed_defense="Advanced", timeout=NET_TIMEOUT).get_data_frames()[0]
    except Exception:
        return {}
    ab = _find(d, "TEAM_ID")
    nr = _find(d, "NET_RATING")
    if ab is None or nr is None:
        return {}
    v = pd.to_numeric(d[nr], errors="coerce")
    return {_abbr(k): float(x) for k, x in zip(d[ab], v) if pd.notna(x)}


def blowout_board(date_str=None):
    date_str = date_str or SLATE_DATE
    sl = get_slate(date_str)
    if sl.empty:
        return pd.DataFrame()
    cur = _net_ratings(SEASON)
    pri = _net_ratings("2025-26")
    logs = get_logs()
    ng = 3.0
    if not logs.empty:
        rec = logs[logs["GAME_DATE"] >= logs["GAME_DATE"].max() - pd.Timedelta(days=120)]
        ng = float(max(rec.groupby("TEAM_ABBREVIATION")["GAME_DATE"].nunique().max(), 1))
    w = min(0.85, ng / (ng + 6.0))
    rows = []
    for _, g in sl.iterrows():
        h = str(g["home"]).upper()
        a2 = str(g["away"]).upper()
        hn = w * cur.get(h, pri.get(h, 0.0)) + (1 - w) * pri.get(h, 0.0)
        an = w * cur.get(a2, pri.get(a2, 0.0)) + (1 - w) * pri.get(a2, 0.0)
        m = (hn - an) * 0.98 + 2.5
        rows.append({"matchup": a2 + " @ " + h, "away_net": round(an, 1), "home_net": round(hn, 1), "margin": round(m, 1), "abs": abs(m)})
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values("abs", ascending=False).reset_index(drop=True)
    _c = out["abs"].quantile([0.85, 0.65, 0.40]).tolist(); out["risk"] = ["SEVERE" if (x >= _c[0] and x >= 6.0) else "HIGH" if (x >= _c[1] and x >= 4.5) else "MEDIUM" if (x >= _c[2] and x >= 3.0) else "LOW" for x in out["abs"]]
    return out


def diagnostics(date_str=None):
    date_str = date_str or SLATE_DATE
    out = {"date": date_str, "has_api_key": has_api_key(), "odds_credits_left": odds_credits_left()}
    try:
        sl = get_slate(date_str)
        out["games_on_slate"] = int(len(sl))
        out["slate_error"] = _last_slate_error
    except Exception as e:
        out["games_on_slate"] = f"ERROR {e}"
    try:
        logs = get_logs()
        out["log_rows"] = int(len(logs))
        if not logs.empty:
            out["newest_game"] = str(logs["GAME_DATE"].max().date())
            try:
                ids, _, _ = _slate_players(logs, slate_teams(get_slate(date_str)))
                out["slate_players_matched"] = int(len(ids))
            except Exception as e:
                out["slate_players_matched"] = f"ERROR {e}"
    except Exception as e:
        out["log_rows"] = f"ERROR {e}"
    try:
        o = read_odds()
        o = o[o["date"].astype(str) == date_str]
        out["lines_on_file"] = int(len(o))
    except Exception as e:
        out["lines_on_file"] = f"ERROR {e}"
    return out


def warm():
    get_logs()
    get_opp_factors()

_CORR_BP = [6.0, 12.0, 18.0, 24.0, 30.0, 40.0]
_CORR = {"AST": [1.60, 1.45, 1.28, 1.12, 1.05, 1.00], "PTS": [1.32, 1.18, 1.02, 0.97, 1.02, 1.03], "PRA": [1.05, 1.00, 1.00, 0.95, 1.00, 1.03], "REB": [1.00, 0.97, 0.97, 1.00, 0.96, 1.04]}
def _mcorr(market, m):
    a = _CORR.get(market)
    if not a:
        return 1.0
    return float(np.interp(m, _CORR_BP, a))

DIST.update({"PTS": "nb", "REB": "nb", "AST": "nb", "PRA": "nb", "FG3M": "nb", "STL": "nb", "BLK": "nb"})

SKEW_W = 0.30
