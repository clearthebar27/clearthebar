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
ODDS_PATH = os.path.join(HERE, "odds_nfl.csv")
DB_PATH = os.path.join(HERE, "propengine.sqlite")

# CONFIG -------------------------------------------------------------------
SEASON = "2026"
SEASONS = ("2026", "2025", "2024")
SLATE_DATE = "10/23/2026"
NET_TIMEOUT = 12
HALF_LIFE_GAMES = 8
SEASON_GAP_DAYS = 250
SEASON_GAP_GAMES = 30
MINUTES_SD_CAP = 22.0
MIN_MINUTES = 1.0
MIN_GAMES = 4
K_MINUTES = 0.2
BLEND_W = 0.75
MINUTES_HALF_LIFE = 3
MINUTES_SEASON_GAP = 40
LOG_CACHE_HOURS = 6
SLATE_CACHE_MIN = 240
RECENT_DAYS = 21
RECENT_FORM_GAMES = 10
RECENT_FORM_MIN_MINUTES = 1.0
EST_BLEND = 0.5

MARKETS = {'REC_YDS': 'rec yds', 'REC': 'rec', 'TGT': 'targets', 'RUSH_YDS': 'rush yds', 'CAR': 'carries', 'PASS_YDS': 'pass yds', 'ATT': 'attempts', 'RR_YDS': 'rush+rec yds'}
SUGGEST_MARKETS = ['REC_YDS', 'REC', 'TGT', 'RUSH_YDS', 'CAR', 'PASS_YDS', 'ATT', 'RR_YDS']

SHRINK = {'REC_YDS': 12.0, 'REC': 6.0, 'TGT': 6.0, 'RUSH_YDS': 20.0, 'CAR': 6.0, 'PASS_YDS': 60.0, 'ATT': 10.0, 'RR_YDS': 30.0}
SPIKINESS = {'REC_YDS': 0.04, 'REC': 0.08, 'TGT': 0.06, 'RUSH_YDS': 0.06, 'CAR': 0.06, 'PASS_YDS': 0.02, 'ATT': 0.03, 'RR_YDS': 0.03}
SPREAD = {'REC_YDS': 1.3, 'REC': 1.3, 'TGT': 1.2, 'RUSH_YDS': 1.3, 'CAR': 1.2, 'PASS_YDS': 1.3, 'ATT': 1.2, 'RR_YDS': 1.3}
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

STAT_COL = {'REC_YDS': 'REC_YDS', 'REC': 'REC', 'TGT': 'TGT', 'RUSH_YDS': 'RUSH_YDS', 'CAR': 'CAR', 'PASS_YDS': 'PASS_YDS', 'ATT': 'ATT', 'RR_YDS': 'RR_YDS'}
VOL_COL = {'PASS_YDS': 'ATT', 'ATT': 'ATT', 'RUSH_YDS': 'CAR', 'CAR': 'CAR', 'REC': 'TGT', 'TGT': 'TGT', 'REC_YDS': 'TGT', 'RR_YDS': 'OPP'}
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
    import ppnfl as prizepicks
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


def _fetch_schedules():
    key = 'nfsched'
    df = _cache_get(key, max_age=12 * 3600)
    if df is not None and not df.empty:
        return df
    try:
        df = pd.read_csv('https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv')
    except Exception as e:
        print('  schedule fetch failed:', e)
        return pd.DataFrame()
    keep = [c for c in ('season', 'week', 'gameday', 'away_team', 'home_team', 'game_id') if c in df.columns]
    df = df[keep].copy()
    _cache_put(key, df)
    return df


def get_slate(date_str):
    global _last_slate_error
    sch = _fetch_schedules()
    if sch.empty:
        _last_slate_error = 'schedule empty'
        return pd.DataFrame(columns=['game_id', 'home', 'away'])
    gd = pd.to_datetime(sch['gameday'], errors='coerce')
    want = pd.Timestamp(_to_date(date_str)).normalize()
    sub = sch[gd.dt.normalize() == want]
    rows = []
    for _, r in sub.iterrows():
        rows.append({'game_id': str(r.get('game_id', '')), 'home': str(r['home_team']).upper(), 'away': str(r['away_team']).upper()})
    _last_slate_error = ''
    if not rows:
        return pd.DataFrame(columns=['game_id', 'home', 'away'])
    return pd.DataFrame(rows)


def get_slate_cached(date_str, max_age=None):
    max_age = SLATE_CACHE_MIN * 60 if max_age is None else max_age
    key = 'nfslate:' + str(date_str)
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


def detect_slate_date(offsets=(0, -1, 1, -2, 2)):
    for off in offsets:
        ds = (datetime.now() + timedelta(days=off)).strftime("%m/%d/%Y")
        if not get_slate_cached(ds).empty:
            return ds
    return datetime.now().strftime("%m/%d/%Y")


# game logs ----------------------------------------------------------------
def _prep_logs(df):
    d = df.copy()
    if 'player_display_name' in d.columns:
        d['PLAYER_NAME'] = d['player_display_name']
    elif 'player_name' in d.columns:
        d['PLAYER_NAME'] = d['player_name']
    if ('team' in d.columns or 'recent_team' in d.columns):
        d['TEAM_ABBREVIATION'] = d['team'] if 'team' in d.columns else d['recent_team']
    if 'player_id' in d.columns:
        d['PLAYER_ID'] = d['player_id'].astype(str)
    if 'GAME_DATE' not in d.columns:
        d['GAME_DATE'] = pd.to_datetime(d['season'].astype(int).astype(str) + '-09-01') + pd.to_timedelta((pd.to_numeric(d['week'], errors='coerce').fillna(1.0) - 1.0) * 7.0, unit='D')
    d['GAME_DATE'] = pd.to_datetime(d['GAME_DATE'], errors='coerce')
    ren = {'attempts': 'ATT', 'carries': 'CAR', 'targets': 'TGT', 'receptions': 'REC', 'passing_yards': 'PASS_YDS', 'rushing_yards': 'RUSH_YDS', 'receiving_yards': 'REC_YDS'}
    d = d.rename(columns={k: v for k, v in ren.items() if k in d.columns and v not in d.columns})
    for c in ('ATT', 'CAR', 'TGT', 'REC', 'PASS_YDS', 'RUSH_YDS', 'REC_YDS'):
        if c not in d.columns:
            d[c] = 0.0
        d[c] = pd.to_numeric(d[c], errors='coerce').fillna(0.0)
    d['OPP'] = d['TGT'] + d['CAR']
    d['RR_YDS'] = d['RUSH_YDS'] + d['REC_YDS']
    d['VOL'] = d['ATT'] + d['CAR'] + d['TGT']
    if float(d['VOL'].sum()) == 0.0 and 'MIN' in d.columns:
        d['VOL'] = pd.to_numeric(d['MIN'].map(_parse_min), errors='coerce').fillna(0.0)
    d = d.dropna(subset=['GAME_DATE'])
    d = d[d['VOL'] > 0]
    return d.sort_values(['PLAYER_ID', 'GAME_DATE'])


def _fetch_logs_one_season(s):
    url = 'https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_' + str(s) + '.csv'
    try:
        raw = pd.read_csv(url)
    except Exception as e:
        print('  log fetch failed:', s, e)
        return None
    if raw is None or raw.empty:
        return None
    keep = [c for c in ('player_id', 'player_display_name', 'player_name', 'position', 'team', 'recent_team', 'season', 'week', 'season_type', 'opponent_team', 'attempts', 'carries', 'targets', 'receptions', 'passing_yards', 'rushing_yards', 'receiving_yards') if c in raw.columns]
    raw = raw[keep].copy()
    if 'season_type' in raw.columns:
        raw = raw[raw['season_type'].astype(str).str.upper().str.startswith('REG')]
    return raw


def get_logs(seasons=SEASONS):
    key = 'nflogs:' + '|'.join(str(s) for s in seasons)
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
    df = df.drop_duplicates(subset=['player_id', 'season', 'week'])
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
    return {}


def _league_priors(logs):
    pri = {}
    vol = {}
    for m in MARKETS:
        col = STAT_COL[m]
        vc = VOL_COL[m]
        tv = float(logs[vc].sum())
        pri[m] = float(logs[col].sum()) / tv if tv > 0 else 0.0
        vol[m] = float(logs[vc].mean())
    return pri, vol, float(logs['VOL'].mean())


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


def _mean_minutes(g, prior_min, volcol='VOL'):
    g = g.sort_values('GAME_DATE', ascending=False).head(40)
    n = len(g)
    if n < MIN_GAMES:
        return None
    w = _decay_weights(g, MINUTES_HALF_LIFE, MINUTES_SEASON_GAP)
    mins = g[volcol].to_numpy(dtype=float)
    return (float(np.sum(w * mins)) + K * prior_min) / (float(np.sum(w)) + K)


def _recent_form(g, market):
    col = STAT_COL[market]
    vc = VOL_COL[market]
    recent = g[g[vc] >= RECENT_FORM_MIN_MINUTES]
    recent = recent.sort_values('GAME_DATE', ascending=False).head(RECENT_FORM_GAMES)
    if len(recent) < 3:
        recent = g.sort_values('GAME_DATE', ascending=False).head(RECENT_FORM_GAMES)
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
    mins = g[VOL_COL[market]].to_numpy(dtype=float)
    col = STAT_COL[market]
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
    mean_min = min(60.0, mean_min * min_mult)
    if mean_min < MIN_MINUTES:
        return None

    var_m = float(np.average((mins - np.average(mins, weights=wm)) ** 2, weights=wm))
    msd = float(np.sqrt(max(var_m, 1.0) + 49.0 / max(n_eff, 1.0)))
    msd = min(msd, MINUTES_SD_CAP)
    msd = min(MINUTES_SD_CAP, msd * blow_mult)

    fac = 1.0 + 0.5 * (opp_fac - 1.0)
    mu = mean_min * rate * fac * LEVEL
    mu = mu * (mean_min / max(prior_min, 0.5)) ** -0.216 * _mcorr(market, mean_min)
    mult = fac * LEVEL * (mean_min / max(prior_min, 0.5)) ** -0.216 * _mcorr(market, mean_min)
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
    priors, prior_vol, prior_vol_all = _league_priors(logs)
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
        _v = float(pd.to_numeric(_gm[_p]['VOL'], errors='coerce').tail(10).mean())
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
            p = _project_player(g, m, priors[m], prior_vol[m], oppfac.get(opp, {}).get(m, 1.0), mult.get(pid, 1.0), _bm.get(_risk.get((team, opp)) or _risk.get((opp, team)), 1.0))
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
    priors, prior_vol, prior_vol_all = _league_priors(logs)

    shortlist = []
    for pid, g in logs.groupby("PLAYER_ID"):
        if pid not in ids:
            continue
        mm = _mean_minutes(g, prior_vol_all)
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
            p = _project_player(g, m, priors[m], prior_vol[m], 1.0)
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
    return {}


def blowout_board(date_str=None):
    date_str = date_str or SLATE_DATE
    sl = get_slate(date_str)
    if sl.empty:
        return pd.DataFrame()
    cur = _net_ratings(SEASON)
    pri = _net_ratings("2025")
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
