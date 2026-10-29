import pandas as pd
import streamlit as st
from datetime import datetime

import engine

st.set_page_config(page_title="Clear The Bar", page_icon="🏀", layout="wide")

st.markdown(
    '<style>'
    '@import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap");'
    ':root{--bg:#07090D;--surface:#0E1626;--border:#1C2740;--lime:#C8FF3D;--coral:#FF6B6B;--text:#E8ECF4;--muted:#7C89A3;}'
    'html,body,[class*="css"]{font-family:"Inter",system-ui,sans-serif;}'
    '.stApp{background:radial-gradient(1100px 560px at 12% -12%,rgba(200,255,61,.07),transparent 60%),radial-gradient(900px 520px at 100% 0,rgba(24,54,110,.35),transparent 58%),var(--bg);}'
    'header[data-testid="stHeader"]{background:transparent;}'
    '#MainMenu,footer{visibility:hidden;}'
    '.block-container{padding-top:1.6rem;max-width:1180px;}'
    '[data-testid="stSidebar"]{background:#0A0E16;border-right:1px solid var(--border);}'
    '[data-testid="stSidebar"] .stButton>button{width:100%;}'
    '[data-testid="stSidebar"] label{color:var(--muted);font-size:.74rem;letter-spacing:.14em;text-transform:uppercase;}'
    'button[kind="primary"]{background:var(--lime)!important;color:#07090D!important;border:0!important;font-weight:700!important;letter-spacing:.08em;text-transform:uppercase;}'
    'button[kind="secondary"]{background:transparent!important;color:var(--text)!important;border:1px solid var(--border)!important;}'
    '.brand{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;}'
    '.wm{font-family:"Space Grotesk";font-weight:700;letter-spacing:.16em;text-transform:uppercase;font-size:1.7rem;color:var(--text);line-height:1;}'
    '.wm i{font-style:normal;color:var(--lime);}'
    '.rule{height:1px;margin:16px 0 22px;background:linear-gradient(90deg,var(--lime),rgba(200,255,61,0) 68%);}'
    '.card{position:relative;background:linear-gradient(180deg,#0F1726,#0B1220);border:1px solid var(--border);border-radius:16px;padding:14px 20px 13px 22px;margin-bottom:12px;overflow:hidden;}'
    '.card:before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--lime);}'
    '.card.under:before{background:var(--coral);}'
    '.card .top{display:flex;gap:14px;align-items:center;}'
    '.card .head{order:2;margin-left:14px;width:94px;height:94px;flex:0 0 94px;object-fit:cover;object-position:top center;border-radius:50%;border:1px solid rgba(255,255,255,.16);background:radial-gradient(circle at 50% 30%,#F4F8FF 0%,#CDDAEE 58%,#A9B9D6 100%);box-shadow:0 4px 14px rgba(0,0,0,.35),inset 0 0 0 5px rgba(255,255,255,.55);}'
    '.card .idcol{flex:1;min-width:0;}'
    '.card .nameline{display:flex;justify-content:space-between;align-items:baseline;gap:12px;}'
    '.card h4{margin:0;font-size:1.04rem;font-weight:600;color:var(--text);}'
    '.card h4 .mu{color:var(--muted);font-weight:400;font-size:.85rem;}'
    '.card .rank{font-family:"Space Grotesk";font-size:.72rem;letter-spacing:.18em;text-transform:uppercase;color:var(--muted);}'
    '.card .play{font-family:"Space Grotesk";font-size:1.32rem;font-weight:700;letter-spacing:.02em;margin-top:3px;}'
    '.card .play.over{color:var(--lime);}'
    '.card .play.under{color:var(--coral);}'
    '.card .grid{display:flex;gap:44px;margin-top:14px;padding-top:13px;border-top:1px solid #17223A;}'
    '.card .g .k{font-size:.74rem;letter-spacing:.22em;text-transform:uppercase;color:var(--muted);}'
    '.card .g .v{font-family:"Space Grotesk";font-size:1.7rem;font-weight:700;color:var(--text);margin-top:2px;}'
    '.grade-a{color:var(--lime)!important;}'
    '.grade-b{color:#DCE4F2!important;}'
    '.grade-c{color:#93A0B8!important;}'
    '.empty{color:var(--muted);padding:26px 0;}'
    '</style>',
    unsafe_allow_html=True)

EDIT_COLS = ["player", "market", "line", "over_price", "under_price"]


@st.cache_data(show_spinner=False)
def _headshots():
    from nba_api.stats.static import players as _p
    exact, norm = {}, {}
    for pl in _p.get_players():
        if not pl.get("is_active"):
            continue
        exact.setdefault(pl["full_name"], pl["id"])
        norm.setdefault("".join(ch for ch in pl["full_name"].lower() if ch.isalnum()), pl["id"])
    return exact, norm


def _head_url(name):
    exact, norm = _headshots()
    pid = exact.get(str(name))
    if pid is None:
        pid = norm.get("".join(ch for ch in str(name).lower() if ch.isalnum()))
    return "https://cdn.nba.com/headshots/nba/latest/260x190/" + str(pid) + ".png" if pid else None


def _score(edge, mins):
    e = max(0.0, float(edge))
    m = float(mins) if mins == mins else 20.0
    s = 100.0 * min(1.0, (e / 0.16) ** 0.6)
    s *= 0.80 + 0.20 * min(1.0, m / 26.0)
    v = int(round(max(1.0, min(99.0, s))))
    cls = 'grade-a' if v >= 80 else ('grade-b' if v >= 60 else 'grade-c')
    return str(v), cls


LOG_MIN_SCORE = 80

def _log_card(sub, dstr=None):
    import csv, os, datetime as _dt
    fp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bets.csv')
    cols = ['logged_at','date','player','team','opp','market','side','line','proj','edge','score','result','units']
    new = not os.path.exists(fp)
    seen = set()
    if not new:
        with open(fp, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                seen.add((row['date'], row['player'], row['market'], row['side']))
    today = str(dstr) if dstr else str(_dt.date.today())
    added = 0
    with open(fp, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if new:
            w.writerow(cols)
        for _, r in sub.iterrows():
            key = (today, str(r['player']), str(r['market']), str(r['side']))
            if key in seen:
                continue
            sc, _c = _score(r['edge'], r['mean_min'])
            if int(sc) < LOG_MIN_SCORE: continue
            w.writerow([_dt.datetime.now().isoformat(timespec='seconds'), today, r['player'], r['team'], r['opp'], r['market'], r['side'], round(float(r['line']), 2), round(float(r['proj']), 2), round(float(r['edge']), 4), sc, '', 1])
            added += 1
    return added


def _bets_ui():
    import os, pandas as pd
    fp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bets.csv')
    if not os.path.exists(fp):
        st.caption('No bets logged yet.')
        return
    df = pd.read_csv(fp)
    if df.empty:
        st.caption('No bets logged yet.')
        return
    if 'units' not in df.columns:
        df['units'] = 1.0
    df['units'] = pd.to_numeric(df['units'], errors='coerce').fillna(1.0)
    df['result'] = df['result'].fillna('')
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    done = df[df['result'].isin(['W', 'L'])]
    won = float(done[done['result'] == 'W']['units'].sum())
    lost = float(done[done['result'] == 'L']['units'].sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Logged', len(df))
    c2.metric('Settled', len(done))
    c3.metric('Hit rate', (str(round(100.0 * (done['result'] == 'W').sum() / len(done), 1)) + '%') if len(done) else '-')
    c4.metric('Units P/L', str(round(won - lost, 2)) if len(done) else '-')
    if len(done):
        st.markdown('**By score**')
        for label, lo, hi in [('90+', 90, 999), ('80-89', 80, 90), ('70-79', 70, 80), ('60-69', 60, 70), ('under 60', 0, 60)]:
            b = done[(done['score'] >= lo) & (done['score'] < hi)]
            if len(b):
                st.write(label + ': ' + str(int((b['result'] == 'W').sum())) + '/' + str(len(b)) + ' (' + str(round(100.0 * (b['result'] == 'W').sum() / len(b), 1)) + '%)')
    st.markdown('**Your bets**')
    df['del'] = False
    ed = st.data_editor(df, hide_index=True, use_container_width=True, height=420, column_config={'del': st.column_config.CheckboxColumn('del'), 'result': st.column_config.SelectboxColumn('result', options=['', 'W', 'L', 'P']), 'units': st.column_config.NumberColumn('units', min_value=0.0, max_value=20.0, step=0.5)})
    cc1, cc2 = st.columns(2)
    if cc1.button('Save changes'):
        ed.drop(columns=['del']).to_csv(fp, index=False)
        st.success('Saved.')
    if cc2.button('Remove ticked'):
        _n = int(ed['del'].fillna(False).sum())
        ed[~ed['del'].fillna(False)].drop(columns=['del']).to_csv(fp, index=False)
        st.success('Removed ' + str(_n) + ' row(s).')
        st.rerun()


def _snap_path(d):
    import os
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'snap_' + str(d).replace('/', '-') + '.csv')

def _card_path(d):
    import os
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'card_' + str(d).replace('/', '-') + '.csv')

def _drop_card(d):
    import os
    try:
        os.remove(_card_path(d))
    except Exception:
        pass

def _frozen_card(board, d):
    import os, pandas as pd
    fp = _card_path(d)
    if os.path.exists(fp):
        try:
            old = pd.read_csv(fp)
            if not old.empty:
                for c in ['line', 'proj', 'edge', 'mean_min', 'rank']:
                    old[c] = pd.to_numeric(old[c], errors='coerce')
                return old
        except Exception:
            pass
    sc = board.apply(lambda r: float(_score(r['edge'], r['mean_min'])[0]), axis=1)
    c = board.assign(_sc=sc).sort_values('_sc', ascending=False)
    c = c.drop_duplicates('player', keep='first').head(6).reset_index(drop=True)
    c['rank'] = range(1, len(c) + 1)
    try:
        c.to_csv(fp, index=False)
    except Exception:
        pass
    return c

def _merge_snap(board, d):
    import os, pandas as pd
    fp = _snap_path(d)
    if os.path.exists(fp):
        try:
            old = pd.read_csv(fp)
        except Exception:
            old = None
        if old is not None and not old.empty:
            board = pd.concat([board, old], ignore_index=True, sort=False)
    board = board.drop_duplicates(['player', 'market', 'side'], keep='first')
    try:
        board.to_csv(fp, index=False)
    except Exception:
        pass
    return board


@st.cache_data(show_spinner=False)
def _teams():
    from nba_api.stats.static import teams as _t
    return {x["abbreviation"]: x["full_name"] for x in _t.get_teams()}


def _full(ab):
    return _teams().get(str(ab), str(ab))


def _review_ui(date_str):
    import os
    import pandas as pd
    st.caption('Graded results for a past day. The daily job grades the previous slate automatically.')
    d = st.text_input('Review date (mm/dd/yyyy)', value=str(date_str))
    fp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_' + str(d).replace('/', '-') + '.csv')
    if not os.path.exists(fp):
        st.caption('No graded results for ' + str(d) + ' yet.')
        return
    try:
        r = pd.read_csv(fp)
    except Exception:
        st.caption('Could not read results for ' + str(d))
        return
    if r.empty:
        st.caption('No plays recorded for ' + str(d))
        return
    g = r[r['result'].isin(['WIN', 'LOSS'])]
    c1, c2, c3 = st.columns(3)
    c1.metric('Plays', len(r))
    c2.metric('Settled', len(g))
    if len(g):
        c3.metric('Hit rate', str(round(100 * (g['result'] == 'WIN').mean(), 1)) + '%')
        st.markdown('**Hit rate by market**')
        t = g.groupby('market')['result'].apply(lambda x: round(100 * (x == 'WIN').mean(), 1)).reset_index()
        t.columns = ['market', 'hit %']
        st.dataframe(t, use_container_width=True, hide_index=True)
        st.markdown('**Projection bias** (mean actual / mean line, 1.00 is perfect)')
        g2 = g.copy()
        g2['line'] = pd.to_numeric(g2['line'], errors='coerce')
        g2['actual'] = pd.to_numeric(g2['actual'], errors='coerce')
        bt = g2.groupby('market').apply(lambda x: round(float(x['actual'].mean() / max(x['line'].mean(), 1e-9)), 3)).reset_index()
        bt.columns = ['market', 'actual/line']
        st.dataframe(bt, use_container_width=True, hide_index=True)
    show = [c for c in ['player', 'market', 'side', 'line', 'proj', 'actual', 'result'] if c in r.columns]
    st.dataframe(r[show], use_container_width=True, hide_index=True, height=520)



def _blow_ui(date_str):
    b = engine.blowout_board(date_str)
    if b is None or b.empty:
        st.caption("No games on this slate.")
        return
    show = b[["matchup", "margin", "risk"]].copy()
    show["margin"] = show["margin"].round(1)
    show.columns = ["Matchup", "Proj margin", "Blowout risk"]
    _pal = {"SEVERE": "background-color:rgba(255,77,77,0.22);color:#FF4D4D;font-weight:700", "HIGH": "background-color:rgba(255,140,80,0.18);color:#FF9F5B;font-weight:600", "MEDIUM": "background-color:rgba(232,192,125,0.14);color:#E8C07D", "LOW": "background-color:rgba(201,191,166,0.10);color:#C9BFA6"}
    _sty = show.style
    _sty = (_sty.map if hasattr(_sty, "map") else _sty.applymap)
    st.dataframe(_sty(lambda v: _pal.get(v, ""), subset=["Blowout risk"]), use_container_width=True, hide_index=True)
    st.caption("Projected from net ratings. Big margins mean starters rest and bench minutes swing.")


def _grade(edge):
    e = float(edge)
    if e >= 0.11:
        return "A+", "grade-a"
    if e >= 0.09:
        return "A", "grade-a"
    if e >= 0.075:
        return "B+", "grade-a"
    if e >= 0.06:
        return "B", "grade-b"
    if e >= 0.045:
        return "C+", "grade-c"
    if e >= 0.03:
        return "C", "grade-c"
    return "D", "grade-c"


def _editor_frame(df):
    out = df.copy()
    for c in EDIT_COLS:
        if c not in out.columns:
            out[c] = pd.NA
    out = out[EDIT_COLS]
    out["player"] = out["player"].fillna("").astype(str)
    out["market"] = out["market"].fillna("").astype(str).str.upper()
    out["line"] = pd.to_numeric(out["line"], errors="coerce")
    out["over_price"] = pd.to_numeric(out["over_price"], errors="coerce").fillna(engine.SUGGEST_PRICE)
    out["under_price"] = pd.to_numeric(out["under_price"], errors="coerce").fillna(engine.SUGGEST_PRICE)
    return out.reset_index(drop=True)


def _persist(df, date_str):
    save = _editor_frame(df)
    save.insert(0, "date", date_str)
    keep = engine.read_odds()
    keep = keep[keep["date"].astype(str) != date_str]
    engine.write_odds(pd.concat([keep, save], ignore_index=True))
    st.session_state["lines"] = save[EDIT_COLS]


def _screen(date_str, markets, min_minutes, max_per_team, teams, min_edge, top_n,
            center, conf, exposure, use_real, version):
    odds = engine.read_odds()
    have = odds[odds["date"].astype(str) == date_str]

    real = engine.fetch_real_odds(date_str, markets, teams) if use_real else None
    if real is not None and not real.empty:
        same = (len(have) == len(real)
                and set(map(tuple, have[["player", "market", "line"]].values.tolist()))
                == set(map(tuple, real[["player", "market", "line"]].values.tolist())))
        if not same:
            keep = odds[odds["date"].astype(str) != date_str]
            engine.write_odds(pd.concat([keep, real], ignore_index=True))
    elif have.empty:
        sug = engine.suggest_lines(date_str, list(markets), min_minutes, max_per_team,
                                   list(teams) if teams else None)
        if sug.empty:
            return pd.DataFrame()
        engine.write_odds(pd.concat([odds, sug], ignore_index=True))

    return engine.build_board(date_str, list(markets), min_edge, 60, engine.DEFAULT_BANKROLL,
                              list(teams) if teams else None, center, conf, exposure)


_screen = st.cache_data(show_spinner="Screening the slate...")(_screen)

version = st.session_state.setdefault("v", 0)

with st.sidebar:
    st.markdown('<div class="wm" style="font-size:1.05rem">Clear <i>The</i> Bar</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="rule" style="margin:10px 0 16px"></div>', unsafe_allow_html=True)

    default_date = datetime.strptime(engine.detect_slate_date(), "%m/%d/%Y").date()
    d = st.date_input("Slate", value=default_date)
    date_str = d.strftime("%m/%d/%Y")

    top_n = st.slider("Plays to show", 5, 80, 40)
    max_per_player = st.slider("Max plays per player", 1, 5, 2)

    if st.button("Refresh", type="primary", use_container_width=True):
        engine.clear_cache()
        st.cache_data.clear()
        st.session_state["v"] = version + 1
        st.rerun()

    with st.expander("Advanced"):
        use_real = st.checkbox("Use real odds", value=True)
        slate = engine.get_slate_cached(date_str)
        team_list = sorted(set(slate["home"]) | set(slate["away"])) if not slate.empty else []
        st.markdown('<div style="font-size:0.78rem;letter-spacing:0.14em;color:#7C89A3;margin:0.7rem 0 0.3rem 0;">TEAMS</div>', unsafe_allow_html=True)
        _c1, _c2 = st.columns(2)
        if _c1.button("All", use_container_width=True, key="t_all"):
            for _t in team_list:
                st.session_state["ck_" + date_str + "_" + _t] = True
            st.rerun()
        if _c2.button("None", use_container_width=True, key="t_none"):
            for _t in team_list:
                st.session_state["ck_" + date_str + "_" + _t] = False
            st.rerun()
        for _t in team_list:
            _k = "ck_" + date_str + "_" + _t
            if _k not in st.session_state:
                st.session_state[_k] = True
        _box = st.container(height=230)
        for _t in team_list:
            _box.checkbox(_full(_t), key="ck_" + date_str + "_" + _t)
        teams_sel = [t for t in team_list if st.session_state.get("ck_" + date_str + "_" + t, True)]
        markets = st.multiselect("Markets", list(engine.MARKETS),
                                 default=["PTS", "REB", "AST", "PRA"])
        min_minutes = 8.0
        max_per_team = 12
        min_edge = st.slider("Min edge", -0.10, 0.20, engine.MIN_EDGE, 0.005)
        conf = st.slider("Confidence shrink", 0.4, 1.0, engine.PROB_SHRINK, 0.05)
        exposure = st.slider("Max total exposure", 0.05, 0.60, engine.MAX_EXPOSURE, 0.05)
        center = st.checkbox("Balance board to slate", value=True)

        if st.button("Refresh odds now", use_container_width=True):
            engine.clear_odds_cache()
            st.cache_data.clear()
            st.rerun()

        if st.button("Regenerate estimates", use_container_width=True):
            odds = engine.read_odds()
            engine.write_odds(odds[odds["date"].astype(str) != date_str])
            st.cache_data.clear()
            st.rerun()

        if st.button("Clear data cache", use_container_width=True):
            engine.clear_cache()
            st.cache_data.clear()
            st.rerun()


st.markdown('<div class="brand"><div class="wm">Clear <i>The</i> Bar</div></div>'
            '<div class="rule"></div>', unsafe_allow_html=True)

board = _screen(date_str, tuple(markets), min_minutes, max_per_team,
                tuple(teams_sel), min_edge, top_n, center, conf, exposure, use_real, version)

if board.empty:
    st.markdown('<div class="empty">No plays for ' + date_str +
                '. Try another date, or lower Min edge in Advanced.</div>',
                unsafe_allow_html=True)
else:
    board_full = board.copy()
    if max_per_player > 0:
        board = board.groupby("player", group_keys=False).head(max_per_player)
    board = board.head(top_n).reset_index(drop=True)
    board["rank"] = range(1, len(board) + 1)

    board = _merge_snap(board, date_str)
    board = board.sort_values('edge', ascending=False).reset_index(drop=True)
    board['rank'] = range(1, len(board) + 1)

    import os as _os, pandas as _pd
    _bfp = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'bets.csv')
    _blog = set()
    if _os.path.exists(_bfp):
        try:
            _bdf = _pd.read_csv(_bfp)
            _blog = set(zip(_bdf['date'].astype(str), _bdf['player'].astype(str), _bdf['market'].astype(str), _bdf['side'].astype(str)))
        except Exception:
            _blog = set()
    for _tab, _sd in zip(_tabs4 := st.tabs(['Best Unders', 'Best Overs', 'All Props', 'Blowout Risk', 'Review']), ['UNDER', 'OVER']):
        _sub = board[board['side'] == _sd].reset_index(drop=True)
        with _tab:
            if _sub.empty:
                st.caption('No plays in this bucket tonight.')
            for _, r in _sub.iterrows():
                side_cls = "under" if r["side"] == "UNDER" else ""
                play_cls = "under" if r["side"] == "UNDER" else "over"
                url = _head_url(r["player"])
                head = '<img class="head" src="' + url + '" alt="" loading="lazy">' if url else '<div class="head"></div>'
                grade, gcls = _score(r["edge"], r["mean_min"])
                st.markdown(
                    '<div class="card ' + side_cls + '">'
                    '<div class="top">' + head +
                    '<div class="idcol">'
                    '<div class="nameline">'
                    '<h4>' + str(r["player"]) + ' <span class="mu">&middot; ' + str(r["team"]) + ' vs ' + str(r["opp"]) + '</span></h4>'
                    '<div class="rank">#' + str(int(r["rank"])) + (' ' + chr(10003) if (str(date_str), str(r["player"]), str(r["market"]), str(r["side"])) in _blog else '') + '</div>'
                    '</div>'
                    '<div class="play ' + play_cls + '">' + str(r["side"]) + ' ' + format(r["line"], ".1f") + ' ' + str(r["market"]) + '</div>'
                    '</div></div>'
                    '<div class="grid">'
                    '<div class="g"><div class="k">Line</div><div class="v">' + format(r["line"], ".1f") + '</div></div>'
                    '<div class="g"><div class="k">Proj</div><div class="v">' + format(r["proj"], ".1f") + '</div></div>'
                    '<div class="g"><div class="k">Score</div><div class="v ' + gcls + '">' + grade + '</div></div>'
                    '</div></div>',
                    unsafe_allow_html=True)
    
    

    with _tabs4[2]:
        st.caption('Every prop for the current filters - no top-N cut.')
        _fc = [c for c in ['rank', 'player', 'team', 'opp', 'market', 'side', 'line', 'proj', 'edge', 'conviction', 'mean_min'] if c in board.columns]
        st.dataframe(board_full[_fc].sort_values('conviction', ascending=False), use_container_width=True, hide_index=True, height=620)
    with _tabs4[3]:
        _blow_ui(date_str)

    with _tabs4[4]:
        _review_ui(date_str)

with st.expander("Lines grid (" + str(len(st.session_state.get("lines", []))) + " rows)"):
    if "lines" not in st.session_state:
        st.session_state["lines"] = _editor_frame(
            engine.read_odds().pipe(lambda x: x[x["date"].astype(str) == date_str])
        )
    edited = st.data_editor(
        st.session_state["lines"],
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "player": st.column_config.TextColumn("player", width="medium"),
            "market": st.column_config.TextColumn("market", width="small"),
            "line": st.column_config.NumberColumn("line", format="%.1f"),
            "over_price": st.column_config.NumberColumn("over_price", format="%d"),
            "under_price": st.column_config.NumberColumn("under_price", format="%d"),
        },
        key="editor",
    )
    c1, c2 = st.columns([1, 3])
    if c1.button("Save lines", use_container_width=True):
        _persist(edited, date_str)
        st.success("Saved.")
    if c2.button("Rebuild board", use_container_width=True):
        _persist(edited, date_str)
        st.cache_data.clear()
        st.rerun()

with st.expander("Diagnostics"):
    st.json(engine.diagnostics(date_str))
    st.markdown("**Calibration by market**")
    st.dataframe(engine.calibration(date_str, markets, teams_sel), use_container_width=True)