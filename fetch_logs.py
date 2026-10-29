import time
from datetime import datetime, timedelta

import pandas as pd
import requests

UA = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'}
SB = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates='
SM = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event='
FIX = {'NY': 'NYK', 'GS': 'GSW', 'NO': 'NOP', 'SA': 'SAS', 'UTAH': 'UTA', 'WSH': 'WAS'}
DAYS = 420

def get(url):
    for _ in range(3):
        try:
            r = requests.get(url, timeout=25, headers=UA)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(1.0)
    return None

rows = []
today = datetime.utcnow()
for i in range(DAYS):
    d = today - timedelta(days=i)
    blob = get(SB + d.strftime('%Y%m%d'))
    if not blob:
        continue
    for ev in (blob.get('events') or []):
        eid = ev.get('id')
        if not eid:
            continue
        s = get(SM + str(eid))
        if not s:
            continue
        gd = str(ev.get('date') or '')[:10]
        for team in ((s.get('boxscore') or {}).get('players') or []):
            tab = str((team.get('team') or {}).get('abbreviation') or '').upper()
            tab = FIX.get(tab, tab)
            for st in (team.get('statistics') or []):
                labels = [str(x).upper() for x in (st.get('labels') or [])]
                for ath in (st.get('athletes') or []):
                    a = ath.get('athlete') or {}
                    nm = a.get('displayName')
                    pid = a.get('id')
                    vals = ath.get('stats') or []
                    if not nm or not pid or not vals:
                        continue
                    rec = {'GAME_ID': str(eid), 'GAME_DATE': gd, 'PLAYER_ID': str(pid), 'PLAYER_NAME': nm, 'TEAM_ABBREVIATION': tab}
                    for lab, v in zip(labels, vals):
                        rec[lab] = v
                    rows.append(rec)
    time.sleep(0.1)

df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit('no rows')
for c in ('MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK'):
    if c not in df.columns:
        df[c] = 0
if '3PT' in df.columns:
    df['FG3M'] = df['3PT'].astype(str).str.split('-').str[0]
else:
    df['FG3M'] = 0
df.to_csv('logs_nba.csv.gz', index=False, compression='gzip')
print('wrote', len(df))