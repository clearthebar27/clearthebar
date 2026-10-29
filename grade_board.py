import os
import sys

import pandas as pd

date = sys.argv[1] if len(sys.argv) > 1 else None
if not date:
    raise SystemExit('date required')
fp = 'board_' + date.replace('/', '-') + '.csv'
if not os.path.exists(fp):
    raise SystemExit('no board for ' + date)
try:
    b = pd.read_csv(fp)
except Exception:
    raise SystemExit('empty board file')
date_iso = pd.to_datetime(date, errors='coerce').strftime('%Y-%m-%d')
logs = pd.read_csv('logs_nba.csv.gz')
logs['GAME_DATE'] = pd.to_datetime(logs['GAME_DATE'], errors='coerce').dt.strftime('%Y-%m-%d')
day = logs[logs['GAME_DATE'] == date_iso].copy()
if day.empty or b.empty:
    raise SystemExit('no actuals for ' + date)
for c in ('PTS', 'REB', 'AST', 'FG3M', 'STL', 'BLK'):
    day[c] = pd.to_numeric(day[c], errors='coerce').fillna(0.0)
day['PRA'] = day['PTS'] + day['REB'] + day['AST']
vals = {}
for _, r in day.iterrows():
    pid = str(r['PLAYER_ID'])
    dd = vals.setdefault(pid, {})
    for m in ('PTS', 'REB', 'AST', 'FG3M', 'STL', 'BLK', 'PRA'):
        dd[m] = float(r[m])
outs = []
for _, r in b.iterrows():
    a = vals.get(str(int(float(r.get('pid')))) if pd.notna(r.get('pid')) else '')
    act = a.get(r['market']) if a else None
    if act is None:
        res = 'NA'
    elif float(act) == float(r['line']):
        res = 'PUSH'
    elif str(r['side']).upper() == 'OVER':
        res = 'WIN' if float(act) > float(r['line']) else 'LOSS'
    else:
        res = 'WIN' if float(act) < float(r['line']) else 'LOSS'
    outs.append({**r.to_dict(), 'actual': act, 'result': res})
o = pd.DataFrame(outs)
pass
o.to_csv('results_' + date.replace('/', '-') + '.csv', index=False)
g = o[o['result'].isin(['WIN', 'LOSS'])]
print('graded', len(o), 'settled', len(g))
if len(g):
    print('hit rate', round(100 * (g['result'] == 'WIN').mean(), 1))
    t = g.groupby('market')['result'].apply(lambda s: round(100 * (s == 'WIN').mean(), 1))
    print(t.to_string())