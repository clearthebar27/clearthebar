import pandas as pd

import engine

frames = []
for s in engine.SEASONS:
    r = engine._fetch_logs_one_season(s)
    if r is not None and not r.empty:
        frames.append(r)
if not frames:
    raise SystemExit('no logs fetched')
d = pd.concat(frames, ignore_index=True)
d.to_csv('logs_nba.csv.gz', index=False, compression='gzip')
print('wrote', len(d))