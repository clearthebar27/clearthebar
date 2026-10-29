import os
import sys

import pandas as pd

import engine
import prizepicks

date = sys.argv[1] if len(sys.argv) > 1 else engine.detect_slate_date()
d = prizepicks.fetch_real_odds(date)
if d is not None and not d.empty:
    keep = engine.read_odds()
    keep = keep[keep['date'].astype(str) != date]
    engine.write_odds(pd.concat([keep, d], ignore_index=True))
b = engine.build_board(date, min_edge=-0.10, max_plays=300)
os.makedirs('boards', exist_ok=True)
fp = 'boards/board_' + date.replace('/', '-') + '.csv'
b.to_csv(fp, index=False)
print('board', date, len(b))