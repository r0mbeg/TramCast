import pandas as pd
from sequence_feasibility import gap_stats

f=pd.DataFrame(dict(vehicle=['a','a','b','a'], time=pd.to_datetime([
    '2025-09-08 06:01:00','2025-09-08 06:00:00','2025-09-08 06:00:00','2025-09-08 06:01:00'])))
s=gap_stats(f,['vehicle'])
assert s['pairs']==2 and s['zero_gap_fraction']==.5
assert s['fraction_at_most_60s']==1 and s['quantiles_seconds']['0.5']==30
assert gap_stats(f.iloc[:1],['vehicle'])['pairs']==0
print('sequence gap checks passed')
