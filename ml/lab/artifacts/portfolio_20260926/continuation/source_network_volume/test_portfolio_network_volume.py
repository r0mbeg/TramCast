"""Small aggregate counts, covariates, feature poison and temporal guard check."""
import numpy as np
import pandas as pd
from experiments.portfolio_network_volume import aggregate,features
from experiments.portfolio_timesfm_errors import daily

def check():
    origin='2025-01-31'
    frame=daily(origin,'2025-04-02');frame=frame.loc[frame.date.le('2025-02-02')].assign(origin=pd.Timestamp(origin))
    result=aggregate(frame,origin,False)
    np.testing.assert_array_equal(result.base,frame.groupby('date').base.sum())
    np.testing.assert_array_equal(result.timesfm,frame.groupby('date').timesfm.sum())
    for days in [7,14,28]:
        expected=frame.assign(value=frame.base*frame[f'ratio{days}']).groupby('date').value.sum()/result.set_index('date').base
        np.testing.assert_allclose(result[f'ratio{days}'],expected,rtol=1e-12)
    observed=frame.assign(boardings=frame.route*100)
    past=aggregate(observed,'2025-04-30',True)
    np.testing.assert_array_equal(past.boardings,observed.groupby('date').boardings.sum())
    assert features(past).shape==(2,28) and features(past,True).shape==(2,34)
    np.testing.assert_array_equal(features(past),features(past.assign(boardings=1e100)))
    cases=[(observed,origin,False),(observed,origin,True),(pd.concat([frame,frame.iloc[:1]]),origin,False),
        (frame.iloc[1:],origin,False),(frame.assign(daytype=frame.route),origin,False)]
    for bad,cut,observed in cases:
        try:aggregate(bad,cut,observed)
        except ValueError:pass
        else:raise AssertionError('Invalid network input accepted')
    print('network aggregation/features/truthpoison/future/grid checks passed')

if __name__=='__main__':check()
