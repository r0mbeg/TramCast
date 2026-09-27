"""Run with PYTHONPATH=ml/lab: regression check for an incomplete first summer week."""
from pathlib import Path
import numpy as np
import pandas as pd
from dense_student.regime import regime_forecast
from experiments.portfolio_structure import regime_forecast as original
from experiments.portfolio_experiment import load_history

if __name__=='__main__':
    history=load_history('artifacts/hourly_clean.csv')
    params=dict(occurrences=32,statistic='mean',regime=True)
    cutoff='2025-06-04';end='2025-08-04';past=history.loc[history.date.le(cutoff)]
    try:original(past,cutoff,end,params)
    except ValueError:pass
    else:raise AssertionError('Expected regression fixture to expose missing summer weekdays')
    fixed=regime_forecast(past,cutoff,end,params)
    assert len(fixed)==14640 and np.isfinite(fixed.prediction).all()
    pd.testing.assert_frame_equal(fixed,regime_forecast(history,cutoff,end,params))
    for cutoff,end in [('2025-05-31','2025-07-31'),('2025-08-31','2025-10-31')]:
        pd.testing.assert_frame_equal(original(history,cutoff,end,params),regime_forecast(history,cutoff,end,params))
    print('Missing seasonal groups repaired; existing full-group origins unchanged; future history ignored')
