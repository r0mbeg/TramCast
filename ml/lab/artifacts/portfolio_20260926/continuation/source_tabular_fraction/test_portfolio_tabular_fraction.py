"""One small check for the shared causal inputs and fixed volume/zero behavior."""
import numpy as np
import pandas as pd
from experiments.portfolio_tabular_fraction import features,prepare
from experiments.portfolio_route_allocation import preserve_network
from experiments.portfolio_timesfm_errors import daily


def check():
    origin='2025-01-31'; frame=daily(origin,'2025-04-02'); frame=frame.loc[frame.date.le('2025-02-02')].copy()
    past=frame.assign(origin=pd.Timestamp(origin),boardings=frame.base*1.05)
    targets=prepare(past,'2025-04-30')
    assert features(targets).shape==(18,30) and np.isfinite(features(targets)).all()
    np.testing.assert_allclose(targets.target,0,atol=1e-16)
    np.testing.assert_array_equal(features(targets),features(targets.assign(boardings=1e99)))
    try:prepare(past,'2025-01-31')
    except ValueError:pass
    else:raise AssertionError('Future past targets accepted')
    base=pd.DataFrame(dict(route=[1,1,5,5],date=pd.Timestamp('2025-02-01'),hour=[0,1,0,1],prediction=[10.,0,0,0]))
    learned=base.assign(prediction=[.2,0,0,0])
    np.testing.assert_array_equal(preserve_network(base,learned).prediction,base.prediction)
    print('tabular fraction causal/feature/volume/zero checks passed')

if __name__=='__main__':check()
