"""One small exact-hour-L1 target and target-free feature check."""
import numpy as np
import pandas as pd
from experiments.portfolio_tabular_volume import features,hourly_targets
from experiments.portfolio_timesfm_errors import daily


def check():
    shape=pd.DataFrame(dict(route=1,date=pd.Timestamp('2025-02-01'),hour=range(24),prediction=[60.,40.]+[0.]*22))
    truth=shape.drop(columns='prediction').assign(boardings=[30,80]+[0]*22)
    target=hourly_targets(shape,truth).optimal_volume.iloc[0];assert target==50
    shares=shape.prediction/100
    loss=lambda volume:abs(shares*volume-truth.boardings).sum()
    assert loss(target)<=loss(110) and loss(target)<=min(loss(v) for v in [0,25,49,51,75,200])
    f=daily('2025-01-31','2025-04-02').iloc[:10].copy();f['base_share']=.1
    profile=np.ones((10,20))/20;volume=np.ones(10)*1000
    assert features(f,profile,volume).shape==(10,51)
    np.testing.assert_array_equal(features(f,profile,volume),features(f.assign(boardings=1e99),profile,volume))
    try:features(f,profile[:,:-1],volume)
    except ValueError:pass
    else:raise AssertionError('Invalid profile shape accepted')
    print('exact hour-L1 target/feature/poison checks passed')

if __name__=='__main__':check()
