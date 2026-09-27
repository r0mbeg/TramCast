"""Small check of route-fraction target, volume weighting and truth-free features."""
import numpy as np
import pandas as pd
from pipeline import ROUTES
from experiments.portfolio_route_fraction import prepare,features


def check():
    frame=pd.DataFrame(dict(route=ROUTES,date=pd.Timestamp('2025-03-01'),origin=pd.Timestamp('2025-01-31'),
        base=[0. if r==5 else 100. for r in ROUTES],boardings=[0. if r==5 else 200. for r in ROUTES],
        horizon=29,effective_weekday=5,daytype=1,off=False,summer=0,timesfm=100.,direct=100.,ridge=100.,regime=100.,
        ratio7=1.,ratio14=1.,ratio28=1.,temperature_2m_mean=1.,precipitation_sum=0.,daylight_duration=36000.))
    frame.loc[frame.route.eq(17),'boardings']=400.
    past=prepare(frame,'2025-03-01')
    assert past.route.ne(5).all() and past.weight.mean()==1.
    np.testing.assert_allclose(past.target,past.boardings/past.network_actual-past.base/past.network_base)
    estimate=np.resize([.02,.09,.31],len(past))
    repeat=past.groupby(['route','date']).base.transform('size')
    normalizer=(past.network_actual/repeat).mean()
    np.testing.assert_allclose(np.sum(abs(estimate-past.target-past.base_share)*past.weight),
        np.sum(abs(estimate*past.network_actual-past.boardings)/repeat)/normalizer)
    before=features(past);assert before.shape==(9,24)
    np.testing.assert_array_equal(before,features(past.assign(boardings=999999.,network_actual=999999.,target=999.)))
    try:prepare(frame.assign(date=pd.Timestamp('2025-03-02')),'2025-03-01')
    except ValueError:pass
    else:raise AssertionError('Future route share target accepted')
    print('Route fraction checks passed: relative error, daily L1 weighting identity, truth-free 24 features, future guard.')


if __name__=='__main__':check()
