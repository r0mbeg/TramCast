"""One causal origin context/partial horizon/target check."""
import numpy as np
import pandas as pd
from experiments.portfolio_tabular_direct import examples,context,features,target_values


def check():
    dates=pd.date_range('2025-01-01','2025-03-31')
    history=pd.DataFrame(dict(route=1,date=dates,hour=0,boardings=1000+dates.dayofweek*10))
    weather=pd.DataFrame(dict(date=dates,temperature_2m_mean=10.,precipitation_sum=0.,daylight_duration=36000.))
    past,daily=examples(history,'2025-03-31',weather)
    assert set(past.origin.dt.strftime('%Y-%m-%d'))=={'2025-01-31','2025-02-28'}
    assert past.date.max()==pd.Timestamp('2025-03-31') and past.horizon.between(1,61).all()
    dates=pd.date_range('2025-02-01','2025-03-31')
    original=context(daily,'2025-01-31',dates,weather)
    poisoned=context(daily.assign(boardings=np.where(daily.date.gt('2025-01-31'),1e99,daily.boardings)),'2025-01-31',dates,weather)
    np.testing.assert_array_equal(features(original),features(poisoned))
    assert features(original).shape==(59,21)
    np.testing.assert_array_equal(features(original),features(original.assign(boardings=1e99)))
    np.testing.assert_array_equal(target_values(past,'count'),past.boardings/10000)
    np.testing.assert_array_equal(target_values(past,'ratio'),past.boardings/past.reference)
    try:target_values(past,'unknown')
    except ValueError:pass
    else:raise AssertionError('Invalid target accepted')
    print('causal origin context, partial horizon, target-free features and target checks passed')


if __name__=='__main__':check()
