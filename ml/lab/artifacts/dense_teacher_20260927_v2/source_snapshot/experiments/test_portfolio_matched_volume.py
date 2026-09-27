"""One completed-horizon gate and teacher alignment check."""
import numpy as np
import pandas as pd
from experiments.portfolio_matched_volume import available_origins,align_teacher,HOURS
from experiments.test_portfolio_tabular_volume import check as feature_check


def check():
    expected=[[],['2025-04-30'],['2025-04-30'],['2025-04-30','2025-06-30'],
              ['2025-04-30','2025-06-30','2025-07-31','2025-08-31']]
    for cutoff,origins in zip(['2025-04-30','2025-06-30','2025-07-31','2025-08-31','2025-10-31'],expected):
        assert available_origins(cutoff)==origins
        assert all(pd.Timestamp(o)+pd.Timedelta(days=61)<=pd.Timestamp(cutoff) for o in origins)
    frame=pd.DataFrame(dict(origin=pd.Timestamp('2025-04-30'),route=1,date=pd.Timestamp('2025-05-01'),
                            hour=HOURS,prediction=5.,base_day=100.,actual_day=420.))
    past=frame[['origin','route','date','base_day','actual_day']].drop_duplicates()
    ref=frame.assign(prediction=np.arange(1,21)).iloc[::-1]
    hourly,paired,shares=align_teacher(frame,past,ref)
    assert paired.base_day.iloc[0]==210 and paired.actual_day.iloc[0]==420
    np.testing.assert_array_equal(hourly.prediction,np.arange(1,21))
    np.testing.assert_array_equal(shares[0],np.arange(1,21)/210)
    try:align_teacher(frame,past,ref.iloc[:-1])
    except ValueError:pass
    else:raise AssertionError('Missing matched teacher accepted')
    feature_check()
    print('completed-horizon gate and matched teacher check passed')


if __name__=='__main__':check()
