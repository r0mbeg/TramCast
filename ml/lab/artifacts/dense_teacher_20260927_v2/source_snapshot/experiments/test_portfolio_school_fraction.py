"""Small calendar coverage, boundaries, availability and target-poison check."""
import numpy as np
import pandas as pd
from experiments.portfolio_school_fraction import calendar_features

def check():
    frame=pd.DataFrame(dict(date=pd.to_datetime(['2025-02-15','2025-02-16','2025-02-23',
        '2025-02-24','2025-05-30','2025-05-31','2025-08-31','2025-09-01',
        '2025-10-04','2025-10-12','2025-10-13','2025-11-15','2025-11-23','2025-12-31']),horizon=30))
    values=calendar_features(frame)
    np.testing.assert_array_equal(values[:,0],[0,1,1,0,0,0,0,0,1,1,0,1,1,1])
    np.testing.assert_array_equal(values[:,1],[0,0,0,0,0,1,1,0,0,0,0,0,0,0])
    assert values.shape==(14,6) and (values>=0).all() and (values<=1).all()
    np.testing.assert_array_equal(values,calendar_features(frame.assign(boardings=1e100)))
    # October schedule was not yet published at a hypothetical May1 origin.
    later=pd.DataFrame(dict(date=pd.to_datetime(['2025-10-04']),horizon=156))
    assert calendar_features(later)[0,0]==0
    assert values[1,2]==0 and values[2,2]==7/61
    for date,horizon in [('2026-01-01',30),('2025-01-01',0)]:
        try:calendar_features(pd.DataFrame(dict(date=pd.to_datetime([date]),horizon=horizon)))
        except ValueError:pass
        else:raise AssertionError('Invalid calendar input accepted')
    print('calendar boundaries/availability/coverage/poison checks passed')

if __name__=='__main__':check()
