"""Small check of the P56 fixed-shape daily-volume transform."""
from unittest.mock import patch
import numpy as np
import pandas as pd
from experiments.portfolio_tirex import hourly


def check():
    shape=pd.DataFrame(dict(route=[1,1,1,5,5,5],date=pd.to_datetime(['2025-05-01']*6),
        hour=[1,6,7,1,6,7],prediction=[0.,1.,3.,0.,0.,0.]))
    daily=pd.DataFrame(dict(route=[1,5],date=pd.to_datetime(['2025-05-01']*2),prediction=[100.,0.]))
    with patch('experiments.portfolio_tirex.raw_frame',return_value=shape), patch('experiments.portfolio_tirex.teacher_daily',return_value=daily) as teacher:
        result=hourly(pd.DataFrame(),'2025-04-30','2025-06-30',np.ones((9,61)),np.ones((9,61)))
    np.testing.assert_array_equal(result.prediction,[0.,25.,75.,0.,0.,0.])
    assert teacher.call_args.args[3].shape==(9,61,10)
    pd.testing.assert_frame_equal(result[['route','date','hour']],shape[['route','date','hour']])
    print('TiRex transform check passed: fixed shares, daily totals, route5/night zeros, grid.')


if __name__=='__main__':
    check()
