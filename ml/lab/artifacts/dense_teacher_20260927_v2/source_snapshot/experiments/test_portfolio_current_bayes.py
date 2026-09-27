"""One risk preference/normalization/causal guard check for the reused engine."""
import numpy as np
import pandas as pd
from experiments.portfolio_bayes import infer_weights,MODELS


def check():
    frame=pd.MultiIndex.from_product([[1,7],pd.date_range('2025-05-01',periods=28),range(24)],
        names=['route','date','hour']).to_frame(index=False).assign(boardings=100)
    for i,name in enumerate(MODELS):frame[name]=100 if i==0 else 200
    posterior=infer_weights(frame,'2025-06-30')
    for item in posterior.values():
        weights=np.asarray(item['weights'])
        assert np.isfinite(weights).all() and (weights>=0).all() and np.isclose(weights.sum(),1)
        assert weights[0]>max(weights[1:])
    try:infer_weights(frame,'2025-05-27')
    except ValueError:pass
    else:raise AssertionError('Future loss accepted')
    print('Bayesian risk preference, normalization and causal guard passed')


if __name__=='__main__':check()
