"""One runnable check for weighted error modes, rank cap, and target-free features."""
import numpy as np
import pandas as pd
from experiments.portfolio_tabular_shape import basis,HOURS
from experiments.portfolio_school_fraction import features
from experiments.portfolio_timesfm_errors import daily


def check():
    rng=np.random.default_rng(42);errors=rng.normal(0,.01,(50,len(HOURS)));errors-=errors.mean(axis=1)[:,None]
    weights=np.arange(1,51,dtype=float);mean,components,coordinates,values=basis(errors,weights)
    assert components.shape==(6,len(HOURS)) and coordinates.shape==(50,6)
    np.testing.assert_allclose(components@components.T,np.eye(6),rtol=0,atol=1e-12)
    np.testing.assert_allclose(mean,np.average(errors,axis=0,weights=weights))
    np.testing.assert_allclose(coordinates,(errors-mean)@components.T)
    assert (components[np.arange(6),np.argmax(abs(components),axis=1)]>=0).all()
    zeros=basis(np.zeros((5,len(HOURS))),np.ones(5));assert zeros[1].shape==(0,len(HOURS))
    one=errors[:1]*np.arange(1,51)[:,None];assert len(basis(one,weights)[1])==1
    for bad,w in [(errors,np.zeros(50)),(errors+1,weights),(errors[:,:-1],weights)]:
        try:basis(bad,w)
        except (ValueError,AssertionError):pass
        else:raise AssertionError('Invalid error matrix accepted')
    f=daily('2025-01-31','2025-04-02').iloc[:10].copy();f['base_share']=.1
    np.testing.assert_array_equal(features(f),features(f.assign(boardings=1e99)))
    print('profile modes/rank/orthogonality/zero/poison checks passed')

if __name__=='__main__':check()
