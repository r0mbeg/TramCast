"""Small check of the two fixed compositions and untouched control parents."""
import numpy as np
import pandas as pd
from experiments.portfolio_fraction_dynamics import compose


def check():
    base=pd.DataFrame(dict(route=[1,1,17,17,5],date=pd.Timestamp('2025-04-01'),hour=[1,6,6,7,6],prediction=[0.,100.,100.,200.,0.]))
    network=base.copy();network.loc[network.route.eq(1),'prediction']*=2
    fractions=base.copy();fractions.loc[fractions.route.eq(17),'prediction']*=2
    for recipe in ['network','relative']:
        result=compose(network,fractions,base,recipe)
        np.testing.assert_allclose(result.prediction.sum(),500.)
        assert result.loc[base.prediction.eq(0),'prediction'].eq(0).all()
        assert result.loc[result.route.eq(17)&result.hour.eq(7),'prediction'].iloc[0]==2*result.loc[result.route.eq(17)&result.hour.eq(6),'prediction'].iloc[0]
    np.testing.assert_allclose(compose(network,fractions,base,'network').prediction,[0.,500/7,1000/7,2000/7,0.])
    np.testing.assert_allclose(compose(network,fractions,base,'relative').prediction,[0.,125.,125.,250.,0.])
    pd.testing.assert_frame_equal(compose(network,fractions,base,'parent025'),network)
    pd.testing.assert_frame_equal(compose(network,fractions,base,'parent028'),fractions)
    invalid=fractions.copy();invalid.loc[invalid.route.eq(5),'prediction']=1.
    try:compose(network,invalid,base,'relative')
    except ValueError:pass
    else:raise AssertionError('Zero baseline route-day revived')
    print('Fraction dynamics checks passed: both compositions, fixed network totals/hour shares, zeros, exact parents.')


if __name__=='__main__':check()
