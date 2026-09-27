"""Small check of relative targets and network-volume preservation."""
import numpy as np
import pandas as pd
from experiments.portfolio_route_allocation import relative_targets,preserve_network
from pipeline import ROUTES


def check():
    past=pd.DataFrame(dict(route=ROUTES,date=pd.Timestamp('2025-03-01'),origin=pd.Timestamp('2025-01-31'),
        base=[0. if r==5 else 100. for r in ROUTES],boardings=[0. if r==5 else 200. for r in ROUTES]))
    past.loc[past.route.eq(17),'boardings']=400.
    result=relative_targets(past,'2025-03-01')
    np.testing.assert_allclose(result.relative_boardings.sum(),900.)
    assert result.loc[result.route.eq(17),'relative_boardings'].iloc[0]>100.
    np.testing.assert_allclose(result.relative_boardings,relative_targets(past.assign(boardings=past.boardings*3),'2025-03-01').relative_boardings)
    for invalid in [past.iloc[:-1],past.assign(date=pd.Timestamp('2025-03-02')),past.assign(boardings=0.)]:
        try: relative_targets(invalid,'2025-03-01')
        except ValueError: pass
        else: raise AssertionError('Invalid relative allocation target accepted')
    base=pd.DataFrame(dict(route=[1,1,17,17,5],date=pd.Timestamp('2025-04-01'),hour=[1,6,6,7,6],prediction=[0.,100.,100.,200.,0.]))
    learned=base.copy();learned.loc[learned.route.eq(17),'prediction']*=2
    corrected=preserve_network(base,learned)
    np.testing.assert_allclose(corrected.prediction.sum(),400.)
    assert corrected.loc[corrected.route.eq(1),'prediction'].sum()<100
    np.testing.assert_allclose(corrected.loc[corrected.route.eq(17),'prediction'].to_numpy(),[400/7,800/7]*np.array([2.,2.]))
    assert corrected.loc[base.prediction.eq(0),'prediction'].eq(0).all()
    try: preserve_network(base,learned.assign(prediction=0.))
    except ValueError: pass
    else: raise AssertionError('Positive network day erased')
    print('Route allocation checks passed: relative target, scale invariance, fixed network totals, hour shares, zeros, future guard.')


if __name__=='__main__':
    check()
