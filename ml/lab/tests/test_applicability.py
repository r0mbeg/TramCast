"""Run with PYTHONPATH=ml .venv/bin/python ml/tests/test_applicability.py."""
import numpy as np
import pandas as pd
from experiments.applicability import forecast, segments, measured, movement_matched
from experiments.portfolio_experiment import load_history
from pipeline import ROOT, postprocess


def check():
    history=load_history(ROOT/'artifacts/hourly_clean.csv')
    cutoff=pd.Timestamp('2025-06-30');end=cutoff+pd.Timedelta(days=7)
    changed=history.copy();changed.loc[changed.date.gt(cutoff),'boardings']=999999
    def predictor(tasks,horizon,cross):
        assert not cross
        return np.repeat(np.asarray([float(np.mean(t[-7:])) for t in tasks])[:,None],horizon,axis=1)
    for method in ['C0','C1']:
        pd.testing.assert_frame_equal(forecast(history,method,cutoff,end,predictor),forecast(changed,method,cutoff,end,predictor))
    for events in [[],['july'],['july','autumn']]:
        pd.testing.assert_frame_equal(movement_matched(history,cutoff,end,events),movement_matched(changed,cutoff,end,events))
    blocks=list(segments('2025-06-15','weekly'))
    assert len(blocks)==9 and sum((e-c).days for c,e in blocks)==61
    assert all(1<=(e-c).days<=7 for c,e in blocks)
    assert all(blocks[i+1][0]==blocks[i][1] for i in range(8))
    p=postprocess(forecast(history,'C0',cutoff,end)).merge(history,on=['route','date','hour'],validate='one_to_one')
    p['month']=p.date.dt.strftime('%Y-%m')
    for level in ['hourly','daily','route_calendar_month']:
        score=measured(p,level)
        assert score['actual_total']==p.boardings.sum() and score['predicted_total']==p.prediction.sum()
    assert measured(p.loc[p.route.eq(5)],'daily')['wape'] is None
    assert p.loc[p.route.eq(5)|p.hour.between(1,4),'prediction'].eq(0).all()
    print('PASS future mutation C0/C1/P10/P11, weekly cutoffs, integer aggregations, structural zero NA')


if __name__=='__main__':check()
