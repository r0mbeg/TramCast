"""Verify frozen activity fit, source coverage and reconstruction scores."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from audit import ROOT,sha
from activity_experiment import load,TRAIN,TEST,compare
from activity_model import reconstruct,parameters


def verify(folder):
    manifest=json.loads((folder/'manifest.json').read_text())
    for name,digest in manifest['inputs'].items():
        assert sha(ROOT/name)==digest,name
    for name,digest in manifest['outputs'].items():
        assert sha(folder/name)==digest,name
    p=json.loads((folder/'parameters.json').read_text())
    s=json.loads((folder/'summary.json').read_text())
    select=json.loads((folder/'selection.json').read_text())
    f,h,r,_=load(TRAIN)
    indices=np.array(p['training_row_indices'])
    assert len(indices)==len(set(indices))==128
    assert f.iloc[indices].date.between('2025-09-08','2025-09-11').all()
    assert p['training_reference_events']==int(r[indices].sum())
    eligible=np.flatnonzero(f.date.between('2025-09-08','2025-09-11'))
    np.testing.assert_array_equal(indices,np.sort(np.random.default_rng(20260927).choice(eligible,128,replace=False)))
    test,th,tr,_=load(TEST)
    assert test.date.between('2025-09-15','2025-09-21').all()
    with np.load(folder/'high_activity_probabilities.npz',allow_pickle=False) as probs:
        for name,frame,held,ref in [('validation',f.loc[f.date.eq('2025-09-12')],h[f.date.eq('2025-09-12')],r[f.date.eq('2025-09-12')]),
                                  ('holdout',test,th,tr)]:
            scored=pd.read_csv(folder/f'{name}_scores.csv',sep=';')
            pd.testing.assert_frame_equal(scored[frame.columns].reset_index(drop=True),frame.reset_index(drop=True),check_exact=False)
            state=probs[name]
            assert state.shape==held.shape and np.isfinite(state).all()
            assert ((state>=0)&(state<=1)).all()
            _,factors,_=parameters(p['theta'])
            intensity=(1-state)*factors[0]+state*factors[1]
            density=.9*intensity/intensity.sum(1,keepdims=True)+.1/360
            np.testing.assert_allclose((held*np.log(density)).sum(1),scored.hsmm_log_score,atol=1e-10)
            # Independent replay of a deterministic spread of full forward/backward traces.
            for i in np.linspace(0,len(ref)-1,min(12,len(ref)),dtype=int):
                _,posterior,_=reconstruct(ref[i],p['theta'])
                np.testing.assert_allclose(posterior,state[i],atol=1e-12)
            for baseline,result in compare(scored).items():
                np.testing.assert_allclose(result['nats_per_event'],s[name][baseline]['nats_per_event'],atol=1e-12)
                np.testing.assert_allclose(result['cluster_bootstrap_95'],s[name][baseline]['cluster_bootstrap_95'],atol=1e-12)
    assert select['selected']==s['selected']==('hsmm' if select['hsmm_log_score']>select['fast_log_score'] else 'fast20s')
    assert s['stop_accuracy'] is None and s['stop_visits_verified']==0
    assert all(v['success'] for v in p['starts']) and max(v['projected_gradient_max'] for v in p['starts'])<.01
    hold_summary=json.loads((TEST/'summary.json').read_text())
    assert hold_summary['used_events']+sum(hold_summary['exclusions'].values())==hold_summary['pilot_clean_successes']
    assert int(th.sum()+tr.sum())==hold_summary['used_events']
    print('Verified hashes, training/validation/holdout split, state probabilities, scores and coverage.')


if __name__=='__main__':
    verify(Path(sys.argv[1]))
