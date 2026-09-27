"""Train shared activity durations; frozen validation and fresh device holdout."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from audit import ROOT,sha,write_json
from activity_model import fit,reconstruct

HERE=Path(__file__).resolve().parent
TRAIN=HERE/'runs/device-signal-20260927-v1'
TEST=HERE/'runs/device-holdout-20260927-v1'


def load(folder):
    manifest=json.loads((folder/'manifest.json').read_text())
    names=['device_hour_scores.csv','binned_counts.npz','summary.json']
    for name in names:
        if sha(folder/name)!=manifest['outputs'][name]:
            raise ValueError('Changed input '+name)
    frame=pd.read_csv(folder/'device_hour_scores.csv',sep=';')
    with np.load(folder/'binned_counts.npz',allow_pickle=False) as f:
        held,reference=f['held'],f['reference']
    if held.shape!=reference.shape or held.shape!=(len(frame),360):
        raise ValueError('Mismatched count matrices')
    return frame,held,reference,[folder/n for n in names+['manifest.json']]


def compare(frame):
    result={}
    for baseline in ['fast','slow','uniform']:
        f=frame.assign(delta=frame.hsmm_log_score-frame[baseline+'_log_score'])
        group=f.groupby('vehicle_day')[['delta','held_events']].sum().to_numpy()
        rng=np.random.default_rng(20260927)
        boot=[]
        for _ in range(1000):
            sample=group[rng.integers(0,len(group),len(group))].sum(0)
            boot.append(sample[0]/sample[1])
        result[baseline]=dict(nats_per_event=float(f.delta.sum()/f.held_events.sum()),
            cluster_bootstrap_95=np.quantile(boot,[.025,.975]).tolist(),
            positive_hour_fraction=float(f.delta.gt(0).mean()))
    return result


def evaluate(frame,held,reference,theta):
    frame=frame.copy();scores=[];states=[]
    for h,r in zip(held,reference):
        density,high,_=reconstruct(r,theta)
        scores.append(float(h@np.log(density)));states.append(high)
    frame['hsmm_log_score']=scores
    return frame,np.asarray(states)


def run(out):
    frame,held,reference,inputs=load(TRAIN)
    eligible=np.flatnonzero(frame.date.between('2025-09-08','2025-09-11'))
    if len(eligible)<128:
        raise ValueError('Insufficient training hours')
    indices=np.sort(np.random.default_rng(20260927).choice(eligible,128,replace=False))
    out.mkdir(parents=True,exist_ok=False)
    theta,diagnostics=fit(reference[indices])
    write_json(out/'parameters.json',dict(version='A20260927-v1',**diagnostics,
        training_row_indices=indices.tolist(),training_hours=128,training_reference_events=int(reference[indices].sum())))
    val_mask=frame.date.eq('2025-09-12').to_numpy()
    validation,val_states=evaluate(frame.loc[val_mask],held[val_mask],reference[val_mask],theta)
    selected='hsmm' if validation.hsmm_log_score.sum()>validation.fast_log_score.sum() else 'fast20s'
    write_json(out/'selection.json',dict(selected=selected,validation_date='2025-09-12',
        hsmm_log_score=float(validation.hsmm_log_score.sum()),fast_log_score=float(validation.fast_log_score.sum())))
    print('Frozen validation selection:',selected,flush=True)
    test,h,r,paths=load(TEST);inputs+=paths
    test,test_states=evaluate(test,h,r,theta)
    validation.to_csv(out/'validation_scores.csv',sep=';',index=False)
    test.to_csv(out/'holdout_scores.csv',sep=';',index=False)
    np.savez_compressed(out/'high_activity_probabilities.npz',validation=val_states,holdout=test_states)
    summary=dict(version='A20260927-v1',selected=selected,validation=compare(validation),holdout=compare(test),
        holdout_per_day={day:compare(g) for day,g in test.groupby('date')},
        holdout_hours=len(test),holdout_vehicle_days=int(test.vehicle_day.nunique()),
        holdout_held_events=int(test.held_events.sum()),holdout_reference_events=int(test.reference_events.sum()),
        stop_accuracy=None,stop_visits_verified=0,status='learned_activity_regimes_not_geographic_stop_model')
    write_json(out/'summary.json',summary)
    inputs+=[HERE/n for n in ['activity_experiment.py','activity_model.py','ACTIVITY_PROTOCOL.md','duration_model.py','audit.py']]
    write_json(out/'manifest.json',dict(version='A20260927-v1',inputs={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps({k:v for k,v in summary.items() if k!='holdout_per_day'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    with threadpool_limits(limits=1):
        run(parser.parse_args().out)
