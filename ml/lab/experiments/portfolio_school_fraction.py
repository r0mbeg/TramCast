"""P61: primary modular school-calendar proxy in fixed route-fraction errors."""
import argparse
from functools import lru_cache
import pandas as pd
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import platform
import resource
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from experiments.portfolio_experiment import FINAL,load_history,run_study,save_candidate,write_json
from experiments.portfolio_combine import mix,raw_frame
from experiments.portfolio_nonlinear_errors import features as daily_features
from experiments.portfolio_route_allocation import relative_targets,preserve_network
from experiments.portfolio_timesfm_errors import examples,SOURCE,TEACHERS,SHAPE

CALENDAR=Path("artifacts/portfolio_20260926/continuation/external/school_calendar/sources.json")
CONTROL=Path("artifacts/portfolio_20260926/continuation/route_fraction/study/route_fraction/selected")

@lru_cache(maxsize=1)
def calendar():
    data=json.loads(CALENDAR.read_text())
    sources={s["id"]:s for s in data["sources"]}
    intervals=[]
    for item in data["intervals"]:
        start,end=pd.Timestamp(item["start"]),pd.Timestamp(item["end"])
        known=pd.Timestamp(sources[item["source"]]["known_at"])
        if start>end or item["kind"] not in ["short","summer"] or known>start:
            raise ValueError("Invalid public calendar interval")
        intervals.append((start,end,item["kind"],known))
    return intervals

def calendar_features(frame):
    dates=pd.to_datetime(frame.date); origins=dates-np.asarray(frame.horizon,dtype="timedelta64[D]")
    coverage = json.loads(CALENDAR.read_text())["coverage"]
    if not dates.between(*coverage).all() or dates.le(origins).any():
        raise ValueError("Calendar date outside verified forecast coverage")
    result=np.zeros((len(frame),6)); result[:,2:4]=1.
    for start,end,kind,known in calendar():
        available=origins.ge(known).to_numpy(); k=0 if kind=="short" else 1
        result[:,k]=np.maximum(result[:,k],(dates.ge(start)&dates.le(end)).to_numpy()&available)
        result[:,4+k]=np.maximum(result[:,4+k],(origins.ge(start)&origins.le(end)).to_numpy()&available)
        distance=(dates-start).dt.days.to_numpy()
        result[:,2]=np.minimum(result[:,2],np.where(available&(distance>=0),np.minimum(distance,61)/61,1.))
        result[:,3]=np.minimum(result[:,3],np.where(available&(distance<0),np.minimum(-distance,61)/61,1.))
    return result


def features(frame):
    return np.column_stack([daily_features(frame),frame.base_share,calendar_features(frame)])


def prepare(past,cutoff):
    past=relative_targets(past,cutoff)
    past=past.loc[past.route.ne(5)&past.base.gt(0)&past.date.gt(np.datetime64(cutoff)-np.timedelta64(224,'D'))].copy()
    if past.empty:raise ValueError('No completed route fraction examples')
    past['base_share']=past.base/past.network_base
    past['target']=past.boardings/past.network_actual-past.base_share
    past['weight']=past.network_actual/past.groupby(['route','date']).base.transform('size')
    past['weight']/=past.weight.mean()
    return past


def fit(past,future,cutoff,out,seed):
    started=time.monotonic();past=prepare(past,cutoff)
    model=HistGradientBoostingRegressor(loss='absolute_error',max_iter=100,learning_rate=.05,
        max_leaf_nodes=15,min_samples_leaf=50,l2_regularization=1,
        categorical_features=[0,1,2],early_stopping=False,random_state=seed)
    model.fit(features(past),past.target,sample_weight=past.weight)
    future=future.copy()
    future['network_base']=future.groupby('date').base.transform('sum')
    if not future.network_base.gt(0).all():raise ValueError('Undefined future route fraction')
    future['base_share']=future.base/future.network_base
    active=future.route.ne(5)&future.base.gt(0)
    future['share_error']=0.
    future.loc[active,'share_error']=model.predict(features(future.loc[active]))
    future['estimated_share']=(future.base_share+future.share_error).clip(lower=0)
    future.loc[~active,'estimated_share']=0.
    if not np.isfinite(future.estimated_share).all():raise ValueError('Invalid estimated route share')
    out.mkdir(parents=True,exist_ok=True)
    past.to_csv(out/'training.csv',sep=';',index=False,date_format='%Y-%m-%d')
    future.to_csv(out/'future.csv',sep=';',index=False,date_format='%Y-%m-%d')
    payload=pickle.dumps(model,protocol=5);(out/'model.pkl').write_bytes(payload)
    np.testing.assert_array_equal(pickle.loads(payload).predict(features(future.loc[active])),future.loc[active,'share_error'])
    write_json(out/'fit.json',dict(cutoff=cutoff,rows=len(past),features=30,seed=seed,
        earliest_target_date=str(past.date.min().date()),latest_target_date=str(past.date.max().date()),
        origins=int(past.origin.nunique()),iterations=int(model.n_iter_),
        model_sha256=hashlib.sha256(payload).hexdigest(),model_reload_exact=True,
        loss='absolute route fraction error, actual network volume/repeat weights; daily L1 surrogate',
        fit_seconds=time.monotonic()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    return future[['route','date','estimated_share']]


def forecast(history,cutoff,end,recipe,out,seed=42):
    base=raw_frame(SHAPE/f'raw_{cutoff}.csv',cutoff,end)
    control=raw_frame(CONTROL/f'raw_{cutoff}.csv',cutoff,end)
    if recipe=='control':return control
    if recipe not in ['half','new']:raise ValueError('Unknown route fraction recipe')
    folder=out/f'fits_{seed}'/cutoff;path=folder/f'raw_{cutoff}.csv'
    if path.exists():learned=raw_frame(path,cutoff,end)
    else:
        past,future=examples(history,cutoff,end)
        shares=fit(past,future,cutoff,folder,seed)
        learned=base.merge(shares,on=['route','date'],validate='many_to_one')
        volume=base.groupby(['route','date']).prediction.transform('sum')
        learned['prediction']=base.prediction.div(volume.where(volume.gt(0))).fillna(0)*learned.estimated_share
        learned=preserve_network(base,learned.drop(columns='estimated_share'))
        learned.to_csv(path,sep=';',index=False,date_format='%Y-%m-%d')
    enhanced=mix(learned,base,.5)
    return mix(enhanced,control,.5 if recipe=='half' else 1.)


def run(args):
    if (os.environ.get('SLURM_JOB_PARTITION')!='ais-cpu' or int(os.environ['SLURM_CPUS_PER_TASK'])!=1
        or len(os.sched_getaffinity(0))>1):raise RuntimeError('Expected one bound ais-cpu core')
    if datetime.now(timezone.utc)>=datetime.fromisoformat('2026-09-27T15:40:40+00:00'):
        raise RuntimeError('Research reserve reached')
    out=Path(args.output)/('pilot' if args.pilot else 'study');out.mkdir(parents=True,exist_ok=True)
    paths=[Path(args.history),Path(__file__),Path('pipeline.py')]
    paths += [Path('experiments')/f'portfolio_{n}.py' for n in
        ['route_fraction','route_allocation','nonlinear_errors','timesfm_errors','timesfm','bayes_volume','combine','experiment','windows']]
    paths += [CALENDAR]+list(CALENDAR.parent.glob('*.pdf'))+list(CONTROL.glob('raw_*.csv'))
    paths += list(SOURCE.glob('*/daily.csv'))+list(TEACHERS.glob('*/*.csv'))+list(TEACHERS.glob('*/*.json'))+list(SHAPE.glob('raw_*.csv'))
    write_json(out/'run_started.json',dict(job_id=os.environ['SLURM_JOB_ID'],command=os.sys.argv,
        versions=dict(python=platform.python_version(),**{n:importlib.metadata.version(n) for n in
            ['numpy','pandas','scikit-learn','optuna']}),sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))
    started=time.monotonic();history=load_history(args.history)
    if args.pilot:forecast(history,*FINAL,'new',out)
    else:
        run_study(history,out,'school_fraction',lambda c,e,p:forecast(history,c,e,p['recipe'],out),
            dict(sampler='grid',space={'recipe':['control','half','new']},trials=3))
        recipe=json.loads((out/'school_fraction/selection.json').read_text())['params']['recipe']
        if recipe!='control':
            (out/'seed73').mkdir(exist_ok=True)
            write_json(out/'seed73/parameters.json',dict(recipe=recipe,seed=73,selection='fixed seed42 recipe; no tuning'))
            save_candidate(history,out/'seed73','school_fraction_seed73',lambda c,e:forecast(history,c,e,recipe,out,73))
    write_json(out/'completed.json',dict(job_id=os.environ['SLURM_JOB_ID'],elapsed_seconds=time.monotonic()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--pilot',action='store_true');run(parser.parse_args())
