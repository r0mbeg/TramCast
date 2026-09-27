"""P62: learn aggregate network daily errors, preserve candidate029 within-day shares."""
import argparse
from datetime import datetime, timezone
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
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from experiments.portfolio_bayes_volume import calibrate
from experiments.portfolio_combine import mix, raw_frame
from experiments.portfolio_experiment import FINAL, load_history, run_study, save_candidate, write_json
from experiments.portfolio_nonlinear_errors import features as daily_features
from experiments.portfolio_school_fraction import CALENDAR, calendar_features
from experiments.portfolio_timesfm_errors import examples, SOURCE, TEACHERS
from pipeline import ROUTES

CONTROL=Path('artifacts/portfolio_20260926/continuation/school_fraction/study/school_fraction/selected')


def aggregate(frame, cutoff, observed):
    keys=['origin','date']
    if (frame.empty or frame.duplicated(keys+['route']).any() or not frame.date.gt(frame.origin).all()
        or not frame.route.isin(ROUTES).all()):
        raise ValueError('Invalid network forecast grid')
    if observed:
        if frame.date.max()>pd.Timestamp(cutoff):raise ValueError('Future target entered network aggregation')
    elif 'boardings' in frame or not frame.origin.eq(pd.Timestamp(cutoff)).all():
        raise ValueError('Future input contains a target or wrong origin')
    groups=frame.groupby(keys,sort=True)
    if not groups.route.nunique().eq(10).all():raise ValueError('Incomplete network routes')
    common=['effective_weekday','daytype','horizon','off','summer','temperature_2m_mean',
        'precipitation_sum','daylight_duration']
    if groups[common].nunique(dropna=False).gt(1).any().any():raise ValueError('Inconsistent common covariates')
    volumes=['base','ridge','regime','direct','timesfm']+(['boardings'] if observed else [])
    if not np.isfinite(frame[volumes]).all().all() or frame[volumes].lt(0).any().any():
        raise ValueError('Invalid network daily counts')
    result=groups[volumes].sum().join(groups[common].first())
    if not result.base.gt(0).all():raise ValueError('Nonpositive network baseline')
    for days in [7,14,28]:
        col=f'ratio{days}'
        weighted=frame.assign(weighted=frame.base*frame[col]).groupby(keys).weighted.sum()
        result[col]=weighted/result.base
    # Synthetic route0 is only the aggregate row identifier for the existing calibrator.
    return result.reset_index().assign(route=0)


def features(frame, bayesian=False):
    values=np.column_stack([daily_features(frame)[:,1:],calendar_features(frame)])
    if bayesian:
        categories=[frame.effective_weekday.eq(d).to_numpy(float) for d in range(1,7)]
        categories += [frame.daytype.eq(d).to_numpy(float) for d in [1,2]]
        values=np.column_stack(categories+[values[:,2:]])
    if not np.isfinite(values).all():raise ValueError('Invalid network features')
    return values


def fit(past, future, cutoff, model_name, out, seed):
    started=time.monotonic()
    past=aggregate(past,cutoff,True)
    past=past.loc[past.date.gt(pd.Timestamp(cutoff)-pd.Timedelta(days=224))].copy()
    future=aggregate(future.assign(origin=pd.Timestamp(cutoff)),cutoff,False)
    if past.empty:raise ValueError('No completed network errors')
    if model_name=='bayes':
        factor,summary=calibrate(past,future,cutoff,
            dict(strength=1.,uncertainty=1,history_days=224,volume_weight=True),
            feature_fn=lambda f:features(f,True))
    elif model_name=='absolute':
        past['target']=past.boardings/past.base
        past['weight']=past.base/past.groupby('date').base.transform('size')
        past['weight']/=past.weight.mean()
        model=HistGradientBoostingRegressor(loss='absolute_error',max_iter=100,learning_rate=.05,
            max_leaf_nodes=7,min_samples_leaf=20,l2_regularization=1,
            categorical_features=[0,1],early_stopping=False,random_state=seed)
        model.fit(features(past),past.target,sample_weight=past.weight)
        future['ratio']=model.predict(features(future))
        factor=future.ratio.clip(.5,2).to_numpy()
        out.mkdir(parents=True,exist_ok=True)
        payload=pickle.dumps(model,protocol=5);(out/'model.pkl').write_bytes(payload)
        np.testing.assert_array_equal(pickle.loads(payload).predict(features(future)),future.ratio)
        summary=dict(model_sha256=hashlib.sha256(payload).hexdigest(),model_reload_exact=True,
            iterations=int(model.n_iter_),seed=seed,loss='absolute actual/base ratio weighted networkbase/repeats')
    else:raise ValueError('Unknown network daily model')
    future['factor']=factor;future['prediction']=future.base*factor
    if not np.isfinite(future.prediction).all() or not future.prediction.ge(0).all():
        raise ValueError('Invalid network prediction')
    out.mkdir(parents=True,exist_ok=True)
    past.to_csv(out/'training.csv',sep=';',index=False,date_format='%Y-%m-%d')
    future.to_csv(out/'future.csv',sep=';',index=False,date_format='%Y-%m-%d')
    summary.update(cutoff=cutoff,model=model_name,features=features(future,model_name=='bayes').shape[1],
        rows=len(past),latest_target_date=str(past.date.max().date()),origins=int(past.origin.nunique()),
        fit_seconds=time.monotonic()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_json(out/'fit.json',summary)
    return future[['date','prediction']]


def forecast(history, cutoff, end, recipe, out, seed=42):
    shape=raw_frame(CONTROL/f'raw_{cutoff}.csv',cutoff,end)
    if recipe=='control':return shape
    model_name,strength=recipe.split('_')
    if strength not in ['half','full']:raise ValueError('Unknown network strength')
    folder=out/f'fits_{model_name}_{seed}'/cutoff;path=folder/f'raw_{cutoff}.csv'
    if path.exists():learned=raw_frame(path,cutoff,end)
    else:
        past,future=examples(history,cutoff,end)
        daily=fit(past,future,cutoff,model_name,folder,seed)
        learned=shape.merge(daily.rename(columns={'prediction':'network_volume'}),on='date',validate='many_to_one')
        denominator=shape.groupby('date').prediction.transform('sum')
        if not denominator.gt(0).all():raise ValueError('Undefined network day shares')
        learned['prediction']*=learned.network_volume/denominator
        learned=learned.drop(columns='network_volume')
        learned.to_csv(path,sep=';',index=False,date_format='%Y-%m-%d')
    return mix(learned,shape,.5 if strength=='half' else 1.)


def run(args):
    if (os.environ.get('SLURM_JOB_PARTITION')!='ais-cpu' or int(os.environ['SLURM_CPUS_PER_TASK'])!=1
        or len(os.sched_getaffinity(0))>1):raise RuntimeError('Expected one bound ais-cpu core')
    if datetime.now(timezone.utc)>=datetime.fromisoformat('2026-09-27T15:40:40+00:00'):
        raise RuntimeError('Research reserve reached')
    out=Path(args.output)/('pilot' if args.pilot else 'study');out.mkdir(parents=True,exist_ok=True)
    paths=[Path(args.history),Path(__file__),Path('pipeline.py'),CALENDAR]
    paths += [Path('experiments')/f'portfolio_{n}.py' for n in
        ['bayes_volume','nonlinear_errors','school_fraction','timesfm_errors','timesfm','combine','experiment','windows']]
    paths += list(CALENDAR.parent.glob('*.pdf'))+list(CONTROL.glob('raw_*.csv'))
    paths += list(SOURCE.glob('*/daily.csv'))+list(TEACHERS.glob('*/*.csv'))+list(TEACHERS.glob('*/*.json'))
    write_json(out/'run_started.json',dict(job_id=os.environ['SLURM_JOB_ID'],command=os.sys.argv,
        versions=dict(python=platform.python_version(),**{n:importlib.metadata.version(n) for n in
            ['numpy','pandas','scikit-learn','optuna']}),sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))
    started=time.monotonic();history=load_history(args.history)
    if args.pilot:
        for model in ['bayes','absolute']:forecast(history,*FINAL,f'{model}_full',out)
    else:
        run_study(history,out,'network_volume',lambda c,e,p:forecast(history,c,e,p['recipe'],out),
            dict(sampler='grid',space={'recipe':['control','bayes_half','bayes_full','absolute_half','absolute_full']},trials=5))
        recipe=json.loads((out/'network_volume/selection.json').read_text())['params']['recipe']
        selected=out/'network_volume/selected'
        if recipe=='control':
            import optuna
            study=optuna.load_study(study_name='network_volume',storage=f"sqlite:///{out/'network_volume/study.db'}")
            best=max((t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE
                and t.params['recipe']!='control'),key=lambda t:t.value)
            recipe=best.params['recipe'];selected=out/'alternative'
            selected.mkdir(exist_ok=True)
            write_json(selected/'parameters.json',dict(params=best.params,trial=best.number,value=best.value,selection='bestnew W1/W2; parent stilloverallwinner'))
            save_candidate(history,selected,'network_volume_alternative',lambda c,e:forecast(history,c,e,recipe,out))
        if recipe.startswith('absolute'):
            (out/'seed73').mkdir(exist_ok=True)
            write_json(out/'seed73/parameters.json',dict(recipe=recipe,seed=73,reference=str(selected),selection='fixedseed42recipe; no tuning'))
            save_candidate(history,out/'seed73','network_volume_seed73',lambda c,e:forecast(history,c,e,recipe,out,73))
    write_json(out/'completed.json',dict(job_id=os.environ['SLURM_JOB_ID'],elapsed_seconds=time.monotonic()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--pilot',action='store_true');run(parser.parse_args())
