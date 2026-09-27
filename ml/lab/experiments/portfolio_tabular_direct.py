"""P69: direct origin-to-day counts. Built with PriorLabs-TabPFN."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import time

import numpy as np
import pandas as pd
from experiments.portfolio_direct_daily import daily_history,context,FEATURES
from experiments.portfolio_school_fraction import calendar_features,CALENDAR
from experiments.portfolio_tabular_fraction import MODEL_SHA
from experiments.portfolio_tabular_volume import CONTROL
from experiments.portfolio_operations import normal_history,apply_operations
from experiments.portfolio_weather_volume import WEATHER_PATH
from experiments.portfolio_combine import raw_frame,mix
from experiments.portfolio_experiment import FINAL,load_history,run_study,save_candidate,write_json
from pipeline import KEYS

ROOT=Path('artifacts/portfolio_20260926/continuation')
OPS=ROOT/'operations/operations/selection.json'
AUGUST=ROOT/'august_operations/august_operations/selection.json'
COLUMNS=[c for c in FEATURES if c!='season']


def features(frame):
    matrix=np.column_stack([frame[COLUMNS].to_numpy(float),calendar_features(frame)])
    if matrix.shape!=(len(frame),21) or not np.isfinite(matrix).all():raise ValueError('Invalid direct count features')
    return matrix


def examples(history,cutoff,weather):
    daily=daily_history(history,cutoff,august7=True,july_verified=True);pieces=[]
    for origin in pd.date_range('2025-01-31',pd.Timestamp(cutoff)-pd.Timedelta(days=1),freq='ME'):
        stop=min(origin+pd.Timedelta(days=61),pd.Timestamp(cutoff))
        frame=context(daily,origin,pd.date_range(origin+pd.Timedelta(days=1),stop),weather)
        pieces.append(frame.merge(daily[['route','date','boardings']],on=['route','date'],validate='one_to_one'))
    if not pieces:raise ValueError('No observed direct count examples')
    past=pd.concat(pieces,ignore_index=True).sort_values(['origin','route','date']).reset_index(drop=True)
    if past.empty or past.date.max()>pd.Timestamp(cutoff) or not past.horizon.between(1,61).all() or not past.date.gt(past.origin).all():raise ValueError('Future target in direct count examples')
    if past.duplicated(['origin','route','date']).any() or not np.isfinite(past.boardings).all() or past.boardings.lt(0).any():raise ValueError('Invalid direct count targets')
    return past,daily


def target_values(past,target):
    if target not in ['count','ratio']:raise ValueError('Unknown direct count target')
    return past.boardings.to_numpy()/(10000 if target=='count' else past.reference.to_numpy())


def fit(history,cutoff,end,target,out,model_path,seed,repeat=False):
    import torch
    from tabpfn import TabPFNRegressor
    started=time.monotonic();weather=pd.read_csv(WEATHER_PATH,sep=';',parse_dates=['date'])
    past,daily=examples(history,cutoff,weather)
    future=context(daily,cutoff,pd.date_range(pd.Timestamp(cutoff)+pd.Timedelta(days=1),end),weather)
    if 'boardings' in future or not future.date.gt(pd.Timestamp(cutoff)).all():raise ValueError('Future target/date violation')
    X=features(past);Y=target_values(past,target);test=features(future)
    model=TabPFNRegressor(n_estimators=4,categorical_features_indices=[0,1],model_path=str(model_path),device='cuda',
        inference_precision=torch.float32,fit_mode='low_memory',memory_saving_mode=True,random_state=seed,n_jobs=2)
    # ponytail: monthly overlapping examples are unweighted in the native prior; revisit if direct counts succeed.
    torch.cuda.reset_peak_memory_stats();model.fit(X,Y);prediction=model.predict(test,output_type='median')
    if repeat:np.testing.assert_array_equal(prediction,model.predict(test,output_type='median'))
    if not np.isfinite(prediction).all():raise ValueError('Invalid predicted count')
    future['normal_volume']=np.maximum(0,prediction)*(10000 if target=='count' else future.reference.to_numpy())
    params={**json.loads(OPS.read_text())['params'],**json.loads(AUGUST.read_text())['params'],'july_verified':True}
    normal=normal_history(history,cutoff,extended=True,august7=True,july_verified=True)
    operated=apply_operations(future[['route','date']].assign(hour=0,prediction=future.normal_volume),history.loc[history.date.le(cutoff)],normal,params)
    future['operated_volume']=operated.prediction.to_numpy()
    base=raw_frame(CONTROL/f'raw_{cutoff}.csv',cutoff,end)
    denominator=base.groupby(['route','date']).prediction.transform('sum')
    raw=base.merge(future[['route','date','operated_volume']],on=['route','date'],how='left',validate='many_to_one')
    pd.testing.assert_frame_equal(raw[KEYS],base[KEYS])
    raw['prediction']=base.prediction.div(denominator.where(denominator.gt(0))).fillna(0)*raw.operated_volume.fillna(0)
    raw=raw[KEYS+['prediction']]
    if not raw.loc[base.prediction.eq(0),'prediction'].eq(0).all():raise ValueError('Revived inactive hour/day')
    if not np.isfinite(raw.prediction).all() or raw.prediction.lt(0).any():raise ValueError('Invalid operated hourly count')
    out.mkdir(parents=True,exist_ok=True)
    past.to_csv(out/'training.csv',sep=';',index=False,date_format='%Y-%m-%d')
    future.to_csv(out/'future.csv',sep=';',index=False,date_format='%Y-%m-%d')
    np.savez(out/'matrices.npz',X=X,y=Y,test=test,prediction=prediction)
    write_json(out/'fit.json',dict(cutoff=cutoff,target=target,seed=seed,rows=len(past),features=21,origins=int(past.origin.nunique()),
        earliest_target_date=str(past.date.min().date()),latest_target_date=str(past.date.max().date()),model_sha256=MODEL_SHA,
        categorical_features=[0,1],n_estimators=4,output_type='median',sample_weight_used=False,operation_params=params,
        repeated_prediction_exact=True if repeat else None,fit_seconds=time.monotonic()-started,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    del model;torch.cuda.empty_cache();return raw


def forecast(history,cutoff,end,recipe,out,model_path,seed=42,repeat=False):
    control=raw_frame(CONTROL/f'raw_{cutoff}.csv',cutoff,end)
    if recipe=='control':return control
    target,strength=recipe.split('_')
    if target not in ['count','ratio'] or strength not in ['half','full']:raise ValueError('Unknown direct recipe')
    folder=out/f'fits_{target}_{seed}'/cutoff;path=folder/f'raw_{cutoff}.csv'
    if path.exists():learned=raw_frame(path,cutoff,end)
    else:
        learned=fit(history,cutoff,end,target,folder,model_path,seed,repeat)
        learned.to_csv(path,sep=';',index=False,date_format='%Y-%m-%d')
    return mix(learned,control,.5 if strength=='half' else 1.)


def run(args):
    if os.environ.get('SLURM_JOB_PARTITION')!='gpu_devel' or int(os.environ.get('SLURM_CPUS_PER_TASK','0'))!=2 or len(os.sched_getaffinity(0))>2:raise RuntimeError('Expected two bound gpu_devel cores')
    if datetime.now(timezone.utc)>=datetime.fromisoformat('2026-09-27T15:40:40+00:00'):raise RuntimeError('Research reserve reached')
    model_path=Path(args.model_path)
    if not model_path.is_file() or hashlib.sha256(model_path.read_bytes()).hexdigest()!=MODEL_SHA:raise ValueError('Unverified checkpoint')
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
    out=Path(args.output)/('pilot' if args.pilot else 'study');out.mkdir(parents=True,exist_ok=True)
    paths=[Path(args.history),Path(__file__),Path('pipeline.py'),model_path,Path(WEATHER_PATH),CALENDAR,Path('artifacts/calendar_sources.json'),OPS,AUGUST]
    paths+=[Path('experiments')/f'portfolio_{n}.py' for n in ['tabular_fraction','tabular_volume','tabular_shape','direct_daily','school_fraction','operations','movement','ridge','structure','weather_volume','combine','experiment']]
    paths+=[Path('experiments/calendar_experiment.py')]+list(CONTROL.glob('raw_*.csv'))
    write_json(out/'run_started.json',dict(job_id=os.environ['SLURM_JOB_ID'],command=os.sys.argv,versions=dict(python=platform.python_version(),
        **{n:importlib.metadata.version(n) for n in ['torch','numpy','pandas','scikit-learn','tabpfn','optuna']}),
        old_tabpfn_gpu_seconds=711,old_tabpfn_trials=13,new_trial_limit=5,total_tabpfn_trials=18,new_allocation_limit_seconds=480,
        sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))
    started=time.monotonic();history=load_history(args.history)
    if args.pilot:
        for target in ['count','ratio']:forecast(history,*FINAL,target+'_full',out,model_path,repeat=True)
    else:
        run_study(history,out,'tabular_direct',lambda c,e,p:forecast(history,c,e,p['recipe'],out,model_path),
            dict(sampler='grid',space={'recipe':['control','count_half','count_full','ratio_half','ratio_full']},trials=5,
                 previous_tabpfn_trials=13,previous_tabpfn_gpu_seconds=711,new_allocation_subbudget_seconds=480))
        selected=json.loads((out/'tabular_direct/selection.json').read_text())['params']['recipe']
        if selected=='control':
            trials=pd.read_csv(out/'tabular_direct/trials.csv',sep=';');selected=trials.loc[trials.params_recipe.ne('control')].sort_values('value',ascending=False).iloc[0].params_recipe
            (out/'alternative').mkdir(exist_ok=True);write_json(out/'alternative/parameters.json',dict(recipe=selected,selection='best new development score; no retuning'))
            save_candidate(history,out/'alternative','tabular_direct_alternative',lambda c,e:forecast(history,c,e,selected,out,model_path))
        (out/'seed73').mkdir(exist_ok=True);write_json(out/'seed73/parameters.json',dict(recipe=selected,seed=73,selection='fixed new recipe; no seed selection'))
        save_candidate(history,out/'seed73','tabular_direct_seed73',lambda c,e:forecast(history,c,e,selected,out,model_path,73))
    write_json(out/'completed.json',dict(job_id=os.environ['SLURM_JOB_ID'],elapsed_seconds=time.monotonic()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--history',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--model-path',required=True);parser.add_argument('--pilot',action='store_true')
    run(parser.parse_args())
