"""P66: conditional absolute route-day volume errors. Built with PriorLabs-TabPFN."""
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
from experiments.portfolio_tabular_shape import training,DAY,HOURS,INNER,DAILY_SOURCE,TEACHERS
from experiments.portfolio_tabular_fraction import MODEL_SHA
from experiments.portfolio_school_fraction import features as daily_features,CALENDAR
from experiments.portfolio_bayes_hourly_target import hourly_targets
from experiments.portfolio_combine import mix,raw_frame
from experiments.portfolio_experiment import FINAL,load_history,run_study,save_candidate,write_json
from pipeline import KEYS

CONTROL=Path('artifacts/portfolio_20260926/continuation/tabular_shape/study/tabular_shape/selected')


def features(frame,shares,volume):
    result=np.column_stack([daily_features(frame),shares,np.log1p(volume)/10])
    if result.shape!=(len(frame),51) or not np.isfinite(result).all():raise ValueError('Invalid volume-error features')
    return result


def past_targets(history,cutoff,end):
    hourly,past,future,shares,errors,weights=training(history,cutoff,end)
    pieces=[]
    for origin in sorted(past.origin.unique()):
        origin=pd.Timestamp(origin);name=str(origin.date());stop=str((origin+pd.Timedelta(days=61)).date())
        if pd.Timestamp(stop)>pd.Timestamp(cutoff):raise ValueError('Future optimal-volume target')
        reference=raw_frame(INNER/name/f'raw_{name}.csv',name,stop)
        truth=history.loc[history.date.gt(origin)&history.date.le(stop),KEYS+['boardings']].reset_index(drop=True)
        pieces.append(hourly_targets(reference,truth).assign(origin=origin))
    targets=pd.concat(pieces,ignore_index=True)
    past=past.merge(targets,on=DAY,validate='one_to_one')
    if len(past)!=len(shares) or not np.isfinite(past.optimal_volume).all() or past.optimal_volume.lt(0).any():raise ValueError('Invalid paired volume targets')
    return hourly,past,future,shares


def fit(history,cutoff,end,target,out,model_path,seed,repeat=False):
    import torch
    from tabpfn import TabPFNRegressor
    if target not in ['actual','optimal']:raise ValueError('Unknown volume target')
    started=time.monotonic();hourly,past,future,shares=past_targets(history,cutoff,end)
    past['model_target']=((past.actual_day if target=='actual' else past.optimal_volume)-past.base_day)/10000
    base=raw_frame(CONTROL/f'raw_{cutoff}.csv',cutoff,end)
    future=future.copy().sort_values(['route','date']).reset_index(drop=True)
    future['origin']=pd.Timestamp(cutoff);future['network_base']=future.groupby('date').base.transform('sum');future['base_share']=future.base/future.network_base
    if 'boardings' in future or not future.date.gt(pd.Timestamp(cutoff)).all():raise ValueError('Future fact/date in model inputs')
    profile=base.pivot(index=['route','date'],columns='hour',values='prediction').reindex(columns=HOURS).rename_axis(columns=None)
    pd.testing.assert_frame_equal(profile.reset_index()[['route','date']],future[['route','date']])
    volume=profile.sum(axis=1).to_numpy();future_shares=np.divide(profile.to_numpy(),volume[:,None],out=np.zeros(profile.shape),where=volume[:,None]>0)
    active=future.route.ne(5).to_numpy()&(volume>0)
    X=features(past,shares,past.base_day.to_numpy());test=features(future.loc[active],future_shares[active],volume[active])
    model=TabPFNRegressor(n_estimators=4,categorical_features_indices=[0,1,2],model_path=str(model_path),device='cuda',
        inference_precision=torch.float32,fit_mode='low_memory',memory_saving_mode=True,random_state=seed,n_jobs=2)
    # ponytail: native prior is unweighted; revisit weighting if absolute volume corrections prove useful.
    torch.cuda.reset_peak_memory_stats();model.fit(X,past.model_target.to_numpy())
    predicted=model.predict(test,output_type='median')
    if repeat:np.testing.assert_array_equal(predicted,model.predict(test,output_type='median'))
    if not np.isfinite(predicted).all():raise ValueError('Invalid passenger correction')
    error=np.zeros(len(future));error[active]=predicted
    corrected=np.clip(volume+10000*error,.5*volume,2*volume)
    factor=np.divide(corrected,volume,out=np.ones(len(volume)),where=volume>0)
    future['source_volume']=volume;future['predicted_error']=error;future['corrected_volume']=corrected;future['factor']=factor
    raw=base.merge(future[['route','date','factor']],on=['route','date'],validate='many_to_one');raw['prediction']*=raw.factor;raw=raw.drop(columns='factor')
    if not raw.loc[raw.route.eq(5)|raw.hour.between(1,4),'prediction'].eq(0).all():raise ValueError('Changed structural zeros')
    out.mkdir(parents=True,exist_ok=True)
    hourly.to_csv(out/'hourly_training.csv',sep=';',index=False,date_format='%Y-%m-%d')
    past.to_csv(out/'daily_training.csv',sep=';',index=False,date_format='%Y-%m-%d')
    future.to_csv(out/'future.csv',sep=';',index=False,date_format='%Y-%m-%d')
    np.savez(out/'matrices.npz',X=X,y=past.model_target.to_numpy(),test=test,past_share=shares,future_share=future_shares,active=active,prediction=predicted)
    write_json(out/'fit.json',dict(cutoff=cutoff,target=target,seed=seed,rows=len(past),features=51,origins=int(past.origin.nunique()),
        earliest_target_date=str(past.date.min().date()),latest_target_date=str(past.date.max().date()),model_sha256=MODEL_SHA,
        target_formula=f'({target} volume minus causal teacher volume)/10000',sample_weight_used=False,n_estimators=4,output_type='median',
        repeated_prediction_exact=True if repeat else None,teacher_final_shape_family_mismatch=True,fit_seconds=time.monotonic()-started,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    del model;torch.cuda.empty_cache();return raw


def forecast(history,cutoff,end,recipe,out,model_path,seed=42,repeat=False):
    control=raw_frame(CONTROL/f'raw_{cutoff}.csv',cutoff,end)
    if recipe=='control':return control
    target,strength=recipe.split('_')
    if target not in ['actual','optimal'] or strength not in ['half','full']:raise ValueError('Unknown volume recipe')
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
    paths=[Path(args.history),Path(__file__),Path('pipeline.py'),model_path,CALENDAR,Path('artifacts/calendar_sources.json')]
    paths += [Path('experiments')/f'portfolio_{n}.py' for n in ['tabular_shape','tabular_fraction','school_fraction','bayes_hourly_target','bayes_direct','bayes_shape','operations','movement','ridge','route_allocation','nonlinear_errors','timesfm_errors','timesfm','bayes_volume','combine','experiment','windows']]
    paths += list(CONTROL.glob('raw_*.csv'))+list(INNER.glob('*/*.csv'))+list(INNER.glob('*/*.json'))+list(DAILY_SOURCE.glob('*/daily.csv'))+list(TEACHERS.glob('*/*.csv'))+list(TEACHERS.glob('*/*.json'))
    write_json(out/'run_started.json',dict(job_id=os.environ['SLURM_JOB_ID'],command=os.sys.argv,versions=dict(python=platform.python_version(),
        **{n:importlib.metadata.version(n) for n in ['torch','numpy','pandas','scikit-learn','tabpfn','optuna']}),sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))
    started=time.monotonic();history=load_history(args.history)
    if args.pilot:
        for target in ['actual','optimal']:forecast(history,*FINAL,target+'_full',out,model_path,repeat=True)
    else:
        run_study(history,out,'tabular_volume',lambda c,e,p:forecast(history,c,e,p['recipe'],out,model_path),
            dict(sampler='grid',space={'recipe':['control','actual_half','actual_full','optimal_half','optimal_full']},trials=5))
        selected=json.loads((out/'tabular_volume/selection.json').read_text())['params']['recipe']
        if selected=='control':
            trials=pd.read_csv(out/'tabular_volume/trials.csv',sep=';');selected=trials.loc[trials.params_recipe.ne('control')].sort_values('value',ascending=False).iloc[0].params_recipe
            (out/'alternative').mkdir(exist_ok=True);write_json(out/'alternative/parameters.json',dict(recipe=selected,selection='best new development score; no retuning'))
            save_candidate(history,out/'alternative','tabular_volume_alternative',lambda c,e:forecast(history,c,e,selected,out,model_path))
        (out/'seed73').mkdir(exist_ok=True);write_json(out/'seed73/parameters.json',dict(recipe=selected,seed=73,selection='fixed new recipe; no seed tuning'))
        save_candidate(history,out/'seed73','tabular_volume_seed73',lambda c,e:forecast(history,c,e,selected,out,model_path,73))
    write_json(out/'completed.json',dict(job_id=os.environ['SLURM_JOB_ID'],elapsed_seconds=time.monotonic()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--history',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--model-path',required=True);parser.add_argument('--pilot',action='store_true');run(parser.parse_args())
