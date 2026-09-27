"""P67: errors of the same fixed 030 forecaster. Built with PriorLabs-TabPFN."""
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
from experiments.portfolio_tabular_volume import features
from experiments.portfolio_combine import mix,raw_frame
from experiments.portfolio_experiment import FINAL,load_history,run_study,save_candidate,write_json
from pipeline import KEYS

CONTROL=Path('artifacts/portfolio_20260926/continuation/tabular_shape/study/tabular_shape/selected')


ORIGINS=['2025-04-30','2025-06-30','2025-07-31','2025-08-31']
FAMILY_STARTED=1790489139.482144


def available_origins(cutoff):
    return [o for o in ORIGINS if pd.Timestamp(o)+pd.Timedelta(days=61)<=pd.Timestamp(cutoff)]


def align_teacher(hourly,past,reference):
    reference=reference[DAY+['hour','prediction']]
    hourly=hourly.drop(columns=['prediction','target','weight'],errors='ignore').merge(reference,on=DAY+['hour'],how='left',validate='one_to_one').sort_values(DAY+['hour']).reset_index(drop=True)
    if not np.isfinite(hourly.prediction).all() or hourly.prediction.lt(0).any():raise ValueError('Missing/invalid matched teacher')
    hourly['base_day']=hourly.groupby(DAY).prediction.transform('sum')
    if not hourly.base_day.gt(0).all():raise ValueError('Nonpositive matched teacher')
    index=hourly[DAY+['base_day']].drop_duplicates(DAY)
    past=past.drop(columns='base_day').merge(index,on=DAY,validate='one_to_one').sort_values(DAY).reset_index(drop=True)
    shares=hourly.pivot(index=DAY,columns='hour',values='prediction').reindex(columns=HOURS).to_numpy()/past.base_day.to_numpy()[:,None]
    if shares.shape!=(len(past),20) or not np.isfinite(shares).all():raise ValueError('Incomplete matched profile')
    np.testing.assert_allclose(shares.sum(axis=1),1,rtol=0,atol=1e-12)
    return hourly,past,shares


def past_targets(history,cutoff,end):
    origins=available_origins(cutoff)
    if not origins:raise ValueError('No completed exact030 teacher')
    hourly,past,future,_,_,_=training(history,cutoff,end)
    hourly=hourly.loc[hourly.origin.isin(pd.to_datetime(origins))].copy()
    past=past.loc[past.origin.isin(pd.to_datetime(origins))].copy()
    refs=[]
    for origin in origins:
        stop=str((pd.Timestamp(origin)+pd.Timedelta(days=61)).date())
        refs.append(raw_frame(CONTROL/f'raw_{origin}.csv',origin,stop).assign(origin=pd.Timestamp(origin)))
    hourly,past,shares=align_teacher(hourly,past,pd.concat(refs,ignore_index=True))
    if past.empty or past.date.max()>pd.Timestamp(cutoff):raise ValueError('Invalid matched past dates')
    return hourly,past,future,shares


def fit(history,cutoff,end,target,out,model_path,seed,repeat=False):
    import torch
    from tabpfn import TabPFNRegressor
    if target!='actual':raise ValueError('Unknown volume target')
    started=time.monotonic();hourly,past,future,shares=past_targets(history,cutoff,end)
    past['model_target']=(past.actual_day-past.base_day)/10000
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
        target_formula='(actual volume minus exact030 causal teacher volume)/10000',sample_weight_used=False,n_estimators=4,output_type='median',
        repeated_prediction_exact=True if repeat else None,teacher_final_shape_family_mismatch=False,fit_seconds=time.monotonic()-started,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    del model;torch.cuda.empty_cache();return raw


def forecast(history,cutoff,end,recipe,out,model_path,seed=42,repeat=False):
    control=raw_frame(CONTROL/f'raw_{cutoff}.csv',cutoff,end)
    if recipe=='control':return control
    if recipe!='matched':raise ValueError('Unknown matched recipe')
    if not available_origins(cutoff):
        folder=out/f'fits_actual_{seed}'/cutoff;folder.mkdir(parents=True,exist_ok=True)
        write_json(folder/'gate.json',dict(cutoff=cutoff,eligible_origins=[],action='exact030 control'))
        return control
    target='actual'
    folder=out/f'fits_{target}_{seed}'/cutoff;path=folder/f'raw_{cutoff}.csv'
    if path.exists():learned=raw_frame(path,cutoff,end)
    else:
        learned=fit(history,cutoff,end,target,folder,model_path,seed,repeat)
        learned.to_csv(path,sep=';',index=False,date_format='%Y-%m-%d')
    return mix(learned,control,.5)


def run(args):
    if os.environ.get('SLURM_JOB_PARTITION')!='gpu_devel' or int(os.environ.get('SLURM_CPUS_PER_TASK','0'))!=2 or len(os.sched_getaffinity(0))>2:raise RuntimeError('Expected two bound gpu_devel cores')
    if datetime.now(timezone.utc)>=datetime.fromisoformat('2026-09-27T15:40:40+00:00'):raise RuntimeError('Research reserve reached')
    if datetime.now(timezone.utc).timestamp()>=FAMILY_STARTED+1800:raise RuntimeError('P66/P67 family wall budget exhausted')
    model_path=Path(args.model_path)
    if not model_path.is_file() or hashlib.sha256(model_path.read_bytes()).hexdigest()!=MODEL_SHA:raise ValueError('Unverified checkpoint')
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
    out=Path(args.output)/('pilot' if args.pilot else 'study');out.mkdir(parents=True,exist_ok=True)
    paths=[Path(args.history),Path(__file__),Path('pipeline.py'),model_path,CALENDAR,Path('artifacts/calendar_sources.json')]
    paths += [Path('experiments')/f'portfolio_{n}.py' for n in ['tabular_volume','tabular_shape','tabular_fraction','school_fraction','bayes_hourly_target','bayes_direct','bayes_shape','operations','movement','ridge','route_allocation','nonlinear_errors','timesfm_errors','timesfm','bayes_volume','combine','experiment','windows']]
    paths += list(CONTROL.glob('raw_*.csv'))+list(INNER.glob('*/*.csv'))+list(INNER.glob('*/*.json'))+list(DAILY_SOURCE.glob('*/daily.csv'))+list(TEACHERS.glob('*/*.csv'))+list(TEACHERS.glob('*/*.json'))
    write_json(out/'run_started.json',dict(job_id=os.environ['SLURM_JOB_ID'],command=os.sys.argv,versions=dict(python=platform.python_version(),
        **{n:importlib.metadata.version(n) for n in ['torch','numpy','pandas','scikit-learn','tabpfn','optuna']}),sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))
    started=time.monotonic();history=load_history(args.history)
    if args.pilot:forecast(history,*FINAL,'matched',out,model_path,repeat=True)
    else:
        import optuna
        family='matched_volume';folder=out/family;folder.mkdir(exist_ok=True)
        study=optuna.create_study(storage=f'sqlite:///{folder / "study.db"}',study_name=family,direction='maximize',load_if_exists=True)
        if study.user_attrs.get('started_at') is None:study.set_user_attr('started_at',FAMILY_STARTED)
        if study.user_attrs['started_at']!=FAMILY_STARTED:raise ValueError('Family budget was reset')
        study.set_user_attr('previous_P66_trials',5);study.set_user_attr('previous_P66_gpu_seconds',135)
        run_study(history,out,family,lambda c,e,p:forecast(history,c,e,p['recipe'],out,model_path),
            dict(sampler='grid',space={'recipe':['control','matched']},trials=2))
        selected=json.loads((folder/'selection.json').read_text())['params']['recipe']
        if selected=='control':
            (out/'alternative').mkdir(exist_ok=True);write_json(out/'alternative/parameters.json',dict(recipe='matched',selection='only new fixed recipe; no retuning'))
            save_candidate(history,out/'alternative','matched_volume_alternative',lambda c,e:forecast(history,c,e,'matched',out,model_path))
        (out/'seed73').mkdir(exist_ok=True);write_json(out/'seed73/parameters.json',dict(recipe='matched',seed=73,selection='fixed new recipe; no seed tuning'))
        save_candidate(history,out/'seed73','matched_volume_seed73',lambda c,e:forecast(history,c,e,'matched',out,model_path,73))
    write_json(out/'completed.json',dict(job_id=os.environ['SLURM_JOB_ID'],elapsed_seconds=time.monotonic()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--history',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--model-path',required=True);parser.add_argument('--pilot',action='store_true');run(parser.parse_args())
