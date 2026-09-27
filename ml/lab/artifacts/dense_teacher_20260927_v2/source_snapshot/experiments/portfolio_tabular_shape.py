"""P65: jointly predict past hourly-share error modes. Built with PriorLabs-TabPFN."""
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

from experiments.portfolio_bayes_shape import HOURS,INNER,SOURCE,inputs,prepare
from experiments.portfolio_school_fraction import features as daily_features,CALENDAR
from experiments.portfolio_route_allocation import relative_targets
from experiments.portfolio_tabular_fraction import CONTROL,MODEL_SHA
from experiments.portfolio_timesfm_errors import examples,SOURCE as DAILY_SOURCE,TEACHERS
from experiments.portfolio_combine import mix,raw_frame,transplant
from experiments.portfolio_experiment import FINAL,load_history,run_study,save_candidate,write_json
from pipeline import KEYS

DAY=['origin','route','date']


def basis(errors,weights):
    errors=np.asarray(errors,float);weights=np.asarray(weights,float)
    if errors.ndim!=2 or errors.shape[1]!=len(HOURS) or len(weights)!=len(errors) or not np.isfinite(errors).all() or not np.isfinite(weights).all() or (weights<=0).any():
        raise ValueError('Invalid profile error matrix/weights')
    np.testing.assert_allclose(errors.sum(axis=1),0,rtol=0,atol=1e-12)
    mean=np.average(errors,axis=0,weights=weights);centered=errors-mean
    covariance=(centered*weights[:,None]).T@centered/weights.sum()
    values,vectors=np.linalg.eigh(covariance);order=np.argsort(values)[::-1]
    values=np.maximum(values[order],0);vectors=vectors[:,order]
    total=float(values.sum())
    rank=0 if total<=1e-24 else min(6,int(np.searchsorted(np.cumsum(values)/total,.9))+1)
    components=vectors[:,:rank].T
    for row in components:
        if row[np.argmax(abs(row))]<0:row*=-1
    coordinates=centered@components.T
    return mean,components,coordinates,values


def training(history,cutoff,end):
    hourly=prepare(inputs(history,cutoff),cutoff).sort_values(DAY+['hour']).reset_index(drop=True)
    if hourly.duplicated(DAY+['hour']).any() or not hourly.groupby(DAY).hour.nunique().eq(len(HOURS)).all():
        raise ValueError('Incomplete reliable hourly profiles')
    daily,future=examples(history,cutoff,end)
    daily=relative_targets(daily,cutoff);daily['base_share']=daily.base/daily.network_base
    index=hourly[DAY+['actual_day','base_day']].drop_duplicates(DAY).reset_index(drop=True)
    paired=index.merge(daily,on=DAY,validate='one_to_one')
    if len(paired)!=len(index):raise ValueError('Unpaired causal daily/profile inputs')
    np.testing.assert_allclose(paired.boardings,paired.actual_day,rtol=0,atol=1e-8)
    predicted=hourly.pivot(index=DAY,columns='hour',values='prediction').reindex(columns=HOURS).to_numpy()
    actual=hourly.pivot(index=DAY,columns='hour',values='boardings').reindex(columns=HOURS).to_numpy()
    shares=predicted/index.base_day.to_numpy()[:,None]
    errors=actual/index.actual_day.to_numpy()[:,None]-shares
    weights=index.actual_day.to_numpy()/index.groupby(['route','date']).route.transform('size').to_numpy()
    weights/=weights.mean()
    return hourly,paired,future,shares,errors,weights


def fit(history,cutoff,end,out,model_path,seed,repeat=False):
    import torch
    from tabpfn import TabPFNRegressor
    started=time.monotonic()
    hourly,past,future,past_share,errors,weights=training(history,cutoff,end)
    mean,components,coordinates,eigenvalues=basis(errors,weights)
    base=raw_frame(SOURCE/f'raw_{cutoff}.csv',cutoff,end)
    future=future.copy();future['origin']=pd.Timestamp(cutoff)
    future['network_base']=future.groupby('date').base.transform('sum');future['base_share']=future.base/future.network_base
    if 'boardings' in future or not future.date.gt(pd.Timestamp(cutoff)).all():raise ValueError('Future target/date violation')
    future=future.sort_values(['route','date']).reset_index(drop=True)
    profile=base.pivot(index=['route','date'],columns='hour',values='prediction').reindex(columns=HOURS).rename_axis(columns=None)
    pd.testing.assert_frame_equal(profile.reset_index()[['route','date']],future[['route','date']])
    source_day=profile.sum(axis=1).to_numpy();future_share=np.divide(profile.to_numpy(),source_day[:,None],out=np.zeros(profile.shape),where=source_day[:,None]>0)
    active=future.route.ne(5).to_numpy()&(source_day>0)
    X=np.column_stack([daily_features(past),past_share]);test=np.column_stack([daily_features(future.loc[active]),future_share[active]])
    if not np.isfinite(X).all() or not np.isfinite(test).all():raise ValueError('Invalid profile features')
    predictions=np.zeros((int(active.sum()),len(components)))
    torch.cuda.reset_peak_memory_stats()
    for i in range(len(components)):
        model=TabPFNRegressor(n_estimators=4,categorical_features_indices=[0,1,2],model_path=str(model_path),
            device='cuda',inference_precision=torch.float32,fit_mode='low_memory',memory_saving_mode=True,random_state=seed,n_jobs=2)
        # ponytail: native prior has no sampleweights; basis is weighted, coordinate models are explicitly unweighted.
        model.fit(X,coordinates[:,i]);predictions[:,i]=model.predict(test,output_type='median')
        if repeat:np.testing.assert_array_equal(predictions[:,i],model.predict(test,output_type='median'))
        del model;torch.cuda.empty_cache()
    if not np.isfinite(predictions).all():raise ValueError('Invalid predicted error coordinates')
    delta=np.zeros_like(future_share);delta[active]=mean+predictions@components
    shares=np.maximum(0,future_share+delta)
    shape=base[KEYS].assign(prediction=0.)
    expanded=future[['route','date']].copy()
    for i,hour in enumerate(HOURS):expanded[str(hour)]=shares[:,i]
    expanded=expanded.melt(id_vars=['route','date'],var_name='hour',value_name='prediction');expanded['hour']=expanded.hour.astype(int)
    shape=shape.drop(columns='prediction').merge(expanded,on=KEYS,how='left',validate='one_to_one').fillna({'prediction':0.})
    control=raw_frame(CONTROL/f'raw_{cutoff}.csv',cutoff,end)
    learned=transplant(control,shape,control)
    np.testing.assert_allclose(learned.groupby(['route','date']).prediction.sum(),control.groupby(['route','date']).prediction.sum(),rtol=1e-12,atol=1e-8)
    if not learned.loc[learned.route.eq(5)|learned.hour.between(1,4),'prediction'].eq(0).all():raise ValueError('Changed structural zeros')
    out.mkdir(parents=True,exist_ok=True)
    hourly.to_csv(out/'hourly_training.csv',sep=';',index=False,date_format='%Y-%m-%d')
    past.to_csv(out/'daily_training.csv',sep=';',index=False,date_format='%Y-%m-%d')
    future.to_csv(out/'future.csv',sep=';',index=False,date_format='%Y-%m-%d')
    np.savez(out/'matrices.npz',X=X,test=test,past_share=past_share,future_share=future_share,errors=errors,weights=weights,
        mean=mean,components=components,coordinates=coordinates,eigenvalues=eigenvalues,active=active,predictions=predictions,delta=delta)
    write_json(out/'fit.json',dict(cutoff=cutoff,seed=seed,rows=len(past),hourly_rows=len(hourly),features=50,rank=len(components),
        variance_retained=float(eigenvalues[:len(components)].sum()/eigenvalues.sum()) if eigenvalues.sum()>0 else 1.,
        earliest_target_date=str(past.date.min().date()),latest_target_date=str(past.date.max().date()),origins=int(past.origin.nunique()),
        n_estimators=4,output_type='median',coordinate_sample_weights=False,basis_weights='actualday / repeated route-date',model_sha256=MODEL_SHA,
        repeated_prediction_exact=True if repeat else None,device=torch.cuda.get_device_name(0),
        fit_seconds=time.monotonic()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    return learned


def forecast(history,cutoff,end,recipe,out,model_path,seed=42,repeat=False):
    control=raw_frame(CONTROL/f'raw_{cutoff}.csv',cutoff,end)
    if recipe=='control':return control
    if recipe not in ['half','new']:raise ValueError('Unknown profile-mode recipe')
    folder=out/f'fits_{seed}'/cutoff;path=folder/f'raw_{cutoff}.csv'
    if path.exists():learned=raw_frame(path,cutoff,end)
    else:
        learned=fit(history,cutoff,end,folder,model_path,seed,repeat)
        learned.to_csv(path,sep=';',index=False,date_format='%Y-%m-%d')
    return mix(learned,control,.25 if recipe=='half' else .5)


def run(args):
    if os.environ.get('SLURM_JOB_PARTITION')!='gpu_devel' or int(os.environ.get('SLURM_CPUS_PER_TASK','0'))!=2 or len(os.sched_getaffinity(0))>2:raise RuntimeError('Expected two bound gpu_devel cores')
    if datetime.now(timezone.utc)>=datetime.fromisoformat('2026-09-27T15:40:40+00:00'):raise RuntimeError('Research reserve reached')
    model_path=Path(args.model_path)
    if not model_path.is_file() or hashlib.sha256(model_path.read_bytes()).hexdigest()!=MODEL_SHA:raise ValueError('Unverified checkpoint')
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
    out=Path(args.output)/('pilot' if args.pilot else 'study');out.mkdir(parents=True,exist_ok=True)
    paths=[Path(args.history),Path(__file__),Path('pipeline.py'),model_path,CALENDAR,Path('artifacts/calendar_sources.json')]
    paths += [Path('experiments')/f'portfolio_{n}.py' for n in ['tabular_fraction','school_fraction','bayes_shape','operations','movement','ridge','route_allocation','nonlinear_errors','timesfm_errors','timesfm','bayes_volume','combine','experiment','windows']]
    paths += list(CONTROL.glob('raw_*.csv'))+list(SOURCE.glob('raw_*.csv'))+list(INNER.glob('*/*.csv'))+list(INNER.glob('*/*.json'))
    paths += list(DAILY_SOURCE.glob('*/daily.csv'))+list(TEACHERS.glob('*/*.csv'))+list(TEACHERS.glob('*/*.json'))
    write_json(out/'run_started.json',dict(job_id=os.environ['SLURM_JOB_ID'],command=os.sys.argv,versions=dict(python=platform.python_version(),
        **{n:importlib.metadata.version(n) for n in ['torch','numpy','pandas','scikit-learn','tabpfn','optuna']}),
        sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},legacy_trusted_checkpoint=True))
    started=time.monotonic();history=load_history(args.history)
    if args.pilot:forecast(history,*FINAL,'new',out,model_path,repeat=True)
    else:
        run_study(history,out,'tabular_shape',lambda c,e,p:forecast(history,c,e,p['recipe'],out,model_path),
            dict(sampler='grid',space={'recipe':['control','half','new']},trials=3))
        selected=json.loads((out/'tabular_shape/selection.json').read_text())['params']['recipe']
        if selected=='control':
            trials=pd.read_csv(out/'tabular_shape/trials.csv',sep=';')
            selected=trials.loc[trials.params_recipe.ne('control')].sort_values('value',ascending=False).iloc[0].params_recipe
            (out/'alternative').mkdir(exist_ok=True);write_json(out/'alternative/parameters.json',dict(recipe=selected,selection='best new development score; no retuning'))
            save_candidate(history,out/'alternative','tabular_shape_alternative',lambda c,e:forecast(history,c,e,selected,out,model_path))
        (out/'seed73').mkdir(exist_ok=True);write_json(out/'seed73/parameters.json',dict(recipe=selected,seed=73,selection='fixed new recipe; no seed tuning'))
        save_candidate(history,out/'seed73','tabular_shape_seed73',lambda c,e:forecast(history,c,e,selected,out,model_path,73))
    write_json(out/'completed.json',dict(job_id=os.environ['SLURM_JOB_ID'],elapsed_seconds=time.monotonic()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history',required=True);parser.add_argument('--output',required=True);parser.add_argument('--model-path',required=True)
    parser.add_argument('--pilot',action='store_true');run(parser.parse_args())
