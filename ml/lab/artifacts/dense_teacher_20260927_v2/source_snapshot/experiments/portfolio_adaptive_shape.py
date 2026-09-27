"""P71: past-data PCA rank, reusing P65 modes. Built with PriorLabs-TabPFN."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import time
import numpy as np
import pandas as pd
from experiments.portfolio_tabular_shape import basis as capped_basis,HOURS,SOURCE,CONTROL as VOLUME
from experiments.portfolio_tabular_fraction import MODEL_SHA
from experiments.portfolio_tabular_volume import CONTROL
from experiments.portfolio_combine import raw_frame,mix,transplant
from experiments.portfolio_experiment import FINAL,load_history,run_study,save_candidate,write_json
from pipeline import KEYS

ROOT=Path('artifacts/portfolio_20260926/continuation');PARENT=ROOT/'tabular_shape/study'
MANIFEST=ROOT/'adaptive_shape/parent_manifest.json'


def basis90(errors,weights):
    mean,old,old_coordinates,values=capped_basis(errors,weights)
    centered=errors-mean;covariance=(centered*weights[:,None]).T@centered/weights.sum()
    eigenvalues,vectors=np.linalg.eigh(covariance);order=np.argsort(eigenvalues)[::-1]
    np.testing.assert_array_equal(np.maximum(eigenvalues[order],0),values)
    rank=0 if values.sum()<=1e-24 else int(np.searchsorted(np.cumsum(values)/values.sum(),.9))+1
    components=vectors[:,order[:rank]].T
    for row in components:
        if row[np.argmax(abs(row))]<0:row*=-1
    np.testing.assert_array_equal(components[:len(old)],old)
    coordinates=centered@components.T
    # Preserve exact saved native targets for reused modes despite matrix-width roundoff.
    coordinates[:,:len(old)]=old_coordinates
    return mean,components,coordinates,values


def hourly(base,future,shares,active,mean,predictions,components,volume):
    delta=np.zeros_like(shares);delta[active]=mean+predictions@components
    profile=np.maximum(0,shares+delta)
    expanded=future[['route','date']].copy()
    for i,hour in enumerate(HOURS):expanded[str(hour)]=profile[:,i]
    expanded=expanded.melt(id_vars=['route','date'],var_name='hour',value_name='prediction');expanded['hour']=expanded.hour.astype(int)
    shape=base[KEYS].merge(expanded,on=KEYS,how='left',validate='one_to_one').fillna({'prediction':0.})
    return transplant(volume,shape,volume),delta


def fit(cutoff,end,out,model_path,seed,repeat=False):
    import torch
    from tabpfn import TabPFNRegressor
    started=time.monotonic();parent=PARENT/f'fits_{seed}'/cutoff
    meta=json.loads((parent/'fit.json').read_text());data=dict(np.load(parent/'matrices.npz'))
    past=pd.read_csv(parent/'daily_training.csv',sep=';',parse_dates=['date'],float_precision='round_trip')
    future=pd.read_csv(parent/'future.csv',sep=';',parse_dates=['date'],float_precision='round_trip')
    if meta['cutoff']!=cutoff or meta['seed']!=seed or past.date.max()>pd.Timestamp(cutoff) or 'boardings' in future or future.date.min()<=pd.Timestamp(cutoff):raise ValueError('Invalid frozen profile parent')
    X=data['X'];test=data['test'];active=data['active'];oldrank=meta['rank']
    if X.shape!=(len(past),50) or test.shape!=(int(active.sum()),50) or not np.isfinite(X).all() or not np.isfinite(test).all():raise ValueError('Invalid parent features')
    mean,components,coordinates,eigenvalues=basis90(data['errors'],data['weights'])
    np.testing.assert_array_equal(mean,data['mean']);np.testing.assert_array_equal(components[:oldrank],data['components'])
    np.testing.assert_array_equal(coordinates[:,:oldrank],data['coordinates'])
    predictions=np.zeros((len(test),len(components)));predictions[:,:oldrank]=data['predictions']
    torch.cuda.reset_peak_memory_stats()
    for i in range(oldrank,len(components)):
        model=TabPFNRegressor(n_estimators=4,categorical_features_indices=[0,1,2],model_path=str(model_path),device='cuda',
            inference_precision=torch.float32,fit_mode='low_memory',memory_saving_mode=True,random_state=seed,n_jobs=2)
        model.fit(X,coordinates[:,i]);predictions[:,i]=model.predict(test,output_type='median')
        if repeat:np.testing.assert_array_equal(predictions[:,i],model.predict(test,output_type='median'))
        del model;torch.cuda.empty_cache()
    if not np.isfinite(predictions).all():raise ValueError('Invalid additional mode predictions')
    base=raw_frame(SOURCE/f'raw_{cutoff}.csv',cutoff,end);volume=raw_frame(VOLUME/f'raw_{cutoff}.csv',cutoff,end)
    pd.testing.assert_frame_equal(future[['route','date']],base[['route','date']].drop_duplicates().reset_index(drop=True))
    oldraw,_=hourly(base,future,data['future_share'],active,mean,data['predictions'],data['components'],volume)
    np.testing.assert_allclose(oldraw.prediction,raw_frame(parent/f'raw_{cutoff}.csv',cutoff,end).prediction,rtol=1e-12,atol=1e-9)
    learned,delta=hourly(base,future,data['future_share'],active,mean,predictions,components,volume)
    raw=mix(learned,volume,.5)
    np.testing.assert_allclose(raw.groupby(['route','date']).prediction.sum(),volume.groupby(['route','date']).prediction.sum(),rtol=1e-12,atol=1e-8)
    if not raw.loc[raw.route.eq(5)|raw.hour.between(1,4),'prediction'].eq(0).all():raise ValueError('Changed structural zeros')
    out.mkdir(parents=True,exist_ok=True)
    for name in ['daily_training.csv','future.csv']:(out/name).write_bytes((parent/name).read_bytes())
    data.update(mean=mean,components=components,coordinates=coordinates,eigenvalues=eigenvalues,predictions=predictions,delta=delta)
    np.savez(out/'matrices.npz',**data)
    write_json(out/'fit.json',dict(cutoff=cutoff,seed=seed,rows=len(past),features=50,rank=len(components),reused_components=oldrank,
        newly_fitted_components=len(components)-oldrank,variance_retained=float(eigenvalues[:len(components)].sum()/eigenvalues.sum()),
        parent=str(parent),latest_target_date=str(past.date.max().date()),coordinate_sample_weights=False,
        model_sha256=MODEL_SHA,n_estimators=4,output_type='median',repeated_prediction_exact=True if repeat else None,
        fit_seconds=time.monotonic()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    return raw


def forecast(cutoff,end,recipe,out,model_path,seed=42,repeat=False):
    control=raw_frame(CONTROL/f'raw_{cutoff}.csv',cutoff,end)
    if recipe=='control':return control
    if recipe not in ['half','full']:raise ValueError('Unknown adaptive profile recipe')
    folder=out/f'fits_{seed}'/cutoff;path=folder/f'raw_{cutoff}.csv'
    if path.exists():raw=raw_frame(path,cutoff,end)
    else:
        raw=fit(cutoff,end,folder,model_path,seed,repeat);raw.to_csv(path,sep=';',index=False,date_format='%Y-%m-%d')
    return mix(raw,control,.5 if recipe=='half' else 1.)


def run(args):
    if os.environ.get('SLURM_JOB_PARTITION')!='gpu_devel' or int(os.environ.get('SLURM_CPUS_PER_TASK','0'))!=2 or len(os.sched_getaffinity(0))>2:raise RuntimeError('Expected two bound gpu_devel cores')
    if datetime.now(timezone.utc)>=datetime.fromisoformat('2026-09-27T15:40:40+00:00'):raise RuntimeError('Research reserve reached')
    model_path=Path(args.model_path)
    if not model_path.is_file() or hashlib.sha256(model_path.read_bytes()).hexdigest()!=MODEL_SHA:raise ValueError('Unverified checkpoint')
    parents=json.loads(MANIFEST.read_text())
    for name,digest in parents.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise ValueError(f'Parent changed: {name}')
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
    started=time.monotonic();out=Path(args.output)/('pilot' if args.pilot else 'study');out.mkdir(parents=True,exist_ok=True)
    paths=[Path(args.history),Path(__file__),Path('pipeline.py'),MANIFEST,model_path]+[Path('experiments')/f'portfolio_{n}.py' for n in ['tabular_shape','tabular_volume','tabular_fraction','bayes_shape','combine','experiment']]
    paths+=list(CONTROL.glob('raw_*.csv'))+list(VOLUME.glob('raw_*.csv'))+list(SOURCE.glob('raw_*.csv'))
    write_json(out/'run_started.json',dict(job_id=os.environ['SLURM_JOB_ID'],command=os.sys.argv,
        versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','torch','tabpfn','optuna']},
        old_tabpfn_trials=18,old_tabpfn_gpu_seconds=806,new_trial_limit=3,total_tabpfn_trials=21,new_gpu_allocation_limit_seconds=540,
        sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},parent_sha256=parents))
    history=load_history(args.history)
    if args.pilot:forecast(*FINAL,'full',out,model_path,repeat=True)
    else:
        run_study(history,out,'adaptive_shape',lambda c,e,p:forecast(c,e,p['recipe'],out,model_path),
            dict(sampler='grid',space={'recipe':['control','half','full']},trials=3))
        selected=json.loads((out/'adaptive_shape/selection.json').read_text())['params']['recipe']
        if selected=='control':
            trials=pd.read_csv(out/'adaptive_shape/trials.csv',sep=';');selected=trials.loc[trials.params_recipe.ne('control')].sort_values('value',ascending=False).iloc[0].params_recipe
            write_json(out/'best_new.json',dict(recipe=selected))
            save_candidate(history,out/'alternative','adaptive_shape_best_new',lambda c,e:forecast(c,e,selected,out,model_path))
        write_json(out/'seed_recipe.json',dict(recipe=selected,seed=73,selection='fixed best new recipe, no seed selection'))
        save_candidate(history,out/'seed73','adaptive_shape_seed73',lambda c,e:forecast(c,e,selected,out,model_path,73))
    write_json(out/'completed.json',dict(job_id=os.environ['SLURM_JOB_ID'],elapsed_seconds=time.monotonic()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def self_check():
    errors=np.eye(20)-.05;mean,components,_,values=basis90(errors,np.ones(20));rank=len(components)
    assert rank==18 and values[:rank].sum()/values.sum()>=.9 and values[:rank-1].sum()/values.sum()<.9
    np.testing.assert_allclose(components.sum(axis=1),0,atol=1e-12)
    assert len(basis90(np.zeros((2,20)),np.ones(2))[1])==0


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--history');parser.add_argument('--output');parser.add_argument('--model-path');parser.add_argument('--pilot',action='store_true');parser.add_argument('--self-check',action='store_true')
    args=parser.parse_args()
    if args.self_check:self_check()
    elif not args.history or not args.output or not args.model_path:parser.error('history, output and model-path are required')
    else:run(args)
