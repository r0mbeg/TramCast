"""P64: TabPFN v2 conditional route-fraction errors. Built with PriorLabs-TabPFN."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import time
import urllib.request

import numpy as np
import pandas as pd

from experiments.portfolio_school_fraction import features,prepare,CALENDAR
from experiments.portfolio_route_allocation import preserve_network
from experiments.portfolio_timesfm_errors import examples,SOURCE,TEACHERS,SHAPE
from experiments.portfolio_combine import mix,raw_frame
from experiments.portfolio_experiment import FINAL,load_history,run_study,save_candidate,write_json

CONTROL=Path('artifacts/portfolio_20260926/continuation/school_fraction/study/school_fraction/selected')
EXTERNAL=Path('artifacts/portfolio_20260926/continuation/external/tabpfn')
MODEL_SHA='2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736'
MODEL_URL='https://huggingface.co/Prior-Labs/TabPFN-v2-reg/resolve/4972a65a1b30806315c6f92499959ffbfc69a673/tabpfn-v2-regressor.ckpt'


def fetch(url,path,sha):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        temporary=path.with_suffix(path.suffix+'.part')
        with urllib.request.urlopen(url,timeout=120) as src,temporary.open('wb') as dst:
            shutil.copyfileobj(src,dst)
        if hashlib.sha256(temporary.read_bytes()).hexdigest()!=sha:raise ValueError('Downloaded file SHA mismatch')
        temporary.replace(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('Changed pinned file')


def fit(past,future,cutoff,out,seed,model_path,repeat=False):
    import torch
    from tabpfn import TabPFNRegressor
    started=time.monotonic();past=prepare(past,cutoff)
    if 'boardings' in future or not future.date.gt(np.datetime64(cutoff)).all():
        raise ValueError('Future targets or nonfuture dates')
    future=future.copy();future['network_base']=future.groupby('date').base.transform('sum')
    if not future.network_base.gt(0).all():raise ValueError('Undefined future route fraction')
    future['base_share']=future.base/future.network_base
    active=future.route.ne(5)&future.base.gt(0)
    X=features(past);Y=past.target.to_numpy();test=features(future.loc[active])
    model=TabPFNRegressor(n_estimators=4,categorical_features_indices=[0,1,2],model_path=str(model_path),
        device='cuda',inference_precision=torch.float32,fit_mode='low_memory',memory_saving_mode=True,
        random_state=seed,n_jobs=2)
    # ponytail: native v2 has no sample weights; reconsider weighted prior if this unweighted model succeeds.
    torch.cuda.reset_peak_memory_stats();model.fit(X,Y)
    errors=model.predict(test,output_type='median')
    if not np.isfinite(errors).all() or errors.shape!=(int(active.sum()),):raise ValueError('Invalid TabPFN errors')
    if repeat:np.testing.assert_array_equal(errors,model.predict(test,output_type='median'))
    future['share_error']=0.;future.loc[active,'share_error']=errors
    future['estimated_share']=(future.base_share+future.share_error).clip(lower=0)
    future.loc[~active,'estimated_share']=0.
    out.mkdir(parents=True,exist_ok=True)
    past.to_csv(out/'training.csv',sep=';',index=False,date_format='%Y-%m-%d')
    future.to_csv(out/'future.csv',sep=';',index=False,date_format='%Y-%m-%d')
    np.savez(out/'matrices.npz',X=X,y=Y,test=test,prediction=errors)
    write_json(out/'fit.json',dict(cutoff=cutoff,rows=len(past),features=30,seed=seed,
        earliest_target_date=str(past.date.min().date()),latest_target_date=str(past.date.max().date()),
        origins=int(past.origin.nunique()),n_estimators=4,output_type='median',sample_weight_used=False,
        repeated_prediction_exact=True if repeat else None,model_sha256=MODEL_SHA,
        target='actual network fraction minus base network fraction; unweighted conditional prior',
        device=torch.cuda.get_device_name(0),fit_seconds=time.monotonic()-started,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    del model;torch.cuda.empty_cache()
    return future[['route','date','estimated_share']]


def forecast(history,cutoff,end,recipe,out,model_path,seed=42,repeat=False):
    base=raw_frame(SHAPE/f'raw_{cutoff}.csv',cutoff,end)
    control=raw_frame(CONTROL/f'raw_{cutoff}.csv',cutoff,end)
    if recipe=='control':return control
    if recipe not in ['half','new']:raise ValueError('Unknown TabPFN recipe')
    folder=out/f'fits_{seed}'/cutoff;path=folder/f'raw_{cutoff}.csv'
    if path.exists():learned=raw_frame(path,cutoff,end)
    else:
        past,future=examples(history,cutoff,end)
        shares=fit(past,future,cutoff,folder,seed,model_path,repeat)
        learned=base.merge(shares,on=['route','date'],validate='many_to_one')
        volume=base.groupby(['route','date']).prediction.transform('sum')
        learned['prediction']=base.prediction.div(volume.where(volume.gt(0))).fillna(0)*learned.estimated_share
        learned=preserve_network(base,learned.drop(columns='estimated_share'))
        learned.to_csv(path,sep=';',index=False,date_format='%Y-%m-%d')
    enhanced=mix(learned,base,.5)
    return mix(enhanced,control,.5 if recipe=='half' else 1.)


def run(args):
    partition=os.environ.get('SLURM_JOB_PARTITION');cpus=int(os.environ.get('SLURM_CPUS_PER_TASK','0'))
    if partition!=('ais-cpu' if args.download else 'gpu_devel') or cpus!=(1 if args.download else 2) or len(os.sched_getaffinity(0))>cpus:
        raise RuntimeError('Unexpected Slurm allocation')
    if datetime.now(timezone.utc)>=datetime.fromisoformat('2026-09-27T15:40:40+00:00'):raise RuntimeError('Research reserve reached')
    root=Path(args.output);model_path=Path(args.model_path)
    if args.download:
        metadata=json.loads((EXTERNAL/'scikit-learn_1.6.1.json').read_text())
        wheel=next(f for f in metadata['urls'] if 'cp312-cp312-manylinux_2_17_x86_64' in f['filename'])
        fetch(wheel['url'],Path('tabpfn_package')/wheel['filename'],wheel['digests']['sha256'])
        fetch(MODEL_URL,model_path,MODEL_SHA)
        write_json(root/'download.json',dict(job_id=os.environ['SLURM_JOB_ID'],model_url=MODEL_URL,model_sha256=MODEL_SHA,
            model_bytes=model_path.stat().st_size,sklearn_wheel=wheel['filename'],sklearn_sha256=wheel['digests']['sha256'],
            authentication=False,attribution='Built with PriorLabs-TabPFN'))
        return
    if not model_path.is_file() or hashlib.sha256(model_path.read_bytes()).hexdigest()!=MODEL_SHA:raise ValueError('Unverified model checkpoint')
    # Trusted pinned legacy checkpoint needs its original loader under Torch2.6; confined to this job.
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
    out=root/('pilot' if args.pilot else 'study');out.mkdir(parents=True,exist_ok=True)
    paths=[Path(args.history),Path(__file__),Path('pipeline.py'),model_path]
    paths += [Path('experiments')/f'portfolio_{n}.py' for n in
        ['school_fraction','route_allocation','nonlinear_errors','timesfm_errors','timesfm','bayes_volume','combine','experiment','windows']]
    paths += [CALENDAR]+list(CONTROL.glob('raw_*.csv'))+list(SOURCE.glob('*/daily.csv'))+list(TEACHERS.glob('*/*.csv'))+list(TEACHERS.glob('*/*.json'))+list(SHAPE.glob('raw_*.csv'))
    write_json(out/'run_started.json',dict(job_id=os.environ['SLURM_JOB_ID'],command=os.sys.argv,
        versions=dict(python=platform.python_version(),**{n:importlib.metadata.version(n) for n in
            ['torch','numpy','pandas','scikit-learn','tabpfn','optuna','huggingface-hub']}),
        sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},legacy_trusted_checkpoint=True))
    started=time.monotonic();history=load_history(args.history)
    if args.pilot:forecast(history,*FINAL,'new',out,model_path,repeat=True)
    else:
        run_study(history,out,'tabular_fraction',lambda c,e,p:forecast(history,c,e,p['recipe'],out,model_path),
            dict(sampler='grid',space={'recipe':['control','half','new']},trials=3))
        selected=json.loads((out/'tabular_fraction/selection.json').read_text())['params']['recipe']
        if selected=='control':
            trials=pd.read_csv(out/'tabular_fraction/trials.csv',sep=';')
            selected=trials.loc[trials.params_recipe.ne('control')].sort_values('value',ascending=False).iloc[0].params_recipe
            write_json(out/'alternative/parameters.json',dict(recipe=selected,selection='best noncontrol development score; no retuning'))
            save_candidate(history,out/'alternative','tabular_fraction_alternative',lambda c,e:forecast(history,c,e,selected,out,model_path))
        write_json(out/'seed73/parameters.json',dict(recipe=selected,seed=73,selection='fixed new recipe; no seed tuning'))
        save_candidate(history,out/'seed73','tabular_fraction_seed73',lambda c,e:forecast(history,c,e,selected,out,model_path,73))
    write_json(out/'completed.json',dict(job_id=os.environ['SLURM_JOB_ID'],elapsed_seconds=time.monotonic()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--model-path',required=True);parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--download',action='store_true');run(parser.parse_args())
