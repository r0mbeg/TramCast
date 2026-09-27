"""Weekly frozen-030 teachers. Run in the existing lab GPU environment; outputs are isolated."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from experiments import portfolio_bayes_volume as bv, portfolio_bayes_direct as bd
from experiments import portfolio_direct_daily as dd, portfolio_bayes_shape as bs
from experiments import portfolio_timesfm as tf, portfolio_timesfm_errors as te
from experiments import portfolio_school_fraction as sf, portfolio_tabular_shape as ts
from experiments.portfolio_combine import mix, raw_frame
from experiments.portfolio_conditional_shape import shape_forecast
from experiments.portfolio_experiment import load_history, write_json

OLD = Path('artifacts/portfolio_20260926/continuation')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run(args):
    import torch
    import timesfm
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('One allocated CUDA device required')
    if len(os.sched_getaffinity(0)) > 2:
        raise RuntimeError('Two allocated CPU cores required')
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    root = out/'cache'
    if not root.exists():
        root.mkdir()
        for p in OLD.rglob('selection.json'):
            dest=root/p.relative_to(OLD); dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(p,dest)
        for name in ['bayes_volume/inner','bayes_volume_verified_july/inner','verified_july/inner',
                     'timesfm_teachers/daily','conditional_shape/conditional_shape/selected']:
            shutil.copytree(OLD/name,root/name)
    for module in [bv,bd,tf]: module.ROOT=root
    dd.ROOT=root/'direct_daily_inputs'
    te.SOURCE=root/'verified_july/inner'; te.TEACHERS=root/'timesfm_teachers/daily'
    bs.SOURCE=ts.SOURCE=root/'021'
    te.SHAPE=sf.SHAPE=root/'024'; ts.CONTROL=root/'029'
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.manual_seed(42)
    history=load_history('artifacts/hourly_clean.csv')
    weather=pd.read_csv(OLD/'external/weather_2025.csv',sep=';',parse_dates=['date'])
    params=lambda name:json.loads((OLD/f'{name}/{name}/selection.json').read_text())['params']
    annual=params('verified_july')
    tm=Path('model-timesfm2p5/model.safetensors'); tab=Path('model-tabpfn/tabpfn-v2-regressor.ckpt')
    if sha(tm)!=tf.WEIGHTS_SHA256 or sha(tab)!=ts.MODEL_SHA: raise ValueError('Checkpoint changed')
    sources=[Path('artifacts/hourly_clean.csv'),Path(__file__),tm,tab,*Path('experiments').glob('*.py'),
             Path('pipeline.py'),Path('constants.py'),Path('artifacts/calendar_sources.json')]
    sources+=list((OLD/'external').rglob('*.json'))+list((OLD/'external').glob('*.csv'))
    sources+=list((root).rglob('selection.json'))
    hashes={str(p):sha(p) for p in sources}
    stamp=time.monotonic()
    write_json(out/f'started_{args.origins[0]}.json',dict(origins=args.origins,source_sha256=hashes,
        recipe='frozen 030 new (.5 learned profile + .5 029)',budget_seconds=args.budget,
        versions={n:importlib.metadata.version(n) for n in ['torch','timesfm','tabpfn','numpy','pandas','scikit-learn']},
        history_policy='history physically truncated to origin for every new teacher',
        caveat='Frozen recipe selected retrospectively; not an unseen model-selection test'))
    def save(folder,cutoff,frame):
        folder.mkdir(parents=True,exist_ok=True)
        path=folder/f'raw_{cutoff}.csv'
        if path.exists(): raise FileExistsError(path)
        frame.to_csv(path,sep=';',index=False,date_format='%Y-%m-%d')
        return frame
    for cutoff in args.origins:
        final=out/'teachers'/f'raw_{cutoff}.csv'
        if final.exists():
            info=json.loads(final.with_suffix('.json').read_text())
            if info['sha256']!=sha(final): raise ValueError('Corrupt completed teacher')
            continue
        if time.monotonic()-stamp>args.budget:
            print('BUDGET STOP',cutoff,flush=True);break
        begun=time.monotonic();end=str((pd.Timestamp(cutoff)+pd.Timedelta(days=61)).date())
        hist=history.loc[history.date.le(cutoff)].copy()
        print('START',cutoff,flush=True)
        # Resume only whole teacher units in a fresh output; partial caches are input-specific.
        v=bv.bayes_volume_forecast(hist,cutoff,end,params('bayes_volume'))
        d=dd.direct_daily_forecast(hist,cutoff,end,params('direct_daily'))
        conditional=shape_forecast(hist,cutoff,params('conditional_shape'),weather,mix(v,d,.75))
        p=root/'conditional_shape/conditional_shape/selected'/f'raw_{cutoff}.csv'
        if not p.exists(): save(p.parent,cutoff,conditional)
        raw21=bd.forecast(hist,cutoff,end,annual,root/'verified_july')
        if not (root/'021'/f'raw_{cutoff}.csv').exists(): save(root/'021',cutoff,raw21)
        raw24=bs.forecast(hist,cutoff,end,'half',root/'bayes_shape')
        if not (root/'024'/f'raw_{cutoff}.csv').exists(): save(root/'024',cutoff,raw24)
        print('PARENTS',cutoff,time.monotonic()-begun,flush=True)
        matrix,restoration=tf.series_inputs(hist,cutoff,end,'normal_ratio',july_verified=True)
        model=timesfm.TimesFM_2p5_200M_torch(torch_compile=False)
        model.load_checkpoint(tm,torch_compile=False)
        model.compile(timesfm.ForecastConfig(max_context=512,max_horizon=128,per_core_batch_size=8,
            normalize_inputs=True,use_continuous_quantile_head=True,force_flip_invariance=True,
            infer_is_positive=True,fix_quantile_crossing=True))
        with torch.inference_mode(): _,q=model.forecast(horizon=61,inputs=[r.copy() for r in matrix])
        folder=te.TEACHERS/cutoff;folder.mkdir(parents=True,exist_ok=True)
        daily=tf.teacher_daily(hist,cutoff,end,q,restoration)
        daily.to_csv(folder/'daily.csv',sep=';',index=False,date_format='%Y-%m-%d')
        np.savez_compressed(folder/'quantiles.npz',inputs=matrix,quantiles=q,restoration=restoration)
        write_json(folder/'source.json',dict(origin=cutoff,end=end,fit_latest_date=cutoff,quantile_index=3,
            july_verified=True,model_sha256=tf.WEIGHTS_SHA256,sha256=sha(folder/'daily.csv')))
        del model;torch.cuda.empty_cache()
        past,future=te.examples(hist,cutoff,end)
        shares=sf.fit(past,future,cutoff,root/'school'/cutoff,42)
        learned=raw24.merge(shares,on=['route','date'],validate='many_to_one')
        volume=raw24.groupby(['route','date']).prediction.transform('sum')
        learned['prediction']=raw24.prediction.div(volume.where(volume.gt(0))).fillna(0)*learned.estimated_share
        from experiments.portfolio_route_allocation import preserve_network
        raw29=mix(preserve_network(raw24,learned.drop(columns='estimated_share')),raw24,.5)
        if not (root/'029'/f'raw_{cutoff}.csv').exists(): save(root/'029',cutoff,raw29)
        raw30=ts.forecast(hist,cutoff,end,'new',root/'tabular',tab)
        save(final.parent,cutoff,raw30)
        write_json(final.with_suffix('.json'),dict(origin=cutoff,end=end,history_max=str(hist.date.max().date()),
            seconds=time.monotonic()-begun,sha256=sha(final),rows=len(raw30),source_manifest=f'started_{args.origins[0]}.json',
            peak_gpu_bytes=torch.cuda.max_memory_allocated()))
        print('DONE',cutoff,time.monotonic()-begun,flush=True)
    # Inputs with root selections are immutable; generated caches are intentionally excluded.
    if any(sha(p)!=h for p,h in hashes.items()):raise ValueError('Source changed during generation')
    write_json(out/f'completed_{args.origins[0]}.json',dict(seconds=time.monotonic()-stamp,sources_unchanged=True,
        requested=args.origins,completed=[o for o in args.origins if (out/'teachers'/f'raw_{o}.csv').exists()]))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--origins',nargs='+',required=True)
    p.add_argument('--budget',type=int,default=3600)
    run(p.parse_args())
