"""P70: fixed-prior Student-t MAP for past annual forecast errors."""
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
from experiments.portfolio_bayes_direct import annual_features
from experiments.portfolio_gp_errors import inputs,ROOT
from experiments.portfolio_combine import raw_frame,mix
from experiments.portfolio_experiment import FINAL,load_history,run_study,save_candidate,write_json
from experiments.portfolio_tabular_volume import CONTROL

NU=4.


def robust_map(x,y,w,sigma2,precision,coef,intercept):
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(w).all() or np.any(w<=0) or sigma2<=0 or precision<=0:
        raise ValueError('Invalid robust regression input')
    def objective(b,a):
        return float(np.sum(w*(NU+1)/2*np.log1p((y-x@b-a)**2/(NU*sigma2)))+precision/2*(b@b))
    trace=[objective(coef,intercept)]
    for iteration in range(100):
        latent=(NU+1)/(NU+(y-x@coef-intercept)**2/sigma2); weights=w*latent
        center=np.average(x,axis=0,weights=weights); ym=np.average(y,weights=weights); xc=x-center
        updated=np.linalg.solve((xc.T*weights)@xc/sigma2+precision*np.eye(x.shape[1]),xc.T@(weights*(y-ym))/sigma2)
        a=float(ym-center@updated); value=objective(updated,a)
        if value>trace[-1]+1e-9*max(1,abs(trace[-1])):raise ValueError('Nonmonotonic t objective')
        delta=max(float(np.max(np.abs(updated-coef))),abs(a-intercept))
        relative=abs(value-trace[-1])/max(1,abs(trace[-1])); trace.append(value)
        coef,intercept=updated,a
        if relative<1e-8 and delta<1e-7:break
    else:raise ValueError('Robust MAP did not converge')
    return coef,intercept,trace


def fit(history,cutoff,end,out):
    started=time.monotonic();past,future=inputs(history,cutoff,end)
    past=past.loc[past.route.ne(5)&past.base.gt(0)].copy()
    meta=json.loads((ROOT/'verified_july'/f'posterior_{cutoff}_1.0_224_route_july_verifiedTrue.json').read_text())
    assert len(past)==meta['rows'] and past.date.max()<=pd.Timestamp(cutoff)
    w=1/past.groupby(['route','date']).base.transform('size').to_numpy()
    x=(annual_features(past,True)-np.array(meta['feature_mean']))/np.array(meta['feature_scale'])
    test=(annual_features(future,True)-np.array(meta['feature_mean']))/np.array(meta['feature_scale'])
    y=np.clip(np.log((past.boardings.to_numpy()+100)/(past.base.to_numpy()+100)),-np.log(2),np.log(2))
    sigma2=1/meta['noise_precision']; precision=meta['coefficient_precision']; old=np.array(meta['coefficient_mean'])
    center=np.average(x,axis=0,weights=w);xc=x-center
    covariance=np.linalg.inv((xc.T*w)@xc/sigma2+precision*np.eye(x.shape[1]))
    sd=np.sqrt(sigma2+np.sum((test@covariance)*test,axis=1)); shrink=1/(1+(sd/.1)**2)
    oldmean=test@old+meta['intercept']; oldfactor=np.exp(np.clip(shrink*oldmean,-np.log(2),np.log(2)))
    cached=pd.read_csv(ROOT/'verified_july'/f'factors_{cutoff}_1.0_224_route_july_verifiedTrue.csv',sep=';',parse_dates=['date'],float_precision='round_trip')
    pd.testing.assert_frame_equal(future[['route','date']],cached[['route','date']])
    np.testing.assert_allclose(oldfactor,cached.factor,rtol=1e-9,atol=1e-11)
    coef,intercept,trace=robust_map(x,y,w,sigma2,precision,old,meta['intercept'])
    newmean=test@coef+intercept;factor=np.exp(np.clip(shrink*newmean,-np.log(2),np.log(2)))
    base=raw_frame(ROOT/'bayes_volume_verified_july/inner'/cutoff/f'raw_{cutoff}.csv',cutoff,end)
    base=base.merge(future[['route','date']].assign(oldfactor=oldfactor,factor=factor),on=['route','date'],validate='many_to_one')
    oldnet=(base.prediction*base.oldfactor).groupby(base.date).sum()
    newnet=(base.prediction*base.factor).groupby(base.date).sum()
    control=raw_frame(CONTROL/f'raw_{cutoff}.csv',cutoff,end)
    np.testing.assert_allclose(control.groupby('date').prediction.sum(),oldnet,rtol=1e-12,atol=1e-8)
    ratio=(newnet/oldnet).where(oldnet.gt(0),1.)
    if not np.isfinite(ratio).all() or ratio.lt(0).any():raise ValueError('Invalid robust volume ratio')
    raw=control.assign(prediction=control.prediction*control.date.map(ratio))
    out.mkdir(parents=True,exist_ok=True)
    past.to_csv(out/'training.csv',sep=';',index=False,date_format='%Y-%m-%d')
    future.assign(oldmean=oldmean,newmean=newmean,attenuation_sd=sd,shrink=shrink,oldfactor=oldfactor,factor=factor).to_csv(out/'future.csv',sep=';',index=False,date_format='%Y-%m-%d')
    pd.DataFrame(dict(old_volume=oldnet,new_volume=newnet,ratio=ratio)).to_csv(out/'network.csv',sep=';',index_label='date')
    np.savez(out/'matrices.npz',x=x,y=y,w=w,test=test,coef=coef,intercept=intercept,oldcoef=old,covariance=covariance)
    write_json(out/'fit.json',dict(cutoff=cutoff,rows=len(x),features=x.shape[1],nu=NU,sigma2=sigma2,precision=precision,
        objective_trace=trace,iterations=len(trace)-1,latest_target_date=str(past.date.max().date()),
        uncertainty='Old Gaussian predictive attenuation preserved; fixed-hyperparameter Student-t MAP, no exact t posterior.',
        fit_seconds=time.monotonic()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    return raw


def forecast(history,cutoff,end,recipe,out):
    control=raw_frame(CONTROL/f'raw_{cutoff}.csv',cutoff,end)
    if recipe=='control':return control
    if recipe not in ['half','full']:raise ValueError('Unknown robust recipe')
    folder=out/'fits'/cutoff;path=folder/f'raw_{cutoff}.csv'
    if path.exists():raw=raw_frame(path,cutoff,end)
    else:
        raw=fit(history,cutoff,end,folder);raw.to_csv(path,sep=';',index=False,date_format='%Y-%m-%d')
    return mix(raw,control,.5 if recipe=='half' else 1.)


def run(args):
    if os.environ.get('SLURM_JOB_PARTITION')!='ais-cpu' or int(os.environ.get('SLURM_CPUS_PER_TASK','0'))!=1 or len(os.sched_getaffinity(0))>1:raise RuntimeError('Expected one bound ais-cpu core')
    if datetime.now(timezone.utc)>=datetime.fromisoformat('2026-09-27T15:40:40+00:00'):raise RuntimeError('Research reserve reached')
    started=time.monotonic();out=Path(args.output)/('pilot' if args.pilot else 'study');out.mkdir(parents=True,exist_ok=True)
    paths=[Path(args.history),Path(__file__),Path('pipeline.py')]+[Path('experiments')/f'portfolio_{n}.py' for n in
        ['bayes_direct','bayes_volume','gp_errors','windows','ridge','combine','experiment','tabular_volume']]
    paths+=list((ROOT/'verified_july').glob('posterior_*_1.0_224_route_july_verifiedTrue.json'))+list((ROOT/'verified_july').glob('factors_*_1.0_224_route_july_verifiedTrue.csv'))
    paths+=list((ROOT/'verified_july/inner').glob('*/daily.csv'))+list((ROOT/'bayes_volume_verified_july/inner').glob('*/raw_*.csv'))+list(CONTROL.glob('raw_*.csv'))
    write_json(out/'run_started.json',dict(job_id=os.environ['SLURM_JOB_ID'],command=os.sys.argv,
        versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','optuna']},
        new_trial_limit=3,new_cpu_allocation_seconds_limit=300,old_phases=['P23','P41','P43'],
        sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))
    history=load_history(args.history)
    if args.pilot:
        raw=forecast(history,*FINAL,'full',out);raw.to_csv(out/'pilot_raw.csv',sep=';',index=False,date_format='%Y-%m-%d')
        repeated=fit(history,*FINAL,out/'repeat');np.testing.assert_array_equal(raw.prediction,repeated.prediction)
    else:
        run_study(history,out,'robust_annual',lambda c,e,p:forecast(history,c,e,p['recipe'],out),
            dict(sampler='grid',space={'recipe':['control','half','full']},trials=3))
        trials=pd.read_csv(out/'robust_annual/trials.csv',sep=';');best=trials.loc[trials.params_recipe.ne('control')].sort_values('value',ascending=False).iloc[0]
        save_candidate(history,out/'alternative','robust_annual_best_new',lambda c,e:forecast(history,c,e,best.params_recipe,out))
        write_json(out/'best_new.json',dict(recipe=best.params_recipe,value=float(best.value)))
    write_json(out/'completed.json',dict(job_id=os.environ['SLURM_JOB_ID'],elapsed_seconds=time.monotonic()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def self_check():
    x=np.zeros((101,1));y=np.r_[np.zeros(100),20.];w=np.ones(101)
    b,a,trace=robust_map(x,y,w,1.,1.,np.zeros(1),float(y.mean()))
    assert abs(a)<.01 and abs(b[0])<1e-12 and trace[-1]<trace[0]


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--history');parser.add_argument('--output');parser.add_argument('--pilot',action='store_true');parser.add_argument('--self-check',action='store_true')
    args=parser.parse_args()
    if args.self_check:self_check()
    else:run(args)
