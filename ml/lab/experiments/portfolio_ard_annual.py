"""P72: native automatic relevance determination of past annual errors."""
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
from sklearn.linear_model import ARDRegression

CONTROL=ROOT/"adaptive_shape/study/adaptive_shape/selected"

def ard(x,y,w):
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(w).all() or np.any(w<=0):
        raise ValueError('Invalid ARD input')
    center=np.average(x,axis=0,weights=w);ym=float(np.average(y,weights=w));root=np.sqrt(w)
    model=ARDRegression(max_iter=1000,tol=1e-5,compute_score=True,fit_intercept=False)
    model.fit((x-center)*root[:,None],(y-ym)*root)
    if model.n_iter_>=1000 or not np.isfinite(model.coef_).all() or not np.isfinite(model.sigma_).all() or not np.isfinite(model.scores_).all():
        raise ValueError('ARD did not converge to a finite posterior')
    return model,center,ym-center@model.coef_


def replace_volume(control,base,oldfactor,factor):
    day=['route','date'];volume=base.groupby(day,as_index=False).prediction.sum().rename(columns={'prediction':'base_volume'})
    volume=volume.merge(oldfactor,on=day,validate='one_to_one')
    volume=volume.merge(factor,on=day,validate='one_to_one')
    current=control.groupby(day,as_index=False).prediction.sum().rename(columns={'prediction':'current_volume'})
    volume=volume.merge(current,on=day,validate='one_to_one')
    volume['old_volume']=volume.base_volume*volume.oldfactor
    volume['new_volume']=volume.base_volume*volume.factor
    for name in ['old','new','current']:
        total=volume.groupby('date')[name+'_volume'].transform('sum')
        if not total.gt(0).all():raise ValueError('Undefined network fraction')
        volume[name+'_share']=volume[name+'_volume']/total
    volume['estimated_share']=(volume.new_share+volume.current_share-volume.old_share).clip(lower=0)
    volume.loc[volume.current_volume.eq(0),'estimated_share']=0.
    denominator=volume.groupby('date').estimated_share.transform('sum')
    if not denominator.gt(0).all():raise ValueError('Empty adjusted network allocation')
    volume['estimated_volume']=volume.estimated_share/denominator*volume.groupby('date').new_volume.transform('sum')
    np.testing.assert_allclose(volume.groupby('date').old_volume.sum(),control.groupby('date').prediction.sum(),rtol=1e-12,atol=1e-8)
    ratios=np.divide(volume.estimated_volume.to_numpy(),volume.current_volume.to_numpy(),out=np.ones(len(volume)),where=volume.current_volume.gt(0).to_numpy())
    result=control.merge(volume[day].assign(ratio=ratios),on=day,validate='many_to_one')
    result['prediction']*=result.ratio
    return result.drop(columns='ratio'),volume


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
    model,center,intercept=ard(x,y,w);coef=model.coef_
    newmean=test@coef+intercept;factor=np.exp(np.clip(shrink*newmean,-np.log(2),np.log(2)))
    base=raw_frame(ROOT/'bayes_volume_verified_july/inner'/cutoff/f'raw_{cutoff}.csv',cutoff,end)
    control=raw_frame(CONTROL/f'raw_{cutoff}.csv',cutoff,end)
    raw,volume=replace_volume(control,base,future[['route','date']].assign(oldfactor=oldfactor),future[['route','date']].assign(factor=factor))
    out.mkdir(parents=True,exist_ok=True)
    past.to_csv(out/'training.csv',sep=';',index=False,date_format='%Y-%m-%d')
    future.assign(oldmean=oldmean,newmean=newmean,attenuation_sd=sd,shrink=shrink,oldfactor=oldfactor,factor=factor).to_csv(out/'future.csv',sep=';',index=False,date_format='%Y-%m-%d')
    volume.to_csv(out/'volume.csv',sep=';',index=False,date_format='%Y-%m-%d')
    np.savez(out/'matrices.npz',x=x,y=y,w=w,test=test,coef=coef,intercept=intercept,oldcoef=old,covariance=covariance,center=center,ard_covariance=model.sigma_,ard_precision=model.lambda_)
    write_json(out/'fit.json',dict(cutoff=cutoff,rows=len(x),features=x.shape[1],sigma2=sigma2,precision=precision,
        ard_params=model.get_params(),noise_precision=float(model.alpha_),iterations=int(model.n_iter_),
        active_features=int((model.lambda_<model.threshold_lambda).sum()),evidence_trace=model.scores_,
        latest_target_date=str(past.date.max().date()),
        uncertainty='Fixed old Gaussian predictive attenuation; ARD relevance is conditional, not hidden-score confidence.',
        fit_seconds=time.monotonic()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    return raw


def forecast(history,cutoff,end,recipe,out):
    control=raw_frame(CONTROL/f'raw_{cutoff}.csv',cutoff,end)
    if recipe=='control':return control
    if recipe not in ['half','full']:raise ValueError('Unknown ARD recipe')
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
        ['bayes_direct','bayes_volume','gp_errors','windows','ridge','combine','experiment']]
    paths+=list((ROOT/'verified_july').glob('posterior_*_1.0_224_route_july_verifiedTrue.json'))+list((ROOT/'verified_july').glob('factors_*_1.0_224_route_july_verifiedTrue.csv'))
    paths+=list((ROOT/'verified_july/inner').glob('*/daily.csv'))+list((ROOT/'bayes_volume_verified_july/inner').glob('*/raw_*.csv'))+list(CONTROL.glob('raw_*.csv'))
    (out/'native_ard_fit.py.txt').write_text(__import__('inspect').getsource(ARDRegression.fit))
    write_json(out/'run_started.json',dict(job_id=os.environ['SLURM_JOB_ID'],command=os.sys.argv,
        versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','optuna']},
        new_trial_limit=3,new_cpu_allocation_seconds_limit=300,old_phases=['P34','P41','P43','P70'],
        sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))
    history=load_history(args.history)
    if args.pilot:
        raw=forecast(history,*FINAL,'full',out);raw.to_csv(out/'pilot_raw.csv',sep=';',index=False,date_format='%Y-%m-%d')
        repeated=fit(history,*FINAL,out/'repeat');np.testing.assert_array_equal(raw.prediction,repeated.prediction)
    else:
        run_study(history,out,'ard_annual',lambda c,e,p:forecast(history,c,e,p['recipe'],out),
            dict(sampler='grid',space={'recipe':['control','half','full']},trials=3))
        trials=pd.read_csv(out/'ard_annual/trials.csv',sep=';');best=trials.loc[trials.params_recipe.ne('control')].sort_values('value',ascending=False).iloc[0]
        save_candidate(history,out/'alternative','ard_annual_best_new',lambda c,e:forecast(history,c,e,best.params_recipe,out))
        write_json(out/'best_new.json',dict(recipe=best.params_recipe,value=float(best.value)))
    write_json(out/'completed.json',dict(job_id=os.environ['SLURM_JOB_ID'],elapsed_seconds=time.monotonic()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def self_check():
    rng=np.random.default_rng(42);x=rng.normal(size=(200,3));x[:,2]=0;y=.2+x[:,0]*.5+rng.normal(size=200)*.05;w=np.linspace(.5,1.5,200)
    model,center,a=ard(x,y,w)
    assert abs(model.coef_[0]-.5)<.02 and abs(a-.2)<.02 and model.coef_[2]==0
    active=model.lambda_<model.threshold_lambda;xc=x-center
    precision=model.alpha_*(xc[:,active].T*w)@xc[:,active]+np.diag(model.lambda_[active])
    np.testing.assert_allclose(precision@model.sigma_,np.eye(active.sum()),atol=1e-8)
    np.testing.assert_allclose(model.coef_[active],model.alpha_*model.sigma_@(xc[:,active].T@(w*(y-np.average(y,weights=w)))),atol=1e-8)
    frame=pd.DataFrame(dict(route=[1,5,7],date=pd.to_datetime(['2025-11-01']*3),hour=[0,0,0],prediction=[4.,0.,6.]))
    old=frame[['route','date']].assign(oldfactor=1.);new=frame[['route','date']].assign(factor=[2.,1.,1.])
    identity,_=replace_volume(frame,frame,old,frame[['route','date']].assign(factor=1.))
    np.testing.assert_allclose(identity.prediction,frame.prediction)
    updated,_=replace_volume(frame,frame,old,new)
    np.testing.assert_allclose(updated.prediction,[8.,0.,6.])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--history');parser.add_argument('--output');parser.add_argument('--pilot',action='store_true');parser.add_argument('--self-check',action='store_true')
    args=parser.parse_args()
    if args.self_check:self_check()
    elif not args.history or not args.output:parser.error("history and output are required")
    else:run(args)
