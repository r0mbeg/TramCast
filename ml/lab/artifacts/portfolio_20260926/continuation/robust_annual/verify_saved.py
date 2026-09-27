"""Independent saved P70 arithmetic, causal input, MAP and shape audit; no fitting."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.portfolio_experiment import FINAL,WINDOWS,load_history,write_json
from experiments.portfolio_bayes_direct import annual_features
from experiments.portfolio_windows import inner_windows

ROOT=Path('artifacts/portfolio_20260926/continuation');OUT=ROOT/'robust_annual'
CONTROL=ROOT/'tabular_shape/study/tabular_shape/selected'


def read(path):return pd.read_csv(path,sep=';',parse_dates=['date'],float_precision='round_trip')


def run():
    history=load_history('artifacts/hourly_clean.csv');verified=[];sha={}
    for phase in ['pilot','study']:
        start=json.loads((OUT/phase/'run_started.json').read_text())
        for name,digest in start['sha256'].items():
            if name.startswith('/beegfs/'):
                name=name.split('/tram-portfolio-20260926/',1)[1]
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
        sha[phase]=len(start['sha256'])
    for folder in sorted(OUT.glob('*/fits/*'))+[OUT/'pilot/repeat']:
        meta=json.loads((folder/'fit.json').read_text());cutoff=meta['cutoff']
        truth=history.loc[history.date.le(cutoff)].groupby(['route','date'],as_index=False).boardings.sum()
        pieces=[]
        for origin,stop in inner_windows(cutoff):
            assert pd.Timestamp(stop)<=pd.Timestamp(cutoff)
            table=read(ROOT/'verified_july/inner'/origin/'daily.csv')
            assert 'boardings' not in table and table.date.min()>pd.Timestamp(origin)
            pieces.append(table.merge(truth,on=['route','date'],validate='one_to_one'))
        past=pd.concat(pieces,ignore_index=True)
        past=past.loc[past.route.ne(5)&past.base.gt(0)&past.date.gt(pd.Timestamp(cutoff)-pd.Timedelta(days=224))].copy()
        saved=read(folder/'training.csv');assert len(saved)==len(past)==meta['rows']
        for column in past:
            if np.issubdtype(past[column].dtype,np.number):np.testing.assert_allclose(past[column],saved[column],rtol=1e-12,atol=1e-12)
            else:assert np.array_equal(past[column].astype(str),saved[column].astype(str)),column
        old=json.loads((ROOT/'verified_july'/f'posterior_{cutoff}_1.0_224_route_july_verifiedTrue.json').read_text())
        matrices=np.load(folder/'matrices.npz');x=matrices['x'];y=matrices['y'];w=matrices['w'];b=matrices['coef'];a=float(matrices['intercept'])
        np.testing.assert_allclose(x,(annual_features(past,True)-old['feature_mean'])/old['feature_scale'],rtol=1e-12,atol=1e-12)
        # libm log differs by one double-precision ULP across macOS/Linux; counts stay exact.
        np.testing.assert_allclose(y,np.clip(np.log((past.boardings.to_numpy()+100)/(past.base.to_numpy()+100)),-np.log(2),np.log(2)),rtol=1e-14,atol=1e-15)
        np.testing.assert_array_equal(w,1/past.groupby(['route','date']).base.transform('size').to_numpy())
        sigma2=1/old['noise_precision'];precision=old['coefficient_precision']
        assert meta['nu']==4 and meta['sigma2']==sigma2 and meta['precision']==precision
        residual=y-x@b-a;latent=5/(4+residual**2/sigma2)
        gradient=-(x.T@(w*latent*residual))/sigma2+precision*b
        assert np.max(np.abs(gradient))/precision<1e-6
        assert abs(np.sum(w*latent*residual))/np.sum(w)<1e-7
        trace=np.array(meta['objective_trace']);assert np.all(np.diff(trace)<=1e-8) and len(trace)<=101
        objective=float(np.sum(w*2.5*np.log1p(residual**2/(4*sigma2)))+precision/2*(b@b))
        np.testing.assert_allclose(objective,trace[-1],rtol=1e-12,atol=1e-10)
        center=np.average(x,axis=0,weights=w);xc=x-center
        covariance=np.linalg.inv((xc.T*w)@xc/sigma2+precision*np.eye(x.shape[1]))
        np.testing.assert_allclose(covariance,matrices['covariance'],rtol=1e-10,atol=1e-13)
        future=read(folder/'future.csv');assert 'boardings' not in future and future.date.min()>pd.Timestamp(cutoff)
        test=(annual_features(future,True)-old['feature_mean'])/old['feature_scale']
        np.testing.assert_allclose(test,matrices['test'],rtol=1e-12,atol=1e-12)
        sd=np.sqrt(sigma2+np.sum((test@covariance)*test,axis=1));shrink=1/(1+(sd/.1)**2)
        oldmean=test@np.array(old['coefficient_mean'])+old['intercept'];newmean=test@b+a
        for name,value in dict(attenuation_sd=sd,shrink=shrink,oldmean=oldmean,newmean=newmean,
            oldfactor=np.exp(np.clip(shrink*oldmean,-np.log(2),np.log(2))),factor=np.exp(np.clip(shrink*newmean,-np.log(2),np.log(2)))).items():
            np.testing.assert_allclose(future[name],value,rtol=1e-11,atol=1e-12)
        base=read(ROOT/'bayes_volume_verified_july/inner'/cutoff/f'raw_{cutoff}.csv')
        base=base.merge(future[['route','date','oldfactor','factor']],on=['route','date'],validate='many_to_one')
        oldnet=(base.prediction*base.oldfactor).groupby(base.date).sum();newnet=(base.prediction*base.factor).groupby(base.date).sum()
        network=read(folder/'network.csv').set_index('date');np.testing.assert_allclose(network.old_volume,oldnet,rtol=1e-12,atol=1e-8)
        np.testing.assert_allclose(network.new_volume,newnet,rtol=1e-12,atol=1e-8)
        np.testing.assert_allclose(network.ratio,newnet/oldnet,rtol=1e-12,atol=1e-12)
        verified.append(str(folder))
    np.testing.assert_array_equal(read(OUT/'pilot/pilot_raw.csv').prediction,read(OUT/'study/fits/2025-10-31/raw_2025-10-31.csv').prediction)
    raw_count=0
    for path in OUT.glob('**/raw_*.csv'):
        cutoff=path.stem[4:];raw=read(path);control=read(CONTROL/path.name)
        pd.testing.assert_frame_equal(raw[['route','date','hour']],control[['route','date','hour']])
        if path.parent.name in ['selected','alternative']:
            recipe=json.loads((OUT/'study/robust_annual/selection.json' if path.parent.name=='selected' else OUT/'study/best_new.json').read_text())
            recipe=recipe.get('params',recipe)['recipe']
        elif path.parent.name.startswith('trial_'):recipe=json.loads((path.parent/'parameters.json').read_text())['recipe']
        else:recipe='full'
        if recipe=='control':expected=control.prediction.to_numpy()
        else:
            phase='pilot' if 'pilot' in path.parts else 'study'
            net=read(OUT/phase/'fits'/cutoff/'network.csv').set_index('date')
            full=control.prediction.to_numpy()*control.date.map(net.ratio).to_numpy()
            strength=.5 if recipe=='half' else 1.
            expected=strength*full+(1-strength)*control.prediction.to_numpy()
        np.testing.assert_allclose(raw.prediction,expected,rtol=1e-12,atol=1e-9)
        assert raw.loc[control.prediction.eq(0),'prediction'].eq(0).all()
        raw_count+=1
    result=dict(sha256_files=sha,verified_fits=verified,raw_forecasts=raw_count,
        checks='Causal completed teachers, fixed targets, inverse repeat weights, saved scaler/X, t MAP stationary gradient and monotonic objective, frozen prior/noise/attenuation, network totals, fixed route/hour allocation, mixes and zeros.',
        audit_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(OUT/'independent_verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':run()
