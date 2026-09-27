"""Independent saved P72 arithmetic, causal input, posterior and shape audit; no fitting."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.portfolio_experiment import FINAL,WINDOWS,load_history,write_json
from experiments.portfolio_bayes_direct import annual_features
from experiments.portfolio_windows import inner_windows

ROOT=Path('artifacts/portfolio_20260926/continuation');OUT=ROOT/'ard_annual'
CONTROL=ROOT/'adaptive_shape/study/adaptive_shape/selected'


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
        assert meta['sigma2']==sigma2 and meta['precision']==precision
        params=meta['ard_params'];assert params==dict(max_iter=1000,tol=1e-5,alpha_1=1e-6,alpha_2=1e-6,lambda_1=1e-6,lambda_2=1e-6,
            compute_score=True,threshold_lambda=10000.,fit_intercept=False,copy_X=True,verbose=False)
        assert meta['iterations']<1000 and np.isfinite(meta['evidence_trace']).all()
        lambdas=matrices['ard_precision'];active=lambdas<params['threshold_lambda'];alpha=meta['noise_precision']
        center=np.average(x,axis=0,weights=w);xc=x-center;ym=np.average(y,weights=w)
        np.testing.assert_allclose(center,matrices['center'],rtol=0,atol=1e-14)
        assert active.sum()==meta['active_features'] and np.all(b[~active]==0)
        posterior=matrices['ard_covariance'];precision_matrix=alpha*(xc[:,active].T*w)@xc[:,active]+np.diag(lambdas[active])
        np.testing.assert_allclose(precision_matrix@posterior,np.eye(active.sum()),rtol=1e-8,atol=1e-8)
        np.testing.assert_allclose(b[active],alpha*posterior@(xc[:,active].T@(w*(y-ym))),rtol=1e-8,atol=1e-10)
        np.testing.assert_allclose(a,ym-center@b,rtol=1e-12,atol=1e-12)
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
        current=read(CONTROL/f'raw_{cutoff}.csv');day=['route','date']
        expected=base.groupby(day,as_index=False).agg(old_volume=('prediction',lambda v:0.))
        expected['old_volume']=(base.prediction*base.oldfactor).groupby([base.route,base.date]).sum().to_numpy()
        expected['new_volume']=(base.prediction*base.factor).groupby([base.route,base.date]).sum().to_numpy()
        expected['current_volume']=current.groupby(day).prediction.sum().to_numpy()
        for name in ['old','new','current']:expected[name+'_share']=expected[name+'_volume']/expected.groupby('date')[name+'_volume'].transform('sum')
        share=np.maximum(0,expected.new_share.to_numpy()+expected.current_share.to_numpy()-expected.old_share.to_numpy())
        share[expected.current_volume.eq(0).to_numpy()]=0
        expected['estimated_share']=share
        expected['estimated_volume']=share/expected.groupby('date').estimated_share.transform('sum').to_numpy()*expected.groupby('date').new_volume.transform('sum').to_numpy()
        volume=read(folder/'volume.csv');pd.testing.assert_frame_equal(volume[day],expected[day])
        for name in expected.columns[2:]:np.testing.assert_allclose(volume[name],expected[name],rtol=1e-11,atol=1e-8)
        np.testing.assert_allclose(volume.groupby('date').old_volume.sum(),current.groupby('date').prediction.sum(),rtol=1e-12,atol=1e-8)
        np.testing.assert_allclose(volume.groupby('date').estimated_volume.sum(),volume.groupby('date').new_volume.sum(),rtol=1e-12,atol=1e-8)
        verified.append(str(folder))
    np.testing.assert_array_equal(read(OUT/'pilot/pilot_raw.csv').prediction,read(OUT/'study/fits/2025-10-31/raw_2025-10-31.csv').prediction)
    raw_count=0
    for path in OUT.glob('**/raw_*.csv'):
        cutoff=path.stem[4:];raw=read(path);control=read(CONTROL/path.name)
        pd.testing.assert_frame_equal(raw[['route','date','hour']],control[['route','date','hour']])
        if path.parent.name in ['selected','alternative']:
            recipe=json.loads((OUT/'study/ard_annual/selection.json' if path.parent.name=='selected' else OUT/'study/best_new.json').read_text())
            recipe=recipe.get('params',recipe)['recipe']
        elif path.parent.name.startswith('trial_'):recipe=json.loads((path.parent/'parameters.json').read_text())['recipe']
        else:recipe='full'
        if recipe=='control':expected=control.prediction.to_numpy()
        else:
            phase='pilot' if 'pilot' in path.parts else 'study'
            volume=read(OUT/phase/'fits'/cutoff/'volume.csv')
            ratio=np.divide(volume.estimated_volume.to_numpy(),volume.current_volume.to_numpy(),out=np.ones(len(volume)),where=volume.current_volume.gt(0).to_numpy())
            pairs=volume[['route','date']].assign(ratio=ratio)
            values=control.merge(pairs,on=['route','date'],validate='many_to_one').ratio.to_numpy()
            full=control.prediction.to_numpy()*values
            strength=.5 if recipe=='half' else 1.
            expected=strength*full+(1-strength)*control.prediction.to_numpy()
        np.testing.assert_allclose(raw.prediction,expected,rtol=1e-12,atol=1e-9)
        assert raw.loc[control.prediction.eq(0),'prediction'].eq(0).all()
        raw_count+=1
    result=dict(sha256_files=sha,verified_fits=verified,raw_forecasts=raw_count,
        checks='Causal completed teachers, fixed targets, inverse repeat weights, saved scaler/X, native ARD Gaussian posterior covariance and normal equations, relevance pruning, frozen old attenuation, route fraction component replacement, network totals, fixed hourly shares, mixes and zeros.',
        audit_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(OUT/'independent_verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':run()
