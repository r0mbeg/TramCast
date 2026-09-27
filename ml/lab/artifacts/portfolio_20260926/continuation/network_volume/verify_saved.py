"""Run from ml with PYTHONPATH=.: independent aggregate targets and network raw audit."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json
import numpy as np
import pandas as pd
from experiments.portfolio_network_volume import CONTROL,features
from experiments.portfolio_timesfm_errors import SOURCE,TEACHERS
from experiments.portfolio_verify import read

OUT=Path('artifacts/portfolio_20260926/continuation/network_volume')
truth=read('artifacts/hourly_clean.csv').groupby(['route','date'],as_index=False).boardings.sum()
hashes={};fits=[]
for phase in ['pilot','study']:
    info=json.loads((OUT/phase/'run_started.json').read_text())
    for name,sha in info['sha256'].items():
        path=Path(name.removeprefix('/beegfs/home/m.persiyanov/codex_runs/tram-portfolio-20260926/'))
        assert hashlib.sha256(path.read_bytes()).hexdigest()==sha,path
    hashes[phase]=len(info['sha256'])

def table(origin, observed, cutoff):
    frame=read(SOURCE/origin/'daily.csv').merge(read(TEACHERS/origin/'daily.csv').rename(columns={'prediction':'timesfm'}),on=['route','date'],validate='one_to_one')
    assert len(frame)==610 and frame.horizon.eq((frame.date-pd.Timestamp(origin)).dt.days).all()
    if observed:
        assert frame.date.max()<=pd.Timestamp(cutoff)
        frame=frame.merge(truth.loc[truth.date.le(cutoff)],on=['route','date'],validate='one_to_one')
        assert len(frame)==610
    counts=['base','ridge','regime','direct','timesfm']+(['boardings'] if observed else [])
    common=['effective_weekday','daytype','horizon','off','summer','temperature_2m_mean','precipitation_sum','daylight_duration']
    result=frame.groupby('date')[counts].sum().join(frame.groupby('date')[common].first())
    for days in [7,14,28]:
        expected=frame.assign(value=frame.base*frame[f'ratio{days}']).groupby('date').value.sum()
        result[f'ratio{days}']=expected/result.base
    result=result.reset_index();result.insert(0,'origin',pd.Timestamp(origin))
    result['origin']=result.origin.astype('datetime64[ns]')
    return result.assign(route=0)

for path in sorted(OUT.glob('*/fits_*/*/training.csv')):
    folder=path.parent;cutoff=folder.name
    past=read(path);past['origin']=pd.to_datetime(past.origin)
    origins=pd.date_range('2025-01-31',pd.Timestamp(cutoff)-pd.Timedelta(days=61),freq='ME')
    expected=pd.concat([table(str(o.date()),True,cutoff) for o in origins],ignore_index=True)
    expected=expected.loc[expected.date.gt(pd.Timestamp(cutoff)-pd.Timedelta(days=224))].reset_index(drop=True)
    info=json.loads((folder/'fit.json').read_text());model=info['model']
    pd.testing.assert_frame_equal(past.drop(columns=['target','weight'],errors='ignore'),expected)
    assert info['rows']==len(past) and info['latest_target_date']<=cutoff
    future=read(folder/'future.csv');future['origin']=pd.to_datetime(future.origin)
    expected=table(cutoff,False,cutoff)
    pd.testing.assert_frame_equal(future.drop(columns=['ratio','factor','prediction'],errors='ignore'),expected)
    if model=='absolute':
        repeats=past.groupby('date').base.transform('size');weights=past.base/repeats
        np.testing.assert_allclose(past.target,past.boardings/past.base,rtol=0,atol=1e-15)
        np.testing.assert_allclose(past.weight,weights/weights.mean(),rtol=1e-12)
        estimate=np.resize([.8,1.,1.2],len(past))
        np.testing.assert_allclose(np.sum(abs(estimate-past.target)*past.weight),
            np.sum(abs(estimate*past.base-past.boardings)/repeats)/weights.mean(),rtol=1e-12)
        np.testing.assert_array_equal(future.factor,future.ratio.clip(.5,2))
        assert info['features']==28 and info['model_reload_exact'] and info['iterations']==100
        assert hashlib.sha256((folder/'model.pkl').read_bytes()).hexdigest()==info['model_sha256']
    else:
        assert model=='bayes' and info['features']==34
        x=(features(past,True)-np.array(info['feature_mean']))/np.array(info['feature_scale'])
        z=(features(future,True)-np.array(info['feature_mean']))/np.array(info['feature_scale'])
        weights=1/past.groupby(['route','date']).base.transform('size').to_numpy()
        weights*=past.base.to_numpy()/np.average(past.base,weights=weights)
        covariance=np.linalg.inv(info['coefficient_precision']*np.eye(34)+info['noise_precision']*(x.T@(weights[:,None]*x)))
        mean=z@np.array(info['coefficient_mean'])+info['intercept']
        variance=np.sum((z@covariance)*z,axis=1)+1/info['noise_precision']
        factor=np.exp(np.clip(mean/(1+variance/.01),-np.log(2),np.log(2)))
        np.testing.assert_allclose(future.factor,factor,rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(future.prediction,future.base*future.factor,rtol=1e-12,atol=1e-8)
    np.testing.assert_array_equal(features(past,model=='bayes'),features(past.assign(boardings=1e100),model=='bayes'))
    raw=read(folder/f'raw_{cutoff}.csv');shape=read(CONTROL/f'raw_{cutoff}.csv')
    total=shape.groupby('date').prediction.transform('sum')
    volume=shape[['date']].merge(future[['date','prediction']],on='date',validate='many_to_one').prediction
    np.testing.assert_allclose(raw.prediction,shape.prediction*volume/total,rtol=1e-12,atol=1e-8)
    np.testing.assert_allclose(raw.groupby('date').prediction.sum(),future.set_index('date').prediction,rtol=1e-12,atol=1e-8)
    assert raw.loc[shape.prediction.eq(0),'prediction'].eq(0).all()
    fits.append(dict(phase=folder.relative_to(OUT).parts[0],model=model,cutoff=cutoff,rows=len(past)))

count=0;study=OUT/'study/network_volume'
folders=sorted(study.glob('trial_*'))+[study/'selected']+[p for p in [OUT/'study/alternative',OUT/'study/seed73'] if p.exists()]
for folder in folders:
    info=json.loads((folder/'parameters.json' if folder.name!='selected' else study/'selection.json').read_text())
    recipe=info.get('params',info)['recipe'];seed=73 if folder.name=='seed73' else 42
    for path in folder.glob('raw_*.csv'):
        actual=read(path);shape=read(CONTROL/path.name)
        if recipe=='control':np.testing.assert_array_equal(actual.prediction,shape.prediction)
        else:
            model,strength=recipe.split('_');value=.5 if strength=='half' else 1.
            learned=read(OUT/f'study/fits_{model}_{seed}'/path.stem[4:]/path.name)
            np.testing.assert_allclose(actual.prediction,value*learned.prediction+(1-value)*shape.prediction,rtol=1e-12,atol=1e-8)
        total=actual.groupby('date').prediction.transform('sum')
        base=shape.groupby('date').prediction.transform('sum')
        np.testing.assert_allclose(actual.prediction/total,shape.prediction/base,rtol=1e-12,atol=1e-12)
        assert actual.loc[shape.prediction.eq(0),'prediction'].eq(0).all()
        count+=1
seed=OUT/'study/seed73';seed_exact=None
if seed.exists():
    meta=json.loads((seed/'parameters.json').read_text());reference=Path(meta['reference'])
    for path in reference.glob('raw_*.csv'):
        np.testing.assert_array_equal(read(path).prediction,read(seed/path.name).prediction)
    pd.testing.assert_frame_equal(read(reference/'submission.csv'),read(seed/'submission.csv'))
    seed_exact=True
result=dict(checked_at=datetime.now(timezone.utc).isoformat(),input_code_sha256=hashes,fits=fits,
    raw_forecasts_checked=count,causal_network_targets_reconstructed=True,network_prediction_totals_preserved=True,
    bayesian_covariance_predictive_factors_reconstructed=True,absolute_daily_loss_identity=True,
    fixed_029_within_day_route_hourshares=True,exact_029_control=True,structural_zeros=True,seed73_all5_raw_exact=seed_exact,
    local_hgb_inference='notloaded: localsklearn1.9 differs fromsaved1.8; ownreload onZhores verified',
    audit_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
