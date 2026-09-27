"""Run from ml with PYTHONPATH=. : audit saved P52 inputs/posterior without refitting."""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib
import numpy as np
import pandas as pd
from experiments.portfolio_timesfm_errors import features, SOURCE, TEACHERS, SHAPE, VOLUME
from experiments.portfolio_verify import read

ROOT=Path('artifacts/portfolio_20260926/continuation')
OUT=ROOT/'timesfm_errors/study'
info=json.loads((OUT/'run_started.json').read_text())
for name,sha in info['sha256'].items():
    path=Path(name.removeprefix('/beegfs/home/m.persiyanov/codex_runs/tram-portfolio-20260926/'))
    assert hashlib.sha256(path.read_bytes()).hexdigest()==sha,path
truth=read('artifacts/hourly_clean.csv').groupby(['route','date'],as_index=False).boardings.sum()
checked=[]
for folder in sorted((OUT/'fits').iterdir()):
    cutoff=folder.name
    posterior=json.loads((folder/'posterior.json').read_text())
    past=read(folder/'past.csv')
    past['origin']=pd.to_datetime(past.origin)
    pieces=[]
    for origin,end in posterior['origins']:
        assert pd.Timestamp(end)<=pd.Timestamp(cutoff)
        frame=read(SOURCE/origin/'daily.csv').merge(read(TEACHERS/origin/'daily.csv').rename(columns={'prediction':'timesfm'}),on=['route','date'],validate='one_to_one')
        frame=frame.merge(truth.loc[truth.date.le(cutoff)],on=['route','date'],validate='one_to_one').assign(origin=pd.Timestamp(origin))
        assert len(frame)==610
        pieces.append(frame)
    expected=pd.concat(pieces,ignore_index=True)
    expected['origin']=expected.origin.astype('datetime64[ns]')
    pd.testing.assert_frame_equal(past,expected)
    future=read(folder/'future.csv')
    assert 'boardings' not in future
    end=str(future.date.max().date())
    expected_future=read(SOURCE/cutoff/'daily.csv').merge(read(TEACHERS/cutoff/'daily.csv').rename(columns={'prediction':'timesfm'}),on=['route','date'],validate='one_to_one')
    pd.testing.assert_frame_equal(future.drop(columns='factor'),expected_future)
    selected=past.loc[past.route.ne(5)&past.base.gt(0)&past.date.gt(pd.Timestamp(cutoff)-pd.Timedelta(days=224))]
    weights=1/selected.groupby(['route','date']).base.transform('size').to_numpy()
    x=features(selected)
    mean=np.average(x,axis=0,weights=weights)
    var=np.average((x-mean)**2,axis=0,weights=weights)
    np.testing.assert_allclose(mean,posterior['feature_mean'],rtol=1e-9,atol=1e-9)
    scale=np.array(posterior['feature_scale'])
    np.testing.assert_allclose(scale,np.where(var<1e-25,1,np.sqrt(var)),rtol=1e-8,atol=1e-8)
    design=(x-mean)/scale
    design-=np.average(design,axis=0,weights=weights)
    precision=posterior['noise_precision']*((design*weights[:,None]).T@design)+posterior['coefficient_precision']*np.eye(x.shape[1])
    covariance=np.linalg.inv(precision)
    new=(features(future)-mean)/scale
    mu=new@np.array(posterior['coefficient_mean'])+posterior['intercept']
    variance=np.einsum('ij,jk,ik->i',new,covariance,new)+1/posterior['noise_precision']
    expected_factor=np.exp(np.clip(mu/(1+variance/.01),-np.log(2),np.log(2)))
    np.testing.assert_allclose(future.factor,expected_factor,rtol=1e-9,atol=1e-9)
    raw=read(folder/f'raw_{cutoff}.csv')
    base=read(VOLUME/cutoff/f'raw_{cutoff}.csv')
    shape=read(SHAPE/f'raw_{cutoff}.csv')
    joined=base.merge(future[['route','date','factor']],on=['route','date'],validate='many_to_one')
    joined['prediction']*=joined.factor
    volume=joined.groupby(['route','date']).prediction.transform('sum')
    denominator=shape.groupby(['route','date']).prediction.transform('sum')
    shares=shape.prediction.div(denominator.where(denominator.gt(0))).fillna(0)
    np.testing.assert_allclose(raw.prediction,volume*shares,rtol=1e-12,atol=1e-8)
    assert raw.loc[raw.route.eq(5)|raw.hour.between(1,4),'prediction'].eq(0).all()
    checked.append(dict(cutoff=cutoff,training_rows=len(selected),feature_count=x.shape[1],posterior_factor_max_error=float(np.max(np.abs(future.factor-expected_factor)))))
count=0
for path in sorted((OUT/'timesfm_errors').rglob('raw_*.csv')):
    params=json.loads((path.parent/('parameters.json' if path.parent.name.startswith('trial_') else '../selection.json')).read_text())
    augmented=params.get('params',params)['augmented']
    reference=OUT/'fits'/path.stem[4:]/path.name if augmented else SHAPE/path.name
    np.testing.assert_array_equal(read(path).prediction,read(reference).prediction)
    count+=1
result=dict(checked_at=datetime.now(timezone.utc).isoformat(),input_code_sha256=len(info['sha256']),fits=checked,raw_forecasts_checked=count,exact_024_control=True,past_targets_teachers_verified=True,posterior_predictive_variance_factors_reconstructed=True,fixed_024_hourshares=True,decision='control won W1/W2; no duplicate archive',audit_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT.parent/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
