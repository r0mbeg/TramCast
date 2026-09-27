"""Run from ml with PYTHONPATH=. : independent P57 target, posterior and raw audit."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import pandas as pd
from experiments.portfolio_bayes_direct import annual_features
from experiments.portfolio_timesfm_errors import SOURCE,TEACHERS,SHAPE,VOLUME
from experiments.portfolio_verify import read

OUT=Path('artifacts/portfolio_20260926/continuation/route_allocation')
truth=read('artifacts/hourly_clean.csv').groupby(['route','date'],as_index=False).boardings.sum()
hashes={};fits=[]
for phase in ['pilot','study']:
    info=json.loads((OUT/phase/'run_started.json').read_text())
    for name,sha in info['sha256'].items():
        path=Path(name.removeprefix('/beegfs/home/m.persiyanov/codex_runs/tram-portfolio-20260926/'))
        assert hashlib.sha256(path.read_bytes()).hexdigest()==sha,path
    hashes[phase]=len(info['sha256'])
for path in sorted(OUT.glob('*/fits/*/past.csv')):
    folder=path.parent;cutoff=folder.name
    past=read(path);past['origin']=pd.to_datetime(past.origin)
    origins=pd.date_range('2025-01-31',pd.Timestamp(cutoff)-pd.Timedelta(days=61),freq='ME')
    pieces=[]
    for origin in origins:
        name=str(origin.date())
        frame=read(SOURCE/name/'daily.csv').merge(read(TEACHERS/name/'daily.csv').rename(columns={'prediction':'timesfm'}),on=['route','date'],validate='one_to_one')
        frame=frame.merge(truth.loc[truth.date.le(cutoff)],on=['route','date'],validate='one_to_one')
        assert len(frame)==610 and origin+pd.Timedelta(days=61)<=pd.Timestamp(cutoff)
        pieces.append(frame.assign(origin=origin))
    expected=pd.concat(pieces,ignore_index=True);expected['origin']=expected.origin.astype('datetime64[ns]')
    pd.testing.assert_frame_equal(past.drop(columns=['network_base','network_actual','relative_boardings']),expected)
    np.testing.assert_array_equal(past.network_base,expected.groupby(['origin','date']).base.transform('sum'))
    np.testing.assert_array_equal(past.network_actual,expected.groupby(['origin','date']).boardings.transform('sum'))
    np.testing.assert_allclose(past.relative_boardings,past.boardings*past.network_base/past.network_actual,rtol=1e-12,atol=1e-8)
    training=past.loc[past.route.ne(5)&past.base.gt(0)&past.date.gt(pd.Timestamp(cutoff)-pd.Timedelta(days=224))].copy()
    info=json.loads((folder/'posterior.json').read_text())
    assert info['rows']==len(training) and info['latest_target_date']<=cutoff and info['feature_count']==97
    future=read(folder/'future.csv')
    expected=read(SOURCE/cutoff/'daily.csv').merge(read(TEACHERS/cutoff/'daily.csv').rename(columns={'prediction':'timesfm'}),on=['route','date'],validate='one_to_one')
    pd.testing.assert_frame_equal(future.drop(columns='factor'),expected)
    x=(annual_features(training,True)-np.array(info['feature_mean']))/np.array(info['feature_scale'])
    z=(annual_features(future,True)-np.array(info['feature_mean']))/np.array(info['feature_scale'])
    w=1/training.groupby(['route','date']).base.transform('size').to_numpy()
    covariance=np.linalg.inv(info['coefficient_precision']*np.eye(97)+info['noise_precision']*(x.T@(w[:,None]*x)))
    mean=z@np.array(info['coefficient_mean'])+info['intercept']
    variance=np.sum((z@covariance)*z,axis=1)+1/info['noise_precision']
    factors=np.exp(np.clip(mean/(1+variance/0.01),-np.log(2),np.log(2)))
    np.testing.assert_allclose(future.factor,factors,rtol=1e-10,atol=1e-12)
    raw=read(folder/f'raw_{cutoff}.csv');shape=read(SHAPE/f'raw_{cutoff}.csv')
    base=read(VOLUME/cutoff/f'raw_{cutoff}.csv').merge(future[['route','date','factor']],on=['route','date'],validate='many_to_one')
    base['prediction']*=base.factor
    volumes=base.groupby(['route','date']).prediction.transform('sum')
    route_total=shape.groupby(['route','date']).prediction.transform('sum')
    result=volumes*shape.prediction.div(route_total.where(route_total.gt(0))).fillna(0)
    result*=shape.groupby('date').prediction.transform('sum')/result.groupby(shape.date).transform('sum')
    np.testing.assert_allclose(raw.prediction,result,rtol=1e-12,atol=1e-8)
    np.testing.assert_allclose(raw.groupby('date').prediction.sum(),shape.groupby('date').prediction.sum(),rtol=1e-12,atol=1e-8)
    assert raw.loc[shape.prediction.eq(0),'prediction'].eq(0).all()
    fits.append(dict(phase=folder.relative_to(OUT).parts[0],cutoff=cutoff,rows=len(training)))
count=0;study=OUT/'study/route_allocation'
for folder in sorted(study.glob('trial_*'))+[study/'selected']:
    info=json.loads((folder/'parameters.json' if folder.name!='selected' else study/'selection.json').read_text())
    recipe=info.get('params',info)['recipe']
    for path in folder.glob('raw_*.csv'):
        actual=read(path);shape=read(SHAPE/path.name)
        if recipe=='control':np.testing.assert_array_equal(actual.prediction,shape.prediction)
        else:
            learned=read(OUT/'study/fits'/path.stem[4:]/path.name);strength=.5 if recipe=='half' else 1.
            np.testing.assert_allclose(actual.prediction,strength*learned.prediction+(1-strength)*shape.prediction,rtol=1e-12,atol=1e-8)
        np.testing.assert_allclose(actual.groupby('date').prediction.sum(),shape.groupby('date').prediction.sum(),rtol=1e-12,atol=1e-8)
        count+=1
result=dict(checked_at=datetime.now(timezone.utc).isoformat(),input_code_sha256=hashes,fits=fits,
    raw_forecasts_checked=count,past_only_relative_targets=True,posterior_covariance_factors_reconstructed=True,
    network_raw_daily_totals_preserved=True,fixed_024_hourshares=True,exact_024_control=True,
    audit_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
