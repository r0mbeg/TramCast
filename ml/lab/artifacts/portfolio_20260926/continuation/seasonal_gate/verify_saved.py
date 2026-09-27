"""Run from ml with PYTHONPATH=. : independent P55 support and forecast audit."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
import numpy as np
import pandas as pd
from experiments.portfolio_timesfm_errors import SOURCE, TEACHERS, SHAPE, VOLUME
from experiments.portfolio_verify import read

OUT=Path('artifacts/portfolio_20260926/continuation/seasonal_gate')
started=json.loads((OUT/'run_started.json').read_text())
for name,sha in started['sha256'].items():
    path=Path(name.removeprefix('/beegfs/home/m.persiyanov/codex_runs/tram-portfolio-20260926/'))
    assert hashlib.sha256(path.read_bytes()).hexdigest()==sha,path
truth=read('artifacts/hourly_clean.csv').groupby(['route','date'],as_index=False).boardings.sum()
fits=[]
for path in sorted(OUT.glob('models/fits_15/*/training.csv')):
    folder=path.parent;cutoff=folder.name
    training=read(path);training['origin']=pd.to_datetime(training.origin)
    origins=pd.date_range('2025-01-31',pd.Timestamp(cutoff)-pd.Timedelta(days=61),freq='ME')
    pieces=[]
    for origin in origins:
        name=str(origin.date())
        assert origin+pd.Timedelta(days=61)<=pd.Timestamp(cutoff)
        frame=read(SOURCE/name/'daily.csv').merge(read(TEACHERS/name/'daily.csv').rename(columns={'prediction':'timesfm'}),on=['route','date'],validate='one_to_one')
        frame=frame.merge(truth.loc[truth.date.le(cutoff)],on=['route','date'],validate='one_to_one')
        assert len(frame)==610
        pieces.append(frame.assign(origin=origin))
    past=pd.concat(pieces,ignore_index=True)
    past=past.loc[past.route.ne(5)&past.base.gt(0)&past.date.gt(pd.Timestamp(cutoff)-pd.Timedelta(days=224))].reset_index(drop=True)
    past['origin']=past.origin.astype('datetime64[ns]')
    pd.testing.assert_frame_equal(training.drop(columns=['target','weight']),past)
    np.testing.assert_allclose(training.target,past.boardings/past.base,rtol=0,atol=1e-12)
    weights=past.base/past.groupby(['route','date']).base.transform('size')
    np.testing.assert_allclose(training.weight,weights/weights.mean(),rtol=0,atol=1e-12)
    info=json.loads((folder/'fit.json').read_text())
    assert info['model_reload_exact'] and info['leaves']==15 and info['features']==23 and info['rows']==len(past)
    assert hashlib.sha256((folder/'model.pkl').read_bytes()).hexdigest()==info['model_sha256']
    future=read(folder/'future.csv')
    expected=read(SOURCE/cutoff/'daily.csv').merge(read(TEACHERS/cutoff/'daily.csv').rename(columns={'prediction':'timesfm'}),on=['route','date'],validate='one_to_one')
    pd.testing.assert_frame_equal(future.drop(columns=['ratio','factor']),expected)
    np.testing.assert_array_equal(future.factor,future.ratio.clip(0.5,2))
    unique=past.drop_duplicates(['route','date']).copy()
    unique['season']=(unique.date.dt.month%12)//3
    counts=unique.groupby(['route','season','effective_weekday']).date.nunique()
    gate=read(OUT/'gates'/f'{cutoff}.csv')
    for row in gate.itertuples():
        assert row.season==(row.date.month%12)//3
        count=int(counts.get((row.route,row.season,row.effective_weekday),0))
        assert row.past_dates==count and row.supported==(count>=2)
    base=read(VOLUME/cutoff/f'raw_{cutoff}.csv').merge(future[['route','date','factor']],on=['route','date'],validate='many_to_one')
    shape=read(SHAPE/f'raw_{cutoff}.csv')
    base['prediction']*=base.factor
    volume=base.groupby(['route','date']).prediction.transform('sum')
    total=shape.groupby(['route','date']).prediction.transform('sum')
    shares=shape.prediction.div(total.where(total.gt(0))).fillna(0)
    learned=read(folder/f'raw_{cutoff}.csv')
    np.testing.assert_allclose(learned.prediction,volume*shares,rtol=1e-12,atol=1e-8)
    fits.append(dict(cutoff=cutoff,rows=len(past),supported_days=int(gate.supported.sum()),total_days=len(gate)))
count=0
study=OUT/'seasonal_gate'
for folder in sorted(study.glob('trial_*'))+[study/'selected']:
    params=json.loads((folder/'parameters.json' if folder.name.startswith('trial_') else study/'selection.json').read_text())
    recipe=params.get('params',params)['recipe']
    for path in folder.glob('raw_*.csv'):
        cutoff=path.stem[4:];shape=read(SHAPE/path.name);actual=read(path)
        pd.testing.assert_frame_equal(actual[['route','date','hour']],shape[['route','date','hour']])
        expected=shape.prediction.copy()
        if recipe=='gated':
            gate=read(OUT/'gates'/f'{cutoff}.csv')
            supported=actual.merge(gate[['route','date','supported']],on=['route','date'],validate='many_to_one').supported
            learned=read(OUT/'models/fits_15'/cutoff/path.name)
            expected.loc[supported]=0.5*learned.loc[supported,'prediction']+0.5*shape.loc[supported,'prediction']
            np.testing.assert_array_equal(actual.loc[~supported,'prediction'],shape.loc[~supported,'prediction'])
        np.testing.assert_allclose(actual.prediction,expected,rtol=1e-12,atol=1e-8)
        assert actual.loc[actual.route.eq(5)|actual.hour.between(1,4),'prediction'].eq(0).all()
        count+=1
result=dict(checked_at=datetime.now(timezone.utc).isoformat(),input_code_sha256=len(started['sha256']),fits=fits,
    raw_forecasts_checked=count,past_only_distinct_season_route_weekday_support=True,
    unsupported_exact_024=True,fixed_024_hourshares=True,model_reloads_exact_on_zhores=True,
    local_model_inference_skipped='sklearn1.9.1 differs from saved1.8.0; saved transforms audited',
    audit_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
