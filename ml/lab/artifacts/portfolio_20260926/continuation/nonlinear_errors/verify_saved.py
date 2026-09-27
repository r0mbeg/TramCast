"""Run from ml with PYTHONPATH=. : audit P53 saved inputs and transformations."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
import numpy as np
import pandas as pd
from experiments.portfolio_timesfm_errors import SOURCE, TEACHERS, SHAPE, VOLUME
from experiments.portfolio_verify import read

ROOT=Path('artifacts/portfolio_20260926/continuation')
OUT=ROOT/'nonlinear_errors'
truth=read('artifacts/hourly_clean.csv').groupby(['route','date'],as_index=False).boardings.sum()
hashes={}
for phase in ['pilot','study']:
    started=json.loads((OUT/phase/'run_started.json').read_text())
    for name,sha in started['sha256'].items():
        path=Path(name.removeprefix('/beegfs/home/m.persiyanov/codex_runs/tram-portfolio-20260926/'))
        assert hashlib.sha256(path.read_bytes()).hexdigest()==sha,path
    hashes[phase]=len(started['sha256'])
fits=[]
for path in sorted(OUT.rglob('training.csv')):
    folder=path.parent
    cutoff=folder.name
    training=read(path)
    training['origin']=pd.to_datetime(training.origin)
    origins=pd.date_range('2025-01-31',pd.Timestamp(cutoff)-pd.Timedelta(days=61),freq='ME')
    pieces=[]
    for origin in origins:
        name=str(origin.date());stop=origin+pd.Timedelta(days=61)
        assert stop<=pd.Timestamp(cutoff)
        frame=read(SOURCE/name/'daily.csv').merge(read(TEACHERS/name/'daily.csv').rename(columns={'prediction':'timesfm'}),on=['route','date'],validate='one_to_one')
        frame=frame.merge(truth.loc[truth.date.le(cutoff)],on=['route','date'],validate='one_to_one')
        assert len(frame)==610
        frame['origin']=origin
        pieces.append(frame)
    past=pd.concat(pieces,ignore_index=True)
    past=past.loc[past.route.ne(5)&past.base.gt(0)&past.date.gt(pd.Timestamp(cutoff)-pd.Timedelta(days=224))].reset_index(drop=True)
    past['origin']=past.origin.astype('datetime64[ns]')
    pd.testing.assert_frame_equal(training.drop(columns=['target','weight']),past)
    np.testing.assert_allclose(training.target,past.boardings/past.base,rtol=0,atol=1e-12)
    repeats=past.groupby(['route','date']).base.transform('size')
    weights=past.base/repeats
    np.testing.assert_allclose(training.weight,weights/weights.mean(),rtol=0,atol=1e-12)
    example_ratio=np.resize([0.7,1.0,1.3],len(past))
    ratio_loss=np.sum(np.abs(example_ratio-training.target)*training.weight)
    day_loss=np.sum(np.abs(example_ratio*past.base-past.boardings)/repeats)/weights.mean()
    np.testing.assert_allclose(ratio_loss,day_loss,rtol=1e-12)
    info=json.loads((folder/'fit.json').read_text())
    assert info['rows']==len(past) and info['latest_target_date']<=cutoff
    assert info['model_reload_exact'] and info['iterations']==100 and info['features']==23
    assert hashlib.sha256((folder/'model.pkl').read_bytes()).hexdigest()==info['model_sha256']
    future=read(folder/'future.csv')
    assert 'boardings' not in future
    expected=read(SOURCE/cutoff/'daily.csv').merge(read(TEACHERS/cutoff/'daily.csv').rename(columns={'prediction':'timesfm'}),on=['route','date'],validate='one_to_one')
    pd.testing.assert_frame_equal(future.drop(columns=['ratio','factor']),expected)
    np.testing.assert_array_equal(future.factor,future.ratio.clip(0.5,2))
    unused=future.route.eq(5)|future.base.eq(0)
    assert future.loc[unused,'factor'].eq(1.).all()
    raw=read(folder/f'raw_{cutoff}.csv')
    base=read(VOLUME/cutoff/f'raw_{cutoff}.csv')
    shape=read(SHAPE/f'raw_{cutoff}.csv')
    base=base.merge(future[['route','date','factor']],on=['route','date'],validate='many_to_one')
    base['prediction']*=base.factor
    volume=base.groupby(['route','date']).prediction.transform('sum')
    total=shape.groupby(['route','date']).prediction.transform('sum')
    shares=shape.prediction.div(total.where(total.gt(0))).fillna(0)
    np.testing.assert_allclose(raw.prediction,volume*shares,rtol=1e-12,atol=1e-8)
    assert raw.loc[raw.route.eq(5)|raw.hour.between(1,4),'prediction'].eq(0).all()
    fits.append(dict(phase=folder.relative_to(OUT).parts[0],cutoff=cutoff,leaves=info['leaves'],rows=len(past),origins=len(origins)))
count=0
study=OUT/'study/nonlinear_errors'
for folder in sorted(study.glob('trial_*'))+[study/'selected']:
    parameters=json.loads((folder/'parameters.json' if folder.name.startswith('trial_') else study/'selection.json').read_text())
    recipe=parameters.get('params',parameters)['recipe']
    for path in folder.glob('raw_*.csv'):
        base=read(SHAPE/path.name)
        actual=read(path)
        if recipe=='control':
            np.testing.assert_array_equal(actual.prediction,base.prediction)
        else:
            strength,leaves=recipe.split('_')
            learned=read(OUT/f'study/fits_{leaves}'/path.stem[4:]/path.name)
            weight=0.5 if strength=='half' else 1.
            np.testing.assert_allclose(actual.prediction,weight*learned.prediction+(1-weight)*base.prediction,rtol=1e-12,atol=1e-8)
        count+=1
result=dict(checked_at=datetime.now(timezone.utc).isoformat(),input_code_sha256=hashes,fits=fits,raw_forecasts_checked=count,past_targets_teachers_verified=True,absolute_daily_loss_identity=True,exact_024_control=True,fixed_024_hourshares=True,model_reloads_exact_on_zhores=True,local_model_inference_skipped='sklearn1.9.1 differs from saved1.8.0; saved transforms audited',decision='control won W1/W2; no duplicate archive',audit_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
