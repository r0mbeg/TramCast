"""Run from ml with PYTHONPATH=. : P61 causal examples, public calendar availability, fractions and raw audit."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import pandas as pd
from experiments.portfolio_timesfm_errors import SOURCE,TEACHERS,SHAPE
from experiments.portfolio_verify import read
from experiments.portfolio_school_fraction import CALENDAR,CONTROL,calendar_features

OUT=Path('artifacts/portfolio_20260926/continuation/school_fraction')
truth=read('artifacts/hourly_clean.csv').groupby(['route','date'],as_index=False).boardings.sum()
hashes={};fits=[]
for phase in ['pilot','study']:
    info=json.loads((OUT/phase/'run_started.json').read_text())
    for name,sha in info['sha256'].items():
        path=Path(name.removeprefix('/beegfs/home/m.persiyanov/codex_runs/tram-portfolio-20260926/'))
        assert hashlib.sha256(path.read_bytes()).hexdigest()==sha,path
    hashes[phase]=len(info['sha256'])
for path in sorted(OUT.glob('*/fits_*/*/training.csv')):
    folder=path.parent;cutoff=folder.name
    training=read(path);training['origin']=pd.to_datetime(training.origin)
    origins=pd.date_range('2025-01-31',pd.Timestamp(cutoff)-pd.Timedelta(days=61),freq='ME')
    pieces=[]
    for origin in origins:
        name=str(origin.date())
        frame=read(SOURCE/name/'daily.csv').merge(read(TEACHERS/name/'daily.csv').rename(columns={'prediction':'timesfm'}),on=['route','date'],validate='one_to_one')
        frame=frame.merge(truth.loc[truth.date.le(cutoff)],on=['route','date'],validate='one_to_one')
        assert len(frame)==610 and origin+pd.Timedelta(days=61)<=pd.Timestamp(cutoff)
        pieces.append(frame.assign(origin=origin))
    past=pd.concat(pieces,ignore_index=True);past['origin']=past.origin.astype('datetime64[ns]')
    groups=past.groupby(['origin','date'])
    past['network_base']=groups.base.transform('sum');past['network_actual']=groups.boardings.transform('sum')
    past['relative_boardings']=past.boardings*past.network_base/past.network_actual
    past=past.loc[past.route.ne(5)&past.base.gt(0)&past.date.gt(pd.Timestamp(cutoff)-pd.Timedelta(days=224))].reset_index(drop=True)
    pd.testing.assert_frame_equal(training.drop(columns=['base_share','target','weight']),past)
    np.testing.assert_allclose(training.base_share,past.base/past.network_base,rtol=0,atol=1e-15)
    np.testing.assert_allclose(training.target,past.boardings/past.network_actual-past.base/past.network_base,rtol=0,atol=1e-15)
    repeats=past.groupby(['route','date']).base.transform('size');weights=past.network_actual/repeats
    np.testing.assert_allclose(training.weight,weights/weights.mean(),rtol=0,atol=1e-12)
    estimate=np.resize([.02,.09,.31],len(past))
    np.testing.assert_allclose(np.sum(abs(estimate-training.target-training.base_share)*training.weight),
        np.sum(abs(estimate*past.network_actual-past.boardings)/repeats)/weights.mean(),rtol=1e-12)
    info=json.loads((folder/'fit.json').read_text())
    assert info['model_reload_exact'] and info['rows']==len(training) and info['features']==30 and info['iterations']==100
    assert info['latest_target_date']<=cutoff and hashlib.sha256((folder/'model.pkl').read_bytes()).hexdigest()==info['model_sha256']
    future=read(folder/'future.csv')
    expected=read(SOURCE/cutoff/'daily.csv').merge(read(TEACHERS/cutoff/'daily.csv').rename(columns={'prediction':'timesfm'}),on=['route','date'],validate='one_to_one')
    pd.testing.assert_frame_equal(future.drop(columns=['network_base','base_share','share_error','estimated_share']),expected)
    np.testing.assert_array_equal(future.network_base,expected.groupby('date').base.transform('sum'))
    np.testing.assert_allclose(future.base_share,expected.base/future.network_base,rtol=0,atol=1e-15)
    estimate=(future.base_share+future.share_error).clip(lower=0)
    estimate.loc[future.route.eq(5)|future.base.eq(0)]=0.
    np.testing.assert_array_equal(future.estimated_share,estimate)
    # Independent calendar matrix; only information published at each origin.
    metadata=json.loads(CALENDAR.read_text())
    sources={s['id']:s for s in metadata['sources']}
    for frame in [training,future]:
        origin=frame.date-pd.to_timedelta(frame.horizon,unit='D')
        matrix=[]
        for date,cut in zip(frame.date,origin):
            intervals=[(pd.Timestamp(i['start']),pd.Timestamp(i['end']),i['kind'])
                for i in metadata['intervals'] if pd.Timestamp(sources[i['source']]['known_at'])<=cut]
            flags=[int(any(a<=date<=b and k==kind for a,b,k in intervals)) for kind in ['short','summer']]
            before=[(date-a).days for a,b,k in intervals if a<=date]
            after=[(a-date).days for a,b,k in intervals if a>date]
            position=[min(before+[61])/61,min(after+[61])/61]
            origin_flags=[int(any(a<=cut<=b and k==kind for a,b,k in intervals)) for kind in ['short','summer']]
            matrix.append(flags+position+origin_flags)
        np.testing.assert_allclose(calendar_features(frame),matrix,rtol=0,atol=0)
        np.testing.assert_array_equal(calendar_features(frame),calendar_features(frame.assign(boardings=1e100)))
    shape=read(SHAPE/f'raw_{cutoff}.csv');raw=read(folder/f'raw_{cutoff}.csv')
    estimated=shape[['route','date']].merge(future[['route','date','estimated_share']],on=['route','date'],validate='many_to_one').estimated_share
    volume=shape.groupby(['route','date']).prediction.transform('sum')
    expected=shape.prediction.div(volume.where(volume.gt(0))).fillna(0)*estimated
    expected*=shape.groupby('date').prediction.transform('sum')/expected.groupby(shape.date).transform('sum')
    np.testing.assert_allclose(raw.prediction,expected,rtol=1e-12,atol=1e-8)
    np.testing.assert_allclose(raw.groupby('date').prediction.sum(),shape.groupby('date').prediction.sum(),rtol=1e-12,atol=1e-8)
    assert raw.loc[shape.prediction.eq(0),'prediction'].eq(0).all()
    fits.append(dict(phase=folder.relative_to(OUT).parts[0],seed=info['seed'],cutoff=cutoff,rows=len(past)))
count=0;study=OUT/'study/school_fraction'
for folder in sorted(study.glob('trial_*'))+[study/'selected',OUT/'study/seed73']:
    info=json.loads((folder/'parameters.json' if folder.name!='selected' else study/'selection.json').read_text())
    recipe=info.get('params',info)['recipe'];seed=73 if folder.name=='seed73' else 42
    for path in folder.glob('raw_*.csv'):
        actual=read(path);shape=read(SHAPE/path.name)
        control=read(CONTROL/path.name)
        if recipe=='control':np.testing.assert_array_equal(actual.prediction,control.prediction)
        else:
            learned=read(OUT/f'study/fits_{seed}'/path.stem[4:]/path.name);strength=.5 if recipe=='half' else 1.
            enhanced=.5*learned.prediction+.5*shape.prediction
            np.testing.assert_allclose(actual.prediction,strength*enhanced+(1-strength)*control.prediction,rtol=1e-12,atol=1e-8)
        np.testing.assert_allclose(actual.groupby('date').prediction.sum(),shape.groupby('date').prediction.sum(),rtol=1e-12,atol=1e-8)
        count+=1
for path in (study/'selected').glob('raw_*.csv'):
    np.testing.assert_array_equal(read(path).prediction,read(OUT/'study/seed73'/path.name).prediction)
pd.testing.assert_frame_equal(read(study/'selected/submission.csv'),read(OUT/'study/seed73/submission.csv'))
result=dict(checked_at=datetime.now(timezone.utc).isoformat(),input_code_sha256=hashes,fits=fits,
    raw_forecasts_checked=count,past_only_fraction_targets=True,public_calendar_matrix_independently_reconstructed=True,calendar_published_before_origin=True,absolute_daily_loss_identity=True,
    network_raw_daily_totals_preserved=True,fixed_024_hourshares=True,exact_028_control=True,seed73_all5_raw_exact=True,
    model_reloads_exact_on_zhores=True,local_model_inference_skipped='sklearn1.9.1 differs from saved1.8.0; saved transforms audited',
    audit_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
