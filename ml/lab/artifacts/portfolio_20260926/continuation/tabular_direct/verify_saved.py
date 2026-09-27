"""P69: independent causal origin contexts, count targets and operated volume replay."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import pandas as pd
from experiments.portfolio_verify import read
from experiments.portfolio_tabular_direct import CONTROL,COLUMNS,MODEL_SHA,WEATHER_PATH,CALENDAR,OPS,AUGUST

OUT=Path('artifacts/portfolio_20260926/continuation/tabular_direct');KEYS=['route','date','hour']
truth=read('artifacts/hourly_clean.csv');weather=read(WEATHER_PATH)
federal=json.loads(Path('artifacts/calendar_sources.json').read_text());school=json.loads(CALENDAR.read_text())
sources={s['id']:s for s in school['sources']}

def mask(frame,event):
    if event=='july':return frame.route.isin([7,50])&frame.date.between('2025-07-10','2025-08-10')
    if event=='autumn':return frame.route.isin([7,50])&frame.date.between('2025-09-06','2025-11-14')&frame.date.dt.dayofweek.ge(5)
    if event=='august':return frame.route.eq(7)&frame.date.between('2025-08-16','2025-09-05')&frame.date.dt.dayofweek.ge(5)
    if event=='april':return frame.route.eq(17)&frame.date.between('2025-04-05','2025-04-30')&frame.date.dt.dayofweek.ge(5)
    raise ValueError(event)

def calendar(frame):
    frame=frame.copy();day=frame.date.dt.dayofweek
    holidays=frame.date.isin(pd.to_datetime(federal['holidays']));transfers=frame.date.isin(pd.to_datetime(list(federal['transfers'].values())))
    work=frame.date.isin(pd.to_datetime(federal['working_weekends']))
    frame['weekday']=day;frame['day_type']='weekday_'+day.astype(str)
    frame.loc[holidays,'day_type']='holiday';frame.loc[transfers,'day_type']='transferred_off';frame.loc[work,'day_type']='transferred_work'
    frame['off']=holidays|transfers;frame['is_workday']=(day.lt(5)&~frame.off)|work
    frame['daytype']=np.where(frame.is_workday,0,np.where(day.eq(5)&~frame.off,1,2))
    frame['effective_weekday']=np.where(frame.is_workday,np.minimum(day,4),np.where(frame.daytype.eq(1),5,6))
    frame['summer']=frame.date.dt.month.between(6,8).astype(int)
    return frame

def context(daily,origin,dates):
    origin=pd.Timestamp(origin);assert pd.Timestamp(federal['known_at'])<=origin
    past=daily.loc[daily.date.le(origin)&~daily.off].copy()
    fallback=past.groupby(['route','effective_weekday']).boardings.median()
    recent=past.loc[past.date.gt(origin-pd.Timedelta(days=56))]
    ref=recent.groupby(['route','effective_weekday']).boardings.median().reindex(fallback.index).fillna(fallback).clip(lower=100)
    normalized=past.merge(ref.rename('reference'),on=['route','effective_weekday'],validate='many_to_one')
    normalized['ratio']=normalized.boardings/normalized.reference
    result=calendar(pd.MultiIndex.from_product([sorted(past.route.unique()),dates],names=['route','date']).to_frame(index=False))
    result=result.merge(ref.rename('reference'),on=['route','effective_weekday'],validate='many_to_one')
    for days in [7,14,28]:
        ratio=normalized.loc[normalized.date.gt(origin-pd.Timedelta(days=days))].groupby('route').ratio.median()
        result[f'ratio{days}']=result.route.map(ratio).fillna(1.)
    result=result.merge(weather,on='date',validate='many_to_one')
    known=weather.loc[weather.date.le(origin)&weather.date.gt(origin-pd.Timedelta(days=14))]
    result['origin_temperature']=known.temperature_2m_mean.mean();result['origin_daylight']=known.daylight_duration.mean()
    result['origin']=origin;result['origin']=result.origin.astype('datetime64[ns]')
    result['horizon']=(result.date-origin).dt.days;result['season']=result.date.dt.month%12//3
    result['annual_sin']=np.sin(2*np.pi*(result.date.dt.dayofyear-1)/365);result['annual_cos']=np.cos(2*np.pi*(result.date.dt.dayofyear-1)/365)
    result['log_reference']=np.log1p(result.reference)
    return result

def compare(frame,expected):
    assert set(frame.columns)==set(expected.columns),(list(frame.columns),list(expected.columns))
    for col in expected:
        if pd.api.types.is_numeric_dtype(expected[col]):np.testing.assert_allclose(frame[col],expected[col],rtol=1e-12,atol=1e-8)
        else:np.testing.assert_array_equal(frame[col],expected[col])

def matrix(frame):
    extra=[]
    for date,origin in zip(frame.date,frame.origin):
        intervals=[(pd.Timestamp(i['start']),pd.Timestamp(i['end']),i['kind']) for i in school['intervals'] if pd.Timestamp(sources[i['source']]['known_at'])<=origin]
        flags=[int(any(a<=date<=b and k==kind for a,b,k in intervals)) for kind in ['short','summer']]
        distance=[min([61]+[(date-a).days for a,b,k in intervals if a<=date])/61,min([61]+[(a-date).days for a,b,k in intervals if a>date])/61]
        originflags=[int(any(a<=origin<=b and k==kind for a,b,k in intervals)) for kind in ['short','summer']]
        extra.append(flags+distance+originflags)
    return np.column_stack([frame[COLUMNS].to_numpy(float),extra])

hashes={};fits=[]
for phase in ['pilot','study']:
    info=json.loads((OUT/phase/'run_started.json').read_text())
    for name,sha in info['sha256'].items():
        path=Path(name.removeprefix('/beegfs/home/m.persiyanov/codex_runs/tram-portfolio-20260926/'))
        if path.name=='tabpfn-v2-regressor.ckpt':assert sha==MODEL_SHA and MODEL_SHA in (OUT.parent/'tabular_fraction/checkpoint_sha256.txt').read_text()
        else:assert hashlib.sha256(path.read_bytes()).hexdigest()==sha,path
    assert info['old_tabpfn_trials']==13 and info['old_tabpfn_gpu_seconds']==711 and info['total_tabpfn_trials']==18
    assert info['versions']['scikit-learn']=='1.6.1' and info['versions']['tabpfn']=='2.0.9'
    hashes[phase]=len(info['sha256'])
for path in sorted(OUT.glob('*/fits_*/*/training.csv')):
    folder=path.parent;cutoff=folder.name;past=read(path);past.origin=pd.to_datetime(past.origin)
    available=truth.loc[truth.date.le(cutoff)]
    ordinary=available.loc[~(mask(available,'july')|mask(available,'autumn')|mask(available,'august')|mask(available,'april'))]
    daily=calendar(ordinary.loc[ordinary.route.ne(5)].groupby(['route','date'],as_index=False).boardings.sum())
    pieces=[]
    origins=pd.date_range('2025-01-31',pd.Timestamp(cutoff)-pd.Timedelta(days=1),freq='ME')
    for origin in origins:
        stop=min(origin+pd.Timedelta(days=61),pd.Timestamp(cutoff))
        ref=context(daily,origin,pd.date_range(origin+pd.Timedelta(days=1),stop))
        pieces.append(ref.merge(daily[['route','date','boardings']],on=['route','date'],validate='one_to_one'))
    expected=pd.concat(pieces,ignore_index=True).sort_values(['origin','route','date']).reset_index(drop=True);compare(past,expected)
    np.testing.assert_array_equal(past.boardings,expected.boardings)
    assert past.date.max()<=pd.Timestamp(cutoff) and past.date.gt(past.origin).all() and past.horizon.between(1,61).all()
    meta=json.loads((folder/'fit.json').read_text());target=meta['target'];matrices=np.load(folder/'matrices.npz')
    np.testing.assert_allclose(matrices['X'],matrix(expected),rtol=1e-12,atol=1e-8)
    Y=past.boardings.to_numpy()/(10000 if target=='count' else past.reference.to_numpy());np.testing.assert_array_equal(matrices['y'],Y)
    assert meta['sample_weight_used'] is False and meta['features']==21 and meta['categorical_features']==[0,1] and meta['model_sha256']==MODEL_SHA
    if folder.relative_to(OUT).parts[0]=='pilot':assert meta['repeated_prediction_exact'] is True
    future=read(folder/'future.csv');future.origin=pd.to_datetime(future.origin)
    end=str(future.date.max().date());expected=context(daily,cutoff,pd.date_range(pd.Timestamp(cutoff)+pd.Timedelta(days=1),end))
    compare(future.drop(columns=['normal_volume','operated_volume']),expected)
    assert 'boardings' not in future and future.date.min()>pd.Timestamp(cutoff)
    np.testing.assert_allclose(matrices['test'],matrix(expected),rtol=1e-12,atol=1e-8)
    volume=np.maximum(0,matrices['prediction'])*(10000 if target=='count' else future.reference.to_numpy())
    np.testing.assert_array_equal(future.normal_volume.to_numpy().astype(volume.dtype),volume)
    params={**json.loads(OPS.read_text())['params'],**json.loads(AUGUST.read_text())['params'],'july_verified':True}
    assert meta['operation_params']==params
    actualday=available.groupby(['route','date'],as_index=False).boardings.sum();actualday['weekday']=actualday.date.dt.dayofweek
    normalday=ordinary.groupby(['route','date'],as_index=False).boardings.sum();normalday['weekday']=normalday.date.dt.dayofweek
    reference=normalday.groupby(['route','weekday']).boardings.median().clip(lower=1)
    operated=volume.copy()
    for event in ['july','autumn','august']:
        observed=actualday.loc[mask(actualday,event)].copy().merge(reference.rename('expected'),on=['route','weekday'],validate='many_to_one')
        observed['ratio']=observed.boardings/observed.expected
        for route in ([7] if event=='august' else [7,50]):
            values=observed.loc[observed.route.eq(route),'ratio']
            prior=params['august7'] if event=='august' else params[f'july{route}'] if event=='july' else .6 if route==7 else 0.
            maximum=2 if event=='july' and route==50 else 1
            factor=float(np.clip(values.median(),0,maximum)) if len(values)>=3 else prior
            if event=='autumn' and route==50:factor=0.
            operated[(mask(future,event)&future.route.eq(route)).to_numpy()]*=factor
    np.testing.assert_array_equal(future.operated_volume.to_numpy().astype(operated.dtype),operated)
    base=read(CONTROL/f'raw_{cutoff}.csv');day=base.groupby(['route','date']).prediction.transform('sum')
    expanded=base.merge(future[['route','date']].assign(operated_volume=operated),on=['route','date'],how='left',validate='many_to_one')
    prediction=base.prediction.div(day.where(day.gt(0))).fillna(0)*expanded.operated_volume.fillna(0)
    raw=read(folder/f'raw_{cutoff}.csv');np.testing.assert_allclose(raw.prediction,prediction,rtol=1e-12,atol=1e-8)
    assert raw.loc[base.prediction.eq(0),'prediction'].eq(0).all()
    fits.append(dict(phase=folder.relative_to(OUT).parts[0],cutoff=cutoff,target=target,seed=meta['seed'],rows=len(past),origins=len(origins)))
count=0;study=OUT/'study/tabular_direct'
for folder in sorted(study.glob('trial_*'))+[study/'selected',OUT/'study/alternative',OUT/'study/seed73']:
    if not folder.exists():continue
    meta=json.loads((folder/'parameters.json' if folder.name!='selected' else study/'selection.json').read_text());recipe=meta.get('params',meta)['recipe'];seed=73 if folder.name=='seed73' else 42
    for path in folder.glob('raw_*.csv'):
        actual=read(path);control=read(CONTROL/path.name)
        if recipe=='control':np.testing.assert_array_equal(actual.prediction,control.prediction)
        else:
            target,strength=recipe.split('_');learned=read(OUT/f'study/fits_{target}_{seed}'/path.stem[4:]/path.name);weight=.5 if strength=='half' else 1.
            np.testing.assert_allclose(actual.prediction,weight*learned.prediction+(1-weight)*control.prediction,rtol=1e-12,atol=1e-8)
        count+=1
chosen=study/'selected'
if json.loads((study/'selection.json').read_text())['params']['recipe']=='control':chosen=OUT/'study/alternative'
meta=json.loads((chosen/'parameters.json' if chosen.name=='alternative' else study/'selection.json').read_text());target=meta.get('params',meta)['recipe'].split('_')[0]
np.testing.assert_array_equal(np.load(OUT/f'pilot/fits_{target}_42/2025-10-31/matrices.npz')['prediction'],np.load(OUT/f'study/fits_{target}_42/2025-10-31/matrices.npz')['prediction'])
seed_difference=[]
for path in chosen.glob('raw_*.csv'):
    a=read(path);b=read(OUT/'study/seed73'/path.name)
    seed_difference.append(dict(cutoff=path.stem[4:],relative_L1=float(abs(a.prediction-b.prediction).sum()/a.prediction.sum())))
result=dict(checked_at=datetime.now(timezone.utc).isoformat(),input_code_sha256=hashes,fits=fits,raw_forecasts_checked=count,
    independent_monthly_context_reference_ratios_weather_checked=True,targets_before_outer_cutoff=True,calendar_publication_and_21features_checked=True,
    future_target_free=True,derived_float32_count_fields_decoded_natively=True,native_unweighted_direct_targets=True,operations_history_only_reconstructed=True,fixed030_shape_and_inactive_zeros=True,
    raw_replay_and_fixed_recipes_checked=True,pilot_repeat_and_study_freshfit_exact=True,seed73_raw_differences=seed_difference,
    old_tabular_budget_and_trials_retained=True,audit_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
