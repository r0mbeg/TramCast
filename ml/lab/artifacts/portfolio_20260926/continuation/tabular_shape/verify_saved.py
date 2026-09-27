"""P65: causal paired profiles, weighted basis, matrices and saved raw audit; run from ml."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import pandas as pd
from experiments.portfolio_verify import read
from experiments.portfolio_bayes_shape import HOURS,INNER,SOURCE
from experiments.portfolio_timesfm_errors import SOURCE as DAILY_SOURCE,TEACHERS
from experiments.portfolio_tabular_fraction import CONTROL,MODEL_SHA
from experiments.portfolio_school_fraction import features,CALENDAR,calendar_features
from experiments.portfolio_movement import disrupted
from experiments.portfolio_ridge import add_calendar
from pipeline import KEYS

OUT=Path('artifacts/portfolio_20260926/continuation/tabular_shape');DAY=['origin','route','date']
truth=read('artifacts/hourly_clean.csv');day_truth=truth.groupby(['route','date'],as_index=False).boardings.sum()
hashes={};fits=[]
for phase in ['pilot','study']:
    info=json.loads((OUT/phase/'run_started.json').read_text())
    for name,sha in info['sha256'].items():
        path=Path(name.removeprefix('/beegfs/home/m.persiyanov/codex_runs/tram-portfolio-20260926/'))
        if path.name=='tabpfn-v2-regressor.ckpt':
            assert sha==MODEL_SHA and MODEL_SHA in (OUT.parent/'tabular_fraction/checkpoint_sha256.txt').read_text()
        else:assert hashlib.sha256(path.read_bytes()).hexdigest()==sha,path
    assert info['versions']['tabpfn']=='2.0.9' and info['versions']['scikit-learn']=='1.6.1'
    hashes[phase]=len(info['sha256'])
for path in sorted(OUT.glob('*/fits_*/*/hourly_training.csv')):
    folder=path.parent;cutoff=folder.name;training=read(path);training.origin=pd.to_datetime(training.origin)
    origins=pd.date_range('2025-01-31',pd.Timestamp(cutoff)-pd.Timedelta(days=61),freq='ME')
    pieces=[];daily_pieces=[]
    for origin in origins:
        name=str(origin.date());stop=origin+pd.Timedelta(days=61)
        assert stop<=pd.Timestamp(cutoff)
        metadata=json.loads((INNER/name/'source.json').read_text());raw_path=INNER/name/f'raw_{name}.csv'
        assert metadata['origin']==name and metadata['fit_latest_date']==name and metadata['end']==str(stop.date())
        assert hashlib.sha256(raw_path.read_bytes()).hexdigest()==metadata['sha256']
        frame=read(raw_path).merge(truth.loc[truth.date.le(cutoff),KEYS+['boardings']],on=KEYS,validate='one_to_one')
        assert len(frame)==14640 and frame.date.min()>origin and frame.date.max()==stop
        pieces.append(frame.assign(origin=origin))
        daily=read(DAILY_SOURCE/name/'daily.csv').merge(read(TEACHERS/name/'daily.csv').rename(columns={'prediction':'timesfm'}),on=['route','date'],validate='one_to_one')
        daily=daily.merge(day_truth.loc[day_truth.date.le(cutoff)],on=['route','date'],validate='one_to_one')
        assert len(daily)==610 and daily.date.max()==stop
        daily_pieces.append(daily.assign(origin=origin))
    past=pd.concat(pieces,ignore_index=True);past.origin=past.origin.astype('datetime64[ns]')
    past=past.loc[past.date.gt(pd.Timestamp(cutoff)-pd.Timedelta(days=224))]
    changed=np.zeros(len(past),dtype=bool)
    for event in ['july_verified','autumn','august7','april17']:changed|=disrupted(past,event).to_numpy()
    past=add_calendar(past.loc[~changed&past.route.ne(5)&past.hour.isin(HOURS)].copy(),cutoff)
    past['actual_day']=past.groupby(DAY).boardings.transform('sum');past['base_day']=past.groupby(DAY).prediction.transform('sum')
    typical=past.drop_duplicates(['route','date']).groupby(['route','daytype']).actual_day.median()
    threshold=typical.reindex(pd.MultiIndex.from_frame(past[['route','daytype']])).to_numpy()
    past=past.loc[past.actual_day.gt(np.maximum(500,.35*threshold))&past.base_day.gt(0)].copy()
    past['target']=np.clip(np.log((past.boardings+1)/(past.actual_day+20))-np.log((past.prediction+1)/(past.base_day+20)),-np.log(2),np.log(2))
    repeat=past.groupby(KEYS).prediction.transform('size');importance=past.prediction.clip(lower=1)
    past['weight']=(1/repeat)*importance/np.average(importance,weights=1/repeat)
    past=past.sort_values(DAY+['hour']).reset_index(drop=True)
    for column in past.select_dtypes(include='integer').columns:past[column]=past[column].astype('int64')
    pd.testing.assert_frame_equal(training,past)
    index=past[DAY+['actual_day','base_day']].drop_duplicates(DAY).reset_index(drop=True)
    daily=pd.concat(daily_pieces,ignore_index=True);daily.origin=daily.origin.astype('datetime64[ns]')
    groups=daily.groupby(['origin','date']);daily['network_base']=groups.base.transform('sum');daily['network_actual']=groups.boardings.transform('sum')
    daily['relative_boardings']=daily.boardings*daily.network_base/daily.network_actual;daily['base_share']=daily.base/daily.network_base
    paired=index.merge(daily,on=DAY,validate='one_to_one');saved=read(folder/'daily_training.csv');saved.origin=pd.to_datetime(saved.origin)
    pd.testing.assert_frame_equal(saved,paired)
    matrices=np.load(folder/'matrices.npz')
    predicted=past.pivot(index=DAY,columns='hour',values='prediction').reindex(columns=HOURS).to_numpy()
    actual=past.pivot(index=DAY,columns='hour',values='boardings').reindex(columns=HOURS).to_numpy()
    shares=predicted/index.base_day.to_numpy()[:,None];errors=actual/index.actual_day.to_numpy()[:,None]-shares
    weights=index.actual_day.to_numpy()/index.groupby(['route','date']).route.transform('size').to_numpy();weights/=weights.mean()
    for key,expected in [('past_share',shares),('errors',errors),('weights',weights)]:np.testing.assert_allclose(matrices[key],expected,rtol=0,atol=1e-14)
    mean=np.average(errors,axis=0,weights=weights);centered=errors-mean;covariance=(centered*weights[:,None]).T@centered/weights.sum()
    eigenvalues=np.maximum(np.linalg.eigvalsh(covariance)[::-1],0);rank=0 if eigenvalues.sum()<=1e-24 else min(6,int(np.searchsorted(np.cumsum(eigenvalues)/eigenvalues.sum(),.9))+1)
    components=matrices['components'];assert components.shape==(rank,len(HOURS))
    np.testing.assert_allclose(matrices['mean'],mean,rtol=0,atol=1e-14)
    np.testing.assert_allclose(matrices['eigenvalues'],eigenvalues,rtol=1e-8,atol=1e-14)
    np.testing.assert_allclose(components@components.T,np.eye(rank),rtol=0,atol=1e-12)
    np.testing.assert_allclose(covariance@components.T,components.T*eigenvalues[:rank],rtol=1e-7,atol=1e-12)
    assert (components[np.arange(rank),np.argmax(abs(components),axis=1)]>=0).all()
    np.testing.assert_allclose(matrices['coordinates'],centered@components.T,rtol=1e-10,atol=1e-14)
    np.testing.assert_allclose(matrices['X'],np.column_stack([features(paired),shares]),rtol=1e-12,atol=1e-14)
    info=json.loads((folder/'fit.json').read_text());assert info['rank']==rank and info['rows']==len(index) and info['features']==50 and info['latest_target_date']<=cutoff
    assert info['model_sha256']==MODEL_SHA and info['coordinate_sample_weights'] is False and info['output_type']=='median'
    if folder.relative_to(OUT).parts[0]=='pilot':assert info['repeated_prediction_exact'] is True
    future=read(folder/'future.csv');future.origin=pd.to_datetime(future.origin)
    expected=read(DAILY_SOURCE/cutoff/'daily.csv').merge(read(TEACHERS/cutoff/'daily.csv').rename(columns={'prediction':'timesfm'}),on=['route','date'],validate='one_to_one')
    expected['origin']=pd.Timestamp(cutoff);expected['origin']=expected.origin.astype('datetime64[ns]');expected['network_base']=expected.groupby('date').base.transform('sum');expected['base_share']=expected.base/expected.network_base
    expected=expected.sort_values(['route','date']).reset_index(drop=True);pd.testing.assert_frame_equal(future,expected)
    assert 'boardings' not in future and future.date.min()>pd.Timestamp(cutoff)
    base=read(SOURCE/f'raw_{cutoff}.csv');profile=base.pivot(index=['route','date'],columns='hour',values='prediction').reindex(columns=HOURS)
    total=profile.sum(axis=1).to_numpy();future_shares=np.divide(profile.to_numpy(),total[:,None],out=np.zeros(profile.shape),where=total[:,None]>0)
    active=future.route.ne(5).to_numpy()&(total>0)
    np.testing.assert_array_equal(matrices['active'],active);np.testing.assert_allclose(matrices['future_share'],future_shares,rtol=0,atol=1e-14)
    np.testing.assert_allclose(matrices['test'],np.column_stack([features(future.loc[active]),future_shares[active]]),rtol=1e-12,atol=1e-14)
    delta=np.zeros_like(future_shares);delta[active]=mean+matrices['predictions']@components
    np.testing.assert_allclose(matrices['delta'],delta,rtol=1e-12,atol=1e-14)
    control=read(CONTROL/f'raw_{cutoff}.csv');raw=read(folder/f'raw_{cutoff}.csv')
    unnormalized=np.maximum(0,future_shares+delta);expanded=future[['route','date']].copy()
    for j,hour in enumerate(HOURS):expanded[str(hour)]=unnormalized[:,j]
    expanded=expanded.melt(id_vars=['route','date'],var_name='hour',value_name='value');expanded.hour=expanded.hour.astype(int)
    shape=control[KEYS].merge(expanded,on=KEYS,how='left',validate='one_to_one').fillna({'value':0.})
    sums=shape.groupby(['route','date']).value.transform('sum');target=control.groupby(['route','date']).prediction.transform('sum')
    backup=control.prediction.div(target.where(target.gt(0))).fillna(0)
    estimate=target*shape.value.div(sums.where(sums.gt(0))).fillna(backup)
    np.testing.assert_allclose(raw.prediction,estimate,rtol=1e-12,atol=1e-8)
    np.testing.assert_allclose(raw.groupby(['route','date']).prediction.sum(),control.groupby(['route','date']).prediction.sum(),rtol=1e-12,atol=1e-8)
    assert raw.loc[raw.route.eq(5)|raw.hour.between(1,4),'prediction'].eq(0).all()
    # Calendar matrix reconstructed using primary publication dates, independently of feature function.
    metadata=json.loads(CALENDAR.read_text());sources={s['id']:s for s in metadata['sources']}
    for frame in [paired,future]:
        matrix=[]
        for date,origin in zip(frame.date,frame.date-pd.to_timedelta(frame.horizon,unit='D')):
            intervals=[(pd.Timestamp(i['start']),pd.Timestamp(i['end']),i['kind']) for i in metadata['intervals'] if pd.Timestamp(sources[i['source']]['known_at'])<=origin]
            flags=[int(any(a<=date<=b and k==kind for a,b,k in intervals)) for kind in ['short','summer']]
            position=[min([61]+[(date-a).days for a,b,k in intervals if a<=date])/61,min([61]+[(a-date).days for a,b,k in intervals if a>date])/61]
            origin_flags=[int(any(a<=origin<=b and k==kind for a,b,k in intervals)) for kind in ['short','summer']]
            matrix.append(flags+position+origin_flags)
        np.testing.assert_array_equal(calendar_features(frame),matrix)
        np.testing.assert_array_equal(features(frame),features(frame.assign(boardings=1e100)))
    fits.append(dict(phase=folder.relative_to(OUT).parts[0],seed=info['seed'],cutoff=cutoff,rows=len(index),rank=rank,variance_retained=info['variance_retained']))
count=0;study=OUT/'study/tabular_shape'
for folder in sorted(study.glob('trial_*'))+[study/'selected',OUT/'study/alternative',OUT/'study/seed73']:
    if not folder.exists():continue
    info=json.loads((folder/'parameters.json' if folder.name!='selected' else study/'selection.json').read_text());recipe=info.get('params',info)['recipe'];seed=73 if folder.name=='seed73' else 42
    for path in folder.glob('raw_*.csv'):
        actual=read(path);control=read(CONTROL/path.name)
        if recipe=='control':np.testing.assert_array_equal(actual.prediction,control.prediction)
        else:
            learned=read(OUT/f'study/fits_{seed}'/path.stem[4:]/path.name);strength=.25 if recipe=='half' else .5
            np.testing.assert_allclose(actual.prediction,strength*learned.prediction+(1-strength)*control.prediction,rtol=1e-12,atol=1e-8)
        np.testing.assert_allclose(actual.groupby(['route','date']).prediction.sum(),control.groupby(['route','date']).prediction.sum(),rtol=1e-12,atol=1e-8);count+=1
for key in ['components','coordinates','predictions','delta']:
    np.testing.assert_array_equal(np.load(OUT/'pilot/fits_42/2025-10-31/matrices.npz')[key],np.load(OUT/'study/fits_42/2025-10-31/matrices.npz')[key])
chosen=study/'selected'
if json.loads((study/'selection.json').read_text())['params']['recipe']=='control':chosen=OUT/'study/alternative'
seed_difference=[]
for path in chosen.glob('raw_*.csv'):
    first=read(path);second=read(OUT/'study/seed73'/path.name)
    seed_difference.append(dict(cutoff=path.stem[4:],relative_l1=float(abs(first.prediction-second.prediction).sum()/first.prediction.sum())))
result=dict(checked_at=datetime.now(timezone.utc).isoformat(),input_code_sha256=hashes,fits=fits,raw_forecasts_checked=count,
    causal_hourly_targets_reconstructed=True,paired_daily_features_reconstructed=True,weighted_mean_covariance_eigenbasis_reconstructed=True,
    adaptive_past_only_rank=True,calendar_publication_and_matrix_checked=True,coordinate_models_unweighted=True,
    saved_input_target_matrices_checked=True,fixed_029_routeday_volumes=True,exact_029_control=True,structural_zeros=True,
    pilot_repeat_and_study_freshfit_exact=True,seed73_raw_differences=seed_difference,
    audit_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
