"""P68: independent posterior algebra, causal panel, support and forecast replay."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,itertools,json
import numpy as np
import pandas as pd
from experiments.portfolio_verify import read

OUT=Path('artifacts/portfolio_20260926/continuation/current_bayes');STUDY=OUT/'study'
ALIASES=['ridge','chronos','movement','regime'];LABELS=['030','029','025','024']
SOURCES=[OUT.parent/p for p in ['tabular_shape/study/tabular_shape/selected','school_fraction/study/school_fraction/selected',
    'bayes_shape_timesfm_blend/bayes_shape_timesfm_blend/selected','bayes_shape/study/bayes_shape/selected']]
ORIGINS=['2025-04-30','2025-06-30','2025-07-31','2025-08-31']
info=json.loads((STUDY/'run_started.json').read_text())
for name,sha in info['sha256'].items():
    path=Path(name.removeprefix('/beegfs/home/m.persiyanov/codex_runs/tram-portfolio-20260926/'))
    assert hashlib.sha256(path.read_bytes()).hexdigest()==sha,path
parent=json.loads((OUT.parent/'bayes/run_started.json').read_text())
old=next(s for p,s in parent['sha256'].items() if p.endswith('experiments/portfolio_bayes.py'))
assert old==info['sha256']['experiments/portfolio_bayes.py']
assert info['posterior_deterministic'] and info['previous_P17_allocation_seconds']==23 and info['cumulative_limit_seconds']==143
history=read(Path('artifacts/hourly_clean.csv'))
calendar=json.loads(Path('artifacts/calendar_sources.json').read_text())
grid=np.array([.05+.1*np.array(x) for x in itertools.product(range(9),repeat=4) if sum(x)==8]);assert len(grid)==165
log_prior=np.log(grid).sum(axis=1)

def annotate(frame):
    frame=frame.copy();dates=frame.date;weekday=dates.dt.dayofweek
    off=dates.isin(pd.to_datetime(calendar['holidays']+list(calendar['transfers'].values())))
    work=(weekday.lt(5)&~off)|dates.isin(pd.to_datetime(calendar['working_weekends']))
    frame['season']=(dates.dt.month%12)//3
    frame['group']=np.where(work,0,np.where(weekday.eq(5)&~off,1,2))
    return frame


def probabilities(prior,loss):
    log=prior-loss.sum(axis=0)/.15
    result=np.exp(log-log.max());return result/result.sum()

checks=[]
for folder in sorted((STUDY/'calibrations').iterdir()):
    cutoff=folder.name;origins=[o for o in ORIGINS if pd.Timestamp(o)+pd.Timedelta(days=61)<=pd.Timestamp(cutoff)]
    assert pd.Timestamp(calendar['known_at'])<=pd.Timestamp(cutoff)
    if not origins:
        assert cutoff=='2025-04-30' and json.loads((folder/'gate.json').read_text())==dict(cutoff=cutoff,origins=[],action='exact030 control')
        np.testing.assert_array_equal(read(folder/f'raw_{cutoff}.csv').prediction,read(SOURCES[0]/f'raw_{cutoff}.csv').prediction)
        continue
    panel=read(folder/'past_panel.csv');panel.origin=pd.to_datetime(panel.origin);pieces=[]
    for origin in origins:
        ref=read(SOURCES[0]/f'raw_{origin}.csv')[['route','date','hour']].copy()
        for alias,source in zip(ALIASES,SOURCES):ref[alias]=read(source/f'raw_{origin}.csv').prediction
        ref=ref.merge(history[['route','date','hour','boardings']],on=['route','date','hour'],validate='one_to_one')
        pieces.append(ref.assign(origin=pd.Timestamp(origin)))
    expected=pd.concat(pieces,ignore_index=True);expected['origin']=expected.origin.astype('datetime64[ns]');pd.testing.assert_frame_equal(panel,expected)
    assert panel.date.max()<=pd.Timestamp(cutoff) and (panel.date>panel.origin).all()
    assert (panel.origin+pd.Timedelta(days=61)<=pd.Timestamp(cutoff)).all()
    panel=panel.loc[panel.route.ne(5)].reset_index(drop=True)
    days=panel[['route','date']].drop_duplicates().sort_values(['route','date']).reset_index(drop=True)
    index=pd.MultiIndex.from_frame(days);code=index.get_indexer(pd.MultiIndex.from_frame(panel[['route','date']]))
    count=np.bincount(code,minlength=len(days))/24
    actual=np.bincount(code,weights=panel.boardings,minlength=len(days))/count
    typical=pd.Series(actual).groupby(days.route).transform('median').clip(lower=100).to_numpy()
    days=annotate(days)
    matrix=panel[ALIASES].to_numpy();truth=panel.boardings.to_numpy()
    loss=np.column_stack([np.bincount(code,weights=abs(np.floor(np.maximum(matrix@w,0)+.5)-truth),minlength=len(days))/count/typical for w in grid])
    posterior=json.loads((folder/'posterior.json').read_text())
    assert posterior['origins']==origins and posterior['interface_aliases']==dict(zip(ALIASES,LABELS))
    for route in sorted(days.route.unique()):
        other=days.route.ne(route).to_numpy()
        pooled=pd.DataFrame(loss[other]).groupby(days.loc[other,'date'].to_numpy()).mean().to_numpy()
        global_p=probabilities(log_prior,pooled)
        for s in range(4):
            for g in range(3):
                selected=(days.route.eq(route)&days.season.eq(s)&days.group.eq(g)).to_numpy()
                if selected.sum()<7:selected=(days.route.eq(route)&days.group.eq(g)).to_numpy()
                local=global_p if selected.sum()<7 else probabilities(np.log(np.maximum(global_p,1e-300)),loss[selected])
                probability=.75*local+.25*global_p;mean=probability@grid
                denominator=(actual[selected]/typical[selected]).sum()
                risks=loss[selected].sum(axis=0)/denominator if denominator else np.ones(165)
                score=float(1-probability@risks)
                item=posterior['groups'][f'{route}:{s}:{g}']
                np.testing.assert_allclose(item['weights'],mean,rtol=1e-12,atol=1e-14)
                np.testing.assert_allclose(item['weight_sd'],np.sqrt(probability@np.square(grid-mean)),rtol=1e-12,atol=1e-14)
                np.testing.assert_allclose(item['expected_score_from_past'],score,rtol=1e-12,atol=1e-14)
                np.testing.assert_allclose(item['score_sd_from_weight_uncertainty'],np.sqrt(probability@np.square((1-risks)-score)),rtol=1e-12,atol=1e-14)
                assert item['past_days']==int(selected.sum())
    weights=read(folder/'daily_weights.csv');future=annotate(weights[['route','date']])
    np.testing.assert_array_equal(weights.season,future.season);np.testing.assert_array_equal(weights.risk_group,future.group)
    support=days.groupby(['route','season','group']).size()
    for row in weights.itertuples():
        if row.route==5:assert sum(getattr(row,'w_'+n) for n in LABELS)==0;continue
        item=posterior['groups'][f'{row.route}:{row.season}:{row.risk_group}']
        np.testing.assert_array_equal([getattr(row,'w_'+n) for n in LABELS],item['weights'])
        assert row.exact_season_days==int(support.get((row.route,row.season,row.risk_group),0)) and row.past_days==item['past_days']
        if row.past_days:assert row.past_score_under_learned_weights==item['expected_score_from_past']
    raw=read(folder/f'raw_{cutoff}.csv');expanded=raw[['route','date','hour']].merge(weights,on=['route','date'],validate='many_to_one')
    forecast=np.column_stack([read(source/f'raw_{cutoff}.csv').prediction for source in SOURCES])
    np.testing.assert_array_equal(raw.prediction,(forecast*expanded[['w_'+n for n in LABELS]].to_numpy()).sum(axis=1))
    assert raw.loc[raw.route.eq(5)|raw.hour.between(1,4),'prediction'].eq(0).all()
    checks.append(dict(cutoff=cutoff,origins=origins,unique_route_days=len(days),posterior_groups=len(posterior['groups'])))
raws=0
for folder in sorted((STUDY/'current_bayes').glob('trial_*'))+[STUDY/'current_bayes/selected',STUDY/'alternative']:
    if not folder.exists():continue
    params=json.loads((folder/'parameters.json' if folder.name!='selected' else folder.parent/'selection.json').read_text());recipe=params.get('params',params)['recipe']
    for path in folder.glob('raw_*.csv'):
        raw=read(path);ref=read(SOURCES[0]/path.name if recipe=='control' else STUDY/'calibrations'/path.stem[4:]/path.name)
        np.testing.assert_array_equal(raw.prediction,ref.prediction);raws+=1
rows=[];predictions=read(STUDY/'alternative/predictions.csv')
for cutoff,part in predictions.groupby('cutoff'):
    folder=STUDY/'calibrations'/cutoff;actual_score=1-abs(part.prediction-part.boardings).sum()/part.boardings.sum()
    if not (folder/'daily_weights.csv').exists():rows.append(dict(cutoff=cutoff,actual_score=actual_score,expected_score_proxy=None,optimism=None,exact_season_supported_fraction=0));continue
    weights=read(folder/'daily_weights.csv').loc[lambda f:f.route.ne(5)]
    volume=read(SOURCES[0]/f'raw_{cutoff}.csv').groupby(['route','date']).prediction.sum().rename('weight_volume').reset_index()
    weights=weights.merge(volume,on=['route','date'],validate='one_to_one')
    proxy=np.average(weights.past_score_under_learned_weights,weights=weights.weight_volume)
    rows.append(dict(cutoff=cutoff,actual_score=actual_score,expected_score_proxy=proxy,optimism=proxy-actual_score,
                     exact_season_supported_fraction=float(weights.exact_season_days.ge(7).mean())))
pd.DataFrame(rows).to_csv(OUT/'quality_calibration.csv',sep=';',index=False)
result=dict(checked_at=datetime.now(timezone.utc).isoformat(),input_code_sha256=len(info['sha256']),causal_component_panels_reconstructed=True,
    repeated_origins_averaged_by_route_date=True,finite_simplex_generalized_posterior_independently_reconstructed=True,
    conditional_past_risk_and_weight_sd_checked=True,future_season_support_and_weights_checked=True,raw_replay_exact=True,
    calibrations=checks,raw_forecasts_checked=raws,quality_calibration=rows,quality_proxy_is_not_hidden_score_confidence=True,
    previous_P17_budget_retained=True,audit_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
