"""P68: reuse generalized Bayesian risk weights for the current four forecasters."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import time

import numpy as np
import pandas as pd
from experiments.portfolio_bayes import infer_weights,MODELS,GRID
from experiments.portfolio_structure import tagged
from experiments.portfolio_windows import season
from experiments.portfolio_combine import raw_frame
from experiments.portfolio_experiment import load_history,run_study,save_candidate,write_json
from pipeline import KEYS

ROOT=Path('artifacts/portfolio_20260926/continuation')
# These are aliases required by the unchanged P17 function, not the component methods.
COMPONENTS=dict(zip(MODELS,[ROOT/p for p in ['tabular_shape/study/tabular_shape/selected',
    'school_fraction/study/school_fraction/selected','bayes_shape_timesfm_blend/bayes_shape_timesfm_blend/selected',
    'bayes_shape/study/bayes_shape/selected']]))
LABELS=['030','029','025','024']
ORIGINS=['2025-04-30','2025-06-30','2025-07-31','2025-08-31']


def group(frame):
    return np.where(frame.is_workday,0,np.where(frame.weekday.eq(5)&~frame.off,1,2))


def component(alias,cutoff,end):
    return raw_frame(COMPONENTS[alias]/f'raw_{cutoff}.csv',cutoff,end)


def forecast(history,cutoff,end,recipe,out):
    control=component(MODELS[0],cutoff,end)
    if recipe=='control':return control
    if recipe!='posterior':raise ValueError('Unknown Bayesian risk recipe')
    folder=out/'calibrations'/cutoff;path=folder/f'raw_{cutoff}.csv'
    if path.exists():return raw_frame(path,cutoff,end)
    folder.mkdir(parents=True,exist_ok=True)
    origins=[o for o in ORIGINS if pd.Timestamp(o)+pd.Timedelta(days=61)<=pd.Timestamp(cutoff)]
    if not origins:
        write_json(folder/'gate.json',dict(cutoff=cutoff,origins=[],action='exact030 control'))
        control.to_csv(path,sep=';',index=False,date_format='%Y-%m-%d');return control
    pieces=[]
    for origin in origins:
        stop=str((pd.Timestamp(origin)+pd.Timedelta(days=61)).date())
        part=component(MODELS[0],origin,stop)[KEYS].copy()
        for alias in MODELS:part[alias]=component(alias,origin,stop).prediction.to_numpy()
        truth=history.loc[history.date.gt(origin)&history.date.le(stop),KEYS+['boardings']]
        part=part.merge(truth,on=KEYS,validate='one_to_one')
        if len(part)!=14640:raise ValueError('Incomplete completed risk reference')
        pieces.append(part.assign(origin=pd.Timestamp(origin)))
    panel=pd.concat(pieces,ignore_index=True)
    if panel.date.max()>pd.Timestamp(cutoff):raise ValueError('Future loss in posterior')
    weights=infer_weights(panel,cutoff)
    past=tagged(panel.loc[panel.route.ne(5),['route','date']].drop_duplicates(),cutoff)
    past['season']=season(past.date);past['risk_group']=group(past)
    support=past.groupby(['route','season','risk_group']).size()
    days=tagged(control[['route','date']].drop_duplicates(),cutoff)
    days['season']=season(days.date);days['risk_group']=group(days)
    rows=[]
    for row in days.itertuples():
        record=dict(route=row.route,date=row.date,season=row.season,risk_group=row.risk_group)
        if row.route==5:
            record.update({f'w_{name}':0. for name in LABELS});record.update(past_days=0,exact_season_days=0)
        else:
            summary=weights[f'{row.route}:{row.season}:{row.risk_group}']
            record.update({f'w_{name}':float(w) for name,w in zip(LABELS,summary['weights'])})
            record.update(past_days=summary['past_days'],exact_season_days=int(support.get((row.route,row.season,row.risk_group),0)),
                past_score_under_learned_weights=summary['expected_score_from_past'] if summary['past_days'] else None,
                sd_from_weight_uncertainty=summary['score_sd_from_weight_uncertainty'] if summary['past_days'] else None)
        rows.append(record)
    daily_weights=pd.DataFrame(rows);columns=[f'w_{name}' for name in LABELS]
    active=daily_weights.route.ne(5)
    np.testing.assert_allclose(daily_weights.loc[active,columns].sum(axis=1),1,rtol=0,atol=1e-12)
    if not np.isfinite(daily_weights[columns]).all().all() or daily_weights[columns].lt(0).any().any():raise ValueError('Invalid posterior weights')
    expanded=control[KEYS].merge(daily_weights[['route','date']+columns],on=['route','date'],validate='many_to_one')
    pd.testing.assert_frame_equal(expanded[KEYS],control[KEYS])
    values=np.column_stack([component(alias,cutoff,end).prediction for alias in MODELS])
    result=control[KEYS].assign(prediction=(values*expanded[columns].to_numpy()).sum(axis=1))
    if not result.loc[result.route.eq(5)|result.hour.between(1,4),'prediction'].eq(0).all():raise ValueError('Changed structural zeros')
    panel.to_csv(folder/'past_panel.csv',sep=';',index=False,date_format='%Y-%m-%d')
    daily_weights.to_csv(folder/'daily_weights.csv',sep=';',index=False,date_format='%Y-%m-%d')
    write_json(folder/'posterior.json',dict(cutoff=cutoff,origins=origins,interface_aliases=dict(zip(MODELS,LABELS)),groups=weights,
        quality_scope='past conditional risk and weight uncertainty, not hidden-score confidence'))
    result.to_csv(path,sep=';',index=False,date_format='%Y-%m-%d');return result


def run(args):
    if os.environ.get('SLURM_JOB_PARTITION')!='ais-cpu' or int(os.environ.get('SLURM_CPUS_PER_TASK','0'))!=1 or len(os.sched_getaffinity(0))>1:raise RuntimeError('Expected one bound ais-cpu core')
    if datetime.now(timezone.utc)>=datetime.fromisoformat('2026-09-27T15:40:40+00:00'):raise RuntimeError('Research reserve reached')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    paths=[Path(args.history),Path(__file__),Path('pipeline.py'),Path('artifacts/calendar_sources.json')]
    paths+=[Path('experiments')/f'{name}.py' for name in ['portfolio_bayes','portfolio_structure','portfolio_windows','portfolio_combine','portfolio_experiment','portfolio_ridge','calendar_experiment']]
    paths += [p for folder in COMPONENTS.values() for p in folder.glob('raw_*.csv')]
    write_json(out/'run_started.json',dict(job_id=os.environ['SLURM_JOB_ID'],command=os.sys.argv,versions=dict(python=platform.python_version(),
        **{n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','optuna']}),
        sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},interface_aliases=dict(zip(MODELS,LABELS)),
        old_fixed_recipe_runs=1,previous_P17_allocation_seconds=23,additional_allocation_total_limit_seconds=120,cumulative_limit_seconds=143,
        grid_points=len(GRID),posterior_deterministic=True))
    started=time.monotonic();history=load_history(args.history)
    run_study(history,out,'current_bayes',lambda c,e,p:forecast(history,c,e,p['recipe'],out),
        dict(sampler='grid',space={'recipe':['control','posterior']},trials=2,
             allocation_subbudget_seconds=120,previous_family_seconds=23,cumulative_family_allocation_seconds=143))
    selected=json.loads((out/'current_bayes/selection.json').read_text())['params']['recipe']
    if selected=='control':
        (out/'alternative').mkdir(exist_ok=True);write_json(out/'alternative/parameters.json',dict(recipe='posterior',selection='only fixed new rule; no retuning'))
        save_candidate(history,out/'alternative','current_bayes_alternative',lambda c,e:forecast(history,c,e,'posterior',out))
    write_json(out/'completed.json',dict(job_id=os.environ['SLURM_JOB_ID'],elapsed_seconds=time.monotonic()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--history',required=True)
    parser.add_argument('--output',required=True);run(parser.parse_args())
