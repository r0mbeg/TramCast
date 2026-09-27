"""P59: fixed route allocation and TimesFM daily dynamics compositions."""
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

from experiments.portfolio_combine import raw_frame
from experiments.portfolio_experiment import load_history,run_study,save_candidate,write_json
from experiments.portfolio_route_allocation import preserve_network
from pipeline import KEYS

ROOT=Path('artifacts/portfolio_20260926/continuation')
SOURCES=dict(network=ROOT/'bayes_shape_timesfm_blend/bayes_shape_timesfm_blend/selected',
    fractions=ROOT/'route_fraction/study/route_fraction/selected',
    baseline=ROOT/'bayes_shape/study/bayes_shape/selected')


def compose(network,fractions,baseline,recipe):
    pd.testing.assert_frame_equal(network[KEYS],fractions[KEYS])
    pd.testing.assert_frame_equal(network[KEYS],baseline[KEYS])
    if recipe=='parent025':return network.copy()
    if recipe=='parent028':return fractions.copy()
    if recipe=='network':return preserve_network(network,fractions)
    if recipe!='relative':raise ValueError('Unknown dynamics composition')
    old=baseline.groupby(['route','date']).prediction.transform('sum').to_numpy()
    new=fractions.groupby(['route','date']).prediction.transform('sum').to_numpy()
    if ((old==0)&(new!=0)).any():raise ValueError('Correction revived a zero baseline route-day')
    ratio=np.divide(new,old,out=np.ones(len(old)),where=old>0)
    result=network.copy();result['prediction']*=ratio
    return preserve_network(network,result)


def run(args):
    if (os.environ.get('SLURM_JOB_PARTITION')!='ais-cpu' or int(os.environ['SLURM_CPUS_PER_TASK'])!=1
        or len(os.sched_getaffinity(0))>1):raise RuntimeError('Expected one bound ais-cpu core')
    if datetime.now(timezone.utc)>=datetime.fromisoformat('2026-09-27T15:40:40+00:00'):
        raise RuntimeError('Research reserve reached')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    paths=[Path(args.history),Path(__file__),Path('pipeline.py')]
    paths += [Path('experiments')/f'portfolio_{n}.py' for n in ['combine','experiment','route_allocation']]
    paths += [p for folder in SOURCES.values() for p in folder.glob('raw_*.csv')]
    write_json(out/'run_started.json',dict(job_id=os.environ['SLURM_JOB_ID'],command=os.sys.argv,
        components={n:str(p) for n,p in SOURCES.items()},versions=dict(python=platform.python_version(),
            **{n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','optuna']}),
        sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))
    def forecast(cutoff,end,recipe):
        frames={n:raw_frame(p/f'raw_{cutoff}.csv',cutoff,end) for n,p in SOURCES.items()}
        return compose(**frames,recipe=recipe)
    started=time.monotonic();history=load_history(args.history)
    run_study(history,out,'fraction_dynamics',lambda c,e,p:forecast(c,e,p['recipe']),
        dict(sampler='grid',space={'recipe':['parent025','parent028','network','relative']},trials=4))
    selection=json.loads((out/'fraction_dynamics/selection.json').read_text())['params']['recipe']
    if selection.startswith('parent'):
        import optuna
        study=optuna.load_study(study_name='fraction_dynamics',storage=f"sqlite:///{out/'fraction_dynamics/study.db'}")
        best=max((t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE
            and t.params['recipe'] in ['network','relative']),key=lambda t:t.value)
        folder=out/'composition_selected';folder.mkdir(exist_ok=True)
        write_json(folder/'parameters.json',dict(params=best.params,trial=best.number,value=best.value,
            selection='W1/W2 bestcomposition; parent still overallstudywinner'))
        save_candidate(history,folder,'fraction_dynamics_composition',lambda c,e:forecast(c,e,best.params['recipe']))
    write_json(out/'completed.json',dict(job_id=os.environ['SLURM_JOB_ID'],elapsed_seconds=time.monotonic()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history',required=True);parser.add_argument('--output',required=True)
    run(parser.parse_args())
