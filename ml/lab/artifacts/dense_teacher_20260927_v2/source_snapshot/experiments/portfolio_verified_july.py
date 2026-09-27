"""P41: replay Bayesian calibration with the official August 11 restoration date."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import os
import platform
from pathlib import Path

from experiments.portfolio_bayes_direct import forecast
from experiments.portfolio_bayes_volume import ROOT
from experiments.portfolio_experiment import load_history, run_study, write_json
from experiments.portfolio_weather_volume import WEATHER_PATH


def run(args):
    if os.environ.get("SLURM_JOB_PARTITION")!="ais-cpu":
        raise RuntimeError("Expected ais-cpu")
    cpus=int(os.environ["SLURM_CPUS_PER_TASK"])
    if not 1<=cpus<=4 or len(os.sched_getaffinity(0))>cpus:
        raise RuntimeError("Unexpected allocation")
    if datetime.now(timezone.utc)>=datetime.fromisoformat("2026-09-27T15:40:40+00:00"):
        raise RuntimeError("Research reserve reached")
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    paths=[Path(args.history),Path(__file__),Path(WEATHER_PATH),ROOT/"external/july_restoration_sources.json"]
    paths += [Path("experiments")/f"portfolio_{n}.py" for n in
        ["experiment","bayes_direct","bayes_volume","direct_daily","operations","movement","ridge","structure","combine","windows","weather_volume"]]
    paths += [ROOT/f"{n}/{n}/selection.json" for n in ["operations","direct_daily","bayes_volume"]]
    paths += list((ROOT/"conditional_shape/conditional_shape/selected").glob("raw_*.csv"))
    if args.uncertainty_components:
        paths += list((ROOT/"verified_july/inner").glob("*/daily.csv"))
        paths += list((ROOT/"bayes_volume_verified_july/inner").glob("*/raw_*.csv"))
    write_json(out/"run_started.json",dict(job_id=os.environ["SLURM_JOB_ID"],command=os.sys.argv,
        source_regime="permitted retrospective official operations; restoration August11 vs estimated August7",
        uncertainty_component_search=args.uncertainty_components,
        versions=dict(python=platform.python_version(), **{n:importlib.metadata.version(n) for n in ["numpy","pandas","scikit-learn","optuna"]}),
        sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))
    history=load_history(args.history)
    space={"july_verified":[False,True],"season":["route"],"strength":[1.],"uncertainty":[1],"history_days":[224]}
    if args.uncertainty_components:
        space.update(july_verified=[True], uncertainty_component=["predictive","epistemic","none"])
    run_study(history,out,"uncertainty_components" if args.uncertainty_components else "verified_july",
        lambda c,e,p:forecast(history,c,e,p,out),dict(sampler="grid",space=space,trials=3 if args.uncertainty_components else 2))
    write_json(out/"completed.json",dict(job_id=os.environ["SLURM_JOB_ID"]))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history",required=True);parser.add_argument("--output",required=True)
    parser.add_argument("--uncertainty-components",action="store_true")
    run(parser.parse_args())
