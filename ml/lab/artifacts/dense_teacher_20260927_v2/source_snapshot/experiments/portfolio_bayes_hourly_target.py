"""P39: causal Bayesian volume correction targeting continuous hourly L1 loss."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.portfolio_bayes_direct import forecast
from experiments.portfolio_bayes_volume import ROOT
from experiments.portfolio_combine import raw_frame
from experiments.portfolio_conditional_shape import shape_forecast
from experiments.portfolio_experiment import load_history, run_study, write_json
from experiments.portfolio_weather_volume import WEATHER_PATH
from pipeline import KEYS


def hourly_targets(shape, truth):
    pd.testing.assert_frame_equal(shape[KEYS].reset_index(drop=True),truth[KEYS].reset_index(drop=True))
    if not np.isfinite(shape.prediction).all() or shape.prediction.lt(0).any():
        raise ValueError("Invalid hourly forecast shape")
    if not np.isfinite(truth.boardings).all() or truth.boardings.lt(0).any():
        raise ValueError("Invalid observed hourly target")
    predicted=shape.pivot(index=["route","date"],columns="hour",values="prediction")
    actual=truth.pivot(index=["route","date"],columns="hour",values="boardings")
    if list(predicted.columns)!=list(range(24)) or not np.isfinite(predicted).all().all():
        raise ValueError("Incomplete hourly target grid")
    weights=predicted.to_numpy(float)
    total=weights.sum(axis=1)
    weights=np.divide(weights,total[:,None],out=np.zeros_like(weights),where=total[:,None]>0)
    ratios=np.divide(actual.to_numpy(float),weights,out=np.full_like(weights,np.inf),where=weights>0)
    order=np.argsort(ratios,axis=1,kind="stable")
    cumulative=np.take_along_axis(weights,order,axis=1).cumsum(axis=1)
    position=np.argmax(cumulative>=weights.sum(axis=1)[:,None]*0.5,axis=1)
    median=np.take_along_axis(ratios,order,axis=1)[np.arange(len(total)),position]
    median[total==0]=0
    return predicted.index.to_frame(index=False).assign(optimal_volume=median)


def prepare_targets(past,history,cutoff,out):
    if past.date.max()>pd.Timestamp(cutoff):
        raise ValueError("Unfinished forecast cannot train hourly target")
    weather=pd.read_csv(WEATHER_PATH,sep=";",parse_dates=["date"])
    params=json.loads((ROOT/"conditional_shape/conditional_shape/selection.json").read_text())["params"]
    origins=past.date-pd.to_timedelta(past.horizon,unit="D")
    pieces=[]
    for origin,frame in past.groupby(origins,sort=True):
        origin=str(origin.date());end=str(frame.date.max().date())
        cache=out/"inner_shape"/origin
        cache.mkdir(parents=True,exist_ok=True)
        path=cache/f"raw_{origin}.csv"
        if not path.exists():
            base=raw_frame(ROOT/"bayes_volume/inner"/origin/f"raw_{origin}.csv",origin,end)
            raw=shape_forecast(history,origin,params,weather,base)
            raw.to_csv(path,sep=";",index=False,date_format="%Y-%m-%d")
            write_json(cache/"source.json",dict(origin=origin,end=end,params=params,
                fit_latest_date=origin,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        shape=raw_frame(path,origin,end)
        if json.loads((cache/"source.json").read_text())["sha256"]!=hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("Changed cached hourly target shape")
        truth=history.loc[history.date.gt(origin)&history.date.le(end),KEYS+["boardings"]]
        target=hourly_targets(shape,truth).merge(frame[KEYS[:2]+["horizon"]],on=KEYS[:2],validate="one_to_one")
        pieces.append(target)
    targets=pd.concat(pieces,ignore_index=True)
    result=past.merge(targets,on=["route","date","horizon"],validate="one_to_one")
    if len(result)!=len(past):
        raise ValueError("Incomplete hourly calibration targets")
    result[["route","date","horizon","boardings","optimal_volume"]].to_csv(
        out/f"training_targets_{cutoff}.csv",sep=";",index=False,date_format="%Y-%m-%d")
    return result


def run(args):
    if os.environ.get("SLURM_JOB_PARTITION")!="ais-cpu":
        raise RuntimeError("Expected ais-cpu")
    cpus=int(os.environ["SLURM_CPUS_PER_TASK"])
    if not 1<=cpus<=4 or len(os.sched_getaffinity(0))>cpus:
        raise RuntimeError("Unexpected allocation")
    if datetime.now(timezone.utc)>=datetime.fromisoformat("2026-09-27T15:40:40+00:00"):
        raise RuntimeError("Research reserve reached")
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    paths=[Path(args.history),Path(__file__),Path(WEATHER_PATH)]+[Path("experiments")/f"portfolio_{n}.py"
        for n in ["experiment","bayes_direct","bayes_volume","conditional_shape","operations","ridge","combine"]]
    paths += list((ROOT/"bayes_direct/inner").glob("*/daily.csv"))
    paths += list((ROOT/"bayes_volume/inner").glob("*/raw_*.csv"))
    paths += [ROOT/"conditional_shape/conditional_shape/selection.json",ROOT/"bayes_annual/bayes_annual/selection.json"]
    write_json(out/"run_started.json",dict(job_id=os.environ["SLURM_JOB_ID"],command=os.sys.argv,
        sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        target="weighted median of actual_hour/share for completed inner forecasts; continuous L1 minimizer",
        limitation="Gaussian clipped log fit is a surrogate; no future benchmark target used"))
    history=load_history(args.history)
    spec=dict(sampler="grid",space={"loss_target":["control","hourly","hourly_weighted"],
        "season":["route"],"strength":[1.],"uncertainty":[1],"history_days":[224]},trials=3)
    run_study(history,out,"bayes_hourly_target",lambda c,e,p:forecast(history,c,e,p,out),spec)
    write_json(out/"completed.json",dict(job_id=os.environ["SLURM_JOB_ID"]))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history",required=True);parser.add_argument("--output",required=True)
    run(parser.parse_args())
