"""P37: Bayesian direct origin/horizon ratios with causal context and smooth seasons."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.portfolio_bayes_volume import ROOT, calibrate
from experiments.portfolio_combine import raw_frame
from experiments.portfolio_direct_daily import context, daily_history, training_examples
from experiments.portfolio_experiment import load_history, run_study, write_json
from experiments.portfolio_operations import normal_history, apply_operations
from experiments.portfolio_movement import disrupted
from experiments.portfolio_ridge import ACTIVE_ROUTES
from experiments.portfolio_weather_volume import WEATHER_PATH
from pipeline import KEYS


def features(frame, route_specific=False):
    phase=2*np.pi*(frame.date.dt.dayofyear.to_numpy()-1)/365
    origin_phase=2*np.pi*(frame.origin.dt.dayofyear.to_numpy()-1)/365
    ratios=np.column_stack([np.log(frame[f"ratio{d}"].to_numpy().clip(0.1,10)) for d in [7,14,28]])
    cycles=np.column_stack([np.sin(phase),np.cos(phase),np.sin(phase)-np.sin(origin_phase),
        np.cos(phase)-np.cos(origin_phase)])
    columns=[np.column_stack([frame.route.eq(r).to_numpy(float) for r in ACTIVE_ROUTES]),
        np.column_stack([frame.effective_weekday.eq(d).to_numpy(float) for d in range(1,7)]),
        np.column_stack([(frame.route.eq(r)&frame.daytype.eq(t)).to_numpy(float)
            for r in ACTIVE_ROUTES for t in [1,2]]),ratios,cycles,
        np.column_stack([frame.horizon.to_numpy()/61,frame.log_reference.to_numpy()/10,
            frame.temperature_2m_mean.to_numpy()/10,np.log1p(frame.precipitation_sum.to_numpy()),
            frame.daylight_duration.to_numpy()/21600,frame.origin_temperature.to_numpy()/10,
            frame.origin_daylight.to_numpy()/21600,frame.off.to_numpy(float)])]
    if route_specific:
        columns += [np.column_stack([ratios,cycles])*frame.route.eq(r).to_numpy(float)[:,None]
            for r in ACTIVE_ROUTES]
    result=np.column_stack(columns)
    if not np.isfinite(result).all():
        raise ValueError("Invalid Bayesian direct horizon features")
    return result


def allocate_volume(shape, daily):
    raw=shape.merge(daily[["route","date","volume"]],on=["route","date"],how="left",validate="many_to_one")
    denominator=raw.groupby(["route","date"]).prediction.transform("sum")
    closed=raw.route.eq(5)|(raw.route.eq(50)&disrupted(raw,"autumn"))
    if not np.isfinite(raw.loc[~closed,"volume"]).all() or denominator[~closed].le(0).any():
        raise ValueError("Missing daily volume or hourly shape for an active route")
    raw["prediction"]=(raw.prediction/denominator.where(denominator.gt(0))).fillna(0)*raw.volume.fillna(0)
    raw.loc[closed,"prediction"]=0
    return raw[KEYS+["prediction"]]


def forecast(history,cutoff,end,params,out):
    weather=pd.read_csv(WEATHER_PATH,sep=";",parse_dates=["date"])
    cache=out/f"examples_{cutoff}.csv"
    if cache.exists():
        past=pd.read_csv(cache,sep=";",parse_dates=["date","origin"],float_precision="round_trip")
    else:
        past=training_examples(history,cutoff,weather,august7=True)
        past.to_csv(cache,sep=";",index=False,date_format="%Y-%m-%d")
    if past.date.max()>pd.Timestamp(cutoff) or not (past.date.gt(past.origin)&past.horizon.between(1,61)).all():
        raise ValueError("Future target in Bayesian horizon fit")
    future=context(daily_history(history,cutoff,august7=True),cutoff,
        pd.date_range(pd.Timestamp(cutoff)+pd.Timedelta(days=1),end),weather)
    past["base"]=past.reference
    future["base"]=future.reference
    feature_fn=lambda frame: features(frame,params["route_specific"])
    factor,summary=calibrate(past,future,cutoff,{**params,"history_days":224},feature_fn=feature_fn)
    future["volume"]=future.reference*factor
    shape=raw_frame(ROOT/"conditional_shape/conditional_shape/selected"/f"raw_{cutoff}.csv",cutoff,end)
    raw=allocate_volume(shape,future)
    base=json.loads((ROOT/"operations/operations/selection.json").read_text())["params"]
    august=json.loads((ROOT/"august_operations/august_operations/selection.json").read_text())["params"]["august7"]
    raw=apply_operations(raw[KEYS+["prediction"]],history.loc[history.date.le(cutoff)],
        normal_history(history,cutoff,extended=True,august7=True),{**base,"august7":august})
    name=f"{cutoff}_{params['route_specific']}_{params['strength']}_{params['uncertainty']}"
    write_json(out/f"posterior_{name}.json",dict(**summary,cutoff=cutoff,params=params,
        origins=int(past.origin.nunique()),training_weight="1/count(route,date); repeated origins not independent",
        target="clipped log((normal daily boardings+100)/(past reference+100))",
        uncertainty="conditional model variance, not confidence in hidden score"))
    return raw


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
        for n in ["experiment","bayes_volume","direct_daily","operations","ridge","combine"]]
    paths += list((ROOT/"conditional_shape/conditional_shape/selected").glob("raw_*.csv"))
    paths += [ROOT/f"{n}/{n}/selection.json" for n in ["operations","august_operations"]]
    write_json(out/"run_started.json",dict(job_id=os.environ["SLURM_JOB_ID"],command=os.sys.argv,
        sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))
    history=load_history(args.history)
    run_study(history,out,"bayes_horizon",lambda c,e,p:forecast(history,c,e,p,out),dict(sampler="grid",
        space={"route_specific":[False,True],"strength":[0.5,1.],"uncertainty":[0,1]},trials=8))
    write_json(out/"completed.json",dict(job_id=os.environ["SLURM_JOB_ID"]))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history",required=True);parser.add_argument("--output",required=True)
    run(parser.parse_args())
