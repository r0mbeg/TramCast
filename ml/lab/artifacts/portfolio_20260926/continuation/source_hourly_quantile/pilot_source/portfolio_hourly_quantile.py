"""P40: regularized linear correction trained on the actual continuous hourly L1 loss."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler

from experiments.portfolio_bayes_direct import annual_features
from experiments.portfolio_bayes_volume import ROOT
from experiments.portfolio_combine import raw_frame, transplant
from experiments.portfolio_experiment import load_history, run_study, write_json
from experiments.portfolio_windows import inner_windows
from pipeline import KEYS


def fit_hourly(past,hourly,future,cutoff,alpha):
    if past.date.max()>pd.Timestamp(cutoff) or hourly.date.max()>pd.Timestamp(cutoff):
        raise ValueError("Future target in hourly quantile fit")
    past=past.loc[past.route.ne(5)&past.base.gt(0)&
        past.date.gt(pd.Timestamp(cutoff)-pd.Timedelta(days=224))].copy().reset_index(drop=True)
    past["index"]=np.arange(len(past))
    repetitions=past.groupby(["route","date"]).base.transform("size")
    past["repeat_weight"]=1/repetitions
    frame=hourly.merge(past[["route","date","horizon","index","repeat_weight"]],
        on=["route","date","horizon"],validate="many_to_one")
    if frame.duplicated(["route","date","hour","horizon"]).any():
        raise ValueError("Duplicate hourly training observation")
    frame=frame.loc[frame.hour_base.gt(0)].copy()
    if not len(frame) or not np.isfinite(frame[["hour_base","boardings"]]).all().all() or frame.boardings.lt(0).any():
        raise ValueError("Invalid hourly quantile input")
    scaler=StandardScaler(with_mean=False).fit(annual_features(past,True),
        sample_weight=past.base.to_numpy()*past.repeat_weight.to_numpy())
    daily_features=scaler.transform(annual_features(past,True))
    x=csc_matrix(daily_features[frame["index"].to_numpy(int)])
    weights=frame.hour_base.to_numpy()*frame.repeat_weight.to_numpy()
    target=frame.boardings.to_numpy()/frame.hour_base.to_numpy()-1
    started=time.monotonic()
    model=QuantileRegressor(quantile=0.5,alpha=alpha,solver="highs",solver_options={"time_limit":120})
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error",ConvergenceWarning)
            model.fit(x,target,sample_weight=weights)
    except ConvergenceWarning as error:
        raise ValueError("Hourly linear program did not converge within its budget") from error
    factor=np.clip(1+model.predict(scaler.transform(annual_features(future,True))),0.5,2.)
    if not np.isfinite(factor).all():
        raise ValueError("Invalid hourly correction")
    fitted=np.clip(1+model.predict(x),0.5,2.)
    before=np.sum(frame.repeat_weight.to_numpy()*np.abs(frame.hour_base.to_numpy()-frame.boardings.to_numpy()))
    after=np.sum(frame.repeat_weight.to_numpy()*np.abs(frame.hour_base.to_numpy()*fitted-frame.boardings.to_numpy()))
    summary=dict(alpha=alpha,coefficient=model.coef_.tolist(),intercept=float(model.intercept_),
        feature_scale=scaler.scale_.tolist(),rows=len(frame),daily_rows=len(past),iterations=int(model.n_iter_),
        seconds=time.monotonic()-started,latest_target=str(frame.date.max().date()),
        training_absolute_error_before=float(before),training_absolute_error_after=float(after),
        factor_range=[float(factor.min()),float(factor.max())],
        objective="hour_base/count(route,date) weighted median loss of actual/hour_base-1 + L1 regularization",
        limitation="factor clipped to [0.5,2]; rounding only at publication; no Bayesian uncertainty estimate")
    return factor,summary


def forecast(history,cutoff,end,alpha,out):
    daily=[];hourly=[]
    for origin,stop in inner_windows(cutoff):
        frame=pd.read_csv(ROOT/"bayes_direct/inner"/origin/"daily.csv",sep=";",
            parse_dates=["date"],float_precision="round_trip")
        raw=raw_frame(ROOT/"bayes_hourly_target/inner_shape"/origin/f"raw_{origin}.csv",origin,stop)
        truth=history.loc[history.date.gt(origin)&history.date.le(stop),KEYS+["boardings"]]
        pd.testing.assert_frame_equal(raw[KEYS],truth[KEYS].reset_index(drop=True))
        raw=raw.rename(columns={"prediction":"hour_base"}).merge(truth,on=KEYS,validate="one_to_one")
        raw["horizon"]=(raw.date-pd.Timestamp(origin)).dt.days
        daily.append(frame);hourly.append(raw)
    future=pd.read_csv(ROOT/"bayes_direct/inner"/cutoff/"daily.csv",sep=";",
        parse_dates=["date"],float_precision="round_trip")
    factor,summary=fit_hourly(pd.concat(daily,ignore_index=True),pd.concat(hourly,ignore_index=True),
        future,cutoff,alpha)
    name=f"{cutoff}_{alpha}"
    write_json(out/f"fit_{name}.json",dict(**summary,cutoff=cutoff,inner_windows=inner_windows(cutoff)))
    future["factor"]=factor
    future[["route","date","factor"]].to_csv(out/f"factors_{name}.csv",sep=";",index=False,date_format="%Y-%m-%d")
    base=raw_frame(ROOT/"bayes_volume/inner"/cutoff/f"raw_{cutoff}.csv",cutoff,end)
    shape=raw_frame(ROOT/"conditional_shape/conditional_shape/selected"/f"raw_{cutoff}.csv",cutoff,end)
    raw=transplant(base,shape,base).merge(future[["route","date","factor"]],on=["route","date"],validate="many_to_one")
    raw["prediction"]*=raw.factor
    return raw[KEYS+["prediction"]]


def run(args):
    if os.environ.get("SLURM_JOB_PARTITION")!="ais-cpu":
        raise RuntimeError("Expected ais-cpu")
    cpus=int(os.environ["SLURM_CPUS_PER_TASK"])
    if not 1<=cpus<=4 or len(os.sched_getaffinity(0))>cpus:
        raise RuntimeError("Unexpected allocation")
    if datetime.now(timezone.utc)>=datetime.fromisoformat("2026-09-27T15:40:40+00:00"):
        raise RuntimeError("Research reserve reached")
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    paths=[Path(args.history),Path(__file__)]+[Path("experiments")/f"portfolio_{n}.py"
        for n in ["experiment","bayes_direct","bayes_volume","combine","windows"]]
    paths += list((ROOT/"bayes_direct/inner").glob("*/daily.csv"))
    paths += list((ROOT/"bayes_hourly_target/inner_shape").glob("*/raw_*.csv"))
    paths += list((ROOT/"conditional_shape/conditional_shape/selected").glob("raw_*.csv"))
    write_json(out/("pilot_started.json" if args.pilot else "run_started.json"),dict(job_id=os.environ["SLURM_JOB_ID"],
        command=os.sys.argv,versions={p:importlib.metadata.version(p) for p in ["scikit-learn","scipy","numpy","optuna"]},
        sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))
    history=load_history(args.history)
    started=time.monotonic()
    if args.pilot:
        if (out/"pilot_completed.json").exists():
            raise ValueError("Pilot already completed; do not repeat")
        forecast(history,"2025-04-30","2025-06-30",0.01,out).to_csv(out/"pilot_raw.csv",sep=";",index=False)
        write_json(out/"pilot_completed.json",dict(job_id=os.environ["SLURM_JOB_ID"],seconds=time.monotonic()-started))
        return
    run_study(history,out,"hourly_quantile",lambda c,e,p:forecast(history,c,e,p["alpha"],out),
        dict(sampler="grid",space={"alpha":[0.001,0.01,0.1]},trials=3))
    write_json(out/"completed.json",dict(job_id=os.environ["SLURM_JOB_ID"],seconds=time.monotonic()-started))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history",required=True);parser.add_argument("--output",required=True)
    parser.add_argument("--pilot",action="store_true")
    run(parser.parse_args())
