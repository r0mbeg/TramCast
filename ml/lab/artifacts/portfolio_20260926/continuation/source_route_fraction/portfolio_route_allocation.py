"""P57: Bayesian relative route demand with a fixed network daily forecast."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import resource
import time

import numpy as np
import pandas as pd

from experiments.portfolio_bayes_direct import annual_features
from experiments.portfolio_bayes_volume import calibrate
from experiments.portfolio_combine import mix, raw_frame, transplant
from experiments.portfolio_experiment import FINAL, load_history, run_study, write_json
from experiments.portfolio_timesfm_errors import examples, SOURCE, TEACHERS, SHAPE, VOLUME
from pipeline import ROUTES


def relative_targets(past, cutoff):
    if (past.empty or past.date.max()>pd.Timestamp(cutoff) or not past.date.gt(past.origin).all()
        or past.duplicated(["origin","route","date"]).any()):
        raise ValueError("Invalid completed route allocation targets")
    if not np.isfinite(past[["base","boardings"]]).all().all() or past[["base","boardings"]].lt(0).any().any():
        raise ValueError("Invalid route allocation values")
    result=past.copy()
    groups=result.groupby(["origin","date"])
    if not groups.route.nunique().eq(10).all() or not result.route.isin(ROUTES).all():
        raise ValueError("Incomplete network target grid")
    result["network_base"]=groups.base.transform("sum")
    result["network_actual"]=groups.boardings.transform("sum")
    if not result.network_actual.gt(0).all() or not result.network_base.gt(0).all():
        raise ValueError("Undefined network allocation")
    result["relative_boardings"]=result.boardings*result.network_base/result.network_actual
    np.testing.assert_allclose(result.groupby(["origin","date"]).relative_boardings.sum(),
        groups.base.sum(),rtol=1e-12,atol=1e-8)
    return result


def preserve_network(base, learned):
    pd.testing.assert_frame_equal(base[["route","date","hour"]],learned[["route","date","hour"]])
    result=learned.copy()
    if not np.isfinite(result.prediction).all() or result.prediction.lt(0).any():
        raise ValueError("Invalid route allocation prediction")
    total=base.groupby("date").prediction.transform("sum")
    denominator=result.groupby("date").prediction.transform("sum")
    if (total.gt(0)&denominator.le(0)).any():
        raise ValueError("Route allocation erased a positive network day")
    result["prediction"]*=np.divide(total.to_numpy(),denominator.to_numpy(),out=np.ones(len(result)),where=denominator.to_numpy()>0)
    np.testing.assert_allclose(result.groupby("date").prediction.sum(),base.groupby("date").prediction.sum(),rtol=1e-12,atol=1e-8)
    if not result.loc[base.prediction.eq(0),"prediction"].eq(0).all():
        raise ValueError("Route allocation changed a zero hourly share")
    return result


def forecast(history, cutoff, end, recipe, out):
    shape=raw_frame(SHAPE/f"raw_{cutoff}.csv",cutoff,end)
    if recipe=="control":
        return shape
    if recipe not in ["half","full"]:
        raise ValueError("Unknown route allocation recipe")
    folder=out/"fits"/cutoff;path=folder/f"raw_{cutoff}.csv"
    if path.exists():
        learned=raw_frame(path,cutoff,end)
    else:
        started=time.monotonic()
        past,future=examples(history,cutoff,end)
        past=relative_targets(past,cutoff)
        factor,posterior=calibrate(past,future,cutoff,dict(strength=1.,uncertainty=1,history_days=224),
            feature_fn=lambda frame:annual_features(frame,True),target_column="relative_boardings")
        folder.mkdir(parents=True,exist_ok=True)
        past.to_csv(folder/"past.csv",sep=";",index=False,date_format="%Y-%m-%d")
        future.assign(factor=factor).to_csv(folder/"future.csv",sep=";",index=False,date_format="%Y-%m-%d")
        write_json(folder/"posterior.json",dict(**posterior,cutoff=cutoff,feature_count=97,
            target="actualroute_day*network_base/network_actual",fit_seconds=time.monotonic()-started,
            uncertainty="conditional dependent errors; not hidden-score confidence"))
        raw=raw_frame(VOLUME/cutoff/f"raw_{cutoff}.csv",cutoff,end).merge(
            future[["route","date"]].assign(factor=factor),on=["route","date"],validate="many_to_one")
        raw["prediction"]*=raw.factor
        learned=preserve_network(shape,transplant(raw.drop(columns="factor"),shape,shape))
        learned.to_csv(path,sep=";",index=False,date_format="%Y-%m-%d")
    return mix(learned,shape,0.5 if recipe=="half" else 1.)


def run(args):
    if (os.environ.get("SLURM_JOB_PARTITION")!="ais-cpu" or int(os.environ["SLURM_CPUS_PER_TASK"])!=1
        or len(os.sched_getaffinity(0))>1):
        raise RuntimeError("Expected one bound ais-cpu core")
    if datetime.now(timezone.utc)>=datetime.fromisoformat("2026-09-27T15:40:40+00:00"):
        raise RuntimeError("Research reserve reached")
    out=Path(args.output)/("pilot" if args.pilot else "study");out.mkdir(parents=True,exist_ok=True)
    paths=[Path(args.history),Path(__file__),Path("pipeline.py")]
    paths += [Path("experiments")/f"portfolio_{n}.py" for n in
        ["bayes_direct","bayes_volume","timesfm_errors","timesfm","combine","experiment","ridge","windows"]]
    paths += list(SOURCE.glob("*/daily.csv"))+list(TEACHERS.glob("*/*.csv"))+list(TEACHERS.glob("*/*.json"))
    paths += list(SHAPE.glob("raw_*.csv"))+list(VOLUME.glob("*/raw_*.csv"))
    write_json(out/"run_started.json",dict(job_id=os.environ["SLURM_JOB_ID"],command=os.sys.argv,
        versions=dict(python=platform.python_version(),**{n:importlib.metadata.version(n) for n in
            ["numpy","pandas","scikit-learn","optuna"]}),
        sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))
    started=time.monotonic();history=load_history(args.history)
    if args.pilot:
        forecast(history,*FINAL,"full",out)
    else:
        run_study(history,out,"route_allocation",lambda c,e,p:forecast(history,c,e,p["recipe"],out),
            dict(sampler="grid",space={"recipe":["control","half","full"]},trials=3))
    write_json(out/"completed.json",dict(job_id=os.environ["SLURM_JOB_ID"],elapsed_seconds=time.monotonic()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history",required=True);parser.add_argument("--output",required=True)
    parser.add_argument("--pilot",action="store_true")
    run(parser.parse_args())
