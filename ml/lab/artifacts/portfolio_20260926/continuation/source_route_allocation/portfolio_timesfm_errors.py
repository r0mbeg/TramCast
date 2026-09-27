"""P52: Bayesian daily error correction with cutoff-safe TimesFM disagreement."""
import argparse
from datetime import datetime, timezone
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

from experiments.portfolio_bayes_direct import annual_features
from experiments.portfolio_bayes_volume import ROOT, calibrate
from experiments.portfolio_combine import raw_frame, transplant
from experiments.portfolio_experiment import FINAL, load_history, run_study, write_json
from experiments.portfolio_ridge import ACTIVE_ROUTES
from experiments.portfolio_timesfm import WEIGHTS_SHA256
from experiments.portfolio_windows import inner_windows

SOURCE = ROOT/"verified_july/inner"
TEACHERS = ROOT/"timesfm_teachers/daily"
SHAPE = ROOT/"bayes_shape/study/bayes_shape/selected"
VOLUME = ROOT/"bayes_volume_verified_july/inner"


def features(frame):
    delta = np.log1p(frame.timesfm.to_numpy())-np.log1p(frame.base.to_numpy())
    columns = [delta]+[delta*frame.route.eq(r).to_numpy(float) for r in ACTIVE_ROUTES]
    result = np.column_stack([annual_features(frame, True)]+columns)
    if not np.isfinite(result).all():
        raise ValueError("Invalid TimesFM disagreement features")
    return result


def daily(origin, end):
    frame = pd.read_csv(SOURCE/origin/"daily.csv", sep=";", parse_dates=["date"], float_precision="round_trip")
    teacher_path = TEACHERS/origin/"daily.csv"
    info = json.loads((teacher_path.parent/"source.json").read_text())
    if (info["origin"] != origin or info["fit_latest_date"] != origin or info["end"] != end
        or info["model_sha256"] != WEIGHTS_SHA256 or info["quantile_index"] != 3 or not info["july_verified"]):
        raise ValueError("Wrong causal TimesFM teacher")
    if hashlib.sha256(teacher_path.read_bytes()).hexdigest() != info["sha256"]:
        raise ValueError("Changed TimesFM daily teacher")
    teacher = pd.read_csv(teacher_path, sep=";", parse_dates=["date"], float_precision="round_trip")
    if not np.isfinite(teacher.prediction).all() or teacher.prediction.lt(0).any():
        raise ValueError("Invalid TimesFM teacher prediction")
    frame = frame.merge(teacher.rename(columns={"prediction":"timesfm"}), on=["route","date"], validate="one_to_one")
    if (len(frame)!=610 or frame.date.min()!=pd.Timestamp(origin)+pd.Timedelta(days=1)
        or frame.date.max()!=pd.Timestamp(end) or "boardings" in frame):
        raise ValueError("Invalid daily forecast input period or future target")
    if not frame.horizon.eq((frame.date-pd.Timestamp(origin)).dt.days).all():
        raise ValueError("Invalid cached origin horizon")
    return frame


def examples(history, cutoff, end):
    truth = history.loc[history.date.le(cutoff)].groupby(["route","date"], as_index=False).boardings.sum()
    pieces = []
    for origin, stop in inner_windows(cutoff):
        if pd.Timestamp(stop)>pd.Timestamp(cutoff):
            raise ValueError("Uncompleted TimesFM error target")
        piece = daily(origin, stop).merge(truth, on=["route","date"], validate="one_to_one")
        if len(piece)!=610:
            raise ValueError("Incomplete past error target")
        pieces.append(piece.assign(origin=pd.Timestamp(origin)))
    past = pd.concat(pieces, ignore_index=True)
    return past, daily(cutoff, end)


def forecast(history, cutoff, end, augmented, out):
    shape = raw_frame(SHAPE/f"raw_{cutoff}.csv", cutoff, end)
    if not augmented:
        return shape
    folder = out/"fits"/cutoff
    path = folder/f"raw_{cutoff}.csv"
    if path.exists():
        return raw_frame(path, cutoff, end)
    past, future = examples(history, cutoff, end)
    params = dict(strength=1., uncertainty=1, history_days=224)
    factors, posterior = calibrate(past, future, cutoff, params, feature_fn=features)
    folder.mkdir(parents=True, exist_ok=True)
    past.to_csv(folder/"past.csv", sep=";", index=False, date_format="%Y-%m-%d")
    future.assign(factor=factors).to_csv(folder/"future.csv", sep=";", index=False, date_format="%Y-%m-%d")
    write_json(folder/"posterior.json", dict(**posterior, cutoff=cutoff, params=params,
        origins=inner_windows(cutoff), feature_count=features(future).shape[1],
        uncertainty="conditional dependent errors; not confidence in hidden score"))
    raw = raw_frame(VOLUME/cutoff/f"raw_{cutoff}.csv", cutoff, end)
    raw = raw.merge(future[["route","date"]].assign(factor=factors), on=["route","date"], validate="many_to_one")
    raw["prediction"] *= raw.factor
    result = transplant(raw.drop(columns="factor"), shape, shape)
    result.to_csv(path, sep=";", index=False, date_format="%Y-%m-%d")
    return result


def run(args):
    if os.environ.get("SLURM_JOB_PARTITION") != "ais-cpu":
        raise RuntimeError("Expected ais-cpu")
    cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
    if not 1 <= cpus <= 4 or len(os.sched_getaffinity(0))>cpus:
        raise RuntimeError("Unexpected allocation")
    if datetime.now(timezone.utc)>=datetime.fromisoformat("2026-09-27T15:40:40+00:00"):
        raise RuntimeError("Research reserve reached")
    out = Path(args.output)/("pilot" if args.pilot else "study")
    out.mkdir(parents=True, exist_ok=True)
    paths = [Path(args.history), Path(__file__), Path("pipeline.py")]
    paths += [Path("experiments")/f"portfolio_{n}.py" for n in
        ["bayes_direct","bayes_volume","combine","experiment","ridge","timesfm","windows"]]
    paths += list(SOURCE.glob("*/daily.csv"))+list(TEACHERS.glob("*/*.csv"))+list(TEACHERS.glob("*/*.json"))
    paths += list(SHAPE.glob("raw_*.csv"))+list(VOLUME.glob("*/raw_*.csv"))
    write_json(out/"run_started.json", dict(job_id=os.environ["SLURM_JOB_ID"], command=os.sys.argv,
        versions=dict(python=platform.python_version(), **{n:importlib.metadata.version(n)
            for n in ["numpy","pandas","scikit-learn","optuna"]}),
        sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))
    history = load_history(args.history)
    started = time.monotonic()
    if args.pilot:
        forecast(history, *FINAL, True, out)
    else:
        run_study(history, out, "timesfm_errors", lambda c,e,p:forecast(history,c,e,p["augmented"],out),
            dict(sampler="grid", space={"augmented":[False,True]}, trials=2))
    write_json(out/"completed.json", dict(job_id=os.environ["SLURM_JOB_ID"], elapsed_seconds=time.monotonic()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pilot", action="store_true")
    run(parser.parse_args())
