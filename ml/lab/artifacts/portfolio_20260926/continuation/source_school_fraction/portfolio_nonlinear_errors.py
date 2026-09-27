"""P53: absolute-loss nonlinear correction of completed daily forecast errors."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import os
from pathlib import Path
import pickle
import platform
import resource
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from experiments.portfolio_combine import mix, raw_frame, transplant
from experiments.portfolio_experiment import FINAL, load_history, run_study, write_json
from experiments.portfolio_timesfm_errors import examples, SOURCE, TEACHERS, SHAPE, VOLUME


def features(frame):
    phase = 2*np.pi*(frame.date.dt.dayofyear.to_numpy()-1)/365
    origin = frame.date-np.asarray(frame.horizon, dtype="timedelta64[D]")
    earlier = 2*np.pi*(origin.dt.dayofyear.to_numpy()-1)/365
    columns = [frame.route, frame.effective_weekday, frame.daytype, frame.horizon/61,
        frame.off.astype(int), frame.summer, np.sin(phase), np.cos(phase), np.sin(earlier),
        np.cos(earlier), np.sin(phase)-np.sin(earlier), np.cos(phase)-np.cos(earlier), np.log1p(frame.base)/10]
    columns += [np.log1p(frame[name])-np.log1p(frame.base) for name in ["timesfm","direct","ridge","regime"]]
    columns += [np.log(frame[f"ratio{days}"].clip(0.1,10)) for days in [7,14,28]]
    columns += [frame.temperature_2m_mean/10, np.log1p(frame.precipitation_sum), frame.daylight_duration/21600]
    result = np.column_stack(columns).astype(float)
    if not np.isfinite(result).all():
        raise ValueError("Invalid nonlinear error features")
    return result


def prepare(past, cutoff):
    if past.empty or past.date.max()>np.datetime64(cutoff) or not past.date.gt(past.origin).all():
        raise ValueError("Invalid completed daily error targets")
    if not np.isfinite(past[["boardings","base"]]).all().all() or past[["boardings","base"]].lt(0).any().any():
        raise ValueError("Invalid daily error values")
    past = past.loc[past.route.ne(5)&past.base.gt(0)&past.date.gt(np.datetime64(cutoff)-np.timedelta64(224,"D"))].copy()
    if past.empty:
        raise ValueError("No positive completed daily references")
    past["target"] = past.boardings/past.base
    past["weight"] = past.base/past.groupby(["route","date"]).base.transform("size")
    past["weight"] /= past.weight.mean()
    return past


def fit(past, future, cutoff, leaves, out):
    started = time.monotonic()
    past = prepare(past, cutoff)
    model = HistGradientBoostingRegressor(loss="absolute_error", max_iter=100,
        learning_rate=0.05, max_leaf_nodes=leaves, min_samples_leaf=50,
        l2_regularization=1, categorical_features=[0,1,2], early_stopping=False, random_state=42)
    model.fit(features(past), past.target, sample_weight=past.weight)
    future = future.copy()
    active = future.route.ne(5)&future.base.gt(0)
    future["ratio"] = 1.
    future.loc[active,"ratio"] = model.predict(features(future.loc[active]))
    future["factor"] = future.ratio.clip(0.5,2)
    if not np.isfinite(future.factor).all():
        raise ValueError("Invalid daily correction factor")
    out.mkdir(parents=True, exist_ok=True)
    past.to_csv(out/"training.csv", sep=";", index=False, date_format="%Y-%m-%d")
    future.to_csv(out/"future.csv", sep=";", index=False, date_format="%Y-%m-%d")
    payload = pickle.dumps(model, protocol=5)
    (out/"model.pkl").write_bytes(payload)
    np.testing.assert_array_equal(pickle.loads(payload).predict(features(future.loc[active])),future.loc[active,"ratio"])
    write_json(out/"fit.json", dict(cutoff=cutoff, leaves=leaves, rows=len(past), features=23,
        earliest_target_date=str(past.date.min().date()), latest_target_date=str(past.date.max().date()),
        origins=int(past.origin.nunique()), iterations=int(model.n_iter_),
        loss="absolute actual/base ratio, weighted base/repeats; daily L1 surrogate",
        model_sha256=hashlib.sha256(payload).hexdigest(), model_reload_exact=True,
        fit_seconds=time.monotonic()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    return future[["route","date","factor"]]


def forecast(history, cutoff, end, recipe, out):
    shape = raw_frame(SHAPE/f"raw_{cutoff}.csv", cutoff, end)
    if recipe == "control":
        return shape
    strength, leaves = recipe.split("_")
    folder = out/f"fits_{leaves}"/cutoff
    path = folder/f"raw_{cutoff}.csv"
    if path.exists():
        learned = raw_frame(path, cutoff, end)
    else:
        past, future = examples(history, cutoff, end)
        factors = fit(past, future, cutoff, int(leaves), folder)
        base = raw_frame(VOLUME/cutoff/f"raw_{cutoff}.csv", cutoff, end).merge(factors,
            on=["route","date"], validate="many_to_one")
        base["prediction"] *= base.factor
        learned = transplant(base.drop(columns="factor"), shape, shape)
        learned.to_csv(path, sep=";", index=False, date_format="%Y-%m-%d")
    return mix(learned, shape, 0.5 if strength=="half" else 1.)


def run(args):
    if os.environ.get("SLURM_JOB_PARTITION") != "ais-cpu":
        raise RuntimeError("Expected ais-cpu")
    cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
    if not 1<=cpus<=4 or len(os.sched_getaffinity(0))>cpus:
        raise RuntimeError("Unexpected allocation")
    if datetime.now(timezone.utc)>=datetime.fromisoformat("2026-09-27T15:40:40+00:00"):
        raise RuntimeError("Research reserve reached")
    out = Path(args.output)/("pilot" if args.pilot else "study")
    out.mkdir(parents=True, exist_ok=True)
    paths = [Path(args.history), Path(__file__), Path("pipeline.py")]
    paths += [Path("experiments")/f"portfolio_{n}.py" for n in
        ["timesfm_errors","timesfm","bayes_volume","combine","experiment","windows"]]
    paths += list(SOURCE.glob("*/daily.csv"))+list(TEACHERS.glob("*/*.csv"))+list(TEACHERS.glob("*/*.json"))
    paths += list(SHAPE.glob("raw_*.csv"))+list(VOLUME.glob("*/raw_*.csv"))
    write_json(out/"run_started.json", dict(job_id=os.environ["SLURM_JOB_ID"], command=os.sys.argv,
        versions=dict(python=platform.python_version(), **{n:importlib.metadata.version(n)
            for n in ["numpy","pandas","scikit-learn","optuna"]}),
        sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))
    history = load_history(args.history)
    started = time.monotonic()
    if args.pilot:
        forecast(history, *FINAL, "full_15", out)
    else:
        run_study(history, out, "nonlinear_errors", lambda c,e,p:forecast(history,c,e,p["recipe"],out),
            dict(sampler="grid", space={"recipe":["control","half_7","full_7","half_15","full_15"]}, trials=5))
    write_json(out/"completed.json", dict(job_id=os.environ["SLURM_JOB_ID"], elapsed_seconds=time.monotonic()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pilot", action="store_true")
    run(parser.parse_args())
