"""P51: absolute-loss corrections of completed forecast hourly-share errors."""
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

from experiments.portfolio_bayes_shape import HOURS, INNER, SOURCE, inputs, prepare
from experiments.portfolio_combine import mix, raw_frame, transplant
from experiments.portfolio_experiment import FINAL, load_history, run_study, write_json
from experiments.portfolio_ridge import add_calendar
from pipeline import KEYS


def targets(past, cutoff):
    past = prepare(past, cutoff)
    past["target"] = past.boardings/past.actual_day - past.prediction/past.base_day
    repeats = past.groupby(KEYS).prediction.transform("size")
    past["weight"] = past.actual_day/repeats
    past["weight"] /= past.weight.mean()
    return past


def features(frame):
    phase = 2*np.pi*(frame.date.dt.dayofyear.to_numpy()-1)/365
    origin = 2*np.pi*(frame.origin.dt.dayofyear.to_numpy()-1)/365
    result = np.column_stack([frame.route, frame.hour, frame.effective_weekday,
        np.sin(phase), np.cos(phase), np.sin(origin), np.cos(origin),
        (frame.date-frame.origin).dt.days/61, frame.prediction.div(frame.base_day.where(frame.base_day.gt(0))).fillna(0),
        np.log1p(frame.base_day/10000), frame.off.astype(int), frame.summer])
    if not np.isfinite(result).all():
        raise ValueError("Invalid hourly share error features")
    return result.astype(float)


def apply_delta(base, delta):
    active = base.route.ne(5) & base.hour.isin(HOURS)
    delta = np.asarray(delta, dtype=float)
    if len(delta) != int(active.sum()) or not np.isfinite(delta).all():
        raise ValueError("Invalid hourly share delta")
    volume = base.groupby(["route", "date"]).prediction.transform("sum")
    shape = base[KEYS].assign(prediction=0.)
    share = base.loc[active, "prediction"].div(volume.loc[active].where(volume.loc[active].gt(0))).fillna(0)
    shape.loc[active, "prediction"] = np.maximum(0, share + delta)
    result = transplant(base, shape, base)
    if not np.isfinite(result.prediction).all() or not result.loc[~active, "prediction"].eq(0).all():
        raise ValueError("Invalid corrected hourly shape")
    return result


def fit(past, base, cutoff, leaves, out):
    started = time.monotonic()
    past = targets(past, cutoff)
    future = add_calendar(base.loc[base.route.ne(5) & base.hour.isin(HOURS)].copy(), cutoff)
    future["origin"] = np.datetime64(cutoff)
    future["base_day"] = future.groupby(["route", "date"]).prediction.transform("sum")
    if not future.base_day.ge(0).all():
        raise ValueError("Invalid active forecast volume")
    model = HistGradientBoostingRegressor(loss="absolute_error", max_iter=100,
        learning_rate=0.05, max_leaf_nodes=leaves, min_samples_leaf=120,
        l2_regularization=1, categorical_features=[0, 2], early_stopping=False, random_state=42)
    model.fit(features(past), past.target, sample_weight=past.weight)
    delta = model.predict(features(future))
    out.mkdir(parents=True, exist_ok=True)
    past.to_csv(out/"training.csv", sep=";", index=False, date_format="%Y-%m-%d")
    future.assign(delta=delta).to_csv(out/"delta.csv", sep=";", index=False, date_format="%Y-%m-%d")
    payload = pickle.dumps(model, protocol=5)
    (out/"model.pkl").write_bytes(payload)
    np.testing.assert_array_equal(pickle.loads(payload).predict(features(future)), delta)
    write_json(out/"fit.json", dict(rows=len(past), features=12, leaves=leaves,
        earliest_target_date=str(past.date.min().date()), latest_target_date=str(past.date.max().date()),
        origin_count=past.origin.nunique(), iterations=int(model.n_iter_),
        model_sha256=hashlib.sha256(payload).hexdigest(), model_reload_exact=True,
        fit_seconds=time.monotonic()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        loss="absolute share error weighted by actual day volume / repeated targets; projected shape surrogate"))
    return apply_delta(base, delta)


def forecast(history, cutoff, end, recipe, out):
    base = raw_frame(SOURCE/f"raw_{cutoff}.csv", cutoff, end)
    if recipe == "control":
        return base
    strength, leaves = recipe.split("_")
    cache = out/f"corrected_{leaves}"/cutoff
    path = cache/f"raw_{cutoff}.csv"
    if path.exists():
        learned = raw_frame(path, cutoff, end)
    else:
        learned = fit(inputs(history, cutoff), base, cutoff, int(leaves), cache)
        learned.to_csv(path, sep=";", index=False, date_format="%Y-%m-%d")
    return mix(learned, base, 0.5 if strength == "half" else 1.)


def run(args):
    if os.environ.get("SLURM_JOB_PARTITION") != "ais-cpu":
        raise RuntimeError("Expected ais-cpu")
    cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
    if not 1 <= cpus <= 4 or len(os.sched_getaffinity(0)) > cpus:
        raise RuntimeError("Unexpected allocation")
    if datetime.now(timezone.utc) >= datetime.fromisoformat("2026-09-27T15:40:40+00:00"):
        raise RuntimeError("Research reserve reached")
    out = Path(args.output)/("pilot" if args.pilot else "study")
    out.mkdir(parents=True, exist_ok=True)
    paths = [Path(args.history), Path(__file__), Path("pipeline.py"), Path("experiments/calendar_experiment.py"),
        Path("artifacts/calendar_sources.json")]
    paths += [Path("experiments")/f"portfolio_{n}.py" for n in
        ["bayes_shape", "experiment", "combine", "operations", "movement", "ridge", "windows"]]
    paths += list(SOURCE.glob("raw_*.csv"))+list(INNER.glob("*/*.csv"))+list(INNER.glob("*/*.json"))
    write_json(out/"run_started.json", dict(job_id=os.environ["SLURM_JOB_ID"], command=os.sys.argv,
        versions=dict(python=platform.python_version(), **{n:importlib.metadata.version(n)
            for n in ["numpy", "pandas", "scikit-learn", "optuna"]}),
        sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))
    started = time.monotonic()
    history = load_history(args.history)
    if args.pilot:
        forecast(history, *FINAL, "full_31", out)
    else:
        spec = dict(sampler="grid", space={"recipe":["control", "half_15", "full_15", "half_31", "full_31"]}, trials=5)
        run_study(history, out, "absolute_shape", lambda c,e,p:forecast(history,c,e,p["recipe"],out), spec)
    write_json(out/"completed.json", dict(job_id=os.environ["SLURM_JOB_ID"], elapsed_seconds=time.monotonic()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pilot", action="store_true")
    run(parser.parse_args())
