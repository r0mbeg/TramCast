"""P29: calendar/weather hourly shares; preserve fixed P28 daily volumes."""
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
from sklearn.ensemble import HistGradientBoostingRegressor

from experiments.portfolio_combine import raw_frame, mix
from experiments.portfolio_experiment import load_history, run_study, write_json
from experiments.portfolio_movement import disrupted
from experiments.portfolio_operations import normal_history
from experiments.portfolio_ridge import add_calendar
from pipeline import KEYS


def operation_features(frame):
    july = disrupted(frame, "july_verified")
    shortened = (disrupted(frame, "april17") | disrupted(frame, "august7") |
        ((july | disrupted(frame, "autumn")) & frame.route.eq(7)))
    return np.column_stack([shortened, july & frame.route.eq(50)]).astype(float)


def features(frame, weather_enabled, operations=False):
    values = [frame.route, frame.hour, frame.effective_weekday, frame.off.astype(int), frame.summer,
        np.sin(2*np.pi*(frame.date.dt.dayofyear-1)/365), np.cos(2*np.pi*(frame.date.dt.dayofyear-1)/365)]
    if weather_enabled:
        values += [frame.temperature_2m_mean/10, np.log1p(frame.precipitation_sum), frame.daylight_duration/21600]
    if operations:
        values += list(operation_features(frame).T)
    result = np.column_stack(values).astype(float)
    if not np.isfinite(result).all():
        raise ValueError("Incomplete conditional hourly context")
    return result


def shape_forecast(history, cutoff, params, weather, base):
    operations = params.get("operations", False)
    train = history.loc[history.date.le(cutoff)].copy() if operations else normal_history(history, cutoff, extended=True)
    if operations:
        train = train.loc[~(disrupted(train, "autumn") & train.route.eq(50))].copy()
    train = add_calendar(train.loc[train.route.ne(5) & ~train.hour.between(1, 4)].copy(), cutoff)
    total = train.groupby(["route", "date"]).boardings.transform("sum")
    typical = total.groupby([train.route, train.daytype]).transform("median")
    retained = total.gt(np.maximum(500, 0.35*typical))
    if operations:
        normal = normal_history(train, cutoff, extended=True, august7=True, july_verified=True)
        typical = total.loc[normal.index].groupby([train.route.loc[normal.index], train.daytype.loc[normal.index]]).median()
        lookup = pd.MultiIndex.from_frame(train[["route", "daytype"]])
        threshold = pd.Series(typical.reindex(lookup).to_numpy(), index=train.index)
        if threshold.isna().any():
            raise ValueError("Missing normal-day volume for regime shape")
        changed = operation_features(train).any(axis=1)
        retained = total.gt(500) & (changed | total.gt(0.35*threshold))
    train = train.loc[retained].copy()
    train["total"] = total.loc[retained]
    train = train.merge(weather, on="date", validate="many_to_one")
    model = HistGradientBoostingRegressor(loss="absolute_error", learning_rate=0.05,
        max_iter=150, max_leaf_nodes=params["leaves"], min_samples_leaf=40,
        l2_regularization=10, categorical_features=[0, 2, 4], early_stopping=False, random_state=42)
    model.fit(features(train, params["weather"], operations), train.boardings.to_numpy()/train.total.to_numpy(),
        sample_weight=train.total.to_numpy()/train.total.mean())
    future = add_calendar(base[KEYS].copy(), cutoff).merge(weather, on="date", validate="many_to_one")
    selected = future.route.ne(5) & ~future.hour.between(1, 4)
    future["share"] = 0.
    future.loc[selected, "share"] = np.maximum(0, model.predict(features(future.loc[selected], params["weather"], operations)))
    denominator = future.groupby(["route", "date"]).share.transform("sum")
    if not denominator[selected].gt(0).all():
        raise ValueError("No positive learned share for a route-day")
    if params.get("normalize", True):
        future["share"] = future.share/denominator.where(denominator.gt(0))
    future.loc[future.route.eq(5), "share"] = 0.
    volume = base.groupby(["route", "date"]).prediction.transform("sum")
    result = base[KEYS].copy()
    learned = future.share.to_numpy()*volume.to_numpy()
    result["prediction"] = params["mix"]*learned+(1-params["mix"])*base.prediction.to_numpy()
    if params.get("normalize", True):
        np.testing.assert_allclose(result.groupby(["route", "date"]).prediction.sum(),
            base.groupby(["route", "date"]).prediction.sum(), rtol=1e-10, atol=1e-8)
    if not np.isfinite(result.prediction).all():
        raise ValueError("Invalid conditional shape")
    return result


def run(args):
    if os.environ.get("SLURM_JOB_PARTITION") != "ais-cpu":
        raise RuntimeError("Expected ais-cpu")
    cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
    if not 1 <= cpus <= 4 or len(os.sched_getaffinity(0)) > cpus:
        raise RuntimeError("Unexpected allocation")
    if datetime.now(timezone.utc) >= datetime.fromisoformat("2026-09-27T15:40:40+00:00"):
        raise RuntimeError("Research reserve reached")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    source = Path("artifacts/portfolio_20260926/continuation") / ("verified_july/verified_july/selected"
        if args.operations else "direct_blend/direct_blend/selected")
    weather = pd.read_csv(args.weather, sep=";", parse_dates=["date"])
    history = load_history(args.history)
    paths = [Path(args.history), Path(args.weather), Path(__file__),
        Path("experiments/portfolio_operations.py"), Path("experiments/portfolio_movement.py"),
        Path("experiments/portfolio_ridge.py"), Path("experiments/portfolio_experiment.py"),
        Path("experiments/portfolio_combine.py")] + list(source.glob("raw_*.csv"))
    write_json(out / "run_started.json", dict(job_id=os.environ["SLURM_JOB_ID"], command=os.sys.argv,
        daily_volume_source=str(source),
        versions=dict(python=platform.python_version(), **{n:importlib.metadata.version(n) for n in ["numpy","pandas","scikit-learn","optuna"]}),
        sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))
    def forecast(cutoff, end, params):
        base = raw_frame(source / f"raw_{cutoff}.csv", cutoff, end)
        if args.operations:
            if params["recipe"] == "control":
                return base
            cache = out / "learned_shape"
            cache.mkdir(exist_ok=True)
            path = cache / f"raw_{cutoff}.csv"
            if path.exists():
                learned = raw_frame(path, cutoff, end)
            else:
                learned = shape_forecast(history, cutoff,
                    dict(leaves=63, weather=True, mix=1., operations=True), weather, base)
                learned.to_csv(path, sep=";", index=False, date_format="%Y-%m-%d")
            return mix(learned, base, 0.5 if params["recipe"] == "half" else 1.)
        return shape_forecast(history, cutoff, params, weather, base)
    family = "operations_shape" if args.operations else "marginal_shape" if args.marginal else "conditional_shape"
    specification = (dict(sampler="grid", space={"leaves": [63], "weather": [True],
        "mix": [0.5, 1.], "normalize": [False, True]}, trials=4) if args.marginal else
        dict(sampler="grid", space={"leaves": [31, 63], "weather": [False, True], "mix": [0.5, 1.]}, trials=8))
    if args.operations:
        specification = dict(sampler="grid", space={"recipe":["control","half","full"]}, trials=3)
    run_study(history, out, family, forecast, specification)
    write_json(out / "completed.json", dict(job_id=os.environ["SLURM_JOB_ID"],
        elapsed_seconds=time.monotonic()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", required=True)
    parser.add_argument("--weather", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--marginal", action="store_true")
    parser.add_argument("--operations", action="store_true")
    args=parser.parse_args()
    if args.operations and args.marginal:
        parser.error("--operations and --marginal are separate studies")
    run(args)
