"""P29: calendar/weather hourly shares; preserve fixed P28 daily volumes."""
import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from experiments.portfolio_combine import raw_frame
from experiments.portfolio_experiment import load_history, run_study, write_json
from experiments.portfolio_operations import normal_history
from experiments.portfolio_ridge import add_calendar
from pipeline import KEYS


def features(frame, weather_enabled):
    values = [frame.route, frame.hour, frame.effective_weekday, frame.off.astype(int), frame.summer,
        np.sin(2*np.pi*(frame.date.dt.dayofyear-1)/365), np.cos(2*np.pi*(frame.date.dt.dayofyear-1)/365)]
    if weather_enabled:
        values += [frame.temperature_2m_mean/10, np.log1p(frame.precipitation_sum), frame.daylight_duration/21600]
    result = np.column_stack(values).astype(float)
    if not np.isfinite(result).all():
        raise ValueError("Incomplete conditional hourly context")
    return result


def shape_forecast(history, cutoff, params, weather, base):
    train = normal_history(history, cutoff, extended=True)
    train = add_calendar(train.loc[train.route.ne(5) & ~train.hour.between(1, 4)].copy(), cutoff)
    total = train.groupby(["route", "date"]).boardings.transform("sum")
    typical = total.groupby([train.route, train.daytype]).transform("median")
    retained = total.gt(np.maximum(500, 0.35*typical))
    train = train.loc[retained].copy()
    train["total"] = total.loc[retained]
    train = train.merge(weather, on="date", validate="many_to_one")
    model = HistGradientBoostingRegressor(loss="absolute_error", learning_rate=0.05,
        max_iter=150, max_leaf_nodes=params["leaves"], min_samples_leaf=40,
        l2_regularization=10, categorical_features=[0, 2, 4], early_stopping=False, random_state=42)
    model.fit(features(train, params["weather"]), train.boardings.to_numpy()/train.total.to_numpy(),
        sample_weight=train.total.to_numpy()/train.total.mean())
    future = add_calendar(base[KEYS].copy(), cutoff).merge(weather, on="date", validate="many_to_one")
    selected = future.route.ne(5) & ~future.hour.between(1, 4)
    future["share"] = 0.
    future.loc[selected, "share"] = np.maximum(0, model.predict(features(future.loc[selected], params["weather"])))
    denominator = future.groupby(["route", "date"]).share.transform("sum")
    if not denominator[selected].gt(0).all():
        raise ValueError("No positive learned share for a route-day")
    future["share"] = future.share/denominator.where(denominator.gt(0))
    future.loc[future.route.eq(5), "share"] = 0.
    volume = base.groupby(["route", "date"]).prediction.transform("sum")
    result = base[KEYS].copy()
    learned = future.share.to_numpy()*volume.to_numpy()
    result["prediction"] = params["mix"]*learned+(1-params["mix"])*base.prediction.to_numpy()
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
    source = Path("artifacts/portfolio_20260926/continuation/direct_blend/direct_blend/selected")
    weather = pd.read_csv(args.weather, sep=";", parse_dates=["date"])
    history = load_history(args.history)
    paths = [Path(args.history), Path(args.weather), Path(__file__),
        Path("experiments/portfolio_operations.py"), Path("experiments/portfolio_movement.py"),
        Path("experiments/portfolio_ridge.py"), Path("experiments/portfolio_experiment.py"),
        Path("experiments/portfolio_combine.py")] + list(source.glob("raw_*.csv"))
    write_json(out / "run_started.json", dict(job_id=os.environ["SLURM_JOB_ID"], command=os.sys.argv,
        daily_volume_source=str(source), sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))
    def forecast(cutoff, end, params):
        base = raw_frame(source / f"raw_{cutoff}.csv", cutoff, end)
        return shape_forecast(history, cutoff, params, weather, base)
    run_study(history, out, "conditional_shape", forecast,
        dict(sampler="grid", space={"leaves": [31, 63], "weather": [False, True], "mix": [0.5, 1.]}, trials=8))
    write_json(out / "completed.json", dict(job_id=os.environ["SLURM_JOB_ID"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", required=True)
    parser.add_argument("--weather", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
