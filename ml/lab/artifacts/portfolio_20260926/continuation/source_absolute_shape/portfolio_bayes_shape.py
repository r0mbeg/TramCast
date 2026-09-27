"""P47: Bayesian corrections of completed forecast hourly-share errors."""
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
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

from experiments.portfolio_bayes_volume import ROOT
from experiments.portfolio_combine import mix, raw_frame
from experiments.portfolio_experiment import FINAL, load_history, run_study, write_json
from experiments.portfolio_operations import normal_history
from experiments.portfolio_ridge import ACTIVE_ROUTES, add_calendar
from experiments.portfolio_windows import inner_windows
from pipeline import KEYS

HOURS = [0] + list(range(5, 24))
SOURCE = ROOT / "verified_july/verified_july/selected"
INNER = ROOT / "bayes_hourly_target/inner_shape"


def features(frame):
    phase = 2*np.pi*(frame.date.dt.dayofyear.to_numpy()-1)/365
    origin = 2*np.pi*(frame.origin.dt.dayofyear.to_numpy()-1)/365
    cycles = np.column_stack([np.sin(phase), np.cos(phase),
        np.sin(phase)-np.sin(origin), np.cos(phase)-np.cos(origin)])
    columns = []
    for hour in HOURS:
        at_hour = frame.hour.eq(hour).to_numpy()
        columns += [(at_hour & frame.route.eq(r).to_numpy()).astype(float) for r in ACTIVE_ROUTES]
        columns += [(at_hour & frame.effective_weekday.eq(d).to_numpy()).astype(float) for d in range(1, 7)]
        columns += list((cycles*at_hour[:, None]).T)
    result = np.column_stack(columns)
    if not np.isfinite(result).all():
        raise ValueError("Invalid hourly correction features")
    return result


def inputs(history, cutoff):
    truth = history.loc[history.date.le(cutoff), KEYS+["boardings"]]
    pieces = []
    for origin, end in inner_windows(cutoff):
        folder = INNER/origin
        path = folder/f"raw_{origin}.csv"
        info = json.loads((folder/"source.json").read_text())
        if info["origin"] != origin or info["end"] != end or info["fit_latest_date"] != origin:
            raise ValueError("Wrong causal shape cache period")
        if hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError("Changed causal shape cache")
        raw = raw_frame(path, origin, end)
        raw = raw.merge(truth, on=KEYS, validate="one_to_one")
        if len(raw) != 14640:
            raise ValueError("Incomplete completed hourly calibration window")
        pieces.append(raw.assign(origin=pd.Timestamp(origin)))
    if not pieces:
        raise ValueError("No completed hourly calibration forecasts")
    past = pd.concat(pieces, ignore_index=True)
    past = past.loc[past.date.gt(pd.Timestamp(cutoff)-pd.Timedelta(days=224))]
    return past


def prepare(past, cutoff):
    if past.empty or past.date.max() > pd.Timestamp(cutoff) or (past.origin >= past.date).any():
        raise ValueError("Invalid completed hourly correction targets")
    if not np.isfinite(past[["prediction", "boardings"]]).all().all() or past[["prediction", "boardings"]].lt(0).any().any():
        raise ValueError("Invalid hourly correction values")
    past = normal_history(past, cutoff, extended=True, august7=True, july_verified=True)
    past = add_calendar(past.loc[past.route.ne(5) & past.hour.isin(HOURS)].copy(), cutoff)
    day_keys = ["origin", "route", "date"]
    past["actual_day"] = past.groupby(day_keys).boardings.transform("sum")
    past["base_day"] = past.groupby(day_keys).prediction.transform("sum")
    days = past.drop_duplicates(["route", "date"])
    typical = days.groupby(["route", "daytype"]).actual_day.median()
    threshold = typical.reindex(pd.MultiIndex.from_frame(past[["route", "daytype"]])).to_numpy()
    keep = past.actual_day.gt(np.maximum(500, 0.35*threshold)) & past.base_day.gt(0)
    past = past.loc[keep].reset_index(drop=True)
    if past.empty:
        raise ValueError("No reliable hourly correction training days")
    target = np.log((past.boardings+1)/(past.actual_day+20)) - np.log((past.prediction+1)/(past.base_day+20))
    past["target"] = np.clip(target, -np.log(2), np.log(2))
    repeats = past.groupby(KEYS).prediction.transform("size")
    importance = past.prediction.clip(lower=1)
    past["weight"] = (1/repeats)*importance/np.average(importance, weights=1/repeats)
    return past


def apply_shape(base, factors):
    result = base.copy()
    active = base.route.ne(5) & base.hour.isin(HOURS)
    factors = np.asarray(factors, dtype=float)
    if len(factors) != int(active.sum()) or not np.isfinite(factors).all() or (factors <= 0).any():
        raise ValueError("Invalid hourly correction factors")
    result.loc[active, "prediction"] *= factors
    volume = base.groupby(["route", "date"]).prediction.transform("sum")
    denominator = result.groupby(["route", "date"]).prediction.transform("sum")
    result["prediction"] *= np.divide(volume.to_numpy(), denominator.to_numpy(),
        out=np.ones(len(base)), where=denominator.to_numpy()>0)
    np.testing.assert_allclose(result.groupby(["route", "date"]).prediction.sum(),
        base.groupby(["route", "date"]).prediction.sum(), rtol=1e-10, atol=1e-8)
    if not result.loc[~active, "prediction"].eq(0).all():
        raise ValueError("Hourly correction changed structural zeros")
    return result


def fit(past, base, cutoff, out):
    started = time.monotonic()
    past = prepare(past, cutoff)
    future = add_calendar(base.loc[base.route.ne(5) & base.hour.isin(HOURS), KEYS].copy(), cutoff)
    future["origin"] = pd.Timestamp(cutoff)
    x, future_x = features(past), features(future)
    scaler = StandardScaler().fit(x, sample_weight=past.weight)
    model = BayesianRidge(max_iter=300, tol=1e-5).fit(scaler.transform(x), past.target, sample_weight=past.weight)
    mean, sd = model.predict(scaler.transform(future_x), return_std=True)
    factors = np.exp(np.clip(mean, -np.log(2), np.log(2)))
    if not np.isfinite(sd).all():
        raise ValueError("Invalid hourly correction posterior")
    out.mkdir(parents=True, exist_ok=True)
    past.to_csv(out/"training.csv", sep=";", index=False, date_format="%Y-%m-%d")
    future.assign(mean=mean, sd=sd, factor=factors).to_csv(out/"factors.csv", sep=";", index=False, date_format="%Y-%m-%d")
    np.savez_compressed(out/"posterior.npz", coefficient=model.coef_, covariance=model.sigma_,
        feature_mean=scaler.mean_, feature_scale=scaler.scale_)
    write_json(out/"fit.json", dict(rows=len(past), features=x.shape[1], latest_target_date=str(past.date.max().date()),
        earliest_target_date=str(past.date.min().date()), origin_count=past.origin.nunique(),
        intercept=float(model.intercept_), noise_precision=float(model.alpha_), coefficient_precision=float(model.lambda_),
        fit_seconds=time.monotonic()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        iterations=int(model.n_iter_), uncertainty="conditional dependent hourly residual variance; not closed-score confidence"))
    return apply_shape(base, factors)


def forecast(history, cutoff, end, recipe, out):
    base = raw_frame(SOURCE/f"raw_{cutoff}.csv", cutoff, end)
    if recipe == "control":
        return base
    cache = out/"corrected_shape"/cutoff
    path = cache/f"raw_{cutoff}.csv"
    if path.exists():
        learned = raw_frame(path, cutoff, end)
    else:
        learned = fit(inputs(history, cutoff), base, cutoff, cache)
        learned.to_csv(path, sep=";", index=False, date_format="%Y-%m-%d")
    return mix(learned, base, 0.5 if recipe == "half" else 1.)


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
    paths += [Path("experiments")/f"portfolio_{n}.py" for n in ["experiment","combine","operations","movement","ridge","windows"]]
    paths += list(SOURCE.glob("raw_*.csv"))+list(INNER.glob("*/*.csv"))+list(INNER.glob("*/*.json"))
    write_json(out/"run_started.json", dict(job_id=os.environ["SLURM_JOB_ID"], command=os.sys.argv,
        versions=dict(python=platform.python_version(), **{n:importlib.metadata.version(n) for n in ["numpy","pandas","scikit-learn","optuna"]}),
        sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))
    started = time.monotonic()
    history = load_history(args.history)
    if args.pilot:
        forecast(history, *FINAL, "full", out)
    else:
        spec = dict(sampler="grid", space={"recipe":["control","half","full"]}, trials=3)
        run_study(history, out, "bayes_shape", lambda c,e,p:forecast(history,c,e,p["recipe"],out), spec)
    write_json(out/"completed.json", dict(job_id=os.environ["SLURM_JOB_ID"], elapsed_seconds=time.monotonic()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pilot", action="store_true")
    run(parser.parse_args())
