"""P44: nonlinear Bayesian bias on completed, weekly aggregated forecast errors."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import resource
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

from experiments.portfolio_bayes_direct import annual_features
from experiments.portfolio_bayes_volume import ROOT
from experiments.portfolio_combine import raw_frame, transplant
from experiments.portfolio_experiment import FINAL, forecast_keys, load_history, run_study, write_json
from experiments.portfolio_windows import inner_windows
from pipeline import KEYS


def weekly(frame, cutoff, observed):
    if observed and frame.date.max() > pd.Timestamp(cutoff):
        raise ValueError("Future target entered GP calibration")
    frame = frame.loc[frame.route.ne(5) & frame.base.gt(0)].copy()
    if frame.empty:
        raise ValueError("No GP calibration samples")
    counts = frame.groupby(["route", "date"]).base.transform("size")
    frame["weight"] = 1 / counts
    frame["origin"] = frame.date - pd.to_timedelta(frame.horizon, unit="D")
    frame["block"] = (frame.horizon - 1) // 7
    x = annual_features(frame, route_specific=True)
    columns = [f"x{i}" for i in range(x.shape[1])]
    table = pd.concat([frame[["route", "origin", "block", "weight"]],
        pd.DataFrame(x * frame.weight.to_numpy()[:, None], index=frame.index, columns=columns)], axis=1)
    if observed:
        if not np.isfinite(frame.boardings).all() or frame.boardings.lt(0).any():
            raise ValueError("Invalid GP target")
        table["target"] = np.clip(np.log((frame.boardings + 100) / (frame.base + 100)),
                                  -np.log(2), np.log(2)) * frame.weight
    group = table.groupby(["route", "origin", "block"], sort=True).sum()
    features = group[columns].div(group.weight, axis=0).to_numpy()
    target = (group.target / group.weight).to_numpy() if observed else None
    return group.reset_index()[["route", "origin", "block"]], features, target, group.weight.to_numpy()


def inputs(history, cutoff, end):
    truth = history.loc[history.date.le(cutoff)].groupby(["route", "date"], as_index=False).boardings.sum()
    past = []
    for origin, stop in inner_windows(cutoff):
        frame = daily_table(origin, stop)
        past.append(frame.merge(truth, on=["route", "date"], validate="one_to_one"))
    if not past:
        raise ValueError("No completed GP inner forecasts")
    past = pd.concat(past, ignore_index=True)
    past = past.loc[past.date.gt(pd.Timestamp(cutoff) - pd.Timedelta(days=224))]
    return past, daily_table(cutoff, end)


def daily_table(origin, end):
    frame = pd.read_csv(ROOT / "verified_july/inner" / origin / "daily.csv", sep=";",
                        parse_dates=["date"], float_precision="round_trip")
    expected = forecast_keys(origin, end)[["route", "date"]].drop_duplicates().reset_index(drop=True)
    pd.testing.assert_frame_equal(frame[["route", "date"]], expected)
    if "boardings" in frame or not frame.horizon.eq((frame.date - pd.Timestamp(origin)).dt.days).all():
        raise ValueError("GP input contains targets or invalid horizon")
    return frame


def fit_gp(past, future, cutoff, kind):
    started = time.monotonic()
    train_keys, x, target, weight = weekly(past, cutoff, True)
    future_keys, future_x, _, _ = weekly(future, cutoff, False)
    scaler = StandardScaler().fit(x, sample_weight=weight)
    smooth = RBF(5., (0.5, 20.)) if kind == "rbf" else Matern(5., (0.5, 20.), nu={"matern15":1.5,"matern25":2.5}[kind])
    kernel = ConstantKernel(0.04, (1e-4, 1.)) * smooth + WhiteKernel(0.01, (1e-4, 0.25))
    # ponytail: exact dense GP on weekly samples; use approximation only if more history makes it too costly.
    model = GaussianProcessRegressor(kernel=kernel, alpha=0.0025 / weight, normalize_y=False,
                                    n_restarts_optimizer=0, random_state=42)
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter("always")
        model.fit(scaler.transform(x), target)
    messages = [str(w.message) for w in notices]
    if any("lbfgs failed" in m.lower() for m in messages):
        raise ValueError("GP likelihood optimizer did not converge")
    mean, sd = model.predict(scaler.transform(future_x), return_std=True)
    if not np.isfinite(mean).all() or not np.isfinite(sd).all():
        raise ValueError("Invalid GP posterior")
    return (future_keys.assign(mean=mean, sd=sd), dict(kernel=str(model.kernel_),
        kernel_theta=model.kernel_.theta.tolist(), log_marginal_likelihood=float(model.log_marginal_likelihood_value_),
        train_rows=len(x), train_weights=weight.tolist(), target_mean=float(np.average(target, weights=weight)),
        latest_target_date=str(past.date.max().date()), feature_mean=scaler.mean_.tolist(),
        feature_scale=scaler.scale_.tolist(), fit_seconds=time.monotonic()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, warnings=messages,
        uncertainty="conditional GP variance on dependent weekly samples; not hidden-score confidence"),
        dict(train_keys=train_keys, x=x, target=target, weight=weight, future_x=future_x))


def forecast(history, cutoff, end, params, out):
    cache = out / "posteriors"
    cache.mkdir(parents=True, exist_ok=True)
    name = f"{cutoff}_{params['kernel']}"
    path = cache / f"{name}.csv"
    past, future = inputs(history, cutoff, end)
    if path.exists():
        posterior = pd.read_csv(path, sep=";", parse_dates=["origin"], float_precision="round_trip")
    else:
        posterior, summary, data = fit_gp(past, future, cutoff, params["kernel"])
        posterior.to_csv(path, sep=";", index=False, date_format="%Y-%m-%d")
        write_json(cache / f"{name}.json", summary)
        np.savez_compressed(cache / f"{name}.npz", x=data["x"], target=data["target"],
                            weight=data["weight"], future_x=data["future_x"])
        data["train_keys"].to_csv(cache / f"{name}_train_keys.csv", sep=";", index=False, date_format="%Y-%m-%d")
    posterior["factor"] = np.exp(np.clip(posterior["mean"] /
        (1 + params["uncertainty"] * np.square(posterior.sd / 0.1)), -np.log(2), np.log(2)))
    future["origin"] = future.date - pd.to_timedelta(future.horizon, unit="D")
    future["block"] = (future.horizon - 1) // 7
    future = future.merge(posterior[["route", "origin", "block", "factor"]],
                          on=["route", "origin", "block"], how="left", validate="many_to_one")
    active = future.route.ne(5) & future.base.gt(0)
    if future.loc[active, "factor"].isna().any():
        raise ValueError("Missing GP future factor")
    future.loc[~active, "factor"] = 1.
    future[["route", "date", "factor"]].to_csv(cache / f"factors_{name}_{params['uncertainty']}.csv",
        sep=";", index=False, date_format="%Y-%m-%d")
    raw = raw_frame(ROOT / "bayes_volume_verified_july/inner" / cutoff / f"raw_{cutoff}.csv", cutoff, end)
    raw = raw.merge(future[["route", "date", "factor"]], on=["route", "date"], validate="many_to_one")
    raw["prediction"] *= raw.factor
    raw = raw.drop(columns="factor")
    shape = raw_frame(ROOT / "conditional_shape/conditional_shape/selected" / f"raw_{cutoff}.csv", cutoff, end)
    return transplant(raw, shape, raw)


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
    paths = [Path(args.history), Path(__file__)] + [Path("experiments") / f"portfolio_{n}.py"
        for n in ["experiment", "bayes_direct", "bayes_volume", "combine", "windows", "ridge"]]
    for folder, pattern in [("verified_july/inner", "*/daily.csv"),
        ("bayes_volume_verified_july/inner", "*/raw_*.csv"),
        ("conditional_shape/conditional_shape/selected", "raw_*.csv")]:
        paths += list((ROOT / folder).glob(pattern))
    started = time.monotonic()
    write_json(out / ("pilot_started.json" if args.pilot else "run_started.json"),
        dict(job_id=os.environ["SLURM_JOB_ID"], command=os.sys.argv,
        versions=dict(python=platform.python_version(), **{n:importlib.metadata.version(n) for n in ["numpy","pandas","scikit-learn","optuna"]}),
        sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))
    history = load_history(args.history)
    if args.pilot:
        folder = out / "pilot"
        folder.mkdir(exist_ok=True)
        raw = forecast(history, *FINAL, dict(kernel="rbf", uncertainty=1), folder)
        raw.to_csv(folder / "pilot_raw.csv", sep=";", index=False, date_format="%Y-%m-%d")
    else:
        run_study(history, out, "gp_errors", lambda c,e,p:forecast(history,c,e,p,out),
            dict(sampler="grid", space={"kernel":["rbf","matern15","matern25"],"uncertainty":[0,1]},trials=6))
    write_json(out / ("pilot_completed.json" if args.pilot else "completed.json"),
        dict(job_id=os.environ["SLURM_JOB_ID"], elapsed_seconds=time.monotonic()-started,
             peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pilot", action="store_true")
    run(parser.parse_args())
