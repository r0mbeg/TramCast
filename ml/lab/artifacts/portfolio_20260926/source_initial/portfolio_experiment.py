"""Bounded CPU studies for P20260926; run only inside an ais-cpu allocation."""
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
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from experiments.calendar_experiment import calendar_predict
from experiments.chronos_experiment import distribute_daily, evaluate, inputs
from pipeline import KEYS, full_grid, metrics, postprocess, predict

WINDOWS = [("2025-04-30", "2025-06-30"), ("2025-06-30", "2025-08-30"),
           ("2025-07-31", "2025-09-30"), ("2025-08-31", "2025-10-31")]
DEVELOPMENT = WINDOWS[:2]
FINAL = ("2025-10-31", "2025-12-31")
HISTORY_SHA256 = "7031c686c149fcb552711df49d82bab23540d8c80ffa6f140c6d0444138384ae"
SPECS = {
    "seasonal": dict(sampler="grid", space={"weeks": [0, 4, 8, 12], "statistic": ["mean", "median"]}, trials=8),
    "adaptive": dict(sampler="tpe", space={"half_life": [14, 112], "trend": [0.0, 1.0], "damping": [14, 61]}, trials=20),
    "daily": dict(sampler="grid", space={"weeks": [0, 4, 8, 12], "statistic": ["mean", "median"]}, trials=8),
    "pooled": dict(sampler="grid", space={"days": [28, 56], "shared": [0.25, 0.5, 0.75]}, trials=6),
    "direct": dict(sampler="tpe", space={"days": [56, 112, 224], "half_life": [28, 112],
                                        "leaves": [7, 15, 31], "l2": [1.0, 10.0]}, trials=12),
}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def load_history(path):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != HISTORY_SHA256:
        raise ValueError("History differs from the frozen snapshot")
    data = pd.read_csv(path, sep=";", parse_dates=["date"])
    expected = full_grid().to_frame(index=False)
    expected["date"] = pd.to_datetime(expected.date)
    pd.testing.assert_frame_equal(data[KEYS].sort_values(KEYS).reset_index(drop=True), expected)
    if not np.isfinite(data.boardings).all() or data.boardings.lt(0).any():
        raise ValueError("Invalid history target")
    if not data.loc[data.route.eq(5) | data.hour.between(1, 4), "boardings"].eq(0).all():
        raise ValueError("History structural zeros differ from the protocol")
    return data


def forecast_keys(cutoff, end):
    result = full_grid(pd.Timestamp(cutoff) + pd.Timedelta(days=1), end).to_frame(index=False)
    result["date"] = pd.to_datetime(result.date)
    return result


def complete_raw(raw, cutoff, end):
    result = forecast_keys(cutoff, end).merge(raw[KEYS + ["prediction"]], on=KEYS,
                                            how="left", validate="one_to_one")
    forced = result.route.eq(5) | result.hour.between(1, 4)
    result.loc[forced, "prediction"] = 0.0
    if not np.isfinite(result.prediction).all():
        raise ValueError("Missing or invalid raw predictions")
    return result


def impute_history(data, cutoff):
    train = data.loc[data.date.le(cutoff)].copy()
    active = train.route.ne(5) & ~train.hour.between(1, 4)
    missing = active & ~train.working_events_observed
    observed = train.loc[active & train.working_events_observed].assign(weekday=lambda x: x.date.dt.dayofweek)
    profile = observed.groupby(["route", "weekday", "hour"]).boardings.median()
    lookup = pd.MultiIndex.from_arrays([train.route, train.date.dt.dayofweek, train.hour])
    estimates = profile.reindex(lookup).to_numpy()
    if not np.isfinite(estimates[missing]).all():
        raise ValueError("Cannot impute absent cells from the available history")
    train["boardings"] = train.boardings.astype(float)
    train.loc[missing, "boardings"] = estimates[missing]
    return train


def daily_profile(train, series, dates, weeks, statistic):
    if weeks:
        series = series.loc[series.index > series.index.max() - pd.Timedelta(days=7 * weeks)]
    profile = series.groupby(series.index.dayofweek).agg(statistic)
    return pd.DataFrame(profile.loc[dates.dayofweek].to_numpy(), index=dates, columns=series.columns)


def cpu_forecast(data, cutoff, end, family, params, seed=42):
    cutoff = pd.Timestamp(cutoff)
    train = data.loc[data.date.le(cutoff)].copy()
    keys = forecast_keys(cutoff, end)
    dates = pd.date_range(cutoff + pd.Timedelta(days=1), end)
    if family in ["seasonal", "calendar", "coverage"]:
        if family == "calendar":
            # The existing calendar implementation includes the canonical rounding.
            return calendar_predict(train, keys, cutoff).astype({"prediction": float})
        if family == "coverage":
            train = impute_history(train, cutoff)
            params = {"weeks": 0, "statistic": "mean"}
        if params["weeks"]:
            train = train.loc[train.date > cutoff - pd.Timedelta(days=7 * params["weeks"])]
        train["weekday"] = train.date.dt.dayofweek
        profile = train.groupby(["route", "weekday", "hour"]).boardings.agg(params["statistic"])
        raw = keys.assign(weekday=keys.date.dt.dayofweek).merge(
            profile.rename("prediction"), on=["route", "weekday", "hour"], validate="many_to_one")
    elif family == "direct":
        active = train.route.ne(5) & ~train.hour.between(1, 4)
        recent = train.loc[active & train.date.gt(cutoff - pd.Timedelta(days=params["days"]))].copy()
        scale = recent.groupby("route").boardings.mean().clip(lower=1)

        def features(frame):
            return pd.DataFrame(dict(route=frame.route, hour=frame.hour, weekday=frame.date.dt.dayofweek,
                                     day=(frame.date - pd.Timestamp("2025-01-01")).dt.days))

        model = HistGradientBoostingRegressor(loss="absolute_error", max_iter=150,
            learning_rate=0.08, max_leaf_nodes=params["leaves"], min_samples_leaf=40,
            l2_regularization=params["l2"], categorical_features=["route", "weekday"],
            early_stopping=False, random_state=seed)
        weight = np.exp2(-(cutoff - recent.date).dt.days.to_numpy() / params["half_life"])
        model.fit(features(recent), recent.boardings / recent.route.map(scale), sample_weight=weight)
        selected = keys.loc[keys.route.ne(5) & ~keys.hour.between(1, 4)].copy()
        selected["prediction"] = model.predict(features(selected)) * selected.route.map(scale)
        raw = selected
    else:
        operating, series = inputs(train, cutoff, "daily_total")
        if family == "daily":
            future = daily_profile(operating, series, dates, **params)
        else:
            weekday = series.groupby(series.index.dayofweek).median().clip(lower=1)
            seasonal = pd.DataFrame(weekday.loc[series.index.dayofweek].to_numpy(),
                                    index=series.index, columns=series.columns)
            ratio = series / seasonal
            if family == "adaptive":
                level = ratio.ewm(halflife=params["half_life"]).mean().iloc[-1].to_numpy()
                # ponytail: trend capped to +/-20% over a month; expand only with longer validation history.
                latest, previous = ratio.iloc[-28:].median(), ratio.iloc[-56:-28].median()
                slope = np.clip((latest - previous).to_numpy() / 28, -0.2 / 28, 0.2 / 28)
                horizon = np.arange(1, len(dates) + 1)
                offset = params["damping"] * (1 - np.exp(-horizon / params["damping"]))
                factors = level[None, :] + params["trend"] * offset[:, None] * slope[None, :]
            elif family == "pooled":
                latest = ratio.iloc[-params["days"]:].median().to_numpy()
                common = float(np.median(latest))
                factors = np.tile((1 - params["shared"]) * latest + params["shared"] * common,
                                  (len(dates), 1))
            else:
                raise ValueError(f"Unknown CPU family {family}")
            future = pd.DataFrame(weekday.loc[dates.dayofweek].to_numpy() * np.maximum(factors, 0),
                                  index=dates, columns=series.columns)
        daily = future.rename_axis(index="date", columns="route").stack().rename("prediction").reset_index()
        raw = distribute_daily(operating, daily)
    return complete_raw(raw, cutoff, end)


def save_candidate(data, out, method, predict_raw, windows=WINDOWS, final=True):
    out.mkdir(parents=True, exist_ok=True)
    rows, details, predictions = [], [], []
    for cutoff, end in windows + ([FINAL] if final else []):
        raw_path = out / f"raw_{cutoff}.csv"
        if raw_path.exists():
            raw = pd.read_csv(raw_path, sep=";", parse_dates=["date"])
            pd.testing.assert_frame_equal(raw[KEYS], forecast_keys(cutoff, end))
        else:
            raw = predict_raw(cutoff, end)
            raw.to_csv(raw_path, sep=";", index=False, date_format="%Y-%m-%d")
        rounded = postprocess(raw)
        if (cutoff, end) == FINAL:
            rounded.to_csv(out / "submission.csv", sep=";", index=False, date_format="%Y-%m-%d")
        else:
            row, detail, prediction = evaluate(data, rounded, cutoff, end, method, 0)
            rows.append(row)
            details.extend(detail)
            predictions.append(prediction)
            pd.DataFrame(rows).to_csv(out / "metrics.csv", sep=";", index=False)
            pd.DataFrame(details).to_csv(out / "breakdown.csv", sep=";", index=False)
            pd.concat(predictions, ignore_index=True).to_csv(out / "predictions.csv", sep=";", index=False)
    return rows


def run_study(data, root, family):
    import optuna

    spec = dict(SPECS[family], family=family, timeout=1800, seed=42,
                objective="mean WAPE-score of both development windows", windows=DEVELOPMENT,
                pruning=False, n_jobs=1)
    out = root / family
    out.mkdir(parents=True, exist_ok=True)
    path = out / "study_spec.json"
    if path.exists():
        if json.loads(path.read_text()) != json.loads(json.dumps(spec)):
            raise ValueError("Cannot change an existing study budget or space")
    else:
        write_json(path, spec)
    sampler = (optuna.samplers.GridSampler(spec["space"], seed=42) if spec["sampler"] == "grid"
               else optuna.samplers.TPESampler(seed=42))
    study = optuna.create_study(storage=f"sqlite:///{out / 'study.db'}", study_name=family,
                                direction="maximize", sampler=sampler, load_if_exists=True)
    if study.user_attrs.get("started_at") is None:
        study.set_user_attr("started_at", datetime.now(timezone.utc).timestamp())
    # A stopped allocation leaves RUNNING trials; count them as failed attempts, never reset the budget.
    for old in study.trials:
        if old.state == optuna.trial.TrialState.RUNNING:
            study.tell(old.number, state=optuna.trial.TrialState.FAIL)
    remaining_trials = max(0, spec["trials"] - len(study.trials))
    remaining_seconds = max(0, spec["timeout"] - (datetime.now(timezone.utc).timestamp() - study.user_attrs["started_at"]))

    def objective(trial):
        params = {}
        for name, values in spec["space"].items():
            if spec["sampler"] == "grid" or name in ["days", "leaves"]:
                params[name] = trial.suggest_categorical(name, values)
            elif name in ["half_life", "damping"]:
                params[name] = trial.suggest_int(name, *values)
            else:
                params[name] = trial.suggest_float(name, *values)
        trial_out = out / f"trial_{trial.number:03d}"
        trial_out.mkdir(exist_ok=True)
        write_json(trial_out / "parameters.json", params)
        rows = save_candidate(data, trial_out, f"{family}_trial{trial.number}",
            lambda cutoff, end: cpu_forecast(data, cutoff, end, family, params), DEVELOPMENT, final=False)
        value = float(np.mean([r["wape_score"] for r in rows]))
        trial.set_user_attr("window_scores", [r["wape_score"] for r in rows])
        return value

    def checkpoint(study, trial):
        study.trials_dataframe().to_csv(out / "trials.csv", sep=";", index=False)

    if remaining_trials and remaining_seconds:
        study.optimize(objective, n_trials=remaining_trials, timeout=remaining_seconds,
                       n_jobs=1, callbacks=[checkpoint], catch=(ValueError,), gc_after_trial=True)
    study.trials_dataframe().to_csv(out / "trials.csv", sep=";", index=False)
    best = study.best_trial
    write_json(out / "selection.json", dict(trial=best.number, value=best.value, params=best.params,
                                            selection_windows=DEVELOPMENT, total_trials=len(study.trials)))
    finalist = out / "selected"
    # Reuse the selected trial's completed development forecasts rather than recomputing them.
    finalist.mkdir(exist_ok=True)
    for cutoff, _ in DEVELOPMENT:
        source = out / f"trial_{best.number:03d}" / f"raw_{cutoff}.csv"
        target = finalist / source.name
        if not target.exists():
            target.write_bytes(source.read_bytes())
    rows = save_candidate(data, finalist, family,
        lambda cutoff, end: cpu_forecast(data, cutoff, end, family, best.params))
    print(family, best.params, [r["wape_score"] for r in rows], flush=True)


def run(args):
    if os.environ.get("SLURM_JOB_PARTITION") != "ais-cpu":
        raise RuntimeError("CPU research must run in ais-cpu")
    cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
    if not 1 <= cpus <= 4 or len(os.sched_getaffinity(0)) > cpus:
        raise RuntimeError("Expected at most four allocated CPU cores")
    if datetime.now(timezone.utc).timestamp() >= datetime.fromisoformat("2026-09-27T15:40:40+00:00").timestamp():
        raise RuntimeError("Research reserve reached; save and verify instead of starting studies")
    start = time.monotonic()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    data = load_history(args.history)
    write_json(root / "run_started.json", dict(job_id=os.environ["SLURM_JOB_ID"],
        partition=os.environ["SLURM_JOB_PARTITION"], node=platform.node(), affinity=sorted(os.sched_getaffinity(0)),
        command=os.sys.argv, history_sha256=HISTORY_SHA256,
        code_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                     [Path(__file__), Path("pipeline.py"), Path("experiments/chronos_experiment.py")]},
        versions={p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "optuna"]}))
    with threadpool_limits(limits=cpus):
        for family in ["seasonal", "adaptive", "daily", "pooled", "direct"]:
            run_study(data, root, family)
        for name, family, params in [("mean_all", "seasonal", dict(weeks=0, statistic="mean")),
                                     ("median_all", "seasonal", dict(weeks=0, statistic="median")),
                                     ("calendar", "calendar", {}), ("coverage", "coverage", {})]:
            rows = save_candidate(data, root / name, name,
                lambda cutoff, end, f=family, p=params: cpu_forecast(data, cutoff, end, f, p))
            print(name, [r["wape_score"] for r in rows], flush=True)
    write_json(root / "completed.json", dict(job_id=os.environ["SLURM_JOB_ID"], seconds=time.monotonic()-start,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
