"""Small ensemble grid and cutoff-safe route selection using saved component forecasts."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.portfolio_experiment import (DEVELOPMENT, WINDOWS, FINAL, forecast_keys,
    load_history, save_candidate, write_json)
from experiments.portfolio_ridge import ridge_forecast
from pipeline import KEYS, metrics, postprocess

PAIRS = {"ridge_chronos": ("ridge", "chronos"), "chronos_hour": ("chronos", "hour"),
         "ridge_adaptive": ("ridge", "adaptive"), "ridge_median": ("ridge", "median")}
WEIGHTS = [0.25, 0.5, 0.75]
INNER = {"2025-04-30": ("2025-02-28", "2025-04-30"),
         "2025-06-30": ("2025-04-30", "2025-06-30"),
         "2025-07-31": ("2025-05-31", "2025-07-31"),
         "2025-08-31": ("2025-06-30", "2025-08-30"),
         "2025-10-31": ("2025-08-31", "2025-10-31")}


def raw_frame(path, cutoff, end):
    frame = pd.read_csv(path, sep=";", parse_dates=["date"], float_precision="round_trip")
    pd.testing.assert_frame_equal(frame[KEYS], forecast_keys(cutoff, end))
    if not np.isfinite(frame.prediction).all():
        raise ValueError("Invalid saved raw forecast")
    return frame.assign(prediction=frame.prediction.clip(lower=0))


def mix(a, b, weight):
    pd.testing.assert_frame_equal(a[KEYS], b[KEYS])
    result = a[KEYS].copy()
    result["prediction"] = weight * a.prediction.to_numpy() + (1 - weight) * b.prediction.to_numpy()
    return result


def transplant(volume, shape, fallback):
    pd.testing.assert_frame_equal(volume[KEYS], shape[KEYS])
    pd.testing.assert_frame_equal(volume[KEYS], fallback[KEYS])
    day = ["route", "date"]
    total = volume.groupby(day).prediction.transform("sum")
    denominator = shape.groupby(day).prediction.transform("sum")
    backup_total = fallback.groupby(day).prediction.transform("sum")
    share = shape.prediction.div(denominator.where(denominator.gt(0)))
    backup = fallback.prediction.div(backup_total.where(backup_total.gt(0))).fillna(0)
    result = volume[KEYS].copy()
    result["prediction"] = total * share.fillna(backup)
    np.testing.assert_allclose(result.groupby(day).prediction.sum(), volume.groupby(day).prediction.sum(), rtol=1e-6)
    return result


def route_weights(data, cutoff, a, b):
    if a.date.max() > pd.Timestamp(cutoff) or b.date.max() > pd.Timestamp(cutoff):
        raise ValueError("Route coefficients cannot use a future validation window")
    truth = data.loc[data.date.between(a.date.min(), a.date.max()), KEYS + ["boardings"]]
    joined = a.merge(b, on=KEYS, validate="one_to_one", suffixes=("_a", "_b"))
    joined = joined.merge(truth, on=KEYS, validate="one_to_one")
    if len(joined) != len(a):
        raise ValueError("Incomplete inner window")
    errors = {}
    for weight in [0.0, 0.5, 1.0]:
        raw = joined[KEYS].assign(prediction=weight * joined.prediction_a + (1 - weight) * joined.prediction_b)
        error = np.abs(postprocess(raw).prediction.to_numpy() - joined.boardings.to_numpy())
        errors[weight] = pd.Series(error).groupby(joined.route.reset_index(drop=True)).sum()
    table = pd.DataFrame(errors)
    global_weight = float(table.sum().idxmin())
    best = table.idxmin(axis=1)
    # ponytail: 50% shrinkage, just three choices on one completed window; richer routing needs more years.
    weights = 0.5 * best + 0.5 * global_weight
    weights.loc[5] = global_weight
    return weights.to_dict(), global_weight


def run(args):
    import optuna

    if os.environ.get("SLURM_JOB_PARTITION") != "ais-cpu":
        raise RuntimeError("Run ensemble research in ais-cpu")
    root = Path(args.root)
    out = root / "combine"
    out.mkdir(parents=True, exist_ok=True)
    data = load_history(args.history)
    sources = {"ridge": root / "ridge_probe/ridge/selected", "chronos": root / "gpu/daily",
               "hour": root / "gpu/route_hour", "adaptive": root / "cpu/adaptive/selected",
               "median": root / "cpu/median_all", "mean": root / "cpu/mean_all",
               "movement": root / "movement"}

    def component(name, cutoff, end):
        return raw_frame(sources[name] / f"raw_{cutoff}.csv", cutoff, end)

    spec = dict(space={"pair": list(PAIRS), "weight": WEIGHTS}, sampler="GridSampler", seed=42,
                trials=12, timeout=1800, objective="mean global WAPE-score on both development windows",
                windows=DEVELOPMENT, pruning=False, components=PAIRS,
                fixed_hybrids=["ridge_volume_chronos_shape", "chronos_volume_ridge_shape", "median_three", "movement_half_chronos"],
                route_rule="three weights, one completed earlier 61-day window, 50% shrinkage", inner=INNER)
    path = out / "study_spec.json"
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(spec)):
        raise ValueError("Existing ensemble specification cannot be changed")
    write_json(path, spec)
    write_json(out / "run_started.json", dict(job_id=os.environ["SLURM_JOB_ID"], command=os.sys.argv,
        versions={p: importlib.metadata.version(p) for p in ["numpy", "pandas", "optuna", "scikit-learn"]},
        sha256={str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                [args.history, __file__, "experiments/portfolio_experiment.py", "experiments/portfolio_ridge.py"]}))
    study = optuna.create_study(storage=f"sqlite:///{out / 'study.db'}", study_name="small_ensembles",
        direction="maximize", load_if_exists=True, sampler=optuna.samplers.GridSampler(spec["space"], seed=42))
    if study.user_attrs.get("started_at") is None:
        study.set_user_attr("started_at", datetime.now(timezone.utc).timestamp())
    for t in study.trials:
        if t.state == optuna.trial.TrialState.RUNNING:
            study.tell(t.number, state=optuna.trial.TrialState.FAIL)

    def objective(trial):
        pair = trial.suggest_categorical("pair", list(PAIRS))
        weight = trial.suggest_categorical("weight", WEIGHTS)
        a, b = PAIRS[pair]
        rows = save_candidate(data, out / f"trial_{trial.number:03d}", pair,
            lambda cutoff, end: mix(component(a, cutoff, end), component(b, cutoff, end), weight),
            DEVELOPMENT, final=False)
        trial.set_user_attr("scores", [r["wape_score"] for r in rows])
        return float(np.mean([r["wape_score"] for r in rows]))

    remaining = max(0, spec["trials"] - len(study.trials))
    seconds = max(0, spec["timeout"] - (datetime.now(timezone.utc).timestamp() - study.user_attrs["started_at"]))
    if remaining and seconds:
        study.optimize(objective, n_trials=remaining, timeout=seconds,
            callbacks=[lambda s, t: s.trials_dataframe().to_csv(out / "trials.csv", sep=";", index=False)])
    study.trials_dataframe().to_csv(out / "trials.csv", sep=";", index=False)
    for pair, (a, b) in PAIRS.items():
        complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.params["pair"] == pair]
        best = max(complete, key=lambda t: t.value)
        candidate = out / pair
        candidate.mkdir(exist_ok=True)
        write_json(candidate / "parameters.json", dict(**best.params, trial=best.number,
            development_score=best.value, selection=DEVELOPMENT))
        save_candidate(data, candidate, pair,
            lambda cutoff, end, x=a, y=b, w=best.params["weight"]:
                mix(component(x, cutoff, end), component(y, cutoff, end), w))

    for name, volume, shape in [("ridge_volume_chronos_shape", "ridge", "chronos"),
                                ("chronos_volume_ridge_shape", "chronos", "ridge")]:
        save_candidate(data, out / name, name, lambda cutoff, end, v=volume, s=shape:
            transplant(component(v, cutoff, end), component(s, cutoff, end), component("mean", cutoff, end)))

    def median_three(cutoff, end):
        frames = [component(n, cutoff, end) for n in ["ridge", "chronos", "mean"]]
        result = frames[0][KEYS].copy()
        result["prediction"] = np.median(np.column_stack([f.prediction for f in frames]), axis=1)
        return result

    save_candidate(data, out / "median_three", "median_three", median_three)
    save_candidate(data, out / "movement_half_chronos", "movement_half_chronos",
        lambda cutoff, end: mix(component("movement", cutoff, end), component("chronos", cutoff, end), 0.5))

    params = json.loads((root / "ridge_probe/ridge/selection.json").read_text())["params"]
    routing = {}

    def specialized(cutoff, end):
        inner_cutoff, inner_end = INNER[cutoff]
        source = sources["chronos"] if inner_cutoff in [w[0] for w in WINDOWS] else root / "inner_gpu/daily"
        chronos = raw_frame(source / f"raw_{inner_cutoff}.csv", inner_cutoff, inner_end)
        path = out / "inner_ridge" / f"raw_{inner_cutoff}.csv"
        if not path.exists():
            path.parent.mkdir(exist_ok=True)
            ridge_forecast(data, inner_cutoff, inner_end, params).to_csv(path, sep=";", index=False)
        ridge = raw_frame(path, inner_cutoff, inner_end)
        weights, global_weight = route_weights(data, cutoff, ridge, chronos)
        routing[cutoff] = dict(inner_cutoff=inner_cutoff, inner_end=inner_end, weights=weights, global_weight=global_weight)
        a, b = component("ridge", cutoff, end), component("chronos", cutoff, end)
        result = mix(a, b, a.route.map(weights).to_numpy())
        return result

    save_candidate(data, out / "route_specialist", "route_specialist", specialized)
    write_json(out / "routing.json", routing)
    write_json(out / "completed.json", dict(job_id=os.environ["SLURM_JOB_ID"], trials=len(study.trials),
        best_development_params=study.best_params, best_development_value=study.best_value))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--history", required=True)
    p.add_argument("--root", required=True)
    run(p.parse_args())
