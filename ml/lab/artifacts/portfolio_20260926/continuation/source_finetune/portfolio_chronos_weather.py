"""Bounded probabilistic Chronos study with retrospective weather, on the frozen windows."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from experiments.chronos_experiment import REVISION, inputs
from experiments.chronos_calendar_experiment import COVARIATES, calendar_for
from experiments.portfolio_chronos import gpu_forecast
from experiments.portfolio_experiment import DEVELOPMENT, load_history, save_candidate, write_json

REPRESENTATIONS = ["daily", "daily_weather", "hour_weather"]
QUANTILES = [0.25, 0.5, 0.75]


def tasks_with_weather(tasks, past_dates, future_dates, cutoff, weather):
    past, future = calendar_for(past_dates, cutoff), calendar_for(future_dates, cutoff)
    indexed = weather.set_index("date")
    fields = ["temperature_2m_mean", "precipitation_sum", "daylight_duration"]
    a = {c: past[c].to_numpy(dtype=np.float32) for c in COVARIATES}
    b = {c: future[c].to_numpy(dtype=np.float32) for c in COVARIATES}
    for field in fields:
        x = indexed.loc[past_dates, field].to_numpy(dtype=np.float32)
        y = indexed.loc[future_dates, field].to_numpy(dtype=np.float32)
        if field == "precipitation_sum":
            x, y = np.log1p(x), np.log1p(y)
        else:
            x, y = x / (10 if field == "temperature_2m_mean" else 3600), y / (10 if field == "temperature_2m_mean" else 3600)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Incomplete weather covariate")
        a[field], b[field] = x, y
    return [dict(target=row.copy(), past_covariates=a.copy(), future_covariates=b.copy()) for row in tasks]


def run(args):
    import optuna
    import torch
    from chronos import Chronos2Pipeline

    if os.environ.get("SLURM_JOB_PARTITION") != "gpu_devel" or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected one GPU in gpu_devel")
    if datetime.now(timezone.utc) >= datetime.fromisoformat("2026-09-27T15:40:40+00:00"):
        raise RuntimeError("Research reserve reached")
    cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
    if not 1 <= cpus <= 4 or len(os.sched_getaffinity(0)) > cpus:
        raise RuntimeError("Expected at most four CPU cores")
    start = time.monotonic()
    torch.set_num_threads(cpus)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    data = load_history(args.history)
    weather = pd.read_csv(args.weather, sep=";", parse_dates=["date"])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    spec = dict(space={"representation": REPRESENTATIONS, "quantile": QUANTILES}, sampler="GridSampler",
        seed=42, trials=9, timeout=600, objective="mean WAPE-score on both development windows",
        windows=DEVELOPMENT, revision=REVISION, dtype="float32", context_length=512,
        batch_size="8 hour_weather; 32 otherwise", cross_learning=False,
        regime="retrospective ERA5 weather; no future targets", pruning=False)
    path = out / "study_spec.json"
    import json
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(spec)):
        raise ValueError("Existing study budget differs")
    write_json(path, spec)
    model = Chronos2Pipeline.from_pretrained(args.model_path, device_map="cuda", torch_dtype=torch.float32)
    model.model.eval()
    write_json(out / "run_started.json", dict(job_id=os.environ["SLURM_JOB_ID"], command=os.sys.argv,
        gpu=torch.cuda.get_device_name(), gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        versions={p: importlib.metadata.version(p) for p in ["torch", "chronos-forecasting", "optuna", "numpy", "pandas"]},
        sha256={str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
            [args.history, args.weather, __file__, "experiments/portfolio_chronos.py", "experiments/portfolio_experiment.py"]}))

    def forecast(cutoff, end, representation, quantile):
        kind = "route_hour" if representation == "hour_weather" else "daily"
        _, series = inputs(data.loc[data.date.le(cutoff)], cutoff,
                           "route_hour" if kind == "route_hour" else "daily_total")
        future = pd.date_range(pd.Timestamp(cutoff) + pd.Timedelta(days=1), end)
        def predictor(tasks, horizon, cross):
            if representation != "daily":
                tasks = tasks_with_weather(tasks, series.index, future, cutoff, weather)
            with torch.inference_mode():
                values, _ = model.predict_quantiles(tasks, prediction_length=horizon, quantile_levels=[quantile],
                    batch_size=8 if representation == "hour_weather" else 32,
                    context_length=512, cross_learning=False)
            return np.stack([v.cpu().numpy().reshape(horizon) for v in values])
        return gpu_forecast(data, cutoff, end, kind, predictor)

    study = optuna.create_study(storage=f"sqlite:///{out / 'study.db'}", study_name="weather_quantiles",
        sampler=optuna.samplers.GridSampler(spec["space"], seed=42), direction="maximize", load_if_exists=True)
    if "started_at" not in study.user_attrs:
        study.set_user_attr("started_at", datetime.now(timezone.utc).timestamp())
    for old in study.trials:
        if old.state == optuna.trial.TrialState.RUNNING:
            study.tell(old.number, state=optuna.trial.TrialState.FAIL)
    def objective(trial):
        representation = trial.suggest_categorical("representation", REPRESENTATIONS)
        q = trial.suggest_categorical("quantile", QUANTILES)
        directory = out / f"trial_{trial.number:03d}"
        directory.mkdir(exist_ok=True)
        write_json(directory / "parameters.json", trial.params)
        rows = save_candidate(data, directory, f"{representation}_q{q}",
            lambda c, e: forecast(c, e, representation, q), DEVELOPMENT, final=False)
        trial.set_user_attr("window_scores", [r["wape_score"] for r in rows])
        return float(np.mean([r["wape_score"] for r in rows]))
    remaining = max(0, spec["trials"] - len(study.trials))
    seconds = max(0, spec["timeout"] - (datetime.now(timezone.utc).timestamp() - study.user_attrs["started_at"]))
    if remaining and seconds:
        study.optimize(objective, n_trials=remaining, timeout=seconds,
            callbacks=[lambda s, t: s.trials_dataframe().to_csv(out / "trials.csv", sep=";", index=False)],
            catch=(ValueError, RuntimeError))
    results = {}
    for representation in REPRESENTATIONS:
        trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.params["representation"] == representation]
        if not trials:
            write_json(out / f"{representation}_error.json", dict(error="no complete trial"))
            continue
        best = max(trials, key=lambda t: t.value)
        target = out / representation
        target.mkdir(exist_ok=True)
        write_json(target / "selection.json", dict(params=best.params, value=best.value, trial=best.number))
        for cutoff, _ in DEVELOPMENT:
            source = out / f"trial_{best.number:03d}" / f"raw_{cutoff}.csv"
            dest = target / source.name
            if not dest.exists():
                dest.write_bytes(source.read_bytes())
        rows = save_candidate(data, target, representation,
            lambda c, e, r=representation, q=best.params["quantile"]: forecast(c, e, r, q))
        results[representation] = [r["wape_score"] for r in rows]
        print(representation, results[representation], flush=True)
    write_json(out / "completed.json", dict(job_id=os.environ["SLURM_JOB_ID"], seconds=time.monotonic()-start,
        scores=results, trials=len(study.trials), peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", required=True)
    parser.add_argument("--weather", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
