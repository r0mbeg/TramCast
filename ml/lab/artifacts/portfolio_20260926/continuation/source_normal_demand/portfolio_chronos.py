"""Fixed representation/coverage/calendar probes; one GPU, checkpoint each cutoff."""
import argparse
import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import resource
import time

import numpy as np
import pandas as pd

from experiments.chronos_experiment import REVISION, inputs, distribute_daily
from experiments.chronos_calendar_experiment import COVARIATES, calendar_for, calendar_distribute
from experiments.portfolio_experiment import (WINDOWS, FINAL, complete_raw, impute_history,
    load_history, save_candidate, write_json)
from pipeline import KEYS

METHODS = ["daily", "route_hour", "calendar", "day_shares", "calendar_shares",
           "recent_shares", "normalized_shared", "coverage", "hourly"]
INNER_WINDOWS = [("2025-02-28", "2025-04-30"), ("2025-05-31", "2025-07-31")]


def gpu_forecast(history, cutoff, end, method, predictor):
    train = history.loc[history.date.le(cutoff)].copy()
    if method == "coverage":
        train = impute_history(train, cutoff)
    operating, series = inputs(train, cutoff, "route_hour" if method == "route_hour" else "daily_total")
    dates = pd.date_range(pd.Timestamp(cutoff) + pd.Timedelta(days=1), end)
    matrix = series.to_numpy(dtype=np.float32).T
    tasks = [row.copy() for row in matrix]
    if method in ["calendar", "calendar_shares"]:
        past, future = calendar_for(series.index, cutoff), calendar_for(dates, cutoff)
        tasks = [dict(target=row.copy(),
                      past_covariates={c: past[c].to_numpy(dtype=np.float32) for c in COVARIATES},
                      future_covariates={c: future[c].to_numpy(dtype=np.float32) for c in COVARIATES})
                 for row in matrix]
    if method == "normalized_shared":
        weekday = series.groupby(series.index.dayofweek).median().clip(lower=1)
        denominator = weekday.loc[series.index.dayofweek].to_numpy(dtype=np.float32).T
        tasks = [row.copy() for row in matrix / denominator]
    if method == "hourly":
        hourly = train.loc[train.route.ne(5)].pivot(index=["date", "hour"], columns="route", values="boardings")
        tasks = [row.copy() for row in hourly.to_numpy(dtype=np.float32).T]
    horizon = len(dates) * 24 if method == "hourly" else len(dates)
    values = predictor(tasks, horizon, method == "normalized_shared")
    if values.shape != (len(tasks), horizon) or not np.isfinite(values).all():
        raise ValueError("Invalid Chronos probe output")
    if method == "normalized_shared":
        values = values * weekday.loc[dates.dayofweek].to_numpy(dtype=np.float32).T
    if method == "route_hour":
        raw = pd.concat([pd.DataFrame(dict(route=r, hour=h, date=dates, prediction=v))
                         for (r, h), v in zip(series.columns, values)], ignore_index=True)
    elif method == "hourly":
        raw = pd.concat([pd.DataFrame(dict(route=r, date=np.repeat(dates, 24),
                                          hour=np.tile(np.arange(24), len(dates)), prediction=v))
                         for r, v in zip(series.columns, values)], ignore_index=True)
    else:
        daily = pd.concat([pd.DataFrame(dict(route=int(r), date=dates, prediction=v))
                           for r, v in zip(series.columns, values)], ignore_index=True)
        if method in ["day_shares", "calendar_shares"]:
            raw = calendar_distribute(operating, daily, cutoff)
        else:
            shape = operating
            if method == "recent_shares":
                shape = shape.loc[shape.date > pd.Timestamp(cutoff) - pd.Timedelta(days=56)]
            raw = distribute_daily(shape, daily)
    return complete_raw(raw, cutoff, end)


def run(args):
    import torch
    from chronos import Chronos2Pipeline

    if os.environ.get("SLURM_JOB_PARTITION") != "gpu_devel":
        raise RuntimeError("Probe requires gpu_devel")
    cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
    if not 1 <= cpus <= 4 or len(os.sched_getaffinity(0)) > cpus:
        raise RuntimeError("Expected at most four allocated CPU cores")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible GPU")
    started = time.monotonic()
    torch.set_num_threads(cpus)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    data = load_history(args.history)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    methods = ["daily"] if args.nested else METHODS
    windows = INNER_WINDOWS if args.nested else WINDOWS
    spec = dict(methods=methods, windows=windows, final=None if args.nested else FINAL, quantile=0.5, context_length=512,
                batch_size=32, seed=42, dtype="float32", revision=REVISION,
                cross_learning="True only for normalized_shared; False otherwise",
                family_timeout_seconds=1800, training_within_horizon=False)
    write_json(root / "probe_spec.json", spec)
    model = Chronos2Pipeline.from_pretrained(args.model_path, device_map="cuda", torch_dtype=torch.float32)
    model.model.eval()
    write_json(root / "run_started.json", dict(spec=spec, job_id=os.environ["SLURM_JOB_ID"],
        partition=os.environ["SLURM_JOB_PARTITION"], node=platform.node(), gpu=torch.cuda.get_device_name(),
        gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        affinity=sorted(os.sched_getaffinity(0)), command=os.sys.argv,
        pretrained_availability="modern weights, not deployable at historical 2025 cutoff",
        sha256={str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                [args.history, __file__, "experiments/portfolio_experiment.py", "pipeline.py",
                 "experiments/chronos_experiment.py", "experiments/chronos_calendar_experiment.py"]},
        versions={p: importlib.metadata.version(p) for p in ["torch", "chronos-forecasting", "numpy", "pandas"]}))

    def predictor(tasks, horizon, cross):
        with torch.inference_mode():
            q, _ = model.predict_quantiles(tasks, prediction_length=horizon, quantile_levels=[0.5],
                batch_size=32, context_length=512, cross_learning=cross)
        return np.stack([v.cpu().numpy().reshape(horizon) for v in q])

    for method in methods:
        t0 = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        try:
            rows = save_candidate(data, root / method, f"chronos_{method}",
                lambda cutoff, end: gpu_forecast(data, cutoff, end, method, predictor), windows, final=not args.nested)
            write_json(root / method / "completed.json", dict(seconds=time.monotonic()-t0,
                scores=[r["wape_score"] for r in rows], peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated()))
            print(method, [r["wape_score"] for r in rows], flush=True)
        except (ValueError, RuntimeError) as error:
            write_json(root / f"{method}_error.json", dict(error=repr(error), seconds=time.monotonic()-t0))
            print(method, repr(error), flush=True)
    write_json(root / "completed.json", dict(job_id=os.environ["SLURM_JOB_ID"], seconds=time.monotonic()-started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--nested", action="store_true")
    run(parser.parse_args())
