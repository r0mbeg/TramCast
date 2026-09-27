"""P27: bounded Chronos adaptation; each cutoff starts from the same pretrained weights."""
import argparse
import gc
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from experiments.chronos_experiment import REVISION, inputs
from experiments.portfolio_chronos_weather import tasks_with_weather
from experiments.portfolio_chronos import gpu_forecast
from experiments.portfolio_experiment import load_history, write_json, run_study, save_candidate


def task_inputs(history, cutoff, end, representation, weather):
    _, series = inputs(history.loc[history.date.le(cutoff)], cutoff, "daily_total")
    future = pd.date_range(pd.Timestamp(cutoff) + pd.Timedelta(days=1), end)
    tasks = [row.copy() for row in series.to_numpy(np.float32).T]
    if representation == "daily":
        return tasks, tasks
    if representation != "daily_weather":
        raise ValueError(representation)
    predict = tasks_with_weather(tasks, series.index, future, cutoff, weather)
    # fit constructs training futures by slicing the available chronological history.
    fit = [{**t, "future_covariates": {c: None for c in t["future_covariates"]}} for t in predict]
    return fit, predict


def study(data, weather, root, base):
    import torch
    from chronos import Chronos2Pipeline

    spec = dict(sampler="grid", space={"steps": [200, 800], "learning_rate": [1e-6, 1e-5]}, trials=4)
    write_json(root / "training_spec.json", dict(**spec, representations=["daily", "daily_weather"],
        context_length=512, prediction_length=61, min_past=28, batch_size=8, dtype="float32",
        quantile=0.5, finetune_mode="full", seeds=[42, 73], total_gpu_allocation_limit_seconds=5400,
        per_study_timeout=1800, adaptation="fresh copy of pretrained model at each cutoff",
        regime="retrospective exogenous ERA5; current weights; targets only through cutoff"))

    def forecast(cutoff, end, representation, params, seed=42):
        destination = root / "models" / f"{representation}_{params['steps']}_{params['learning_rate']}_seed{seed}" / cutoff
        destination.mkdir(parents=True, exist_ok=True)
        fit, predict = task_inputs(data, cutoff, end, representation, weather)
        expected = dict(cutoff=cutoff, representation=representation, params=params, seed=seed)
        marker = destination / "completed.json"
        model = None
        start = time.monotonic()
        try:
            if marker.exists():
                recorded = json.loads(marker.read_text())
                if recorded["fit"] != expected:
                    raise ValueError("Existing fit configuration differs")
                for relative, digest in recorded["checkpoints"].items():
                    if hashlib.sha256((destination / relative).read_bytes()).hexdigest() != digest:
                        raise ValueError("Changed fine-tuned checkpoint")
                model = Chronos2Pipeline.from_pretrained(destination / "finetuned-ckpt", device_map="cuda", torch_dtype=torch.float32)
            else:
                write_json(destination / "started.json", expected)
                model = base.fit(fit, prediction_length=61, context_length=512,
                    learning_rate=params["learning_rate"], num_steps=params["steps"], batch_size=8,
                    min_past=28, output_dir=destination, seed=seed, data_seed=seed,
                    disable_tqdm=True, report_to="none", logging_steps=100)
                checkpoints = {str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (destination / "finetuned-ckpt").rglob("*") if p.is_file()}
                write_json(marker, dict(fit=expected, seconds=time.monotonic()-start,
                    checkpoints=checkpoints, last_target_date=cutoff))
            model.model.eval()
            def predictor(tasks, horizon, cross):
                if horizon != 61:
                    raise ValueError("Unexpected adaptation horizon")
                with torch.inference_mode():
                    q, _ = model.predict_quantiles(predict, prediction_length=horizon,
                        quantile_levels=[0.5], context_length=512, batch_size=8, cross_learning=False)
                return np.stack([v.cpu().numpy().reshape(horizon) for v in q])
            return gpu_forecast(data, cutoff, end, "daily", predictor)
        except RuntimeError as error:
            write_json(destination / "error.json", dict(error=repr(error), fit=expected))
            raise ValueError(str(error)) from error
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()

    for representation in ["daily", "daily_weather"]:
        family = f"finetune_{representation}"
        run_study(data, root, family,
            lambda c, e, p: forecast(c, e, representation, p), specification=spec)
        selected = json.loads((root / family / "selection.json").read_text())["params"]
        save_candidate(data, root / f"{family}_seed73", f"{family}_seed73",
            lambda c, e: forecast(c, e, representation, selected, seed=73), final=False)


def run(args):
    import torch
    from chronos import Chronos2Pipeline

    if os.environ.get("SLURM_JOB_PARTITION") != "gpu_devel" or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one GPU in gpu_devel")
    cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
    if not 1 <= cpus <= 4 or len(os.sched_getaffinity(0)) > cpus:
        raise RuntimeError("Expected at most four allocated CPU cores")
    if datetime.now(timezone.utc) >= datetime.fromisoformat("2026-09-27T15:40:40+00:00"):
        raise RuntimeError("Research reserve reached")
    torch.set_num_threads(cpus)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    data = load_history(args.history)
    weather = pd.read_csv(args.weather, sep=";", parse_dates=["date"])
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    if not args.pilot:
        base = Chronos2Pipeline.from_pretrained(args.model_path, device_map="cuda", torch_dtype=torch.float32)
        write_json(root / "study_started.json", dict(job_id=os.environ["SLURM_JOB_ID"], command=os.sys.argv,
            gpu=torch.cuda.get_device_name(), revision=REVISION,
            versions={p: importlib.metadata.version(p) for p in ["chronos-forecasting", "torch", "transformers", "accelerate", "optuna"]},
            sha256={str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                [args.history, args.weather, __file__, "experiments/portfolio_experiment.py",
                 "experiments/portfolio_chronos.py", "experiments/portfolio_chronos_weather.py",
                 "experiments/chronos_experiment.py", "experiments/chronos_calendar_experiment.py"]}))
        started = time.monotonic()
        study(data, weather, root, base)
        write_json(root / "completed.json", dict(job_id=os.environ["SLURM_JOB_ID"],
            seconds=time.monotonic()-started, peak_gpu_bytes=torch.cuda.max_memory_allocated()))
        return
    destination = root / "pilot"
    if (destination / "completed.json").exists():
        raise ValueError("Pilot already completed; do not repeat")
    destination.mkdir(exist_ok=True)
    spec = dict(seed=42, steps=20, learning_rate=1e-6, min_past=28, context_length=512,
        batch_size=8, prediction_length=61, representation="daily", cutoff="2025-04-30",
        end="2025-06-30", revision=REVISION, objective="cost/memory/API check only",
        fit_targets="only dates<=cutoff; contiguous time order within each sample")
    write_json(destination / "started.json", dict(**spec, job_id=os.environ["SLURM_JOB_ID"],
        gpu=torch.cuda.get_device_name(), command=os.sys.argv,
        versions={p: importlib.metadata.version(p) for p in ["chronos-forecasting", "torch", "transformers", "accelerate"]},
        sha256={str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
            [args.history, args.weather, __file__, "experiments/portfolio_experiment.py",
             "experiments/portfolio_chronos_weather.py", "experiments/chronos_experiment.py"]}))
    base = Chronos2Pipeline.from_pretrained(args.model_path, device_map="cuda", torch_dtype=torch.float32)
    fit, predict = task_inputs(data, spec["cutoff"], spec["end"], "daily", weather)
    start = time.monotonic()
    model = base.fit(fit, prediction_length=61, context_length=512, learning_rate=1e-6,
        num_steps=20, batch_size=8, min_past=28, output_dir=destination,
        seed=42, data_seed=42, disable_tqdm=True, report_to="none", logging_steps=20)
    model.model.eval()
    with torch.inference_mode():
        q, _ = model.predict_quantiles(predict, prediction_length=61, quantile_levels=[0.5],
            context_length=512, batch_size=8, cross_learning=False)
    values = np.stack([v.cpu().numpy().reshape(61) for v in q])
    assert values.shape == (9, 61) and np.isfinite(values).all()
    checkpoints = {str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (destination / "finetuned-ckpt").rglob("*") if p.is_file()}
    result = dict(seconds=time.monotonic()-start, peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        forecast_shape=list(values.shape), checkpoints=checkpoints)
    write_json(destination / "completed.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", required=True)
    parser.add_argument("--weather", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pilot", action="store_true")
    run(parser.parse_args())
