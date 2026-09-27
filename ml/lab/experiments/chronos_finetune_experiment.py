"""One fixed fine-tuning configuration versus daily zero-shot Chronos-2."""
import argparse
import gc
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

from experiments.chronos_experiment import REVISION, WINDOWS, evaluate, forecast, inputs
from pipeline import KEYS, full_grid

FIT = dict(prediction_length=62, finetune_mode="full", context_length=512,
           learning_rate=1e-6, num_steps=200, batch_size=8, min_past=56,
           seed=42, data_seed=42, bf16=False, tf32=False, optim="adamw_torch",
           logging_steps=50, disable_tqdm=True, report_to="none")
LORA_FIT = dict(FIT, finetune_mode="lora", learning_rate=1e-5,
    lora_config=dict(r=8, lora_alpha=16, target_modules=["self_attention.q", "self_attention.v",
        "self_attention.k", "self_attention.o", "output_patch_embedding.output_layer"]))


def fit_before_cutoff(base, history, cutoff, output, lora=False):
    _, series = inputs(history, cutoff, "daily_total")
    assert len(series) >= FIT["min_past"] + FIT["prediction_length"]
    matrix = series.to_numpy(dtype=np.float32).T
    return base.fit(inputs=[row.copy() for row in matrix], output_dir=output, **(LORA_FIT if lora else FIT))


def run(args):
    import torch
    from chronos import Chronos2Pipeline

    if args.lora:
        import peft  # Fail explicitly rather than let Chronos silently fall back to full fine-tuning.

    started = time.monotonic()
    cpu_count = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    assert 1 <= cpu_count <= 4 and len(os.sched_getaffinity(0)) <= cpu_count
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.set_num_threads(cpu_count)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    history = pd.read_csv(args.history, sep=";", parse_dates=["date"])
    expected = full_grid().to_frame(index=False)
    expected["date"] = pd.to_datetime(expected.date)
    pd.testing.assert_frame_equal(history[KEYS].sort_values(KEYS).reset_index(drop=True), expected)
    assert np.isfinite(history.boardings).all() and history.boardings.ge(0).all()
    info = dict(model="amazon/chronos-2", revision=REVISION, fit=LORA_FIT if args.lora else FIT, windows=WINDOWS,
        selection="fixed 200-step final checkpoint; validation windows never passed to fit",
        features="nine daily totals; no covariates; unchanged historical weekday/hour shares",
        inference=dict(quantile=0.5, context_length=512, batch_size=32, cross_learning=False, dtype="float32"),
        history_sha256=hashlib.sha256(Path(args.history).read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        adapter_sha256=hashlib.sha256(Path("experiments/chronos_experiment.py").read_bytes()).hexdigest(),
        gpu=torch.cuda.get_device_name(0), cpu_affinity=sorted(os.sched_getaffinity(0)),
        job_id=os.environ.get("SLURM_JOB_ID"), partition=os.environ.get("SLURM_JOB_PARTITION"),
        python=platform.python_version(),
        versions={p: importlib.metadata.version(p) for p in
            ["numpy", "pandas", "torch", "chronos-forecasting", "transformers", "accelerate"]},
        missing_policy="absent counts = 0; validation unchanged",
        postprocessing="floor(max(0,p)+0.5); route5 and hours1-4 forced0",
        pretrained_availability="modern weights; input history restricted by historical cutoff",
        status="running")
    if args.lora:
        info["versions"]["peft"] = peft.__version__
    (out / "run.json").write_text(json.dumps(info, indent=2))
    rows, details, predictions = [], [], []
    for cutoff, end in WINDOWS:
        # Each window starts with the same pretrained weights, never the previous fine-tune.
        base = Chronos2Pipeline.from_pretrained(args.model_path, device_map="cuda", torch_dtype=torch.float32)
        for method in ["chronos_daily_zero_shot", "chronos_daily_lora" if args.lora else "chronos_daily_finetuned"]:
            torch.manual_seed(42)
            np.random.seed(42)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.monotonic()
            adapted = method != "chronos_daily_zero_shot"
            model = fit_before_cutoff(base, history, cutoff, out / cutoff, args.lora) if adapted else base
            torch.cuda.synchronize()
            fit_seconds = time.monotonic() - t0 if adapted else 0
            trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad) if adapted else 0
            total = sum(p.numel() for p in model.model.parameters())
            if args.lora and adapted:
                assert hasattr(model.model, "peft_config") and 0 < trainable < total / 10
            model.model.eval()

            def predictor(matrix, horizon):
                with torch.inference_mode():
                    quantiles, _ = model.predict_quantiles(
                        [torch.from_numpy(row.copy()) for row in matrix], prediction_length=horizon,
                        quantile_levels=[0.5], batch_size=32, context_length=512, cross_learning=False)
                return np.stack([q.cpu().numpy().reshape(horizon) for q in quantiles])

            result = forecast(history, cutoff, end, "daily_total", predictor)
            torch.cuda.synchronize()
            row, detail, prediction = evaluate(history, result, cutoff, end, method, time.monotonic() - t0)
            row.update(fit_seconds=fit_seconds, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                       trainable_parameters=trainable, total_parameters=total)
            rows.append(row)
            details.extend(detail)
            predictions.append(prediction)
            pd.DataFrame(rows).to_csv(out / "metrics.csv", sep=";", index=False)
            pd.DataFrame(details).to_csv(out / "breakdown.csv", sep=";", index=False)
            pd.concat(predictions, ignore_index=True).to_csv(out / "predictions.csv", sep=";", index=False)
            print(json.dumps(row), flush=True)
            del model
        del base
        gc.collect()
        torch.cuda.empty_cache()
    info.update(status="complete", seconds=time.monotonic() - started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    (out / "run.json").write_text(json.dumps(info, indent=2))
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", default="artifacts/results")
    parser.add_argument("--lora", action="store_true", help="Use fixed rank-8 LoRA instead of full fine-tuning")
    run(parser.parse_args())
