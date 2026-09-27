"""Calendar ablation: daily Chronos-2 covariates x hourly day-type shares."""
import argparse
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

from experiments.calendar_experiment import MIN_DAYS, SOURCES, calendar_table
from experiments.chronos_experiment import REVISION, WINDOWS, distribute_daily, evaluate, inputs
from pipeline import KEYS, full_grid, metrics, postprocess

COVARIATES = ["weekday_sin", "weekday_cos", "is_holiday", "is_workday"]
DIAGNOSTIC_WINDOW = ("2025-04-30", "2025-06-30")


def calendar_for(dates, cutoff):
    calendar = calendar_table().set_index("date")
    if calendar.known_at.gt(cutoff).any():
        raise ValueError("Calendar not available at cutoff")
    result = calendar.reindex(pd.DatetimeIndex(dates))
    if result.isna().any().any():
        raise ValueError("Calendar covers 2025 only")
    result["weekday_sin"] = np.sin(2 * np.pi * result.weekday / 7)
    result["weekday_cos"] = np.cos(2 * np.pi * result.weekday / 7)
    result["is_holiday"] = result.day_type.eq("holiday")
    return result


def model_inputs(history, cutoff, end, covariates):
    train, series = inputs(history, cutoff, "daily_total")
    dates = pd.date_range(pd.Timestamp(cutoff) + pd.Timedelta(days=1), end)
    matrix = series.to_numpy(dtype=np.float32).T
    if covariates:
        past = calendar_for(series.index, cutoff)
        future = calendar_for(dates, cutoff)
        tasks = [dict(target=row.copy(),
                      past_covariates={c: past[c].to_numpy(dtype=np.float32) for c in COVARIATES},
                      future_covariates={c: future[c].to_numpy(dtype=np.float32) for c in COVARIATES})
                 for row in matrix]
    else:
        tasks = [row.copy() for row in matrix]
    return train, series.columns, dates, tasks


def calendar_distribute(train, daily, cutoff):
    train = train.loc[train.date.le(cutoff), KEYS + ["boardings"]].copy()
    cal = calendar_for(pd.concat([train.date, daily.date]).drop_duplicates(), cutoff)
    train = train.merge(cal[["day_type", "is_workday"]], left_on="date", right_index=True, validate="many_to_one")
    group = ["route", "day_type", "hour"]
    profile = train.groupby(group).boardings.agg(volume="sum", days="count").reset_index()
    total = profile.groupby(["route", "day_type"]).volume.transform("sum")
    profile["share"] = (profile.volume / total.where(total.gt(0))).where(profile.days.ge(MIN_DAYS))
    broad = train.groupby(["route", "is_workday", "hour"]).boardings.sum().rename("volume").reset_index()
    denominator = broad.groupby(["route", "is_workday"]).volume.transform("sum")
    broad["fallback"] = broad.volume / denominator.where(denominator.gt(0))
    # Reuse the original expansion, then replace only its within-day proportions.
    result = distribute_daily(train, daily).drop(columns="prediction")
    result = result.merge(cal[["day_type", "is_workday"]], left_on="date", right_index=True, validate="many_to_one")
    result = result.merge(profile[group + ["share"]], on=group, how="left", validate="many_to_one")
    result = result.merge(broad[["route", "is_workday", "hour", "fallback"]],
                          on=["route", "is_workday", "hour"], how="left", validate="many_to_one")
    result["share"] = result.share.fillna(result.fallback)
    if result.share.isna().any():
        raise ValueError("No valid day-type or workday profile")
    np.testing.assert_allclose(result.groupby(["route", "date"]).share.sum(), 1)
    result = result.merge(daily, on=["route", "date"], validate="many_to_one")
    result["prediction"] = result.prediction.clip(lower=0) * result.share
    return result[KEYS + ["prediction"]]


def hourly_forecast(train, daily, cutoff, use_day_type):
    raw = calendar_distribute(train, daily, cutoff) if use_day_type else distribute_daily(train, daily)
    expected = daily.assign(prediction=daily.prediction.clip(lower=0)).set_index(["route", "date"]).prediction.sort_index()
    np.testing.assert_allclose(raw.groupby(["route", "date"]).prediction.sum().sort_index(), expected, rtol=1e-6)
    grid = full_grid(daily.date.min(), daily.date.max()).to_frame(index=False)
    grid["date"] = pd.to_datetime(grid.date)
    result = grid.merge(raw, on=KEYS, how="left", validate="one_to_one")
    forced = result.route.eq(5) | result.hour.between(1, 4)
    if result.loc[~forced, "prediction"].isna().any():
        raise ValueError("Missing hourly forecast")
    result.loc[forced, "prediction"] = 0
    return postprocess(result)


def run(args):
    import torch
    from chronos import Chronos2Pipeline

    started = time.monotonic()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) > 8 or int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) != 8:
        raise RuntimeError("Expected 8 allocated CPUs with affinity of at most 8")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name():
        raise RuntimeError("Expected exactly one A100")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    data = pd.read_csv(args.history, sep=";", parse_dates=["date"])
    expected = full_grid().to_frame(index=False)
    expected["date"] = pd.to_datetime(expected.date)
    pd.testing.assert_frame_equal(data[KEYS].sort_values(KEYS).reset_index(drop=True), expected)
    if not np.isfinite(data.boardings).all() or data.boardings.lt(0).any():
        raise ValueError("Invalid target")
    reference = pd.read_csv(args.reference, sep=";", parse_dates=["date"])
    t0 = time.monotonic()
    model = Chronos2Pipeline.from_pretrained(args.model_path, device_map="cuda", torch_dtype=torch.float32)
    model.model.eval()
    torch.cuda.synchronize()
    load_seconds = time.monotonic() - t0
    rows, details, predictions, daily_predictions, inference = [], [], [], [], []
    # Four configurations are fixed before seeing scores: 2x2 isolates each change.
    for cutoff, end in WINDOWS + [DIAGNOSTIC_WINDOW]:
        primary = (cutoff, end) in WINDOWS
        for use_covariates in [False, True]:
            train, routes, dates, tasks = model_inputs(data, cutoff, end, use_covariates)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.monotonic()
            with torch.inference_mode():
                q, _ = model.predict_quantiles(tasks, prediction_length=len(dates), quantile_levels=[0.5],
                    batch_size=32, context_length=512, cross_learning=False)
            torch.cuda.synchronize()
            inference.append(dict(cutoff=cutoff, covariates=use_covariates, seconds=time.monotonic()-t0,
                                  peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated()))
            values = np.stack([v.cpu().numpy().reshape(len(dates)) for v in q])
            if values.shape != (9, len(dates)) or not np.isfinite(values).all():
                raise ValueError("Invalid daily prediction")
            daily = pd.concat([pd.DataFrame(dict(route=int(r), date=dates, prediction=v))
                               for r, v in zip(routes, values)], ignore_index=True)
            daily_predictions.append(daily.assign(cutoff=cutoff, covariates=use_covariates))
            for use_day_type in [False, True]:
                method = f"cov{int(use_covariates)}_shares{int(use_day_type)}"
                t0 = time.monotonic()
                result = hourly_forecast(train, daily, cutoff, use_day_type)
                if primary and not use_covariates and not use_day_type:
                    old = reference.loc[reference.method.eq("chronos_daily_total") & reference.cutoff.eq(cutoff), KEYS+["prediction"]]
                    pd.testing.assert_frame_equal(result.sort_values(KEYS).reset_index(drop=True), old.sort_values(KEYS).reset_index(drop=True))
                row, detail, compared = evaluate(data, result, cutoff, end, method, time.monotonic()-t0)
                row["primary"] = primary
                rows.append(row)
                details.extend(detail)
                predictions.append(compared.assign(primary=primary))
                print(json.dumps(row), flush=True)
                pd.DataFrame(rows).to_csv(out/"metrics.csv", sep=";", index=False)
                pd.DataFrame(details).to_csv(out/"breakdown.csv", sep=";", index=False)
                pd.concat(predictions, ignore_index=True).to_csv(out/"predictions.csv", sep=";", index=False)
    pd.concat(daily_predictions, ignore_index=True).to_csv(out/"daily_predictions.csv", sep=";", index=False)
    pd.DataFrame(inference).to_csv(out/"inference.csv", sep=";", index=False)
    combined = pd.concat(predictions, ignore_index=True)
    pooled = [dict(method=m, **metrics(g.boardings, g.prediction))
              for m, g in combined[combined.primary].groupby("method")]
    pd.DataFrame(pooled).to_csv(out/"pooled_primary.csv", sep=";", index=False)
    calendar_table().to_csv(out/"calendar_2025.csv", sep=";", index=False)
    files = [Path(args.history), SOURCES, Path(__file__), Path("experiments/chronos_experiment.py"),
             Path("experiments/calendar_experiment.py"), Path("pipeline.py")]
    metadata = dict(model="amazon/chronos-2", revision=REVISION, primary_windows=WINDOWS,
        diagnostic_window=DIAGNOSTIC_WINDOW, parameters=dict(seed=42, batch_size=32, context_length=512,
        quantile=0.5, cross_learning=False, dtype="float32", covariates=COVARIATES, share_min_days=MIN_DAYS),
        sources=json.loads(SOURCES.read_text()), sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        job_id=os.environ.get("SLURM_JOB_ID"), reservation=os.environ.get("SLURM_JOB_RESERVATION"),
        node=platform.node(), gpu=torch.cuda.get_device_name(), cpu_affinity=affinity,
        seconds=time.monotonic()-started, model_load_seconds=load_seconds,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        versions={p: importlib.metadata.version(p) for p in ["numpy","pandas","torch","chronos-forecasting","transformers","accelerate"]},
        python=platform.python_version(), control_exactly_matches_previous=True,
        limitations="modern pretrained weights; reused validation; no historical deployment claim",
        postprocessing="floor(max(0,p)+0.5); route5 and hours1-4 forced0; missing counts remain0")
    (out/"run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--history", default="artifacts/hourly_clean.csv")
    p.add_argument("--model-path", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--output", default="artifacts/chronos_calendar_zhores")
    run(p.parse_args())
