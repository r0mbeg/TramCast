"""Fixed-cutoff Chronos-2 zero-shot experiment; no external covariates or fine-tuning."""
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

from pipeline import KEYS, ROUTES, full_grid, metrics, postprocess, predict

REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
WINDOWS = [("2025-06-30", "2025-08-31"), ("2025-08-31", "2025-10-31")]
WORK_HOURS = [0] + list(range(5, 24))


def inputs(history, cutoff, representation):
    train = history.loc[history.date.le(cutoff) & history.route.ne(5)
                        & history.hour.isin(WORK_HOURS), KEYS + ["boardings"]].copy()
    columns = ["route"] if representation == "daily_total" else ["route", "hour"]
    series = train.groupby(["date"] + columns).boardings.sum().unstack(columns)
    expected_dates = pd.date_range(history.date.min(), cutoff)
    if not series.index.equals(expected_dates.rename("date")) or series.isna().any().any():
        raise ValueError("Incomplete daily history")
    expected_count = 9 if representation == "daily_total" else 180
    if len(series.columns) != expected_count:
        raise ValueError("Unexpected series count")
    return train, series


def distribute_daily(train, daily):
    train = train.assign(weekday=train.date.dt.dayofweek)
    profile = train.groupby(["route", "weekday", "hour"]).boardings.sum().rename("volume").reset_index()
    totals = profile.groupby(["route", "weekday"]).volume.transform("sum")
    # ponytail: a zero-volume weekday uses the route profile; revisit with validated coverage imputation.
    route_profile = train.groupby(["route", "hour"]).boardings.sum().rename("fallback").reset_index()
    route_total = route_profile.groupby("route").fallback.transform("sum")
    route_profile["fallback"] = route_profile.fallback.div(route_total.where(route_total.gt(0))).fillna(1 / 20)
    profile["share"] = profile.volume.div(totals.where(totals.gt(0)))
    profile = profile.merge(route_profile, on=["route", "hour"], validate="many_to_one")
    profile["share"] = profile.share.fillna(profile.fallback)
    np.testing.assert_allclose(profile.groupby(["route", "weekday"]).share.sum(), 1)
    result = daily.assign(weekday=daily.date.dt.dayofweek).merge(
        profile[["route", "weekday", "hour", "share"]], on=["route", "weekday"], validate="many_to_many")
    result["prediction"] = result.prediction.clip(lower=0) * result.share
    return result[KEYS + ["prediction"]]


def forecast(history, cutoff, end, representation, predictor):
    train, series = inputs(history, cutoff, representation)
    dates = pd.date_range(pd.Timestamp(cutoff) + pd.Timedelta(days=1), end)
    values = predictor(series.to_numpy(dtype=np.float32).T, len(dates))
    if values.shape != (len(series.columns), len(dates)) or not np.isfinite(values).all():
        raise ValueError("Invalid model output")
    parts = []
    for key, values_for_series in zip(series.columns, values):
        if representation == "daily_total":
            parts.append(pd.DataFrame(dict(route=int(key), date=dates, prediction=values_for_series)))
        else:
            route, hour = key
            parts.append(pd.DataFrame(dict(route=route, hour=hour, date=dates, prediction=values_for_series)))
    result = pd.concat(parts, ignore_index=True)
    if representation == "daily_total":
        result = distribute_daily(train, result)
    grid = full_grid(dates.min(), dates.max()).to_frame(index=False)
    grid["date"] = pd.to_datetime(grid.date)
    result = grid.merge(result, on=KEYS, how="left", validate="one_to_one")
    forced_zero = result.route.eq(5) | result.hour.between(1, 4)
    if result.loc[~forced_zero, "prediction"].isna().any():
        raise ValueError("Missing operating-hour prediction")
    result.loc[forced_zero, "prediction"] = 0
    return postprocess(result)


def evaluate(data, forecast_frame, cutoff, end, method, seconds):
    valid = data[data.date.gt(cutoff) & data.date.le(end)]
    compared = valid[KEYS + ["boardings"]].merge(forecast_frame, on=KEYS, validate="one_to_one")
    if len(compared) != len(valid) or len(forecast_frame) != len(valid):
        raise ValueError("Forecast keys do not match validation")
    spec = dict(method=method, cutoff=cutoff, end=end)
    row = dict(spec, seconds=seconds, **metrics(compared.boardings, compared.prediction))
    compared["month"] = compared.date.dt.strftime("%Y-%m")
    horizon = (compared.date - pd.Timestamp(cutoff)).dt.days
    compared["horizon_group"] = pd.cut(horizon, [0, 7, 28, 62], labels=["1-7", "8-28", "29-62"])
    details = []
    for dimension in ["route", "hour", "month", "horizon_group"]:
        for value, group in compared.groupby(dimension, observed=True):
            details.append(dict(spec, dimension=dimension, value=str(value),
                                **metrics(group.boardings, group.prediction)))
    return row, details, compared[KEYS + ["boardings", "prediction"]].assign(**spec)


def run(args):
    import torch
    from chronos import Chronos2Pipeline

    started = time.monotonic()
    cpu_count = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    if not 1 <= cpu_count <= 8:
        raise RuntimeError("Expected a Slurm allocation of 1–8 CPUs")
    torch.set_num_threads(cpu_count)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)
    cuda = args.device == "cuda"
    if cuda and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError("This experiment requires exactly one visible CUDA GPU")
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) > cpu_count:
        raise RuntimeError(f"CPU affinity exceeds allocation: {affinity}")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.history, sep=";", parse_dates=["date"])
    expected = full_grid().to_frame(index=False)
    expected["date"] = pd.to_datetime(expected.date)
    pd.testing.assert_frame_equal(data[KEYS].sort_values(KEYS).reset_index(drop=True), expected)
    if data.boardings.isna().any() or data.boardings.lt(0).any():
        raise ValueError("Invalid target")
    load_started = time.monotonic()
    model = Chronos2Pipeline.from_pretrained(args.model_path, device_map=args.device, torch_dtype=torch.float32)
    model.model.eval()
    if cuda:
        torch.cuda.synchronize()
    load_seconds = time.monotonic() - load_started

    def predictor(matrix, horizon):
        with torch.inference_mode():
            quantiles, _ = model.predict_quantiles(
                [torch.from_numpy(row.copy()) for row in matrix], prediction_length=horizon,
                quantile_levels=[0.5], batch_size=32, context_length=512, cross_learning=False)
        return np.stack([q.cpu().numpy().reshape(horizon) for q in quantiles])

    rows, details, predictions = [], [], []
    for cutoff, end in WINDOWS:
        keys = data.loc[data.date.gt(cutoff) & data.date.le(end), KEYS]
        for method in ["mean_all", "chronos_daily_total", "chronos_route_hour"]:
            t0 = time.monotonic()
            if method == "mean_all":
                result = predict(data, keys, cutoff, 0, "mean")
            else:
                if cuda:
                    torch.cuda.reset_peak_memory_stats()
                result = forecast(data, cutoff, end, method.removeprefix("chronos_"), predictor)
                if cuda:
                    torch.cuda.synchronize()
            row, detail, prediction = evaluate(data, result, cutoff, end, method, time.monotonic() - t0)
            row["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated() if cuda and method != "mean_all" else 0
            rows.append(row)
            details.extend(detail)
            predictions.append(prediction)
            pd.DataFrame(rows).to_csv(out / "metrics.csv", sep=";", index=False)
            pd.DataFrame(details).to_csv(out / "breakdown.csv", sep=";", index=False)
            pd.concat(predictions, ignore_index=True).to_csv(out / "predictions.csv", sep=";", index=False)
            print(json.dumps(row), flush=True)
    combined = pd.concat(predictions, ignore_index=True)
    pooled = [dict(method=method, **metrics(g.boardings, g.prediction)) for method, g in combined.groupby("method")]
    pd.DataFrame(pooled).to_csv(out / "pooled_metrics.csv", sep=";", index=False)
    if args.submission:
        final = forecast(data, "2025-10-31", "2025-12-31", "daily_total", predictor)
        final.to_csv(out / "submission.csv", sep=";", index=False, date_format="%Y-%m-%d")
        pd.testing.assert_frame_equal(final, pd.read_csv(out / "submission.csv", sep=";", parse_dates=["date"]))
    run_info = dict(
        model="amazon/chronos-2", revision=REVISION, mode="zero-shot; no covariates; no fine-tuning",
        history_sha256=hashlib.sha256(Path(args.history).read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        windows=WINDOWS, seed=42, quantile=0.5, batch_size=32, context_length=512,
        cross_learning=False, dtype="float32", cpu_affinity=affinity,
        device=args.device, gpu=torch.cuda.get_device_name(0) if cuda else None,
        gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory if cuda else 0,
        job_id=os.environ.get("SLURM_JOB_ID"), reservation=os.environ.get("SLURM_JOB_RESERVATION"),
        node=platform.node(), python=platform.python_version(),
        versions={p: importlib.metadata.version(p) for p in
                  ["numpy", "pandas", "torch", "chronos-forecasting", "transformers", "accelerate"]},
        model_load_seconds=load_seconds, seconds=time.monotonic() - started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        missing_policy="absent counts = 0; observed validation unchanged",
        postprocessing="floor(max(0,p)+0.5); route5 and hours1-4 forced0",
        pretrained_availability="modern weights; not deployable at historical 2025 cutoff")
    if args.submission:
        run_info["submission"] = dict(cutoff="2025-10-31", end="2025-12-31",
            representation="daily_total", rows=len(final), prediction_total=int(final.prediction.sum()),
            sha256=hashlib.sha256((out / "submission.csv").read_bytes()).hexdigest())
    (out / "run.json").write_text(json.dumps(run_info, indent=2))
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", default="artifacts/hourly_clean.csv")
    parser.add_argument("--output", default="artifacts/chronos_zhores")
    parser.add_argument("--model-path", required=True, help="Local snapshot of the pinned model revision")
    parser.add_argument("--submission", action="store_true", help="Also forecast November–December using daily totals")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    run(parser.parse_args())
