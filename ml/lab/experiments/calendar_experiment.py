"""Run: python -m experiments.calendar_experiment. Fixed calendar ablation of mean_all."""
import hashlib
import json
from pathlib import Path
import platform
import resource
import time

import numpy as np
import pandas as pd

from pipeline import KEYS, OUT, full_grid, metrics, postprocess, predict

SOURCES = OUT / "calendar_sources.json"
WINDOWS = [("2025-06-30", "2025-08-31"), ("2025-08-31", "2025-10-31")]
MIN_DAYS = 3  # Fixed before evaluation; not tuned on the validation windows.


def calendar_table():
    spec = json.loads(SOURCES.read_text(encoding="utf-8"))
    if "calendars" in spec:
        tables = [_calendar_table(item) for item in spec["calendars"]]
        result = pd.concat(tables, ignore_index=True).sort_values("date").reset_index(drop=True)
        if result.date.duplicated().any():
            raise ValueError("Overlapping calendar coverage")
        return result
    return _calendar_table(spec)


def _calendar_table(spec):
    dates = pd.date_range(*spec["coverage"])
    table = pd.DataFrame({"date": dates, "weekday": dates.dayofweek})
    table["day_type"] = "weekday_" + table.weekday.astype(str)
    table["is_workday"] = table.weekday.lt(5)
    for kind, days, work in [("holiday", spec["holidays"], False),
                             ("transferred_off", list(spec["transfers"].values()), False),
                             ("transferred_work", spec["working_weekends"], True)]:
        selected = table.date.isin(pd.to_datetime(days))
        table.loc[selected, "day_type"] = kind
        table.loc[selected, "is_workday"] = work
    table["known_at"] = pd.Timestamp(spec["known_at"])
    return table


def calendar_predict(history, keys, cutoff):
    cutoff = pd.Timestamp(cutoff)
    if keys.empty or keys.duplicated(KEYS).any() or not keys.date.gt(cutoff).all():
        raise ValueError("Expected unique forecast keys after cutoff")
    # Filter before joins/aggregations: later facts cannot affect any fallback.
    train = history.loc[history.date.le(cutoff), KEYS + ["boardings"]].copy()
    if train.empty or train.duplicated(KEYS).any() or not np.isfinite(train.boardings).all() or train.boardings.lt(0).any():
        raise ValueError("Invalid training history")
    calendar = calendar_table()
    if calendar.loc[calendar.date.isin(pd.concat([history.date, keys.date])), "known_at"].gt(cutoff).any():
        raise ValueError("Calendar was not available at cutoff")
    train = train.merge(calendar, on="date", how="left", validate="many_to_one")
    result = keys[KEYS].merge(calendar, on="date", how="left", validate="many_to_one")
    if train.day_type.isna().any() or result.day_type.isna().any():
        raise ValueError("Calendar covers 2025 only")
    group = ["route", "day_type", "hour"]
    profile = train.groupby(group).boardings.agg(["mean", "count"])
    profile["prediction"] = profile["mean"].where(profile["count"].ge(MIN_DAYS))
    result = result.merge(profile[["prediction"]], on=group, how="left", validate="many_to_one")
    # ponytail: all holidays share one type; separate seasons only with enough history.
    fallback = train.groupby(["route", "is_workday", "hour"]).boardings.mean().rename("fallback")
    result = result.merge(fallback, on=["route", "is_workday", "hour"], how="left", validate="many_to_one")
    result["prediction"] = result.prediction.fillna(result.fallback)
    return postprocess(result)  # Missing broad profile is an error, never an invented zero.


def run():
    started = time.monotonic()
    data = pd.read_csv(OUT / "hourly_clean.csv", sep=";", parse_dates=["date"])
    expected = full_grid().to_frame(index=False)
    expected["date"] = pd.to_datetime(expected.date)
    pd.testing.assert_frame_equal(data[KEYS].sort_values(KEYS).reset_index(drop=True), expected)
    if not np.isfinite(data.boardings).all() or data.boardings.lt(0).any():
        raise ValueError("Invalid cleaned target")
    calendar = calendar_table()
    calendar.to_csv(OUT / "calendar_2025.csv", sep=";", index=False)
    rows, details, predictions = [], [], []
    for cutoff, end in WINDOWS:
        valid = data.loc[data.date.gt(cutoff) & data.date.le(end), KEYS + ["boardings"]]
        for method in ["mean_all", "calendar_mean_all"]:
            forecast = (predict(data, valid, cutoff, 0, "mean") if method == "mean_all"
                        else calendar_predict(data, valid, cutoff))
            compared = valid.merge(forecast, on=KEYS, validate="one_to_one")
            assert len(compared) == len(valid) == len(forecast)
            assert compared.loc[compared.route.eq(5) | compared.hour.between(1, 4), "prediction"].eq(0).all()
            spec = dict(method=method, cutoff=cutoff, end=end)
            rows.append(dict(spec, **metrics(compared.boardings, compared.prediction)))
            predictions.append(compared.assign(**spec))
            compared["month"] = compared.date.dt.strftime("%Y-%m")
            compared["horizon_group"] = pd.cut((compared.date - pd.Timestamp(cutoff)).dt.days,
                [0, 7, 28, 62], labels=["1-7", "8-28", "29-62"])
            for dimension in ["route", "hour", "month", "horizon_group"]:
                for value, group in compared.groupby(dimension, observed=True):
                    details.append(dict(spec, dimension=dimension, value=str(value),
                                        **metrics(group.boardings, group.prediction)))
    results = pd.DataFrame(rows)
    reference = pd.read_csv(OUT / "baseline_metrics.csv", sep=";")
    columns = ["cutoff", "wape_score", "absolute_error", "actual_total", "bias"]
    pd.testing.assert_frame_equal(results.loc[results.method.eq("mean_all"), columns].reset_index(drop=True),
        reference.loc[reference.weeks.eq(0) & reference.statistic.eq("mean"), columns].reset_index(drop=True))
    export = pd.concat(predictions, ignore_index=True)
    path = OUT / "calendar_predictions.csv"
    export.to_csv(path, sep=";", index=False)
    pd.testing.assert_frame_equal(export, pd.read_csv(path, sep=";", parse_dates=["date"]))
    results.to_csv(OUT / "calendar_metrics.csv", sep=";", index=False)
    pd.DataFrame(details).to_csv(OUT / "calendar_breakdown.csv", sep=";", index=False)
    pooled = [dict(method=method, **metrics(g.boardings, g.prediction))
              for method, g in export.groupby("method")]
    pd.DataFrame(pooled).to_csv(OUT / "calendar_pooled_metrics.csv", sep=";", index=False)
    run_info = dict(
        command="python -m experiments.calendar_experiment", windows=WINDOWS,
        features=["route", "day_type", "hour"], fallback_features=["route", "is_workday", "hour"],
        parameters=dict(statistic="mean", weeks=0, min_days=MIN_DAYS, update_within_horizon=False),
        sources=json.loads(SOURCES.read_text(encoding="utf-8")), seed=None,
        missing_policy="absent counts = 0; observation masks retained; validation unchanged",
        postprocessing="floor(max(0,p)+0.5); route5 and hours1-4 forced0",
        python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__,
        sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [OUT / "hourly_clean.csv", SOURCES, OUT / "calendar_2025.csv", Path(__file__), Path(__file__).resolve().parents[1] / "pipeline.py"]},
        seconds=time.monotonic() - started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024))
    (OUT / "calendar_run.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(results.to_string(index=False))
    print(json.dumps({k: run_info[k] for k in ["seconds", "peak_rss_bytes"]}))


if __name__ == "__main__":
    run()
