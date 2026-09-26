"""python pipeline.py prepare | evaluate | repeat-week; Python + pandas + numpy."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import resource
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts"
ROUTES = [1, 5, 7, 11, 12, 17, 25, 26, 28, 50]
KEYS = ["route", "date", "hour"]


def working_time(timestamp):
    minute = timestamp.dt.hour * 60 + timestamp.dt.minute
    return (minute < 60) | (minute >= 330)


def aggregate(frame, start, end):
    timestamp = pd.to_datetime(frame.tran_date_time, format="%Y-%m-%d %H:%M:%S", errors="raise")
    route = frame.ngpt_route.str.extract(r"^(\d+) трамвай$", expand=False)
    if timestamp.isna().any() or route.isna().any() or frame.validation_result.isna().any():
        raise ValueError("Missing timestamp, route or validation result")
    route = route.astype(int)
    if not route.isin(ROUTES).all():
        raise ValueError(f"Unexpected routes: {route[~route.isin(ROUTES)].unique()}")
    in_period = timestamp.ge(start) & timestamp.lt(end)
    success = frame.validation_result.eq(1)
    work = working_time(timestamp)
    events = pd.DataFrame({"route": route, "date": timestamp.dt.strftime("%Y-%m-%d"),
                           "hour": timestamp.dt.hour})
    raw = events[in_period & success].groupby(KEYS).size()
    clean = events[in_period & success & work & route.ne(5)].groupby(KEYS).size()
    present = events[in_period & work].groupby(KEYS).size()
    counts = dict(rows=len(frame), outside_period=int((~in_period).sum()),
                  unsuccessful_in_period=int((in_period & ~success).sum()),
                  successful_in_period=int((in_period & success).sum()),
                  removed_nonworking=int((in_period & success & ~work).sum()),
                  removed_route5_working=int((in_period & success & work & route.eq(5)).sum()))
    return raw, clean, present, counts


def full_grid(start="2025-01-01", end="2025-10-31"):
    return pd.MultiIndex.from_product(
        [ROUTES, pd.date_range(start, end).strftime("%Y-%m-%d"), range(24)], names=KEYS)


def metrics(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    if y.shape != prediction.shape or not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError("Invalid metric inputs")
    absolute = float(np.abs(y - prediction).sum())
    total = float(y.sum())
    return dict(wape_score=max(0, 1 - absolute / total) if total else None,
                absolute_error=absolute, actual_total=total,
                bias=float((prediction - y).sum()))


def predict(history, keys, cutoff, weeks, statistic):
    # Filter here as well as in the caller, so later observations cannot enter the profile.
    train = history[history.date.le(cutoff)].copy()
    if weeks:
        train = train[train.date.gt(pd.Timestamp(cutoff) - pd.Timedelta(days=weeks * 7))]
    train["weekday"] = train.date.dt.dayofweek
    result = keys[KEYS].copy()
    if not result.date.gt(cutoff).all():
        raise ValueError("Forecast keys must be after cutoff")
    result["weekday"] = result.date.dt.dayofweek
    profile = train.groupby(["route", "weekday", "hour"]).boardings.agg(statistic)
    result = result.merge(profile.rename("prediction"), on=["route", "weekday", "hour"],
                          how="left", validate="many_to_one")
    return postprocess(result)


def postprocess(result):
    result = result.copy()
    if not np.isfinite(result.prediction).all():
        raise ValueError("Missing seasonal profile")
    result["prediction"] = np.floor(result.prediction.clip(lower=0) + 0.5).astype("int64")
    result.loc[result.route.eq(5) | result.hour.between(1, 4), "prediction"] = 0
    return result[KEYS + ["prediction"]]


def repeat_week(history, keys, cutoff):
    cutoff = pd.Timestamp(cutoff)
    week_end = cutoff - pd.Timedelta(days=(cutoff.dayofweek + 1) % 7)
    week_start = week_end - pd.Timedelta(days=6)
    week = history[history.date.between(week_start, week_end)]
    expected = full_grid(week_start, week_end).to_frame(index=False)
    expected["date"] = pd.to_datetime(expected.date)
    if len(week) != len(expected) or set(week[KEYS].itertuples(index=False, name=None)) != set(expected.itertuples(index=False, name=None)):
        raise ValueError("Repeat requires a complete Monday–Sunday grid")
    if not keys.date.gt(cutoff).all():
        raise ValueError("Forecast keys must be after cutoff")
    return predict(week, keys, cutoff, 0, "mean")


def prepare():
    raw_counts, clean_counts, present_counts = Counter(), Counter(), Counter()
    audit, boundary = {}, {}
    for filename in ["train.csv", "test.csv"]:
        path = ROOT / "dataset" / filename
        totals, boundary_parts = Counter(), []
        for chunk in pd.read_csv(path, sep=";", chunksize=250_000,
                                 dtype={"tran_no": str, "device_no": str,
                                        "validation_result": "int64"},
                                 usecols=["tran_no", "device_no", "tran_date_time",
                                          "ngpt_route", "validation_result"]):
            # Files have next-month tails: filter the combined history by event time.
            raw, clean, present, counts = aggregate(chunk, "2025-01-01", "2025-11-01")
            raw_counts.update(raw.to_dict())
            clean_counts.update(clean.to_dict())
            present_counts.update(present.to_dict())
            totals.update(counts)
            day = chunk.tran_date_time.str[:10]
            boundary_parts.append(chunk.loc[day.eq("2025-09-01")].copy())
            if totals["rows"] % 5_000_000 == 0:
                print(filename, dict(totals), flush=True)
        boundary[filename] = pd.concat(boundary_parts, ignore_index=True)
        audit[filename] = dict(totals, bytes=path.stat().st_size,
                              mtime_ns=path.stat().st_mtime_ns)
        print(filename, "complete", audit[filename], flush=True)
    # These are candidate identities, not proof of duplicate full raw rows.
    a, b = boundary["train.csv"], boundary["test.csv"]
    columns = ["tran_no", "device_no", "tran_date_time", "ngpt_route", "validation_result"]
    left, right = Counter(a[columns].itertuples(index=False, name=None)), Counter(b[columns].itertuples(index=False, name=None))
    audit["september_boundary"] = dict(train_rows=len(a), test_rows=len(b),
        matching_five_field_occurrences=sum((left & right).values()),
        train_unmatched_occurrences=sum((left - right).values()),
        train_repeated_tran_no=int(a.tran_no.duplicated().sum()),
        test_repeated_tran_no=int(b.tran_no.duplicated().sum()))
    if (left & right):
        raise ValueError("Potential boundary duplicates: inspect full events before counting twice")
    grid = full_grid()
    data = pd.DataFrame(index=grid)
    # ponytail: missing counts become zero; retain masks until coverage/imputation is validated.
    for name, counter in [("raw_boardings", raw_counts), ("boardings", clean_counts),
                           ("working_events", present_counts)]:
        data[name] = pd.Series(counter).reindex(grid, fill_value=0).astype("int64")
    supplied = pd.concat([pd.read_csv(p, sep=";") for p in (ROOT / "dataset/labels").glob("*.csv")])
    if supplied.duplicated(KEYS).any():
        raise ValueError("Duplicate supplied labels")
    expected = supplied.set_index(KEYS).boardings.reindex(grid, fill_value=0)
    data["supplied_label_present"] = grid.isin(supplied.set_index(KEYS).index)
    data["successful_working_observed"] = data.boardings.gt(0)
    data["working_events_observed"] = data.working_events.gt(0)
    data["removed_boardings"] = data.raw_boardings - data.boardings
    difference = data.raw_boardings - expected
    audit["reconciliation"] = dict(mismatched_keys=int(difference.ne(0).sum()),
        raw_total=int(data.raw_boardings.sum()), supplied_total=int(expected.sum()),
        cleaned_total=int(data.boardings.sum()), removed_total=int(data.removed_boardings.sum()))
    data.assign(supplied_boardings=expected, delta=difference).loc[difference.ne(0)].to_csv(
        OUT / "label_mismatches.csv", sep=";")
    (OUT / "preparation_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if difference.ne(0).any():
        raise ValueError("Raw/label mismatch: inspect preparation_audit.json and label_mismatches.csv")
    assert len(data) == 72960 and data.index.is_unique
    assert data.boardings.ge(0).all() and data.removed_boardings.ge(0).all()
    assert data.loc[5, "boardings"].eq(0).all()
    data.to_csv(OUT / "hourly_clean.csv", sep=";")
    data.reset_index().groupby(["route", "hour"])[["raw_boardings", "boardings", "removed_boardings"]].sum().to_csv(
        OUT / "cleaning_by_route_hour.csv", sep=";")
    print(json.dumps(audit, indent=2), flush=True)


def evaluate(repeat=False):
    data = pd.read_csv(OUT / "hourly_clean.csv", sep=";", parse_dates=["date"])
    if len(data) != 72960 or data.duplicated(KEYS).any() or data.isna().any().any():
        raise ValueError("Invalid cleaned grid")
    supplied = pd.concat([pd.read_csv(p, sep=";", parse_dates=["date"])
                          for p in (ROOT / "dataset/labels").glob("*.csv")])
    data = data.merge(supplied.rename(columns={"boardings": "supplied_boardings"}),
                      on=KEYS, how="left", validate="one_to_one")
    data["supplied_boardings"] = data.supplied_boardings.fillna(0)
    rows, breakdowns, predictions = [], [], []
    for cutoff, end in [("2025-06-30", "2025-08-31"), ("2025-08-31", "2025-10-31")]:
        valid = data[data.date.gt(cutoff) & data.date.le(end)]
        for weeks in ([1, 0] if repeat else [4, 8, 12, 0]):
            for statistic in (["mean"] if repeat else ["mean", "median"]):
                forecast = (repeat_week(data, valid, cutoff) if repeat and weeks == 1
                            else predict(data, valid, cutoff, weeks, statistic))
                comparison = valid.merge(forecast, on=KEYS, validate="one_to_one")
                spec = dict(cutoff=cutoff, end=end, weeks=weeks, statistic=statistic)
                if repeat:
                    spec["method"] = "repeat_week" if weeks == 1 else "mean_all"
                    source_end = pd.Timestamp(cutoff) - pd.Timedelta(days=(pd.Timestamp(cutoff).dayofweek + 1) % 7)
                    spec["source_start"] = str((source_end - pd.Timedelta(days=6)).date()) if weeks else "2025-01-01"
                    spec["source_end"] = str(source_end.date()) if weeks else cutoff
                    assert len(comparison) == len(valid)
                    assert comparison.loc[comparison.route.eq(5) | comparison.hour.between(1, 4), "prediction"].eq(0).all()
                    predictions.append(comparison[KEYS + ["boardings", "prediction"]].assign(**spec))
                rows.append(dict(spec, **metrics(comparison.boardings, comparison.prediction),
                    score_against_original_labels=metrics(comparison.supplied_boardings, comparison.prediction)["wape_score"]))
                for dimension in ["route", "hour", "month"]:
                    comparison["month"] = comparison.date.dt.strftime("%Y-%m")
                    for value, group in comparison.groupby(dimension):
                        breakdowns.append(dict(spec, dimension=dimension, value=value,
                                               **metrics(group.boardings, group.prediction)))
    results = pd.DataFrame(rows)
    prefix = "repeat_week" if repeat else "baseline"
    if repeat:
        reference = pd.read_csv(OUT / "baseline_metrics.csv", sep=";")
        reference = reference[reference.weeks.eq(0) & reference.statistic.eq("mean")]
        columns = ["cutoff", "wape_score", "absolute_error", "actual_total", "bias"]
        pd.testing.assert_frame_equal(results.loc[results.method.eq("mean_all"), columns].reset_index(drop=True),
                                      reference[columns].reset_index(drop=True))
        export = pd.concat(predictions, ignore_index=True)
        path = OUT / "repeat_week_predictions.csv"
        export.to_csv(path, sep=";", index=False)
        pd.testing.assert_frame_equal(export, pd.read_csv(path, sep=";", parse_dates=["date"]))
    results.to_csv(OUT / f"{prefix}_metrics.csv", sep=";", index=False)
    pd.DataFrame(breakdowns).to_csv(OUT / f"{prefix}_breakdown.csv", sep=";", index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "evaluate", "repeat-week"])
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    started = time.monotonic()
    {"prepare": prepare, "evaluate": evaluate, "repeat-week": lambda: evaluate(repeat=True)}[args.command]()
    run = dict(command=args.command, seconds=time.monotonic() - started,
               peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024),
               python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__,
               missing_policy="absent counts = 0, observation masks retained",
               seed=None, rounding="floor(max(0, prediction) + 0.5)")
    if args.command == "repeat-week":
        run.update(history_sha256=hashlib.sha256((OUT / "hourly_clean.csv").read_bytes()).hexdigest(),
                   features=["route", "weekday", "hour"],
                   parameters={"source": "last complete Monday–Sunday", "update_within_horizon": False},
                   windows=[{"cutoff": "2025-06-30", "end": "2025-08-31"},
                            {"cutoff": "2025-08-31", "end": "2025-10-31"}])
    (OUT / f"{args.command}_run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
