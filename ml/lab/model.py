"""Train one fixed gradient boosting model per temporal fold: python model.py."""
import argparse
import hashlib
import json
import pickle
import platform
import resource
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from pipeline import KEYS, OUT, full_grid, metrics, postprocess

FEATURES = ["route", "hour", "weekday", "month", "day_of_year", "horizon",
            "origin_day_of_year", "mean_all", "mean_28", "median_28"]
ABSOLUTE_CALENDAR = ["month", "day_of_year", "origin_day_of_year"]
RESIDUAL_LIMIT = 0.20
PARAMETERS = dict(loss="absolute_error", max_iter=150, learning_rate=0.08,
                  max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=1.0,
                  categorical_features=["route", "weekday"],
                  early_stopping=False, random_state=42)


def features(history, keys, origin):
    origin = pd.Timestamp(origin)
    x = keys[KEYS].copy().reset_index(drop=True)
    x["horizon"] = (x.date - origin).dt.days
    if not x.horizon.between(1, 62).all() or x.duplicated(KEYS).any():
        raise ValueError("Expected unique forecast keys 1–62 days after origin")
    x["weekday"], x["month"] = x.date.dt.dayofweek, x.date.dt.month
    x["day_of_year"], x["origin_day_of_year"] = x.date.dt.dayofyear, origin.dayofyear
    past = history.loc[history.date.le(origin), KEYS + ["boardings"]].copy()
    past["weekday"] = past.date.dt.dayofweek
    group = ["route", "weekday", "hour"]
    for name, days, statistic in [("mean_all", None, "mean"), ("mean_28", 28, "mean"),
                                   ("median_28", 28, "median")]:
        source = past if days is None else past[past.date.gt(origin - pd.Timedelta(days=days))]
        profile = source.groupby(group).boardings.agg(statistic).rename(name)
        x = x.merge(profile, on=group, how="left", validate="many_to_one")
    if not np.isfinite(x[FEATURES]).all().all():
        raise ValueError("Missing or invalid historical features")
    return x


def training_examples(data, cutoff, exclude_january_origin=False):
    past = data[data.date.le(cutoff)]
    xs, ys, origins = [], [], []
    # ponytail: origins every 28 days, overlapping 62-day targets; more history needed for annual effects.
    for origin in pd.date_range(past.date.min() + pd.Timedelta(days=30),
                                pd.Timestamp(cutoff) - pd.Timedelta(days=62), freq="28D"):
        if exclude_january_origin and origin == pd.Timestamp("2025-01-31"):
            continue
        target = past[past.date.gt(origin) & past.date.le(origin + pd.Timedelta(days=62))]
        target = target[target.route.ne(5) & ~target.hour.between(1, 4)]
        block = features(past, target, origin).merge(target[KEYS + ["boardings"]],
                                                   on=KEYS, validate="one_to_one")
        xs.append(block[FEATURES])
        ys.append(block.boardings)
        origins.append(origin.strftime("%Y-%m-%d"))
    if not xs:
        raise ValueError("Not enough history for a complete training horizon")
    return pd.concat(xs, ignore_index=True), pd.concat(ys, ignore_index=True), origins


def forecast(model, history, keys, cutoff):
    x = features(history, keys, cutoff)
    x["prediction"] = 0.0
    active = x.route.ne(5) & ~x.hour.between(1, 4)
    x.loc[active, "prediction"] = model.predict(x.loc[active, model.feature_names_in_])
    limit = getattr(model, "residual_limit_", None)
    if limit is not None:
        # ponytail: a fixed cap protects the baseline; tune only on separate temporal validation.
        x["prediction"] = x.mean_all + np.clip(x.prediction, -limit * x.mean_all, limit * x.mean_all)
    return postprocess(x)


def run(no_absolute_calendar=False, residual_correction=False, exclude_january_origin=False):
    if exclude_january_origin and not residual_correction:
        raise ValueError("January-origin experiment requires --residual-correction")
    started = time.monotonic()
    no_absolute_calendar = no_absolute_calendar or residual_correction
    selected = [name for name in FEATURES if not no_absolute_calendar or name not in ABSOLUTE_CALENDAR]
    prefix = "ml_no_calendar" if no_absolute_calendar else "ml"
    model_prefix = "model_no_calendar" if no_absolute_calendar else "model"
    if residual_correction:
        prefix, model_prefix = "ml_residual", "model_residual"
    if exclude_january_origin:
        prefix, model_prefix = "ml_residual_no_january", "model_residual_no_january"
    source = OUT / "hourly_clean.csv"
    data = pd.read_csv(source, sep=";", parse_dates=["date"])
    expected = full_grid().to_frame(index=False)
    expected["date"] = pd.to_datetime(expected.date)
    if (not pd.MultiIndex.from_frame(data[KEYS]).equals(pd.MultiIndex.from_frame(expected))
            or not np.isfinite(data.boardings).all() or data.boardings.lt(0).any()):
        raise ValueError("Invalid cleaned history")
    baseline = pd.read_csv(OUT / "baseline_metrics.csv", sep=";")
    rows, parts, predictions, folds = [], [], [], []
    for cutoff, end in [("2025-06-30", "2025-08-31"), ("2025-08-31", "2025-10-31")]:
        begin = time.monotonic()
        x, y, origins = training_examples(data, cutoff, exclude_january_origin)
        model = HistGradientBoostingRegressor(**PARAMETERS)
        model.fit(x[selected], y - x.mean_all if residual_correction else y)
        if residual_correction:
            model.residual_limit_ = RESIDUAL_LIMIT
        valid = data[data.date.gt(cutoff) & data.date.le(end)]
        pred = forecast(model, data, valid, cutoff)
        scored = valid.merge(pred, on=KEYS, validate="one_to_one")
        reference = baseline[(baseline.cutoff == cutoff) & (baseline.weeks == 0)
                             & (baseline.statistic == "mean")].iloc[0]
        score = metrics(scored.boardings, scored.prediction)
        rows.append(dict(cutoff=cutoff, end=end, **score,
                         baseline_mean_all=reference.wape_score,
                         delta=score["wape_score"] - reference.wape_score))
        for dimension in ["route", "hour", "month"]:
            scored["month"] = scored.date.dt.strftime("%Y-%m")
            for value, group in scored.groupby(dimension):
                parts.append(dict(cutoff=cutoff, dimension=dimension, value=value,
                                  **metrics(group.boardings, group.prediction)))
        predictions.append(scored[KEYS + ["boardings", "prediction"]].assign(cutoff=cutoff))
        path = OUT / f"{model_prefix}_{cutoff}.pkl"
        path.write_bytes(pickle.dumps(model))
        # Only our just-written artifact is loaded; verify persistence preserves predictions.
        loaded = pickle.loads(path.read_bytes())
        pd.testing.assert_frame_equal(pred, forecast(loaded, data, valid, cutoff))
        folds.append(dict(cutoff=cutoff, end=end, training_rows=len(x), origins=origins,
                          seconds=time.monotonic() - begin, artifact=path.name))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / f"{prefix}_metrics.csv", sep=";", index=False)
    pd.DataFrame(parts).to_csv(OUT / f"{prefix}_breakdown.csv", sep=";", index=False)
    pd.concat(predictions).to_csv(OUT / f"{prefix}_predictions.csv", sep=";", index=False)
    metadata = dict(model="HistGradientBoostingRegressor", parameters=PARAMETERS,
        excluded_origins=["2025-01-31"] if exclude_january_origin else [],
        target="boardings - mean_all" if residual_correction else "boardings",
        residual_limit=RESIDUAL_LIMIT if residual_correction else None,
        features=selected, folds=folds, seconds=time.monotonic() - started, threads=4,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024),
        python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__,
        sklearn=sklearn.__version__, input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        missing_policy="same zero-filled grid as baseline; no imputation",
        rounding="floor(max(0, prediction)+0.5); route5 and hours1-4 forced zero")
    (OUT / f"{prefix}_run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-absolute-calendar", action="store_true",
                        help="Remove month, day_of_year and origin_day_of_year; preserve the original outputs")
    parser.add_argument("--residual-correction", action="store_true",
                        help="Learn residuals to mean_all with a fixed 20%% cap; implies --no-absolute-calendar")
    parser.add_argument("--exclude-january-origin", action="store_true",
                        help="Exclude only training origin 2025-01-31; requires --residual-correction")
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        run(args.no_absolute_calendar, args.residual_correction, args.exclude_january_origin)
