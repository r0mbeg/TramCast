"""CPU-only 030 student. Run from the repository root with python -m ml.lab.cpu_student.model."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd

from ml.runtime.constants import KEYS, ROUTES

HISTORY_FEATURES = ["profile_all", "profile_112", "profile_28", "route_hour", "route_level", "history_age", "history_days"]
CATEGORICAL = ["route", "weekday"]


def read_history(path):
    data = pd.read_csv(path, sep=";", usecols=KEYS + ["boardings", "working_events_observed"], parse_dates=["date"])
    if data.empty or data.date.isna().any() or not data.date.eq(data.date.dt.normalize()).all():
        raise ValueError("History must contain dated calendar-hour observations")
    if data.duplicated(KEYS).any() or not data.route.isin(ROUTES).all() or not data.hour.isin(range(24)).all():
        raise ValueError("Invalid or duplicate history keys")
    if not np.isfinite(data.boardings).all() or data.boardings.lt(0).any() or not data.boardings.eq(np.floor(data.boardings)).all():
        raise ValueError("History requires nonnegative integer observed counts")
    if data.working_events_observed.dtype != bool:
        raise ValueError("Observation mask must contain True/False")
    if not data.loc[data.route.eq(5) | data.hour.between(1, 4), "boardings"].eq(0).all():
        raise ValueError("History violates structural zeros")
    return data


def date_start(value):
    if not isinstance(value, str) or not re.fullmatch(r"2025-\d{2}-\d{2}", value):
        raise ValueError("Start must be a YYYY-MM-DD date in 2025")
    return pd.Timestamp(value)


def grid(start):
    start = date_start(start)
    return pd.MultiIndex.from_product([ROUTES, pd.date_range(start, periods=61), range(24)], names=KEYS).to_frame(index=False)


def features(history, start, calendar, movement):
    keys = grid(start)
    begin = pd.Timestamp(start)
    frame = keys.copy()
    frame["weekday"] = frame.date.dt.dayofweek
    frame["month"] = frame.date.dt.month
    frame["day_of_year"] = frame.date.dt.dayofyear
    frame["horizon"] = (frame.date - begin).dt.days + 1
    frame["origin_day"] = begin.dayofyear
    frame["target_year"] = frame.date.dt.year
    frame["is_workday"] = frame.weekday.lt(5).astype(float)
    frame.loc[frame.date.isin(pd.to_datetime(calendar["holidays"] + list(calendar["transfers"].values()))), "is_workday"] = 0.
    frame.loc[frame.date.isin(pd.to_datetime(calendar["working_weekends"])), "is_workday"] = 1.
    covered = frame.date.between(*calendar["coverage"])
    frame.loc[~covered, "is_workday"] = np.nan
    for name in ["april17", "july_verified", "august7", "autumn"]:
        flag = pd.Series(False, index=frame.index)
        for event in movement["events"][name]:
            selected = frame.route.isin(event["routes"]) & frame.date.between(event["start"], event["end"])
            if "weekdays" in event:
                selected &= frame.weekday.isin(event["weekdays"])
            flag |= selected
        frame[f"movement_{name}"] = flag.astype(float)
        frame.loc[~frame.date.between(*movement["coverage"]), f"movement_{name}"] = np.nan
    # Only pre-start observations contribute, including when the input contains later facts.
    past = history.loc[history.date.lt(begin) & history.working_events_observed & history.route.ne(5) & ~history.hour.between(1, 4)].copy()
    past["weekday"] = past.date.dt.dayofweek
    lookup = pd.MultiIndex.from_frame(frame[["route", "weekday", "hour"]])
    for name, days in [("profile_all", None), ("profile_112", 112), ("profile_28", 28)]:
        pool = past if days is None else past.loc[past.date.ge(begin-pd.Timedelta(days=days))]
        profile = pool.groupby(["route", "weekday", "hour"]).boardings.mean()
        frame[name] = profile.reindex(lookup).to_numpy()
    profile = past.groupby(["route", "hour"]).boardings.mean()
    frame["route_hour"] = profile.reindex(pd.MultiIndex.from_frame(frame[["route", "hour"]])).to_numpy()
    frame["route_level"] = frame.route.map(past.groupby("route").boardings.mean())
    last = frame.route.map(past.groupby("route").date.max())
    frame["history_age"] = (begin-last).dt.days
    frame["history_days"] = frame.route.map(past.groupby("route").date.nunique()).fillna(0)
    frame = frame.drop(columns="date")
    for name in CATEGORICAL:
        frame[name] = frame[name].astype(str)
    if np.isinf(frame.drop(columns=CATEGORICAL).to_numpy(float)).any():
        raise ValueError("Infinite model features")
    return keys, frame


def rounded(keys, values):
    values = np.asarray(values, dtype=float)
    if values.shape != (len(keys),) or not np.isfinite(values).all() or (values >= float(2**63)-2048).any():
        raise ValueError("Invalid or out-of-range model output")
    result = keys.copy()
    result["prediction"] = np.floor(np.maximum(values, 0)+0.5).astype("int64")
    result.loc[result.route.eq(5) | result.hour.between(1, 4), "prediction"] = 0
    return result


def load_bundle(path):
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text())
    model_path = path / "student.cbm"
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != metadata["model_sha256"]:
        raise ValueError("Model checksum mismatch")
    model = CatBoostRegressor()
    model.load_model(str(model_path))
    if model.feature_names_ != metadata["features"]:
        raise ValueError("Model feature contract mismatch")
    return model, metadata


def predict(model, metadata, history, start):
    keys, matrix = features(history, start, metadata["calendar"], metadata["movement"])
    if matrix.columns.tolist() != metadata["features"]:
        raise ValueError("Feature contract mismatch")
    scales = keys.route.map({int(k): v for k, v in metadata["route_scales"].items()}).to_numpy()
    values = model.predict(matrix, thread_count=2, task_type="CPU") * scales
    result = rounded(keys, values)
    begin = pd.Timestamp(start)
    past = history.loc[history.date.lt(begin) & history.working_events_observed & history.route.ne(5)]
    latest = past.date.max()
    warnings = ["fixed_year_distillation_not_historical_as_of_validation"]
    if begin < pd.Timestamp(metadata["teacher_start_range"][0]) or begin > pd.Timestamp(metadata["teacher_start_range"][1]):
        warnings.append("start_outside_teacher_examples")
    if result.date.max().year != 2025:
        warnings.append("horizon_outside_teacher_year_calendar_and_movement_unknown")
    if past.empty:
        warnings.append("no_past_observations_calendar_only_extrapolation")
    elif (begin-latest).days > 1:
        warnings.append("stale_history")
    if matrix.loc[keys.route.ne(5), "history_days"].lt(28).any():
        warnings.append("less_than_28_observed_days_for_some_routes")
    info = dict(model_version=metadata["model_version"], timezone="Europe/Moscow", from_date=start,
                to_exclusive=str((begin+pd.Timedelta(days=61)).date()), rows=len(result),
                history_latest_used=None if pd.isna(latest) else str(latest.date()),
                teacher_start_range=metadata["teacher_start_range"], warnings=warnings,
                status="research_surrogate", route_5_reason="organizer_required_zero_no_history")
    return result, info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    started = time.perf_counter()
    date_start(args.start)
    model, metadata = load_bundle(args.bundle)
    prediction, info = predict(model, metadata, read_history(args.history), args.start)
    info["load_history_features_predict_seconds"] = time.perf_counter()-started
    args.output.mkdir(parents=True, exist_ok=False)
    prediction.to_csv(args.output / "forecast.csv", sep=";", index=False, date_format="%Y-%m-%d")
    (args.output / "forecast.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(json.dumps(info, indent=2, ensure_ascii=False))
