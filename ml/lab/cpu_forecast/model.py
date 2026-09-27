"""CPU direct 61-day forecast trained on observed counts, not teacher predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd

from ml.lab.cpu_student.model import features, grid, read_history, rounded


def reference(matrix, scales):
    return matrix.profile_all.fillna(matrix.route_hour).fillna(matrix.route.map(scales)).clip(lower=1).to_numpy()


def publish(keys, matrix, values):
    values = np.asarray(values, float).copy()
    # Existing retrospective movement scenario: route 50 does not run on these autumn weekends.
    values[keys.route.eq(50) & matrix.movement_autumn.eq(1)] = 0
    return rounded(keys, values)


def load_bundle(path):
    path = Path(path)
    metadata = json.loads((path/"metadata.json").read_text())
    if hashlib.sha256((path/"forecast.cbm").read_bytes()).hexdigest() != metadata["model_sha256"]:
        raise ValueError("Model checksum mismatch")
    model = CatBoostRegressor()
    model.load_model(str(path/"forecast.cbm"))
    if model.feature_names_ != metadata["features"]:
        raise ValueError("Feature contract mismatch")
    return model, metadata


def predict(model, metadata, history, start):
    keys, matrix = features(history, start, metadata["calendar"], metadata["movement"])
    if matrix.columns.tolist() != metadata["features"]:
        raise ValueError("Feature contract mismatch")
    base = reference(matrix, metadata["route_scales"])
    ratio = model.predict(matrix, task_type="CPU", thread_count=2)
    weight = metadata["model_weight"]
    result = publish(keys, matrix, base*(weight*ratio + 1-weight))
    begin = pd.Timestamp(start)
    past = history.loc[history.date.lt(begin) & history.working_events_observed & history.route.ne(5) & ~history.hour.between(1,4)]
    latest = past.date.max()
    warnings = []
    if begin <= pd.Timestamp(metadata["training_target_end"]):
        warnings.append("retrospective_request_weights_trained_through_"+metadata["training_target_end"])
    if past.empty:
        warnings.append("no_past_history_unvalidated_cold_start")
    elif (begin-latest).days > 1:
        warnings.append("stale_history")
    if matrix.loc[keys.route.ne(5), "history_days"].lt(28).any():
        warnings.append("less_than_28_observed_days_for_some_routes")
    if result.date.max().year != 2025:
        warnings.append("2026_extrapolation_calendar_and_movement_unknown")
    info = dict(model_version=metadata["model_version"], dataset_version=metadata.get("dataset_version"),
        timezone="Europe/Moscow", from_date=start, to_exclusive=str((begin+pd.Timedelta(days=61)).date()),
        rows=len(result), training_target_end=metadata["training_target_end"],
        history_latest_used=None if pd.isna(latest) else str(latest.date()), warnings=warnings,
        status="experimental_observation_model", route_5_reason="organizer_required_zero_no_history",
        external_regime="retrospective_2025_calendar_and_movement")
    return result, info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    model, metadata = load_bundle(args.bundle)
    forecast, info = predict(model, metadata, read_history(args.history), args.start)
    info["load_features_predict_seconds"] = time.perf_counter()-started
    args.output.mkdir(parents=True, exist_ok=False)
    forecast.to_csv(args.output/"forecast.csv", sep=";", index=False, date_format="%Y-%m-%d")
    (args.output/"forecast.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
