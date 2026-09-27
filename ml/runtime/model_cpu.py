"""Real CatBoost CPU inference from prepared features, without torch or lab."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from constants import KEYS, ROUTES


def run(spec_path, output):
    spec_path = Path(spec_path)
    spec = json.loads(spec_path.read_text())
    weights = Path(spec["model_file"])
    if hashlib.sha256(weights.read_bytes()).hexdigest() != spec["model_sha256"]:
        raise ValueError("CPU checkpoint checksum mismatch")
    frame = pd.read_csv(spec_path.parent/"features.csv", sep=";",
                        dtype={"route": str, "weekday": str, "route_hour_key": str},
                        parse_dates=["date"], float_precision="round_trip")
    if frame.columns.tolist() != spec["features"] + ["date", "reference"]:
        raise ValueError("CPU feature schema mismatch")
    keys = frame[KEYS].copy()
    keys["route"] = keys.route.astype(int)
    start = pd.Timestamp(spec["forecast_from"]).tz_convert("Europe/Moscow").tz_localize(None)
    end = pd.Timestamp(spec["forecast_to"]).tz_convert("Europe/Moscow").tz_localize(None)
    grid = pd.MultiIndex.from_product([ROUTES, pd.date_range(start, end, inclusive="left"), range(24)], names=KEYS).to_frame(index=False)
    pd.testing.assert_frame_equal(keys, grid)
    if not np.isfinite(frame.reference).all() or frame.reference.lt(1).any():
        raise ValueError("Invalid CPU reference scale")
    model = CatBoostRegressor(thread_count=spec["threads"])
    model.load_model(str(weights))
    if model.feature_names_ != spec["features"]:
        raise ValueError("Model feature contract mismatch")
    raw = np.asarray(model.predict(frame[spec["features"]], task_type="CPU",
                                  thread_count=spec["threads"])) * frame.reference.to_numpy()
    raw[keys.route.eq(50) & frame.movement_autumn.eq(1)] = 0
    if raw.shape != (len(keys),) or not np.isfinite(raw).all() or (raw >= float(2**63)-2048).any():
        raise ValueError("Invalid CPU model output")
    keys["prediction"] = np.floor(np.maximum(raw, 0)+.5).astype("int64")
    keys.loc[keys.route.eq(5) | keys.hour.between(1, 4), "prediction"] = 0
    keys.to_csv(output, sep=";", index=False, date_format="%Y-%m-%d")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.spec, args.output)
