"""P65/030 inference from frozen prepared inputs. Built with PriorLabs-TabPFN.

Extracted from lab/experiments/portfolio_tabular_shape.py and portfolio_combine.py.
Parent forecasts, features and error basis are prepared offline; no model search here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from constants import ROUTES, KEYS

HOURS = [0] + list(range(5, 24))


def load_inputs(directory, spec):
    with np.load(directory / "inputs.npz", allow_pickle=False) as data:
        arrays = {name: data[name] for name in
                  ("X", "coordinates", "test", "mean", "components", "future_share", "active")}
    x, y, test = (arrays[n] for n in ("X", "coordinates", "test"))
    start = pd.Timestamp(spec["forecast_from"]).tz_convert("Europe/Moscow").tz_localize(None)
    end = pd.Timestamp(spec["forecast_to"]).tz_convert("Europe/Moscow").tz_localize(None)
    dates = pd.date_range(start, end, inclusive="left", freq="D")
    days = len(ROUTES) * len(dates)
    active = arrays["active"]
    rank = arrays["components"].shape[0]
    if (x.ndim != 2 or x.shape[1] != 50 or not 1 <= len(x) <= 10000
            or y.shape != (len(x), rank) or not 1 <= rank <= 6
            or arrays["components"].shape != (rank, 20) or arrays["mean"].shape != (20,)
            or active.dtype != np.bool_ or active.shape != (days,)
            or test.shape != (int(active.sum()), 50) or arrays["future_share"].shape != (days, 20)
            or any(not np.isfinite(a).all() for a in arrays.values())
            or (arrays["future_share"] < 0).any()):
        raise ValueError("Invalid prepared TabPFN input dimensions or values")
    base = pd.read_csv(directory / "base.csv", sep=";", parse_dates=["date"], float_precision="round_trip")
    grid = pd.MultiIndex.from_product([ROUTES, dates, range(24)], names=KEYS).to_frame(index=False)
    pd.testing.assert_frame_equal(base[KEYS], grid)
    if not np.isfinite(base.prediction).all() or base.prediction.lt(0).any():
        raise ValueError("Invalid parent forecast")
    if not base.loc[base.route.eq(5) | base.hour.between(1, 4), "prediction"].eq(0).all():
        raise ValueError("Parent structural zeros violated")
    day = base.groupby(["route", "date"]).prediction.sum()
    expected_active = (day.index.get_level_values("route") != 5) & day.gt(0).to_numpy()
    if not np.array_equal(active, expected_active):
        raise ValueError("Prepared active routes do not match the parent")
    return arrays, base


def compose(arrays, base, coordinates, weight):
    """Same normalization, fallback, 50% blend and half-up as submission 030."""
    if coordinates.shape != (int(arrays["active"].sum()), len(arrays["components"])) or not np.isfinite(coordinates).all():
        raise ValueError("Invalid TabPFN prediction")
    delta = np.zeros_like(arrays["future_share"])
    delta[arrays["active"]] = arrays["mean"] + coordinates @ arrays["components"]
    shares = np.maximum(0, arrays["future_share"] + delta)
    shape = base[KEYS].copy()
    shape["prediction"] = 0.
    shape.loc[shape.hour.isin(HOURS), "prediction"] = shares.reshape(-1)
    group = ["route", "date"]
    total = base.groupby(group).prediction.transform("sum")
    denominator = shape.groupby(group).prediction.transform("sum")
    share = shape.prediction.div(denominator.where(denominator.gt(0)))
    backup = base.prediction.div(total.where(total.gt(0))).fillna(0)
    learned = total * share.fillna(backup)
    result = base[KEYS].copy()
    raw = weight * learned + (1 - weight) * base.prediction
    rounded = np.floor(raw.clip(lower=0) + .5)
    if not np.isfinite(rounded).all() or (rounded >= float(2**63)).any():
        raise ValueError("Prediction exceeds int64")
    result["prediction"] = rounded.astype("int64")
    result.loc[result.route.eq(5) | result.hour.between(1, 4), "prediction"] = 0
    return result.to_csv(sep=";", index=False, date_format="%Y-%m-%d").encode()


def run(spec_path, output):
    spec_path = Path(spec_path)
    spec = json.loads(spec_path.read_text())
    weights = Path(spec["model_file"])
    # The legacy checkpoint is pickle-based: only load the exact audited file.
    if hashlib.file_digest(weights.open("rb"), "sha256").hexdigest() != spec["model_sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from tabpfn import TabPFNRegressor

    torch.set_num_threads(spec["threads"])
    torch.set_num_interop_threads(1)
    torch.manual_seed(spec["seed"])
    np.random.seed(spec["seed"])
    arrays, base = load_inputs(spec_path.parent, spec)
    predictions = np.empty((len(arrays["test"]), len(arrays["components"])))
    for i in range(predictions.shape[1]):
        model = TabPFNRegressor(n_estimators=spec["n_estimators"], categorical_features_indices=[0, 1, 2],
            model_path=str(weights), device=spec["device"], inference_precision=torch.float32,
            fit_mode="low_memory", memory_saving_mode=True, random_state=spec["seed"], n_jobs=spec["threads"],
            ignore_pretraining_limits=spec["device"] == "cpu")
        model.fit(arrays["X"], arrays["coordinates"][:, i])
        predictions[:, i] = model.predict(arrays["test"], output_type="median")
        del model
        if spec["device"] == "cuda":
            torch.cuda.empty_cache()
    Path(output).write_bytes(compose(arrays, base, predictions, spec["shape_weight"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.spec, args.output)
