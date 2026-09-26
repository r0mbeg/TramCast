"""Fixed CatBoost/LightGBM/XGBoost comparison: python -m experiments.boost_experiment."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import platform
import resource
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.chronos_experiment import WINDOWS, evaluate
from model import ABSOLUTE_CALENDAR, FEATURES, features, training_examples
from pipeline import KEYS, OUT, ROUTES, full_grid, metrics, postprocess, predict

DEST = OUT / "boost_comparison"
FEATURE_SETS = {"full": FEATURES,
                "no_absolute_calendar": [c for c in FEATURES if c not in ABSOLUTE_CALENDAR]}
PARAMETERS = {
    "catboost": dict(loss_function="MAE", iterations=500, depth=6, learning_rate=0.05,
                     l2_leaf_reg=5, random_seed=42, thread_count=4, verbose=False,
                     allow_writing_files=False, cat_features=["route", "weekday"]),
    "lightgbm": dict(objective="regression_l1", n_estimators=500, num_leaves=31,
                     learning_rate=0.05, min_child_samples=50, reg_lambda=5,
                     random_state=42, n_jobs=4, deterministic=True, force_col_wise=True,
                     verbosity=-1),
    "xgboost": dict(objective="reg:absoluteerror", n_estimators=500, max_depth=6,
                    learning_rate=0.05, min_child_weight=50, reg_lambda=5,
                    random_state=42, n_jobs=4, tree_method="hist", enable_categorical=True),
}


def matrix(frame, selected, algorithm):
    result = frame[selected].copy()
    if not np.isfinite(result).all().all():
        raise ValueError("Non-finite model inputs")
    if algorithm != "catboost":
        # Fixed domains preserve category codes between training and inference.
        for column, categories in [("route", ROUTES), ("weekday", list(range(7)))]:
            if not result[column].isin(categories).all():
                raise ValueError(f"Unknown {column}")
            result[column] = pd.Categorical(result[column], categories=categories)
    return result


def estimator(algorithm, rounds=500):
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    params = PARAMETERS[algorithm].copy()
    params["iterations" if algorithm == "catboost" else "n_estimators"] = rounds
    return {"catboost": CatBoostRegressor, "lightgbm": LGBMRegressor,
            "xgboost": XGBRegressor}[algorithm](**params)


def forecast(model, history, keys, cutoff, selected, algorithm):
    frame = features(history, keys, cutoff)
    active = frame.route.ne(5) & ~frame.hour.between(1, 4)
    frame["prediction"] = 0.0
    frame.loc[active, "prediction"] = model.predict(matrix(frame.loc[active], selected, algorithm))
    return postprocess(frame)


def run():
    started = time.monotonic()
    DEST.mkdir(exist_ok=True)
    source = OUT / "hourly_clean.csv"
    data = pd.read_csv(source, sep=";", parse_dates=["date"])
    expected = full_grid().to_frame(index=False)
    expected["date"] = pd.to_datetime(expected.date)
    pd.testing.assert_frame_equal(data[KEYS].sort_values(KEYS).reset_index(drop=True), expected)
    if not np.isfinite(data.boardings).all() or data.boardings.lt(0).any():
        raise ValueError("Invalid cleaned target")
    rows, details, predictions, fits = [], [], [], []
    for cutoff, end in WINDOWS:
        x, y, origins = training_examples(data, cutoff)
        keys = data.loc[data.date.gt(cutoff) & data.date.le(end), KEYS]
        configurations = [("mean_all", None, None)] + [
            (f"{algorithm}_{feature_set}", algorithm, feature_set)
            for algorithm in PARAMETERS for feature_set in FEATURE_SETS]
        for method, algorithm, feature_set in configurations:
            t0 = time.monotonic()
            if algorithm is None:
                result = predict(data, keys, cutoff, 0, "mean")
            else:
                selected = FEATURE_SETS[feature_set]
                model = estimator(algorithm)
                model.fit(matrix(x, selected, algorithm), y)
                result = forecast(model, data, keys, cutoff, selected, algorithm)
                path = DEST / f"{method}_{cutoff}.pkl"
                path.write_bytes(pickle.dumps(model))
                # Only a locally just-written artifact is deserialized.
                loaded = pickle.loads(path.read_bytes())
                pd.testing.assert_frame_equal(result, forecast(loaded, data, keys, cutoff, selected, algorithm))
                fits.append(dict(method=method, cutoff=cutoff, training_rows=len(x), origins=origins,
                                 features=selected, parameters=model.get_params(), artifact=path.name))
            assert result.loc[result.route.eq(5) | result.hour.between(1, 4), "prediction"].eq(0).all()
            row, detail, prediction = evaluate(data, result, cutoff, end, method, time.monotonic() - t0)
            rows.append(row)
            details.extend(detail)
            predictions.append(prediction)
            pd.DataFrame(rows).to_csv(DEST / "metrics.csv", sep=";", index=False)
            print(json.dumps(row), flush=True)
    export = pd.concat(predictions, ignore_index=True)
    path = DEST / "predictions.csv"
    export.to_csv(path, sep=";", index=False)
    pd.testing.assert_frame_equal(export, pd.read_csv(path, sep=";", parse_dates=["date"]))
    pd.DataFrame(details).to_csv(DEST / "breakdown.csv", sep=";", index=False)
    results = pd.DataFrame(rows)
    reference = pd.read_csv(OUT / "baseline_metrics.csv", sep=";")
    columns = ["cutoff", "wape_score", "absolute_error", "actual_total", "bias"]
    pd.testing.assert_frame_equal(results.loc[results.method.eq("mean_all"), columns].reset_index(drop=True),
        reference.loc[reference.weeks.eq(0) & reference.statistic.eq("mean"), columns].reset_index(drop=True))
    pooled = [dict(method=method, **metrics(g.boardings, g.prediction)) for method, g in export.groupby("method")]
    pd.DataFrame(pooled).to_csv(DEST / "pooled_metrics.csv", sep=";", index=False)
    summary = results.pivot(index="method", columns="cutoff", values="wape_score")
    summary["mean_window_score"] = summary.mean(axis=1)
    summary.to_csv(DEST / "summary.csv", sep=";")
    run_info = dict(fits=fits, windows=WINDOWS, feature_sets=FEATURE_SETS, seed=42, threads=4,
        early_stopping=False, validation_used_for_training=False,
        selection="Fixed six configurations; no tuning on either evaluation window",
        missing_policy="absent counts = 0; no imputation; validation unchanged",
        postprocessing="floor(max(0,p)+0.5); route5 and hours1-4 forced0",
        versions={n: importlib.metadata.version(n) for n in
                  ["numpy", "pandas", "scikit-learn", "scipy", "catboost", "lightgbm", "xgboost", "threadpoolctl"]},
        python=platform.python_version(), platform=platform.platform(),
        sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [source, Path(__file__), Path("model.py"), Path("pipeline.py"), Path("experiments/chronos_experiment.py")]},
        seconds=time.monotonic() - started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024))
    (DEST / "run.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        run()
