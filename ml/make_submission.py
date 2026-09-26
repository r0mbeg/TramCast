"""Export the alternative median submission: python make_submission.py."""
import hashlib
import json
import platform
import resource
import time

import numpy as np
import pandas as pd

from pipeline import DATA, KEYS, OUT, ROOT, full_grid, metrics, predict


def main():
    started = time.monotonic()
    history_path = OUT / "hourly_clean.csv"
    history = pd.read_csv(history_path, sep=";", parse_dates=["date"])
    history = history[history.date.between("2025-01-01", "2025-10-31")].copy()
    expected_history = full_grid().to_frame(index=False)
    expected_history["date"] = pd.to_datetime(expected_history.date)
    pd.testing.assert_frame_equal(history[KEYS].sort_values(KEYS).reset_index(drop=True),
                                  expected_history)
    assert np.isfinite(history.boardings).all() and history.boardings.ge(0).all()

    # Reproduce the selected validation score before fitting the final profile.
    valid = history[history.date.gt("2025-08-31")]
    comparison = valid.merge(predict(history, valid, "2025-08-31", 0, "median"),
                            on=KEYS, validate="one_to_one")
    score = metrics(comparison.boardings, comparison.prediction)
    assert abs(score["wape_score"] - 0.8794502200500636) < 1e-12

    template = pd.read_csv(DATA / "test_submission.csv", sep=";", parse_dates=["date"])
    keys = template[KEYS]
    expected = full_grid("2025-11-01", "2025-12-31").to_frame(index=False)
    expected["date"] = pd.to_datetime(expected.date)
    pd.testing.assert_frame_equal(keys.sort_values(KEYS).reset_index(drop=True), expected)
    forecast = predict(history, keys, "2025-10-31", 0, "median")
    submission = keys.merge(forecast, on=KEYS, how="left", validate="one_to_one")
    path = OUT / "submission_median" / "submission.csv"
    path.parent.mkdir(exist_ok=True)
    submission.to_csv(path, sep=";", index=False, encoding="utf-8", date_format="%Y-%m-%d")
    exported = pd.read_csv(path, sep=";", parse_dates=["date"])
    pd.testing.assert_frame_equal(submission, exported)
    assert list(exported.columns) == KEYS + ["prediction"]
    assert len(exported) == 14640 and not exported.duplicated(KEYS).any()
    pd.testing.assert_frame_equal(exported[KEYS], keys)
    assert exported.prediction.dtype.kind in "iu" and exported.prediction.ge(0).all()
    assert np.isfinite(exported.prediction).all()
    assert len(exported[exported.route.eq(5)]) == 1464
    assert exported.loc[exported.route.eq(5) | exported.hour.between(1, 4), "prediction"].eq(0).all()
    history.assign(weekday=history.date.dt.dayofweek).groupby(
        ["route", "weekday", "hour"]).boardings.median().rename("prediction").to_csv(
            OUT / "submission_median_profile.csv", sep=";")
    run = dict(model="median_all", features=["route", "weekday", "hour"], weeks=0,
               selection="best cleaned September–October WAPE-score among tried models",
               validation=score, history_start="2025-01-01", cutoff="2025-10-31",
               forecast_start="2025-11-01", forecast_end="2025-12-31", seed=None,
               missing_policy="absent counts = 0; no imputation",
               postprocessing="floor(max(0, x) + 0.5); route 5 and hours 1–4 forced to 0",
               history_sha256=hashlib.sha256(history_path.read_bytes()).hexdigest(),
               submission_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
               rows=len(exported), prediction_total=int(exported.prediction.sum()),
               python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__,
               seconds=time.monotonic() - started,
               peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
               (1 if platform.system() == "Darwin" else 1024))
    (path.parent / "submission_run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
