"""Run CPU bundle checks: python -m ml.lab.cpu_student.test_model --bundle ..."""
import argparse
import json
from pathlib import Path
import resource
import shutil
import statistics
import sys
import tempfile
import time

import numpy as np
import pandas as pd

from ml.lab.cpu_student.model import date_start, features, grid, load_bundle, predict, read_history, rounded


def rejects(call):
    try:
        call()
    except (ValueError, OverflowError):
        return
    raise AssertionError("Invalid input was accepted")


def run(bundle, history_path):
    begin = time.perf_counter()
    model, metadata = load_bundle(bundle)
    history = read_history(history_path)
    loading = time.perf_counter()-begin
    for start in ["2025-01-01", "2025-01-29", "2025-05-17", "2025-09-15", "2025-11-01", "2025-12-31"]:
        result, info = predict(model, metadata, history, start)
        pd.testing.assert_frame_equal(result.drop(columns="prediction"), grid(start))
        assert len(result) == 14640 and result.prediction.dtype == np.dtype("int64")
        assert result.prediction.ge(0).all()
        assert result.loc[result.route.eq(5) | result.hour.between(1, 4), "prediction"].eq(0).all()
        if start == "2025-01-01":
            assert info["history_latest_used"] is None
            assert "no_past_observations_calendar_only_extrapolation" in info["warnings"]
        if start == "2025-12-31":
            assert result.date.max() == pd.Timestamp("2026-03-01")
            assert "horizon_outside_teacher_year_calendar_and_movement_unknown" in info["warnings"]
    start = "2025-09-15"
    expected, _ = predict(model, metadata, history, start)
    poisoned = history.copy()
    mask = poisoned.date.ge(start) | ~poisoned.working_events_observed
    poisoned.loc[mask, "boardings"] = 99999999
    poisoned.loc[poisoned.date.ge(start), "working_events_observed"] = False
    result, _ = predict(model, metadata, poisoned, start)
    pd.testing.assert_frame_equal(expected, result)
    keys, matrix = features(history, start, metadata["calendar"], metadata["movement"])
    future_poison = features(poisoned, start, metadata["calendar"], metadata["movement"])[1]
    pd.testing.assert_frame_equal(matrix, future_poison)
    times = []
    for _ in range(3):
        before = time.perf_counter()
        result, _ = predict(model, metadata, history, start)
        times.append(time.perf_counter()-before)
        pd.testing.assert_frame_equal(expected, result)
    reloaded, other_meta = load_bundle(bundle)
    pd.testing.assert_frame_equal(expected, predict(reloaded, other_meta, history, start)[0])
    for invalid in ["2024-12-31", "2026-01-01", "2025-02-30", "2025-01-01T00:00:00", "2025-1-1"]:
        rejects(lambda: date_start(invalid))
    rejects(lambda: rounded(keys, np.full(len(keys), np.inf)))
    rejects(lambda: rounded(keys, np.full(len(keys), float(2**63))))
    values = np.zeros(len(keys))
    at = keys.index[(keys.route==1)&(keys.hour==6)][0]
    values[at] = .5
    assert rounded(keys, values).prediction.iloc[at] == 1
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary)
        duplicate = pd.concat([history, history.iloc[:1]], ignore_index=True)
        duplicate.to_csv(path/"bad.csv", sep=";", index=False)
        rejects(lambda: read_history(path/"bad.csv"))
        shutil.copy(bundle/"metadata.json", path/"metadata.json")
        (path/"student.cbm").write_bytes(b"corrupt")
        rejects(lambda: load_bundle(path))
    assert not any(name in sys.modules for name in ["torch", "timesfm", "tabpfn"])
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps(dict(checks="passed", load_model_and_history_seconds=loading,
        features_predict_median_seconds=statistics.median(times),
        peak_rss_mib=rss/(1024**2 if sys.platform=="darwin" else 1024),
        boundary_dates_checked=6, gpu_libraries_loaded=False), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--history", type=Path, default=Path("ml/lab/artifacts/hourly_clean.csv"))
    args = parser.parse_args()
    run(args.bundle, args.history)
