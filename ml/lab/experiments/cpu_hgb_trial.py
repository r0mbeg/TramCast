"""Local CPU-only HGB trial; run from any directory with --output NEW_DIRECTORY."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import signal
import sys
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.portfolio_experiment import (
    FINAL, WINDOWS, cpu_forecast, forecast_keys, load_history, write_json,
)
from pipeline import KEYS, metrics, postprocess

LAB = Path(__file__).resolve().parents[1]
SELECTION = LAB / "artifacts/portfolio_20260926/cpu/direct/selection.json"
CONTROL = LAB / "artifacts/portfolio_20260926/continuation/tabular_shape/study/tabular_shape/selected"


def forecast(history, cutoff, end, params, family="direct"):
    # Missing counts are excluded, never used as zero-demand training labels.
    train = history.loc[history.date.le(cutoff) & history.working_events_observed].copy()
    raw = cpu_forecast(train, cutoff, end, family, params)
    result = postprocess(raw).sort_values(KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(result[KEYS], forecast_keys(cutoff, end))
    assert len(result) == 14640 and result.prediction.ge(0).all()
    assert result.loc[result.route.eq(5) | result.hour.between(1, 4), "prediction"].eq(0).all()
    return result


def score(frame):
    result = metrics(frame.boardings, frame.prediction)
    result.update(rows=len(frame), mae=result["absolute_error"] / len(frame),
                  wape=result["absolute_error"] / result["actual_total"] if result["actual_total"] else None)
    return result


def run(output):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    history_path = LAB / "artifacts/hourly_clean.csv"
    params = json.loads(SELECTION.read_text())["params"]
    paths = [history_path, SELECTION, Path(__file__), LAB / "pipeline.py",
             LAB / "constants.py", LAB / "experiments/portfolio_experiment.py"]
    paths += [CONTROL / f"raw_{cutoff}.csv" for cutoff, _ in WINDOWS]
    hashes = {str(p.relative_to(LAB)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    write_json(output / "protocol.json", dict(
        hypothesis="Existing direct hourly HGB can forecast the full horizon on CPU without foundation models",
        params=params, windows=WINDOWS, final=FINAL, seed=42, threads=2,
        budget_seconds=600, search_trials=0, controls=["route-weekday-hour mean", "saved 030"],
        features=["route", "hour", "weekday", "days since 2025-01-01"],
        training="Only working_events_observed before cutoff; last 112 days for HGB",
        evaluation="Observed active cells only; identical mask for all candidates; half-up first",
        caveats=["Previously viewed overlapping windows, not independent validation",
                 "Presence of events does not prove complete counting",
                 "030 uses retrospective external data; HGB uses no external data",
                 "Raw CSV aggregation excluded from timing; frozen cleaned history reused"],
        sources_sha256=hashes,
        environment=dict(platform=platform.platform(), processor=platform.processor(),
                         logical_cpus=os.cpu_count(), python=platform.python_version(),
                         versions={n: importlib.metadata.version(n) for n in ["numpy", "pandas", "scikit-learn", "threadpoolctl"]})))
    history = load_history(history_path)
    rows, breakdown, timings = [], [], []
    reference = None
    with threadpool_limits(limits=2):
        for cutoff, end in WINDOWS + [FINAL]:
            before = time.perf_counter()
            predicted = forecast(history, cutoff, end, params)
            elapsed = time.perf_counter() - before
            timings.append(dict(cutoff=cutoff, prepare_fit_predict_seconds=elapsed))
            predicted.to_csv(output / f"hgb_{cutoff}.csv", sep=";", index=False, date_format="%Y-%m-%d")
            print(f"HGB {cutoff}: {elapsed:.3f}s, {len(predicted)} rows", flush=True)
            if (cutoff, end) == FINAL:
                continue
            baseline = forecast(history, cutoff, end, dict(weeks=0, statistic="mean"), "seasonal")
            saved = postprocess(pd.read_csv(CONTROL / f"raw_{cutoff}.csv", sep=";", parse_dates=["date"]))
            pd.testing.assert_frame_equal(saved[KEYS], forecast_keys(cutoff, end))
            truth = history.loc[history.date.gt(cutoff) & history.date.le(end)]
            observed = truth.loc[truth.working_events_observed & truth.route.ne(5) & ~truth.hour.between(1, 4)]
            for name, candidate in [("hgb", predicted), ("mean", baseline), ("030", saved)]:
                compared = observed.merge(candidate, on=KEYS, validate="one_to_one")
                assert len(compared) == len(observed)
                rows.append(dict(model=name, cutoff=cutoff, end=end, coverage=len(observed)/(9*61*20), **score(compared)))
                compared["horizon_week"] = ((compared.date-pd.Timestamp(cutoff)).dt.days-1)//7+1
                for dimension in ["route", "hour", "horizon_week"]:
                    for value, group in compared.groupby(dimension):
                        breakdown.append(dict(model=name, cutoff=cutoff, dimension=dimension, value=int(value), **score(group)))
            if cutoff == "2025-08-31":
                reference = predicted
        # Runnable leakage check: both future targets and future observation flags change.
        poisoned = history.copy()
        future = poisoned.date.gt("2025-08-31")
        poisoned.loc[future, "boardings"] = 99999999
        poisoned.loc[future, "working_events_observed"] = False
        pd.testing.assert_frame_equal(reference, forecast(poisoned, "2025-08-31", "2025-10-31", params))
    assert "torch" not in sys.modules and "timesfm" not in sys.modules and "tabpfn" not in sys.modules
    for p in paths:
        assert hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p.relative_to(LAB))]
    pd.DataFrame(rows).to_csv(output / "metrics.csv", sep=";", index=False)
    pd.DataFrame(breakdown).to_csv(output / "breakdown.csv", sep=";", index=False)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(output / "completed.json", dict(
        elapsed_seconds=time.perf_counter()-started,
        peak_rss_mib=rss/(1024**2 if sys.platform == "darwin" else 1024),
        timings=timings, future_poison_check="passed", grid_checks="passed",
        input_hashes_unchanged=True, gpu_libraries_loaded=False))
    print(pd.DataFrame(rows)[["model", "cutoff", "wape_score", "mae"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    signal.alarm(600)
    run(args.output.resolve())
