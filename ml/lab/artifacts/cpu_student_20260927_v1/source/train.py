"""Bounded distillation from saved 030 forecasts; never calls a GPU model."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import signal
import sys
import time

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd

from ml.lab.cpu_student.model import CATEGORICAL, HISTORY_FEATURES, features, grid, predict, read_history, rounded
from ml.runtime.constants import KEYS

LAB = Path(__file__).resolve().parents[1]
SELECTED = LAB / "artifacts/portfolio_20260926/continuation/tabular_shape/study/tabular_shape/selected"
EXTRA = LAB / "artifacts/selection_audit_20260927/new/030"
ORIGINS = ["2025-04-30", "2025-05-31", "2025-06-30", "2025-07-31", "2025-08-14", "2025-08-31", "2025-10-31"]
CANDIDATES = [dict(depth=6, iterations=500), dict(depth=8, iterations=700)]


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(y, prediction):
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    if len(y) == 0 or y.shape != prediction.shape or not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError("Invalid evaluation vectors")
    error = np.abs(y-prediction)
    wape = float(error.sum()/y.sum()) if y.sum() else None
    return dict(rows=len(y), mae=float(error.mean()), wape=wape,
                score=None if wape is None else max(0., 1-wape), bias=float((prediction-y).sum()))


def fit(examples, params):
    data = pd.concat(examples, ignore_index=True)
    scales = data.groupby("route").teacher.median().clip(lower=1).to_dict()
    matrix = data.drop(columns="teacher")
    scale = matrix.route.map(scales).to_numpy()
    # ponytail: 10% history-free weight permits cold-start requests; early-year quality needs teacher examples.
    cold = matrix.copy()
    cold[HISTORY_FEATURES] = np.nan
    cold["history_days"] = 0.
    cold["history_age"] = np.nan
    weights = scale/scale.mean()
    model = CatBoostRegressor(**params, loss_function="MAE", learning_rate=.06,
        random_seed=42, thread_count=2, task_type="CPU", allow_writing_files=False, verbose=False)
    model.fit(pd.concat([matrix, cold], ignore_index=True), np.tile(data.teacher.to_numpy()/scale, 2),
              cat_features=CATEGORICAL, sample_weight=np.concatenate([weights*.9, weights*.1]))
    return model, {**scales, "5": 1.}


def run(out):
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    history_path = LAB / "artifacts/hourly_clean.csv"
    calendar_path = LAB / "artifacts/calendar_sources.json"
    movement_path = LAB.parent / "preparation/movement_calendar.json"
    teacher_paths = {o: (EXTRA if o in ["2025-05-31", "2025-08-14"] else SELECTED) / f"raw_{o}.csv" for o in ORIGINS}
    source_paths = [history_path, calendar_path, movement_path, Path(__file__), Path(__file__).with_name("model.py"),
                    LAB.parent / "runtime/constants.py", *teacher_paths.values()]
    source_hashes = {str(p.relative_to(LAB.parent.parent)): digest(p) for p in source_paths}
    save_json(out / "protocol.json", dict(
        hypothesis="Compact CPU CatBoost reproduces saved 030 on held-out forecast origins with CPU-only features",
        train_origins=ORIGINS[:3], selection_origin=ORIGINS[3], test_origins=ORIGINS[4:],
        final_refit_origins=ORIGINS, candidates=CANDIDATES, target="Published rounded 030 counts, route-scaled MAE",
        selection="Lowest teacher WAPE on selection origin; no actual target based selection",
        common_params=dict(loss="MAE", learning_rate=.06, seed=42, threads=2, cold_start_weight=.1),
        budget_seconds=1800, gpu=False, future_facts_in_features=False,
        caveats=["Seven origins only; historical teacher recipe selected with hindsight",
                 "Final all-origin refit is a retrospective fixed-year surrogate, not an as-of backtest",
                 "January-April and 2026 have no teacher labels; cold-start extrapolation is unvalidated",
                 "Calendar and movement snapshots are retrospective external scenarios",
                 "Teacher input contexts are fixed; response to arbitrary changed history is unvalidated"],
        versions={n: importlib.metadata.version(n) for n in ["catboost", "numpy", "pandas"]},
        platform=platform.platform(), python=platform.python_version(), sources_sha256=source_hashes))
    history = read_history(history_path)
    if digest(history_path) != "7031c686c149fcb552711df49d82bab23540d8c80ffa6f140c6d0444138384ae":
        raise ValueError("Expected the frozen cleaned history")
    calendar = json.loads(calendar_path.read_text())
    movement = json.loads(movement_path.read_text())
    examples, teachers, frames = {}, {}, {}
    for origin, path in teacher_paths.items():
        start = str((pd.Timestamp(origin)+pd.Timedelta(days=1)).date())
        keys, matrix = features(history, start, calendar, movement)
        raw = pd.read_csv(path, sep=";", parse_dates=["date"], float_precision="round_trip").sort_values(KEYS).reset_index(drop=True)
        pd.testing.assert_frame_equal(raw[KEYS], keys)
        if not np.isfinite(raw.prediction).all() or raw.prediction.lt(0).any():
            raise ValueError("Invalid teacher values")
        forced = keys.route.eq(5) | keys.hour.between(1, 4)
        if not raw.loc[forced, "prediction"].eq(0).all():
            raise ValueError("Teacher structural zeros differ")
        teacher = rounded(keys, raw.prediction)
        teachers[origin], frames[origin] = teacher, matrix
        examples[origin] = matrix.loc[~forced].assign(teacher=teacher.loc[~forced, "prediction"].to_numpy())
    # Final teacher must be precisely the selected, published 030 rather than an unmixed correction.
    published = pd.read_csv(LAB.parent / "bundles/030/forecast.csv", sep=";", parse_dates=["date"])
    pd.testing.assert_frame_equal(teachers[ORIGINS[-1]], published.sort_values(KEYS).reset_index(drop=True))
    common = dict(calendar=calendar, movement=movement, features=frames[ORIGINS[0]].columns.tolist(),
                  teacher_start_range=["2025-05-01", "2025-11-01"], model_version="evaluation-only")
    selection = []
    selected_model = selected_meta = None
    best = float("inf")
    for index, params in enumerate(CANDIDATES):
        before = time.perf_counter()
        model, scales = fit([examples[o] for o in ORIGINS[:3]], params)
        meta = dict(common, route_scales=scales)
        result, _ = predict(model, meta, history, "2025-08-01")
        metric = measure(teachers[ORIGINS[3]].prediction, result.prediction)
        selection.append(dict(candidate=index, params=params, seconds=time.perf_counter()-before, **metric))
        print(f"Candidate {index}: teacher WAPE {metric['wape']:.5f}, {selection[-1]['seconds']:.2f}s", flush=True)
        if metric["wape"] < best:
            best, selected_model, selected_meta, selected_index = metric["wape"], model, meta, index
    save_json(out / "selection.json", dict(selected=selected_index, candidates=selection))
    rows, details = [], []
    for origin in ORIGINS[3:]:
        start = str((pd.Timestamp(origin)+pd.Timedelta(days=1)).date())
        before = time.perf_counter()
        result, _ = predict(selected_model, selected_meta, history, start)
        seconds = time.perf_counter()-before
        result.to_csv(out / f"heldout_{start}.csv", sep=";", index=False, date_format="%Y-%m-%d")
        teacher = teachers[origin]
        baseline = rounded(teacher[KEYS], frames[origin].profile_all.fillna(frames[origin].route_hour).fillna(0))
        split = "selection" if origin == ORIGINS[3] else "test"
        for name, candidate in [("student", result), ("mean", baseline)]:
            rows.append(dict(origin=origin, split=split, model=name, target="teacher", seconds=seconds if name=="student" else None,
                             **measure(teacher.prediction, candidate.prediction)))
        observed = history.loc[history.working_events_observed & history.route.ne(5) & ~history.hour.between(1, 4)]
        for name, candidate in [("student", result), ("teacher", teacher), ("mean", baseline)]:
            paired = observed.merge(candidate, on=KEYS, validate="one_to_one")
            if len(paired):
                rows.append(dict(origin=origin, split=split, model=name, target="observed_actual", seconds=None, **measure(paired.boardings, paired.prediction)))
        compared = teacher.rename(columns={"prediction":"teacher"}).merge(result, on=KEYS, validate="one_to_one")
        compared["horizon_week"] = ((compared.date-pd.Timestamp(start)).dt.days//7)+1
        for dimension in ["route", "hour", "horizon_week"]:
            for value, group in compared.groupby(dimension):
                details.append(dict(origin=origin, dimension=dimension, value=int(value), **measure(group.teacher, group.prediction)))
        print(f"Held-out {start}: {seconds:.3f}s, teacher WAPE {measure(teacher.prediction,result.prediction)['wape']:.5f}", flush=True)
    pd.DataFrame(rows).to_csv(out / "metrics.csv", sep=";", index=False)
    pd.DataFrame(details).to_csv(out / "breakdown.csv", sep=";", index=False)
    selected_model.save_model(str(out / "evaluation.cbm"))
    save_json(out / "evaluation_metadata.json", selected_meta)
    before = time.perf_counter()
    model, scales = fit([examples[o] for o in ORIGINS], CANDIDATES[selected_index])
    refit_seconds = time.perf_counter()-before
    bundle = out / "bundle"
    bundle.mkdir()
    model.save_model(str(bundle / "student.cbm"))
    sha = digest(bundle / "student.cbm")
    metadata = dict(common, route_scales=scales, model_sha256=sha, model_version=f"cpu-student-030-2025-{sha[:12]}",
                    params=CANDIDATES[selected_index], teacher_origins=ORIGINS, history_max_date="2025-10-31",
                    role="retrospective_year_surrogate", selected_on="2025-08-01",
                    heldout_metrics_apply_to="evaluation.cbm before final refit, not the final all-origin model",
                    sources_sha256=source_hashes, license_note="Distilled from 030. Built with PriorLabs-TabPFN; see ml/recipes/tabpfn-030/LICENSE.txt")
    save_json(bundle / "metadata.json", metadata)
    fidelity = []
    for origin in ORIGINS:
        start = str((pd.Timestamp(origin)+pd.Timedelta(days=1)).date())
        result, _ = predict(model, metadata, history, start)
        fidelity.append(dict(origin=origin, split="final_refit_training_fidelity_only", **measure(teachers[origin].prediction,result.prediction)))
    pd.DataFrame(fidelity).to_csv(out / "refit_fidelity.csv", sep=";", index=False)
    for path in source_paths:
        if digest(path) != source_hashes[str(path.relative_to(LAB.parent.parent))]:
            raise ValueError("Input or code changed during training")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    save_json(out / "completed.json", dict(elapsed_seconds=time.perf_counter()-started, refit_seconds=refit_seconds,
        peak_rss_mib=rss/(1024**2 if sys.platform=="darwin" else 1024), model_bytes=(bundle/"student.cbm").stat().st_size,
        gpu_used=False, sources_unchanged=True))
    print(f"Saved {bundle}; total {time.perf_counter()-started:.2f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    signal.alarm(1800)
    run(args.output.resolve())
