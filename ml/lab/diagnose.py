"""Diagnose training-origin residuals and saved validation corrections: python diagnose.py."""
import hashlib
import json

import numpy as np
import pandas as pd

from model import features
from pipeline import KEYS, OUT, metrics, postprocess


def summary(frame):
    needed = frame.boardings - frame.baseline
    actual, base = float(frame.boardings.sum()), float(frame.baseline.sum())
    result = dict(rows=len(frame), actual_total=actual, baseline_total=base,
                  needed_correction=float(needed.sum()),
                  needed_percent_of_baseline=100 * needed.sum() / base if base else None,
                  positive_residual_share=float(needed.gt(0).mean()),
                  negative_residual_share=float(needed.lt(0).mean()))
    if "prediction" in frame:
        applied = frame.prediction - frame.baseline
        changed = needed.ne(0) & applied.ne(0)
        result.update(applied_correction=float(applied.sum()),
                      applied_percent_of_baseline=100 * applied.sum() / base if base else None,
                      direction_agreement=float((np.sign(needed[changed]) == np.sign(applied[changed])).mean()) if changed.any() else None,
                      compared_nonzero_hours=int(changed.sum()),
                      absolute_error_change=float((frame.boardings - frame.prediction).abs().sum() - needed.abs().sum()))
    return result


def run():
    sample = pd.DataFrame(dict(boardings=[80, 120], baseline=[100, 100], prediction=[110, 110]))
    check = summary(sample)
    assert check["needed_correction"] == 0 and check["applied_correction"] == 20
    assert check["direction_agreement"] == 0.5 and check["absolute_error_change"] == 0
    meta = json.loads((OUT / "ml_residual_run.json").read_text())
    source = OUT / "hourly_clean.csv"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == meta["input_sha256"]
    data = pd.read_csv(source, sep=";", parse_dates=["date"])
    predictions = pd.read_csv(OUT / "ml_residual_predictions.csv", sep=";", parse_dates=["date"])
    baseline_metrics = pd.read_csv(OUT / "baseline_metrics.csv", sep=";")
    totals, routes, coverage = [], [], []
    for fold in meta["folds"]:
        cutoff = fold["cutoff"]
        train_keys = []
        for origin in fold["origins"]:
            end = pd.Timestamp(origin) + pd.Timedelta(days=62)
            assert end <= pd.Timestamp(cutoff)
            target = data[data.date.gt(origin) & data.date.le(end)]
            target = target[target.route.ne(5) & ~target.hour.between(1, 4)]
            profile = features(data, target, origin)
            frame = target.merge(profile[KEYS + ["mean_all"]].rename(columns={"mean_all": "baseline"}),
                                 on=KEYS, validate="one_to_one")
            spec = dict(cutoff=cutoff, stage="training", origin=origin)
            totals.append(dict(spec, **summary(frame)))
            for route, group in frame.groupby("route"):
                routes.append(dict(spec, route=route, **summary(group)))
            train_keys.append(target[KEYS])
        occurrences = pd.concat(train_keys)
        assert len(occurrences) == fold["training_rows"]
        coverage.append(dict(cutoff=cutoff, origin_count=len(fold["origins"]),
                             training_rows=len(occurrences), unique_target_hours=len(occurrences.drop_duplicates()),
                             first_target=str(occurrences.date.min().date()),
                             last_target=str(occurrences.date.max().date())))
        valid = predictions[predictions.cutoff.eq(cutoff)]
        profile = features(data, valid, cutoff)
        base = postprocess(profile.assign(prediction=profile.mean_all)).rename(columns={"prediction": "baseline"})
        frame = valid.merge(base, on=KEYS, validate="one_to_one")
        reference = baseline_metrics[(baseline_metrics.cutoff == cutoff) & baseline_metrics.weeks.eq(0)
                                     & baseline_metrics.statistic.eq("mean")].iloc[0]
        assert abs(metrics(frame.boardings, frame.baseline)["wape_score"] - reference.wape_score) < 1e-12
        frame = frame[frame.route.ne(5) & ~frame.hour.between(1, 4)]
        spec = dict(cutoff=cutoff, stage="validation", origin=cutoff)
        totals.append(dict(spec, **summary(frame)))
        for route, group in frame.groupby("route"):
            routes.append(dict(spec, route=route, **summary(group)))
    pd.DataFrame(totals).to_csv(OUT / "diagnostic_origins.csv", sep=";", index=False)
    pd.DataFrame(routes).to_csv(OUT / "diagnostic_routes.csv", sep=";", index=False)
    pd.DataFrame(coverage).to_csv(OUT / "diagnostic_coverage.csv", sep=";", index=False)
    print(pd.DataFrame(totals).to_string(index=False))
    print(pd.DataFrame(coverage).to_string(index=False))


if __name__ == "__main__":
    run()
