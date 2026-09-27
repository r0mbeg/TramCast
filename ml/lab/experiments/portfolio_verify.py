"""Independently rescore saved forecasts; optionally archive the fixed seven finalists."""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from archive_submission import archive
from pipeline import KEYS, ROOT, full_grid

OUT = ROOT / "artifacts/portfolio_20260926"
FINALISTS = [
    ("031_adaptive_hourly_profile_errors", "continuation/adaptive_shape/study/adaptive_shape/selected"),
    ("029_school_calendar_route_fractions", "continuation/school_fraction/study/school_fraction/selected"),
    ("025_timesfm_volume_bayesian_shape", "continuation/bayes_shape_timesfm_blend/bayes_shape_timesfm_blend/selected"),
    ("024_bayesian_hourly_share_errors", "continuation/bayes_shape/study/bayes_shape/selected"),
    ("023_gaussian_process_errors", "continuation/gp_errors/gp_errors/selected"),
    ("026_timesfm_verified_operations", "continuation/timesfm_verified_july/timesfm_verified_july/selected"),
    ("005_movement_chronos", "combine/movement_half_chronos"),
]


def read(path):
    return pd.read_csv(path, sep=";", parse_dates=["date"], float_precision="round_trip")


def score(frame):
    # Independent implementation of the frozen metric, using integer published values.
    actual = int(frame.boardings.sum())
    absolute = int((frame.prediction - frame.boardings).abs().sum())
    bias = int(frame.prediction.sum()) - actual
    return dict(wape_score=max(0, 1 - absolute / actual) if actual else None,
                absolute_error=absolute, actual_total=actual, bias=bias,
                bias_fraction=bias / actual if actual else None)


def validate(frame, start, end, raw):
    expected = full_grid(start, end).to_frame(index=False)
    expected["date"] = pd.to_datetime(expected.date)
    pd.testing.assert_frame_equal(frame[KEYS].reset_index(drop=True), expected)
    pd.testing.assert_frame_equal(raw[KEYS], expected)
    values = frame.prediction
    assert values.dtype.kind in "iu" and values.ge(0).all() and np.isfinite(raw.prediction).all()
    rounded = np.floor(raw.prediction.clip(lower=0).to_numpy() + 0.5).astype("int64")
    forced = frame.route.eq(5) | frame.hour.between(1, 4)
    rounded[forced.to_numpy()] = 0
    np.testing.assert_array_equal(values, rounded)
    assert values[forced].eq(0).all()


def run(export=False):
    history = read(ROOT / "artifacts/hourly_clean.csv")
    snapshot = json.loads((OUT / "snapshot.json").read_text())
    protected = [p for p in snapshot["files"] if p.startswith("ml/")]
    for name in protected:
        assert hashlib.sha256((ROOT.parent / name).read_bytes()).hexdigest() == snapshot["files"][name]["sha256"], name
    rows, routes, finals = [], [], {}
    checked_windows = 0
    for path in sorted(OUT.rglob("predictions.csv")):
        if any(p in path.parts for p in ["before_precision_fix", "github"]):
            continue
        candidate = str(path.parent.relative_to(OUT))
        predictions = read(path)
        recorded = pd.read_csv(path.parent / "metrics.csv", sep=";")
        for (cutoff, end), frame in predictions.groupby(["cutoff", "end"], sort=False):
            frame = frame.reset_index(drop=True)
            raw = read(path.parent / f"raw_{cutoff}.csv")
            validate(frame, pd.Timestamp(cutoff) + pd.Timedelta(days=1), end, raw)
            truth = history.loc[history.date.gt(cutoff) & history.date.le(end), KEYS + ["boardings"]]
            pd.testing.assert_frame_equal(frame[KEYS + ["boardings"]], truth.reset_index(drop=True))
            measured = score(frame)
            saved = recorded.loc[recorded.cutoff.eq(cutoff) & recorded.end.eq(end)]
            assert len(saved) == 1
            for key in ["wape_score", "absolute_error", "actual_total", "bias"]:
                assert np.isclose(measured[key], saved.iloc[0][key], rtol=0, atol=1e-12), (candidate, cutoff, key)
            rows.append(dict(candidate=candidate, cutoff=cutoff, end=end, **measured))
            for route, group in frame.groupby("route"):
                routes.append(dict(candidate=candidate, cutoff=cutoff, route=route, **score(group)))
            checked_windows += 1
        final_path = path.parent / "submission.csv"
        if final_path.exists():
            final = read(final_path)
            assert list(final.columns) == KEYS + ["prediction"]
            validate(final, "2025-11-01", "2025-12-31", read(path.parent / "raw_2025-10-31.csv"))
            finals[candidate] = final
    window_scores = pd.DataFrame(rows)
    window_scores.to_csv(OUT / "verified_windows.csv", sep=";", index=False)
    route_scores = pd.DataFrame(routes)
    route_scores.to_csv(OUT / "verified_routes.csv", sep=";", index=False)
    ranks = {candidate: rank for rank, (_, candidate) in enumerate(FINALISTS, 1)}
    for table, filename in [(window_scores, "finalist_window_metrics.csv"), (route_scores, "finalist_routes.csv")]:
        selected = table.loc[table.candidate.isin(ranks)].copy()
        selected["rank"] = selected.candidate.map(ranks)
        assert selected.candidate.nunique() == len(FINALISTS)
        selected.sort_values(["rank", "cutoff"]).to_csv(OUT / filename, sep=";", index=False)
    summaries = []
    for candidate in finals:
        part = window_scores.loc[window_scores.candidate.eq(candidate)]
        assert len(part) == 4
        summaries.append(dict(candidate=candidate, mean=part.wape_score.mean(), worst=part.wape_score.min(),
            final_total=int(finals[candidate].prediction.sum()),
            **{str(r.cutoff): r.wape_score for r in part.itertuples()}))
    pd.DataFrame(summaries).sort_values("mean", ascending=False).to_csv(OUT / "verified_summary.csv", sep=";", index=False)
    old = read(ROOT / "artifacts/chronos_submission/predictions.csv")
    reference = old.loc[old.method.eq("chronos_daily_total") & old.cutoff.eq("2025-08-31"), KEYS + ["prediction"]].reset_index(drop=True)
    control = read(OUT / "gpu/daily/predictions.csv")
    control = control.loc[control.cutoff.eq("2025-08-31"), KEYS + ["prediction"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(control, reference)
    archived = []
    for rank, (attempt, candidate) in enumerate(FINALISTS, 1):
        target = ROOT / "submissions" / f"{attempt}.csv"
        if export and not target.exists():
            archive(OUT / candidate / "submission.csv", attempt)
        if target.exists():
            pd.testing.assert_frame_equal(read(target), finals[candidate])
            archived.append(dict(rank=rank, attempt=attempt, candidate=candidate, rows=14640,
                sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
        elif export:
            raise AssertionError(target)
    differences = []
    for (a, ca), (b, cb) in itertools.combinations(FINALISTS, 2):
        left, right = finals[ca].prediction, finals[cb].prediction
        delta = (left - right).abs()
        assert delta.sum() > 0, (a, b)
        differences.append(dict(a=a, b=b, l1=int(delta.sum()),
            relative_l1=float(delta.sum() / ((left.sum() + right.sum()) / 2)), changed_rows=int(delta.gt(0).sum())))
    pd.DataFrame(differences).to_csv(OUT / "finalist_diversity.csv", sep=";", index=False)
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), checked_windows=checked_windows,
        checked_final_forecasts=len(finals), protected_sha256_files=len(protected),
        unchanged_autumn_chronos_control=True, finalists=archived,
        complete=len(archived) == len(FINALISTS), audit_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", action="store_true", help="archive missing fixed finalists; never overwrite")
    run(parser.parse_args().export)
