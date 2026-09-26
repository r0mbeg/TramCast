"""Verify downloaded Chronos results and export submission.csv locally."""
import hashlib
import json

import numpy as np
import pandas as pd

from pipeline import DATA, KEYS, OUT, ROOT, full_grid, metrics


def main():
    source = OUT / "chronos_submission"
    run = json.loads((source / "run.json").read_text())
    assert run["history_sha256"] == hashlib.sha256((OUT / "hourly_clean.csv").read_bytes()).hexdigest()
    sources = [ROOT / "experiments/chronos_experiment.py",
               OUT / "source_snapshot_09cab47/chronos_experiment.py"]
    assert run["code_sha256"] in {hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    assert run["submission"]["sha256"] == hashlib.sha256((source / "submission.csv").read_bytes()).hexdigest()
    history = pd.read_csv(OUT / "hourly_clean.csv", sep=";", parse_dates=["date"])
    predictions = pd.read_csv(source / "predictions.csv", sep=";", parse_dates=["date"])
    reference = pd.read_csv(OUT / "chronos_zhores/metrics.csv", sep=";")
    reported = pd.read_csv(source / "metrics.csv", sep=";")
    previous = pd.read_csv(OUT / "chronos_zhores/predictions.csv", sep=";", parse_dates=["date"])
    comparison_keys = ["method", "cutoff"] + KEYS
    paired = previous.merge(predictions, on=comparison_keys, suffixes=("_old", "_new"), validate="one_to_one")
    assert len(paired) == len(previous) == len(predictions)
    delta = paired.prediction_new - paired.prediction_old
    # Different GPUs can cross a half-integer rounding boundary; retain the exact differences.
    assert delta.abs().max() <= 1 and delta.abs().sum() <= 2
    paired.loc[delta.ne(0), comparison_keys + ["prediction_old", "prediction_new"]].to_csv(
        source / "hardware_rounding_differences.csv", sep=";", index=False)
    for (method, cutoff), group in predictions.groupby(["method", "cutoff"]):
        truth = history[history.date.between(group.date.min(), group.date.max())]
        compared = truth[KEYS + ["boardings"]].merge(group[KEYS + ["prediction"]], on=KEYS, validate="one_to_one")
        assert len(compared) == len(truth) == len(group)
        score = metrics(compared.boardings, compared.prediction)
        expected_score = reference.loc[reference.method.eq(method) & reference.cutoff.eq(cutoff), "wape_score"].item()
        reported_score = reported.loc[reported.method.eq(method) & reported.cutoff.eq(cutoff), "wape_score"].item()
        assert abs(score["wape_score"] - reported_score) < 1e-12
        assert abs(score["wape_score"] - expected_score) <= 1 / score["actual_total"] + 1e-12

    template = pd.read_csv(DATA / "test_submission.csv", sep=";", parse_dates=["date"])[KEYS]
    forecast = pd.read_csv(source / "submission.csv", sep=";", parse_dates=["date"])
    assert list(forecast.columns) == KEYS + ["prediction"]
    expected = full_grid("2025-11-01", "2025-12-31").to_frame(index=False)
    expected["date"] = pd.to_datetime(expected.date)
    pd.testing.assert_frame_equal(forecast[KEYS].sort_values(KEYS).reset_index(drop=True), expected)
    result = template.merge(forecast, on=KEYS, how="left", validate="one_to_one")
    assert len(result) == 14640 and not result.duplicated(KEYS).any()
    assert np.isfinite(result.prediction).all() and result.prediction.ge(0).all()
    assert result.prediction.dtype.kind in "iu"
    assert len(result[result.route.eq(5)]) == 1464
    assert result.loc[result.route.eq(5) | result.hour.between(1, 4), "prediction"].eq(0).all()
    target = ROOT / "submission.csv"
    result.to_csv(target, sep=";", index=False, encoding="utf-8", date_format="%Y-%m-%d")
    pd.testing.assert_frame_equal(result, pd.read_csv(target, sep=";", parse_dates=["date"]))
    run["export_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    run["local_verification"] = dict(scores_recomputed=6, changed_historical_cells=int(delta.ne(0).sum()),
        maximum_cell_difference=int(delta.abs().max()), submission_contract="passed")
    (OUT / "submission_run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(json.dumps(run["submission"], indent=2))
    print("Verified: historical scores, history/code hashes, template keys, integers, zeros, CSV readback.")


if __name__ == "__main__":
    main()
