"""Gibbs/Dirichlet forecast ensemble inferred from completed daily blocks of earlier forecasts."""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.portfolio_experiment import WINDOWS, FINAL, load_history, save_candidate, forecast_keys, write_json
from experiments.portfolio_ridge import ridge_forecast
from experiments.portfolio_movement import movement_forecast
from experiments.portfolio_structure import regime_forecast, tagged
from experiments.portfolio_windows import season
from pipeline import KEYS

MODELS = ["ridge", "chronos", "movement", "regime"]
EARLY = [("2025-02-28", "2025-04-30"), ("2025-05-31", "2025-07-31")] + WINDOWS
GRID = np.array([0.05 + 0.1 * np.array(x) for x in itertools.product(range(9), repeat=4) if sum(x) == 8])


def posterior(log_prior, loss):
    # Generalized Bayes on daily loss blocks, not an independence claim about 24 hourly observations.
    log_weight = log_prior - loss.sum(axis=0) / 0.15
    weight = np.exp(log_weight - log_weight.max())
    return weight / weight.sum()


def infer_weights(panel, cutoff):
    if panel.date.max() > pd.Timestamp(cutoff):
        raise ValueError("Future target entered Bayesian weight inference")
    panel = panel.loc[panel.route.ne(5)].copy()
    code = panel.groupby(["route", "date"], sort=True).ngroup().to_numpy()
    dates = panel.groupby(["route", "date"], sort=True).size().rename("rows").reset_index()
    dates["season"] = season(dates.date)
    calendar = tagged(dates, cutoff)
    dates["group"] = np.where(calendar.is_workday, 0,
        np.where(calendar.weekday.eq(5) & ~calendar.off, 1, 2))
    # Average repeated origins for the same route-day, preserving daily blocks under overlapping windows.
    origin_count = dates.rows.to_numpy() / 24
    daily_actual = np.bincount(code, weights=panel.boardings.to_numpy(), minlength=len(dates)) / origin_count
    typical = pd.Series(daily_actual).groupby(dates.route).transform("median").clip(lower=100).to_numpy()
    values = panel[MODELS].to_numpy()
    truth = panel.boardings.to_numpy()
    loss = np.empty((len(dates), len(GRID)))
    for i, weights in enumerate(GRID):
        predicted = np.floor(np.maximum(values @ weights, 0) + 0.5)
        loss[:, i] = np.bincount(code, weights=np.abs(predicted - truth), minlength=len(dates)) / origin_count / typical
    log_prior = np.log(GRID).sum(axis=1)  # Dirichlet(alpha=2) on the fixed discrete simplex.
    summaries = {}
    for route in sorted(dates.route.unique()):
        other = dates.route.ne(route).to_numpy()
        pooled = pd.DataFrame(loss[other]).groupby(dates.loc[other, "date"].to_numpy()).mean().to_numpy()
        global_p = posterior(log_prior, pooled)
        for s in range(4):
            for group in range(3):
                selected = dates.route.eq(route) & dates.season.eq(s) & dates.group.eq(group)
                if selected.sum() < 7:
                    selected = dates.route.eq(route) & dates.group.eq(group)
                if selected.sum() < 7:
                    local_p = global_p
                else:
                    local_p = posterior(np.log(np.maximum(global_p, 1e-300)), loss[selected.to_numpy()])
                # ponytail: 25% common-risk shrinkage; richer routing needs several annual cycles.
                probability = 0.75 * local_p + 0.25 * global_p
                mean = probability @ GRID
                sd = np.sqrt(probability @ np.square(GRID - mean))
                selected_loss = loss[selected.to_numpy()]
                denominator = float((daily_actual[selected] / typical[selected]).sum())
                risks = selected_loss.sum(axis=0) / denominator if denominator else np.ones(len(GRID))
                expected_score = float(1 - probability @ risks)
                summaries[f"{route}:{s}:{group}"] = dict(weights=mean.tolist(), weight_sd=sd.tolist(),
                    past_days=int(selected.sum()), expected_score_from_past=expected_score,
                    score_sd_from_weight_uncertainty=float(np.sqrt(probability @ np.square((1-risks)-expected_score))))
    return summaries


def run(args):
    if os.environ.get("SLURM_JOB_PARTITION") != "ais-cpu":
        raise RuntimeError("Run Bayesian ensemble in ais-cpu")
    if datetime.now(timezone.utc) >= datetime.fromisoformat("2026-09-27T15:40:40+00:00"):
        raise RuntimeError("Research reserve reached")
    root = Path(args.root)
    out = root / "continuation/bayes"
    out.mkdir(parents=True, exist_ok=True)
    data = load_history(args.history)
    ridge_params = json.loads((root / "ridge_probe/ridge/selection.json").read_text())["params"]
    regime_params = json.loads((root / "continuation/cpu/regime/selection.json").read_text())["params"]
    spec = dict(models=MODELS, inner_windows=EARLY, outer_windows=WINDOWS, final=FINAL,
        prior="Dirichlet(2,2,2,2)", grid=GRID.tolist(), loss="daily absolute error / typical route-day volume",
        likelihood_temperature=0.15, common_shrinkage=0.25, minimum_local_days=7,
        no_outer_targets=True, uncertainty="conditional weights and past risk; not a hidden score confidence interval",
        duplicate_origins="average per route-date; network prior pooled per date, leave selected route out",
        ridge_params=ridge_params, regime_params=regime_params, job_id=os.environ["SLURM_JOB_ID"])
    write_json(out / "spec.json", spec)
    write_json(out / "run_started.json", dict(command=os.sys.argv, job_id=os.environ["SLURM_JOB_ID"],
        sha256={str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
            [args.history, __file__, "experiments/portfolio_movement.py", "experiments/portfolio_ridge.py", "experiments/portfolio_structure.py"]}))
    paths = {"ridge": root / "ridge_probe/ridge/selected", "movement": root / "movement",
             "regime": root / "continuation/cpu/regime/selected", "chronos": root / "gpu/daily"}

    def component(name, origin, end):
        source = paths[name] / f"raw_{origin}.csv"
        if name == "chronos" and not source.exists():
            source = root / "inner_gpu/daily" / f"raw_{origin}.csv"
        if not source.exists():
            cache = out / "inner" / name / origin
            predictor = {"ridge": lambda c, e: ridge_forecast(data, c, e, ridge_params),
                "movement": lambda c, e: movement_forecast(data, c, e, ridge_params),
                "regime": lambda c, e: regime_forecast(data, c, e, regime_params)}[name]
            save_candidate(data, cache, name, predictor, windows=[(origin, end)], final=False)
            source = cache / f"raw_{origin}.csv"
        frame = pd.read_csv(source, sep=";", parse_dates=["date"], float_precision="round_trip")
        pd.testing.assert_frame_equal(frame[KEYS], forecast_keys(origin, end))
        return frame

    def forecast(cutoff, end):
        pieces = []
        earlier = [(a, b) for a, b in EARLY if pd.Timestamp(b) <= pd.Timestamp(cutoff)]
        for origin, stop in earlier:
            part = component("ridge", origin, stop)[KEYS].copy()
            for name in MODELS:
                part[name] = component(name, origin, stop).prediction.to_numpy()
            part = part.merge(data[KEYS + ["boardings"]], on=KEYS, validate="one_to_one")
            pieces.append(part)
        if not pieces:
            raise ValueError("No completed past component forecasts")
        weights = infer_weights(pd.concat(pieces, ignore_index=True), cutoff)
        write_json(out / f"posterior_{cutoff}.json", dict(cutoff=cutoff, earlier_windows=earlier, groups=weights))
        result = component("ridge", cutoff, end)[KEYS].copy()
        future = tagged(result, cutoff)
        groups = np.where(future.is_workday, 0, np.where(future.weekday.eq(5) & ~future.off, 1, 2))
        seasons = season(future.date)
        array = np.column_stack([component(name, cutoff, end).prediction for name in MODELS])
        result["prediction"] = 0.
        quality = []
        for i in range(len(result)):
            r = int(result.route.iloc[i])
            if r == 5:
                continue
            summary = weights[f"{r}:{seasons.iloc[i]}:{groups[i]}"]
            result.loc[i, "prediction"] = array[i] @ np.array(summary["weights"])
            if int(result.hour.iloc[i]) == 0:
                quality.append(dict(route=r, date=str(result.date.iloc[i].date()),
                    expected_score_from_past=summary["expected_score_from_past"],
                    conditional_sd=summary["score_sd_from_weight_uncertainty"], past_days=summary["past_days"]))
        pd.DataFrame(quality).to_csv(out / f"expected_quality_{cutoff}.csv", sep=";", index=False)
        return result

    rows = save_candidate(data, out / "selected", "bayesian_ensemble", forecast)
    write_json(out / "completed.json", dict(job_id=os.environ["SLURM_JOB_ID"], scores=[r["wape_score"] for r in rows]))
    print([r["wape_score"] for r in rows], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", required=True)
    parser.add_argument("--root", required=True)
    run(parser.parse_args())
