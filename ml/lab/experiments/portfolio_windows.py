"""Select historical windows using completed inner 61-day backtests, never outer targets."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from experiments.portfolio_experiment import (WINDOWS, FINAL, load_history, forecast_keys,
    complete_raw, save_candidate, write_json)
from experiments.portfolio_structure import tagged
from pipeline import KEYS, postprocess

RULES = [(kind, size) for kind, sizes in [("recent", [28, 56, 112]),
          ("season", [28, 56, 112]), ("weather", [7, 14, 28])] for size in sizes]
WEATHER = ["temperature_2m_mean", "precipitation_sum", "daylight_duration"]


def season(dates):
    return dates.dt.month.mod(12).floordiv(3)


def inner_windows(cutoff):
    origins = pd.date_range("2025-01-31", pd.Timestamp(cutoff) - pd.Timedelta(days=61), freq="ME")
    return [(str(o.date()), str((o + pd.Timedelta(days=61)).date())) for o in origins]


def analogue_forecast(history, weather, cutoff, end, rule):
    kind, size = rule
    train = tagged(history.loc[history.date.le(cutoff) & history.route.ne(5)], cutoff)
    train = train.loc[~train.off]
    target = tagged(forecast_keys(cutoff, end), cutoff)
    weather = weather.copy().set_index("date")
    # Scaling depends on past weather only; actual forecast weather is a permitted external scenario.
    weather["precipitation_sum"] = np.log1p(weather.precipitation_sum)
    weather["daylight_duration"] /= 3600
    observed = weather.loc[weather.index <= pd.Timestamp(cutoff), WEATHER]
    scales = (observed - observed.median()).abs().median().clip(lower=pd.Series([4, 0.5, 2], index=WEATHER))
    parts = []
    for route, future in target.loc[target.route.ne(5)].groupby("route"):
        past = train.loc[train.route.eq(route)]
        matrix = past.pivot(index="date", columns="hour", values="boardings").reindex(columns=range(24)).sort_index()
        dates = matrix.index
        sums = matrix.sum(axis=1)
        dows = dates.dayofweek.to_numpy()
        factor = sums.groupby(dates.dayofweek).mean()
        factor = factor / max(float(factor.loc[factor.index < 5].mean()), 1)
        values = matrix.to_numpy(float)
        norm = np.where(dows < 5, factor.reindex(dows).to_numpy(), 1.)
        normalised = values / np.maximum(norm[:, None], 1e-6)
        old_weather = weather.loc[dates, WEATHER].to_numpy(float)
        for date, day in future.groupby("date", sort=True):
            off = bool(day.off.iloc[0])
            dow = int(day.dow.iloc[0])
            groups = [5, 6] if off else [0 if dow < 5 else dow]
            prediction = np.zeros(24)
            for group in groups:
                usable = (dows < 5) if group == 0 else (dows == group)
                if kind == "season":
                    same = usable & ((dates.month % 12 // 3) == date.month % 12 // 3)
                    if same.sum() >= 4:
                        usable = same
                ids = np.flatnonzero(usable)
                if len(ids) == 0:
                    raise ValueError("No historical calendar group")
                if kind == "weather":
                    distance = np.square((old_weather[ids] - weather.loc[date, WEATHER].to_numpy(float)) / scales.to_numpy()).sum(axis=1)
                    chosen = np.argsort(distance, kind="stable")[:size]
                    ids, distance = ids[chosen], distance[chosen]
                    weights = np.exp(-np.minimum(distance / 2, 50))
                else:
                    edge = dates[ids].max() - pd.Timedelta(days=size)
                    ids = ids[dates[ids] > edge]
                    age = (dates[ids].max() - dates[ids]).days.to_numpy()
                    weights = np.exp2(-age / max(size / 2, 1))
                forecast = np.average(normalised[ids], axis=0, weights=weights)
                if group == 0:
                    forecast *= float(factor.get(dow, 1))
                prediction += forecast / len(groups)
            if off:
                prediction *= 0.95
            parts.append(day[KEYS].assign(prediction=prediction))
    return complete_raw(pd.concat(parts, ignore_index=True), cutoff, end)


def run(args):
    import optuna
    if os.environ.get("SLURM_JOB_PARTITION") != "ais-cpu":
        raise RuntimeError("Run adaptive window selection in ais-cpu")
    if datetime.now(timezone.utc) >= datetime.fromisoformat("2026-09-27T15:40:40+00:00"):
        raise RuntimeError("Research reserve reached")
    start = time.monotonic()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    data = load_history(args.history)
    weather = pd.read_csv(args.weather, sep=";", parse_dates=["date"])
    expected = pd.date_range("2025-01-01", "2025-12-31")
    assert weather.date.tolist() == expected.tolist() and np.isfinite(weather[WEATHER]).all().all()
    spec = dict(rules=RULES, sampler="GridSampler", trials_per_cutoff=9, timeout_per_cutoff=1200,
        seed=42, objective="pooled score on completed inner 61-day windows ending <= outer cutoff",
        selection="season score 75%, global score 25%; global fallback if <14 unique seasonal days",
        outer_windows=WINDOWS, final=FINAL, no_outer_targets=True,
        regime="retrospective external ERA5 weather; no future target validations")
    spec_path = out / "study_spec.json"
    if spec_path.exists() and json.loads(spec_path.read_text()) != json.loads(json.dumps(spec)):
        raise ValueError("Existing adaptive window budget differs")
    write_json(spec_path, spec)
    write_json(out / "run_started.json", dict(job_id=os.environ["SLURM_JOB_ID"], command=os.sys.argv,
        sha256={str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                [args.history, args.weather, __file__, "experiments/portfolio_structure.py", "experiments/portfolio_experiment.py"]}))
    choices = {}

    def select(cutoff, end):
        windows = inner_windows(cutoff)
        if not windows:
            raise ValueError("No completed 61-day inner windows")
        root = out / f"cutoff_{cutoff}"
        root.mkdir(exist_ok=True)
        study = optuna.create_study(storage=f"sqlite:///{root / 'study.db'}", study_name=cutoff,
            direction="maximize", load_if_exists=True,
            sampler=optuna.samplers.GridSampler({"rule": list(range(len(RULES)))}, seed=42))
        if "started_at" not in study.user_attrs:
            study.set_user_attr("started_at", datetime.now(timezone.utc).timestamp())
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.RUNNING:
                study.tell(trial.number, state=optuna.trial.TrialState.FAIL)

        def objective(trial):
            rule = trial.suggest_categorical("rule", list(range(len(RULES))))
            frames = []
            for origin, stop in windows:
                cache = out / "inner" / f"rule_{rule}" / origin
                save_candidate(data, cache, f"analogue_{rule}", lambda c, e:
                    analogue_forecast(data, weather, c, e, RULES[rule]), windows=[(origin, stop)], final=False)
                frames.append(pd.read_csv(cache / "predictions.csv", sep=";", parse_dates=["date"]))
            joined = pd.concat(frames, ignore_index=True)
            # Each prediction was made from its own origin, all facts finish before outer cutoff.
            assert joined.date.max() <= pd.Timestamp(cutoff)
            error = (joined.prediction - joined.boardings).abs()
            global_score = float(1 - error.sum() / joined.boardings.sum())
            scores = {}
            for s in range(4):
                group = season(joined.date).eq(s)
                score = float(1 - error[group].sum() / joined.loc[group, "boardings"].sum()) if group.any() else global_score
                scores[str(s)] = (0.75 * score + 0.25 * global_score
                                  if joined.loc[group, "date"].nunique() >= 14 else global_score)
            trial.set_user_attr("season_scores", scores)
            return global_score

        remaining = max(0, spec["trials_per_cutoff"] - len(study.trials))
        seconds = max(0, spec["timeout_per_cutoff"] - (datetime.now(timezone.utc).timestamp() - study.user_attrs["started_at"]))
        if remaining and seconds:
            study.optimize(objective, n_trials=remaining, timeout=seconds,
                callbacks=[lambda s, t: s.trials_dataframe().to_csv(root / "trials.csv", sep=";", index=False)])
        complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        mapping = {s: max(complete, key=lambda t: t.user_attrs["season_scores"][str(s)]).params["rule"] for s in range(4)}
        write_json(root / "selection.json", dict(inner_windows=windows, selected_rule_by_season=mapping,
            best_global_rule=study.best_params, total_trials=len(study.trials)))
        choices[cutoff] = mapping
        future = forecast_keys(cutoff, end)
        forecasts = {r: analogue_forecast(data, weather, cutoff, end, RULES[r]) for r in set(mapping.values())}
        result = future.copy()
        result["prediction"] = 0.
        for s, rule in mapping.items():
            group = season(future.date).eq(s)
            result.loc[group, "prediction"] = forecasts[rule].loc[group, "prediction"].to_numpy()
        return result

    rows = save_candidate(data, out / "selected", "automatic_window", select)
    write_json(out / "completed.json", dict(job_id=os.environ["SLURM_JOB_ID"], seconds=time.monotonic()-start,
        choices=choices, scores=[r["wape_score"] for r in rows]))
    print([r["wape_score"] for r in rows], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", required=True)
    parser.add_argument("--weather", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
