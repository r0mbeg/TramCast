"""P23: empirical Bayes on robust latent levels; cutoff-only evidence selects persistence."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.portfolio_experiment import write_json
from experiments.portfolio_movement import disrupted
from experiments.portfolio_operations import operations_forecast
from experiments.portfolio_ridge import add_calendar
from experiments.portfolio_weather_volume import WEATHER_PATH, fit_weather_volume

ROOT = Path("artifacts/portfolio_20260926/continuation")
SPACE = {"q": [1e-5, 1e-4, 1e-3], "jump_probability": [0., 0.02, 0.1]}


def filter_level(values, gaps, q, jump_probability, particles=1024):
    values = np.asarray(values, float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Invalid level observations")
    changes = np.diff(values)
    sigma = max(0.025, float(1.4826 * np.median(np.abs(changes - np.median(changes))) / np.sqrt(2))) if len(changes) else 0.1
    rng = np.random.default_rng(42)
    states = rng.normal(values[0], 0.1, particles)
    evidence = 0.
    constant = math.lgamma(2.5) - math.lgamma(2) - 0.5 * math.log(4 * math.pi) - math.log(sigma)
    for value, gap in zip(values, gaps):
        jump = rng.random(particles) < 1 - (1 - jump_probability) ** gap
        states += rng.normal(0, np.where(jump, 0.35, np.sqrt(q * gap)), particles)
        log_likelihood = constant - 2.5 * np.log1p(np.square((value - states) / sigma) / 4)
        weights = np.exp(log_likelihood - log_likelihood.max())
        evidence += float(log_likelihood.max() + np.log(weights.mean()))
        weights /= weights.sum()
        positions = (np.arange(particles) + rng.random()) / particles
        ancestors = np.searchsorted(np.cumsum(weights), positions).clip(max=particles - 1)
        states = states[ancestors]
    return dict(level=float(np.median(states)), sd=float(states.std()), evidence=evidence, sigma=sigma)


def states_for(fit, cutoff, grouped):
    import optuna

    directory = ROOT / "particle_level" / "levels" / f"{cutoff}_{int(grouped)}"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "posterior.json"
    if target.exists():
        return json.loads(target.read_text())
    if fit.date.max() > pd.Timestamp(cutoff):
        raise ValueError("Future level observation")
    keys = ["route", "daytype"] if grouped else ["route"]
    series = {}
    for group, section in fit.sort_values("date").groupby(keys):
        label = ":".join(str(int(x)) for x in (group if isinstance(group, tuple) else [group]))
        gaps = section.date.diff().dt.days.fillna(1).clip(lower=1).to_numpy()
        series[label] = (section.residual.to_numpy(), gaps)
    spec = dict(space=SPACE, trials=9, timeout=300, seed=42, sampler="GridSampler",
        objective="pooled past marginal log likelihood", trial_particles=512, final_particles=1024,
        cutoff=cutoff, grouped=grouped, latest_observation=str(fit.date.max().date()))
    spec_path = directory / "study_spec.json"
    if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
        raise ValueError("Cannot change an existing evidence study")
    write_json(spec_path, spec)
    study = optuna.create_study(storage=f"sqlite:///{directory/'study.db'}", study_name="evidence",
        direction="maximize", sampler=optuna.samplers.GridSampler(SPACE, seed=42), load_if_exists=True)
    if "started_at" not in study.user_attrs:
        study.set_user_attr("started_at", datetime.now(timezone.utc).timestamp())
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.RUNNING:
            study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
    remaining = max(0, 300 - (datetime.now(timezone.utc).timestamp() - study.user_attrs["started_at"]))

    def objective(trial):
        params = {k: trial.suggest_categorical(k, v) for k, v in SPACE.items()}
        return sum(filter_level(x, gaps, **params, particles=512)["evidence"] for x, gaps in series.values())

    if remaining and len(study.trials) < 9:
        study.optimize(objective, n_trials=9-len(study.trials), timeout=remaining, n_jobs=1)
    study.trials_dataframe().to_csv(directory / "trials.csv", sep=";", index=False)
    selected = study.best_params
    posterior = {label: filter_level(x, gaps, **selected) for label, (x, gaps) in series.items()}
    write_json(target, dict(spec=spec, selected=selected, groups=posterior))
    return json.loads(target.read_text())


def particle_forecast(history, cutoff, end, params):
    base_params = json.loads((ROOT / "operations/operations/selection.json").read_text())["params"]
    weather = pd.read_csv(WEATHER_PATH, sep=";", parse_dates=["date"])
    train = history.loc[history.date.le(cutoff)]
    normal = train.loc[~(disrupted(train, "july") | disrupted(train, "autumn"))]
    _, fit, _ = fit_weather_volume(normal, cutoff, dict(alpha=1, mode="rain", residual_strength=0,
        route_season=base_params["route_season"]), weather)
    posterior = states_for(fit, cutoff, params["grouped"])
    raw = operations_forecast(history, cutoff, end, base_params, weather)
    calendar = add_calendar(raw, cutoff)
    factors = np.ones(len(raw))
    for (route, daytype), indices in calendar.groupby(["route", "daytype"]).groups.items():
        if route == 5:
            continue
        label = f"{route}:{daytype}" if params["grouped"] else str(route)
        state = posterior["groups"][label]
        shrink = 1 / (1 + params["uncertainty"] * (state["sd"] / 0.1) ** 2)
        factors[np.asarray(indices)] = np.exp(np.clip(params["strength"] * shrink * state["level"], -np.log(2), np.log(2)))
    raw["prediction"] *= factors
    return raw
