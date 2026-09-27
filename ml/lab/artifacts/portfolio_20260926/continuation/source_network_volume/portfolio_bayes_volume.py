"""P21: Bayesian daily bias learned only from completed out-of-origin forecasts."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

from experiments.portfolio_experiment import save_candidate, write_json
from experiments.portfolio_operations import operations_forecast
from experiments.portfolio_ridge import ACTIVE_ROUTES, add_calendar, ridge_forecast
from experiments.portfolio_structure import regime_forecast
from experiments.portfolio_weather_volume import WEATHER_PATH
from experiments.portfolio_windows import inner_windows

ROOT = Path("artifacts/portfolio_20260926/continuation")


def features(frame):
    columns = [frame.route.eq(r).to_numpy(float) for r in ACTIVE_ROUTES]
    columns += [(frame.route.eq(r) & frame.daytype.eq(t)).to_numpy(float)
                for r in ACTIVE_ROUTES for t in [1, 2]]
    columns += [frame.effective_weekday.eq(d).to_numpy(float) for d in range(1, 7)]
    columns += [(frame.date.dt.month % 12 // 3).eq(s).to_numpy(float) for s in range(1, 4)]
    columns += [frame.horizon.to_numpy() / 61, np.log1p(frame.base.to_numpy()) / 10,
        np.log1p(frame.ridge.to_numpy()) - np.log1p(frame.base.to_numpy()),
        np.log1p(frame.regime.to_numpy()) - np.log1p(frame.base.to_numpy()),
        frame.temperature_2m_mean.to_numpy() / 10, np.log1p(frame.precipitation_sum.to_numpy()),
        frame.daylight_duration.to_numpy() / 21600, frame.off.to_numpy(float)]
    return np.column_stack(columns)


def calibrate(past, future, cutoff, params, feature_fn=features, target_column="boardings"):
    component = params.get("uncertainty_component", "predictive")
    if component not in ["predictive", "epistemic", "none"]:
        raise ValueError("Unknown Bayesian uncertainty component")
    if past.date.max() > pd.Timestamp(cutoff):
        raise ValueError("Future target entered daily Bayesian calibration")
    past = past.loc[past.route.ne(5) & past.base.gt(0)].copy()
    if params["history_days"]:
        past = past.loc[past.date.gt(pd.Timestamp(cutoff) - pd.Timedelta(days=params["history_days"]))]
    counts = past.groupby(["route", "date"]).base.transform("size")
    weights = 1 / counts.to_numpy()
    if params.get("volume_weight", False):
        volumes=past.base.to_numpy()
        weights *= volumes / np.average(volumes,weights=weights)
    scaler = StandardScaler().fit(feature_fn(past), sample_weight=weights)
    if not np.isfinite(past[target_column]).all() or past[target_column].lt(0).any():
        raise ValueError("Invalid Bayesian calibration target")
    target = np.log((past[target_column].to_numpy() + 100) / (past.base.to_numpy() + 100))
    target = np.clip(target, -np.log(2), np.log(2))
    model = BayesianRidge(max_iter=300, tol=1e-5).fit(scaler.transform(feature_fn(past)), target,
        sample_weight=weights)
    mean, sd = model.predict(scaler.transform(feature_fn(future)), return_std=True)
    noise_variance = 1 / model.alpha_
    epistemic_sd = np.sqrt(np.maximum(0, np.square(sd) - noise_variance))
    attenuation_sd = sd if component == "predictive" else epistemic_sd if component == "epistemic" else np.zeros_like(sd)
    # ponytail: conditional coefficient variance omits regime shifts; retain held-out temporal checks.
    shrink = 1 / (1 + params["uncertainty"] * np.square(attenuation_sd / 0.1))
    delta = np.clip(params["strength"] * shrink * mean, -np.log(2), np.log(2))
    return np.exp(delta), dict(coefficient_mean=model.coef_.tolist(),
        intercept=float(model.intercept_), noise_precision=float(model.alpha_),
        coefficient_precision=float(model.lambda_), days=int(past.date.nunique()),
        rows=len(past), latest_target_date=str(past.date.max().date()),
        feature_mean=scaler.mean_.tolist(), feature_scale=scaler.scale_.tolist(),
        mean_predictive_sd=float(sd.mean()), noise_sd=float(np.sqrt(noise_variance)),
        mean_epistemic_sd=float(epistemic_sd.mean()), uncertainty_component=component)


def bayes_volume_forecast(history, cutoff, end, params):
    root = ROOT / ("bayes_volume_verified_july" if params.get("july_verified",False) else "bayes_volume")
    root.mkdir(exist_ok=True)
    base_params = json.loads((ROOT / "operations/operations/selection.json").read_text())["params"]
    if params.get("july_verified",False):
        base_params={**base_params,"july_verified":True}
    weather = pd.read_csv(WEATHER_PATH, sep=";", parse_dates=["date"])

    def daily(origin, stop, observed):
        cache = root / "inner" / origin
        table = cache / "daily.csv"
        if table.exists():
            result = pd.read_csv(table, sep=";", parse_dates=["date"], float_precision="round_trip")
        else:
            base = operations_forecast(history, origin, stop, base_params, weather)
            save_candidate(history, cache, "operations_inner", lambda c, e: base,
                windows=[(origin, stop)], final=False)
            result = base.groupby(["route", "date"], as_index=False).prediction.sum().rename(columns={"prediction": "base"})
            for name, raw in [
                ("ridge", ridge_forecast(history, origin, stop, dict(correction_days=14, shape_mix=0.5, route_season=False))),
                ("regime", regime_forecast(history, origin, stop, dict(occurrences=32, statistic="mean", regime=True)))]:
                volume = raw.groupby(["route", "date"], as_index=False).prediction.sum().rename(columns={"prediction": name})
                result = result.merge(volume, on=["route", "date"], validate="one_to_one")
            result = add_calendar(result, origin).merge(weather, on="date", validate="many_to_one")
            result["horizon"] = (result.date - pd.Timestamp(origin)).dt.days
            result.to_csv(table, sep=";", index=False, date_format="%Y-%m-%d")
        if result.date.min() != pd.Timestamp(origin) + pd.Timedelta(days=1) or result.date.max() != pd.Timestamp(stop):
            raise ValueError("Cached inner forecast has wrong dates")
        if observed:
            if pd.Timestamp(stop) > pd.Timestamp(cutoff):
                raise ValueError("Unfinished inner forecast used for calibration")
            actual = history.loc[history.date.le(cutoff)].groupby(["route", "date"], as_index=False).boardings.sum()
            result = result.merge(actual, on=["route", "date"], validate="one_to_one")
        return result

    earlier = inner_windows(cutoff)
    past = pd.concat([daily(a, b, True) for a, b in earlier], ignore_index=True)
    future = daily(cutoff, end, False)
    factor, summary = calibrate(past, future, cutoff, params)
    summary.update(cutoff=cutoff, inner_windows=earlier, params=params, base_params=base_params,
        uncertainty="conditional Bayesian predictive variance; not hidden-score calibration")
    name = f"{params['strength']}_{params['uncertainty']}_{params['history_days']}"
    write_json(root / f"posterior_{cutoff}_{name}.json", summary)
    future["factor"] = factor
    future[["route", "date", "factor"]].to_csv(root / f"factors_{cutoff}_{name}.csv", sep=";", index=False)
    base = pd.read_csv(root / "inner" / cutoff / f"raw_{cutoff}.csv", sep=";", parse_dates=["date"], float_precision="round_trip")
    result = base.merge(future[["route", "date", "factor"]], on=["route", "date"], validate="many_to_one")
    result["prediction"] *= result.factor
    return result.drop(columns="factor")
