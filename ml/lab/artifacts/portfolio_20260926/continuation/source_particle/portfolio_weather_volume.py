"""Weather-conditioned daily volume with existing Ridge hourly shapes; retrospective external regime."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from experiments.portfolio_ridge import ACTIVE_ROUTES, add_calendar, design, ridge_forecast
from pipeline import KEYS

WEATHER_PATH = Path("artifacts/portfolio_20260926/continuation/external/weather_2025.csv")


def weather_design(frame, mode, route_season=False):
    base = design(frame, route_season)
    rain = np.log1p(frame.precipitation_sum.to_numpy())
    if mode == "rain":
        return np.column_stack([base, rain])
    # Replace the binary summer effect with continuous climate, scaled in fixed physical units.
    base[:, len(ACTIVE_ROUTES) * 3 + 6] = 0
    temperature = frame.temperature_2m_mean.to_numpy() / 10
    daylight = frame.daylight_duration.to_numpy() / 21600
    work = frame.is_workday.to_numpy(float)
    extra = [rain, temperature, daylight, temperature * work, daylight * work]
    if mode == "spline":
        extra += [np.maximum(temperature - 1, 0), np.maximum(temperature - 2, 0)]
    return np.column_stack([base, *extra])


def fit_weather_volume(history, cutoff, params, weather):
    if weather.date.duplicated().any():
        raise ValueError("Duplicate weather date")
    train = history.loc[history.date.le(cutoff) & history.route.ne(5)].copy()
    daily = add_calendar(train.groupby(["route", "date"], as_index=False).boardings.sum(), cutoff)
    daily = daily.merge(weather, on="date", validate="many_to_one")
    if not np.isfinite(daily[["temperature_2m_mean", "precipitation_sum", "daylight_duration"]]).all().all():
        raise ValueError("Incomplete training weather")
    normal = daily.groupby(["route", "daytype"]).boardings.transform("median")
    fit = daily.loc[~daily.off & daily.boardings.gt(np.maximum(500, 0.35 * normal))].copy()
    model = Ridge(alpha=params["alpha"]).fit(weather_design(fit, params["mode"], params.get("route_season", False)), np.log1p(fit.boardings))
    fit["residual"] = np.log1p(fit.boardings) - model.predict(weather_design(fit, params["mode"], params.get("route_season", False)))
    return train, fit, model


def weather_forecast(history, cutoff, end, params, weather=None):
    if weather is None:
        weather = pd.read_csv(WEATHER_PATH, sep=";", parse_dates=["date"])
    train, fit, model = fit_weather_volume(history, cutoff, params, weather)
    correction = fit.loc[fit.date.gt(pd.Timestamp(cutoff) - pd.Timedelta(days=14))].groupby("route").residual.median()
    shape = ridge_forecast(train, cutoff, end, dict(correction_days=14, shape_mix=0.5, route_season=False))
    future = add_calendar(shape.loc[shape.route.ne(5), ["route", "date"]].drop_duplicates(), cutoff)
    future = future.merge(weather, on="date", validate="many_to_one")
    original_off = future.off.to_numpy()
    future.loc[original_off, "weekday"] = 6
    future.loc[original_off, "daytype"] = 2
    future.loc[original_off, "is_workday"] = False
    future.loc[original_off, "off"] = False
    # The forecast receives only exogenous future weather, never future boardings.
    future["volume"] = np.maximum(0, np.expm1(model.predict(weather_design(future, params["mode"], params.get("route_season", False)))
        + params["residual_strength"] * future.route.map(correction).fillna(0)))
    future.loc[original_off, "volume"] *= 0.95
    if not np.isfinite(future.volume).all():
        raise ValueError("Incomplete forecast weather")
    result = shape.merge(future[["route", "date", "volume"]], on=["route", "date"], how="left", validate="many_to_one")
    denominator = result.groupby(["route", "date"]).prediction.transform("sum")
    result["prediction"] = result.prediction / denominator.where(denominator.gt(0)) * result.volume
    result.loc[result.route.eq(5), "prediction"] = 0
    if not np.isfinite(result.prediction).all():
        raise ValueError("Missing daily shape or volume")
    return result[KEYS + ["prediction"]]
