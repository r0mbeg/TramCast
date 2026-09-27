"""P20: documented operations, route summer levels and climate-selected seasonal shape."""
import numpy as np
import pandas as pd

from experiments.portfolio_movement import disrupted
from experiments.portfolio_structure import regime_forecast, tagged
from experiments.portfolio_weather_volume import WEATHER_PATH, weather_forecast
from pipeline import KEYS


def seasonal_shape(history, cutoff, end, weather):
    train = history.loc[history.date.le(cutoff)].copy()
    train["season"] = train.date.dt.month % 12 // 3
    calendar_train = tagged(train, cutoff)
    climate = weather.copy()
    climate["season"] = climate.date.dt.month % 12 // 3
    cols = ["temperature_2m_mean", "daylight_duration"]
    past = climate.loc[climate.date.isin(train.date)].groupby("season")[cols].mean()
    pieces = []
    future_dates = pd.date_range(pd.Timestamp(cutoff) + pd.Timedelta(days=1), end)
    for quarter in np.unique(future_dates.month % 12 // 3):
        selected = quarter
        if quarter not in past.index:
            target = climate.loc[climate.date.isin(future_dates) & climate.season.eq(quarter), cols].mean()
            distances = ((past - target) / [10, 21600]).pow(2).sum(axis=1)
            selected = int(distances.idxmin())
        pool = train.loc[train.season.eq(selected)]
        available = tagged(pool, cutoff).loc[lambda x: ~x.off, ["route", "dow"]].drop_duplicates()
        fallback = calendar_train.merge(available.assign(present=True), on=["route", "dow"], how="left")
        pool = pd.concat([pool, fallback.loc[fallback.present.isna(), train.columns]], ignore_index=True)
        # ponytail: one climate analogue per unseen quarter; revise with multiple years of history.
        raw = regime_forecast(pool, cutoff, end, dict(occurrences=32, statistic="mean", regime=False))
        pieces.append(raw.loc[(raw.date.dt.month % 12 // 3).eq(quarter)])
    return pd.concat(pieces, ignore_index=True).sort_values(KEYS).reset_index(drop=True)


def normal_history(history, cutoff, extended=False, august7=False):
    train = history.loc[history.date.le(cutoff)].copy()
    changed = disrupted(train, "july") | disrupted(train, "autumn")
    if august7:
        changed |= disrupted(train, "august7")
    if extended:
        changed |= train.route.eq(17) & train.date.between("2025-04-05", "2025-04-30") & train.date.dt.dayofweek.ge(5)
    return train.loc[~changed].copy()


def operations_forecast(history, cutoff, end, params, weather=None):
    if weather is None:
        weather = pd.read_csv(WEATHER_PATH, sep=";", parse_dates=["date"])
    train = history.loc[history.date.le(cutoff)].copy()
    normal = normal_history(history, cutoff, params.get("extended_ops", False), "august7" in params)
    volume_params = dict(alpha=1, mode="rain", residual_strength=0, route_season=params["route_season"])
    volume_params.update(params.get("volume_params", {}))
    raw = weather_forecast(normal, cutoff, end, volume_params, weather)
    if params["shape"] == "season":
        shape = seasonal_shape(normal, cutoff, end, weather)
        volume = raw.groupby(["route", "date"]).prediction.sum().rename("volume")
        shape = shape.merge(volume, on=["route", "date"], validate="many_to_one")
        denominator = shape.groupby(["route", "date"]).prediction.transform("sum")
        shape["prediction"] = shape.prediction / denominator.where(denominator.gt(0)) * shape.volume
        shape.loc[shape.route.eq(5), "prediction"] = 0
        raw = shape[KEYS + ["prediction"]]
    return apply_operations(raw, train, normal, params)


def apply_operations(raw, train, normal, params):
    daily = train.groupby(["route", "date"], as_index=False).boardings.sum()
    ordinary = normal.groupby(["route", "date"], as_index=False).boardings.sum()
    ordinary["weekday"] = ordinary.date.dt.dayofweek
    profile = ordinary.groupby(["route", "weekday"]).boardings.median().clip(lower=1)
    for event in ["july", "autumn"] + (["august7"] if "august7" in params else []):
        observed = daily.loc[disrupted(daily, event)].copy()
        observed["weekday"] = observed.date.dt.dayofweek
        observed = observed.merge(profile.rename("expected"), on=["route", "weekday"], validate="many_to_one")
        observed["ratio"] = observed.boardings / observed.expected
        for route in ([7] if event == "august7" else [7, 50]):
            group = observed.loc[observed.route.eq(route)]
            prior = (params["august7"] if event == "august7" else
                params[f"july{route}"] if event == "july" else (0.6 if route == 7 else 0))
            maximum = 2 if event == "july" and route == 50 else 1
            factor = float(np.clip(group.ratio.median(), 0, maximum)) if len(group) >= 3 else prior
            if event == "autumn" and route == 50:
                factor = 0
            raw.loc[disrupted(raw, event) & raw.route.eq(route), "prediction"] *= factor
    if not np.isfinite(raw.prediction).all():
        raise ValueError("Invalid operations forecast")
    return raw
