"""P25: daily direct multi-horizon regression from strictly causal origin contexts."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from experiments.portfolio_experiment import write_json
from experiments.portfolio_operations import normal_history, seasonal_shape, apply_operations
from experiments.portfolio_ridge import add_calendar
from experiments.portfolio_weather_volume import WEATHER_PATH
from pipeline import KEYS

ROOT = Path("artifacts/portfolio_20260926/continuation/direct_daily")
FEATURES = ["route", "effective_weekday", "season", "off", "horizon", "annual_sin", "annual_cos",
    "log_reference", "ratio7", "ratio14", "ratio28", "temperature_2m_mean", "precipitation_sum",
    "daylight_duration", "origin_temperature", "origin_daylight"]


def daily_history(history, cutoff):
    normal = normal_history(history, cutoff, extended=True)
    return add_calendar(normal.loc[normal.route.ne(5)].groupby(["route", "date"], as_index=False).boardings.sum(), cutoff)


def context(daily, origin, dates, weather):
    origin = pd.Timestamp(origin)
    past = daily.loc[daily.date.le(origin) & ~daily.off].copy()
    recent = past.loc[past.date.gt(origin - pd.Timedelta(days=56))]
    fallback = past.groupby(["route", "effective_weekday"]).boardings.median()
    reference = recent.groupby(["route", "effective_weekday"]).boardings.median().reindex(fallback.index).fillna(fallback).clip(lower=100)
    past = past.merge(reference.rename("reference"), on=["route", "effective_weekday"], validate="many_to_one")
    past["ratio"] = past.boardings / past.reference
    result = pd.MultiIndex.from_product([sorted(past.route.unique()), dates], names=["route", "date"]).to_frame(index=False)
    result = add_calendar(result, origin).merge(reference.rename("reference"), on=["route", "effective_weekday"], validate="many_to_one")
    for days in [7, 14, 28]:
        ratios = past.loc[past.date.gt(origin-pd.Timedelta(days=days))].groupby("route").ratio.median()
        result[f"ratio{days}"] = result.route.map(ratios).fillna(1.)
    result = result.merge(weather, on="date", validate="many_to_one")
    known_weather = weather.loc[weather.date.le(origin) & weather.date.gt(origin - pd.Timedelta(days=14))]
    result["origin_temperature"] = known_weather.temperature_2m_mean.mean()
    result["origin_daylight"] = known_weather.daylight_duration.mean()
    result["origin"] = origin
    result["horizon"] = (result.date-origin).dt.days
    result["season"] = result.date.dt.month % 12 // 3
    result["annual_sin"] = np.sin(2*np.pi*(result.date.dt.dayofyear-1)/365)
    result["annual_cos"] = np.cos(2*np.pi*(result.date.dt.dayofyear-1)/365)
    result["log_reference"] = np.log1p(result.reference)
    if not np.isfinite(result[FEATURES].to_numpy(float)).all():
        raise ValueError("Incomplete direct forecast context")
    return result


def training_examples(history, cutoff, weather):
    daily = daily_history(history, cutoff)
    pieces = []
    for origin in pd.date_range("2025-01-31", pd.Timestamp(cutoff)-pd.Timedelta(days=1), freq="7D"):
        end = min(origin+pd.Timedelta(days=61), pd.Timestamp(cutoff))
        frame = context(daily, origin, pd.date_range(origin+pd.Timedelta(days=1), end), weather)
        frame = frame.merge(daily[["route", "date", "boardings"]], on=["route", "date"], validate="one_to_one")
        pieces.append(frame)
    examples = pd.concat(pieces, ignore_index=True)
    if examples.date.max() > pd.Timestamp(cutoff) or not (examples.date.gt(examples.origin) & examples.horizon.between(1, 61)).all():
        raise ValueError("Training target outside the available origin/horizon")
    return examples


def direct_daily_forecast(history, cutoff, end, params):
    ROOT.mkdir(exist_ok=True)
    weather = pd.read_csv(WEATHER_PATH, sep=";", parse_dates=["date"])
    cache = ROOT / f"examples_{cutoff}.csv"
    if cache.exists():
        examples = pd.read_csv(cache, sep=";", parse_dates=["date", "origin"], float_precision="round_trip")
    else:
        examples = training_examples(history, cutoff, weather)
        examples.to_csv(cache, sep=";", index=False, date_format="%Y-%m-%d")
    if examples.date.max() > pd.Timestamp(cutoff) or not examples.date.gt(examples.origin).all():
        raise ValueError("Future target in direct daily fit")
    repetitions = examples.groupby(["route", "date"]).reference.transform("size")
    weights = examples.reference.to_numpy() / repetitions.to_numpy()
    weights /= weights.mean()
    model = HistGradientBoostingRegressor(loss="absolute_error", max_leaf_nodes=params["leaves"],
        l2_regularization=params["l2"], max_iter=params["max_iter"], min_samples_leaf=50,
        categorical_features=[0, 1, 2], early_stopping=False, random_state=42)
    model.fit(examples[FEATURES].to_numpy(float), examples.boardings.to_numpy()/examples.reference.to_numpy(), sample_weight=weights)
    future = context(daily_history(history, cutoff), cutoff, pd.date_range(pd.Timestamp(cutoff)+pd.Timedelta(days=1), end), weather)
    future["volume"] = future.reference * np.maximum(0, model.predict(future[FEATURES].to_numpy(float)))
    normal = normal_history(history, cutoff, extended=True)
    shape = seasonal_shape(normal, cutoff, end, weather)
    raw = shape.merge(future[["route", "date", "volume"]], on=["route", "date"], how="left", validate="many_to_one")
    denominator = raw.groupby(["route", "date"]).prediction.transform("sum")
    raw["prediction"] = raw.prediction/denominator.where(denominator.gt(0))*raw.volume
    raw.loc[raw.route.eq(5), "prediction"] = 0
    base_params = json.loads(Path("artifacts/portfolio_20260926/continuation/operations/operations/selection.json").read_text())["params"]
    raw = apply_operations(raw[KEYS+["prediction"]], history.loc[history.date.le(cutoff)], normal, base_params)
    write_json(ROOT / f"fit_{cutoff}_{params['leaves']}_{params['l2']}_{params['max_iter']}.json",
        dict(cutoff=cutoff, latest_target=str(examples.date.max().date()), rows=len(examples),
        origins=int(examples.origin.nunique()), target_days=int(examples.date.nunique()), features=FEATURES,
        params=params, loss="MAE normalized ratio weighted reference/count(route,date)", base_params=base_params))
    return raw
