"""Own cutoff-safe implementation inspired by Manticore's public daily Ridge recipe.

Source: github.com/Manticore-MT/tram-forecast, commit e1fbd17 (MIT).
Uses TramCast cleaned history, calendar snapshot and half-up, not their fitted artifact.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from experiments.calendar_experiment import calendar_table
from experiments.portfolio_experiment import complete_raw
from pipeline import KEYS, ROUTES

ACTIVE_ROUTES = [r for r in ROUTES if r != 5]


def add_calendar(frame, cutoff):
    calendar = calendar_table()
    if calendar.loc[calendar.date.isin(frame.date), "known_at"].gt(cutoff).any():
        raise ValueError("Calendar unavailable at cutoff")
    result = frame.merge(calendar[["date", "weekday", "day_type", "is_workday"]],
                         on="date", how="left", validate="many_to_one")
    if result.day_type.isna().any():
        raise ValueError("Dates outside the calendar snapshot")
    result["off"] = result.day_type.isin(["holiday", "transferred_off"])
    result["daytype"] = np.where(result.is_workday, 0, np.where(result.weekday.eq(5) & ~result.off, 1, 2))
    result["effective_weekday"] = np.where(result.is_workday, np.minimum(result.weekday, 4),
                                          np.where(result.daytype.eq(1), 5, 6))
    result["summer"] = result.date.dt.month.isin([6, 7, 8]).astype(int)
    return result


def design(frame, route_season):
    columns = [frame.route.eq(r).astype(float).to_numpy() for r in ACTIVE_ROUTES]
    columns += [(frame.route.eq(r) & frame.daytype.eq(t)).astype(float).to_numpy()
                for r in ACTIVE_ROUTES for t in [1, 2]]
    columns += [frame.weekday.eq(d).astype(float).to_numpy() for d in range(1, 7)]
    columns += [frame.summer.to_numpy(float), frame.off.to_numpy(float),
                ((frame.date.dt.month == 1) & (frame.date.dt.day <= 8)).to_numpy(float)]
    if route_season:
        columns += [(frame.route.eq(r) * frame.summer).to_numpy(float) for r in ACTIVE_ROUTES]
    return np.column_stack(columns)


def ridge_forecast(history, cutoff, end, params):
    train = history.loc[history.date.le(cutoff) & history.route.ne(5)].copy()
    train = add_calendar(train, cutoff)
    daily = add_calendar(train.groupby(["route", "date"], as_index=False).boardings.sum(), cutoff)
    normal = daily.groupby(["route", "daytype"]).boardings.transform("median")
    usable = daily.boardings.gt(np.maximum(500, 0.35 * normal))
    fit = daily.loc[usable].copy()
    # ponytail: one summer indicator from one year; no claim of a learned annual cycle.
    model = Ridge(alpha=1.0).fit(design(fit, params["route_season"]), np.log1p(fit.boardings))
    fit["residual"] = np.log1p(fit.boardings) - model.predict(design(fit, params["route_season"]))
    recent = fit.loc[fit.date > pd.Timestamp(cutoff) - pd.Timedelta(days=params["correction_days"])]
    correction = recent.groupby("route").residual.median()
    dates = pd.date_range(pd.Timestamp(cutoff) + pd.Timedelta(days=1), end)
    future = pd.MultiIndex.from_product([ACTIVE_ROUTES, dates], names=["route", "date"]).to_frame(index=False)
    future = add_calendar(future, cutoff)
    future["prediction"] = np.maximum(0, np.expm1(model.predict(design(future, params["route_season"]))
                                                  + future.route.map(correction).fillna(0)))

    total = train.groupby(["route", "date"]).boardings.transform("sum")
    shape = train.loc[total.gt(500)].copy()
    shape["share"] = shape.boardings / total.loc[shape.index]
    group = ["route", "effective_weekday", "hour"]
    seasonal = shape.groupby(["route", "summer", "effective_weekday", "hour"]).share.median().rename("seasonal")
    fallback = shape.groupby(group).share.median().rename("fallback")
    broad = shape.groupby(["route", "hour"]).share.median().rename("broad")
    short = shape.loc[shape.date > pd.Timestamp(cutoff) - pd.Timedelta(days=56)]
    recent_shape = short.groupby(group).share.median().rename("recent")
    raw = future.merge(pd.DataFrame({"hour": np.arange(24)}), how="cross")
    raw = raw.merge(seasonal, on=["route", "summer", "effective_weekday", "hour"], how="left", validate="many_to_one")
    raw = raw.merge(fallback, on=group, how="left", validate="many_to_one")
    raw = raw.merge(recent_shape, on=group, how="left", validate="many_to_one")
    raw = raw.merge(broad, on=["route", "hour"], how="left", validate="many_to_one")
    raw["share"] = raw.seasonal.fillna(raw.fallback).fillna(raw.broad)
    raw["share"] = params["shape_mix"] * raw.share + (1 - params["shape_mix"]) * raw.recent.fillna(raw.share)
    if not np.isfinite(raw.share).all():
        raise ValueError("Missing calendar/hour profile")
    raw["share"] /= raw.groupby(["route", "date"]).share.transform("sum")
    raw["prediction"] *= raw.share
    return complete_raw(raw[KEYS + ["prediction"]], cutoff, end)
