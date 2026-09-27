"""Own implementations of public regime and level/shape ideas, on the frozen clean target."""
import numpy as np
import pandas as pd

from experiments.calendar_experiment import calendar_table
from experiments.portfolio_experiment import complete_raw, forecast_keys
from pipeline import KEYS


def tagged(frame, cutoff):
    calendar = calendar_table()
    if calendar.loc[calendar.date.isin(frame.date), "known_at"].gt(cutoff).any():
        raise ValueError("Calendar unavailable at cutoff")
    result = frame.merge(calendar, on="date", validate="many_to_one")
    result["off"] = result.day_type.isin(["holiday", "transferred_off"])
    result["dow"] = np.where(result.is_workday, np.minimum(result.weekday, 4), result.weekday)
    result["group"] = np.where(result.dow.lt(4), 0, result.dow - 3)
    result["summer"] = result.date.dt.month.between(6, 8)
    return result


def trimmed(values):
    ordered = np.sort(values)
    return float(ordered[1:-1].mean()) if len(ordered) >= 5 else float(ordered.mean())


def regime_forecast(history, cutoff, end, params):
    train = tagged(history.loc[history.date.le(cutoff) & history.route.ne(5)], cutoff)
    train = train.loc[~train.off]
    future = tagged(forecast_keys(cutoff, end), cutoff)
    pieces = []
    for season, target in future.groupby("summer"):
        same = train.loc[train.summer.eq(season)] if params["regime"] else train
        pool = same if len(same) else train
        pool = pool.sort_values("date").groupby(["route", "dow", "hour"]).tail(params["occurrences"])
        statistic = pool.groupby(["route", "dow", "hour"]).boardings.agg(
            "mean" if params["statistic"] == "mean" else trimmed)
        result = target.merge(statistic.rename("prediction"), on=["route", "dow", "hour"], how="left")
        weekend = statistic.loc[statistic.index.get_level_values("dow").isin([5, 6])].groupby(["route", "hour"]).mean()
        holiday = result[KEYS].merge(weekend.rename("holiday"), on=["route", "hour"], how="left").holiday
        result.loc[result.off, "prediction"] = 0.95 * holiday[result.off].to_numpy()
        result.loc[result.route.eq(5), "prediction"] = 0.
        pieces.append(result[KEYS + ["prediction"]])
    return complete_raw(pd.concat(pieces, ignore_index=True), cutoff, end)


def keep_days(values):
    """Remove isolated large log deviations; retain adjacent same-sign shifts."""
    values = np.asarray(values)
    if len(values) < 4:
        return np.ones(len(values), dtype=bool)
    logged = np.log1p(values)
    median = np.median(logged)
    mad = max(1.4826 * np.median(np.abs(logged - median)), 0.02)
    z = (logged - median) / mad
    flags = np.abs(z) > 3
    for i in np.flatnonzero(flags):
        neighbours = [j for j in [i-1, i+1] if 0 <= j < len(z)]
        if any(np.sign(z[j]) == np.sign(z[i]) and abs(z[j]) > 1.5 for j in neighbours):
            flags[i] = False
    return ~flags


def structure_forecast(history, cutoff, end, params):
    train = tagged(history.loc[history.date.le(cutoff) & history.route.ne(5)], cutoff)
    train = train.loc[~train.off]
    daily = train.groupby(["route", "date", "dow", "group", "summer"], as_index=False).boardings.sum()
    weekdays = daily.loc[daily.dow.lt(4)].copy()
    weekdays["normal"] = weekdays.groupby("route").boardings.transform("median").clip(lower=1)
    weekday_factors = (weekdays.boardings / weekdays.normal).groupby(weekdays.dow).median()
    weekday_factors = (weekday_factors / weekday_factors.mean()).clip(0.85, 1.15)
    future = tagged(forecast_keys(cutoff, end), cutoff)
    pieces = []
    for route, target in future.loc[future.route.ne(5)].groupby("route"):
        past = daily.loc[daily.route.eq(route)].sort_values("date")
        past_hourly = train.loc[train.route.eq(route)].copy()
        past_hourly["total"] = past_hourly.groupby("date").boardings.transform("sum")
        past_hourly["share"] = past_hourly.boardings / past_hourly.total.where(past_hourly.total.gt(0))
        for season, section in target.groupby("summer"):
            same = past.loc[past.summer.eq(season)]
            pool = same if params["regime"] and len(same) >= 14 else past
            shapes = past_hourly.loc[past_hourly.date.isin(pool.date)]
            levels, profiles = {}, {}
            for group in range(4):
                chosen = pool.loc[pool.group.eq(group)].tail(params["days"] if group == 0 else max(4, params["days"] // 7))
                if chosen.empty:
                    chosen = past.loc[past.group.eq(group)]
                selected = chosen.boardings.to_numpy(float)
                if group == 0:
                    selected /= chosen.dow.map(weekday_factors).to_numpy()
                retained = keep_days(selected)
                weights = np.exp2(-(chosen.date.max() - chosen.date).dt.days.to_numpy() / params["half_life"])
                levels[group] = float(np.average(selected[retained], weights=weights[retained]))
                if group >= 2 and levels[group] < 0.1 * levels[0]:
                    levels[group] = float(np.median(selected))
                shape = shapes.loc[shapes.group.eq(group)]
                shape = shape.loc[shape.date.gt(shape.date.max() - pd.Timedelta(weeks=params["shape_weeks"]))]
                profile = shape.groupby("hour").share.median().fillna(0).reindex(range(24), fill_value=0).to_numpy()
                if profile.sum() == 0:
                    profile = past_hourly.loc[past_hourly.group.eq(group)].groupby("hour").boardings.sum().reindex(range(24), fill_value=0).to_numpy(float)
                profiles[group] = profile / max(profile.sum(), 1e-12)
            result = section[KEYS].copy()
            factors = np.where(section.dow.lt(4), section.dow.map(weekday_factors).fillna(1), 1)
            prediction = np.array([levels[int(g)] * profiles[int(g)][int(h)] for g, h in zip(section.group, section.hour)]) * factors
            off = section.off.to_numpy()
            # ponytail: pooled Saturday/Sunday holiday prior; distinguish individual holidays with more years.
            prediction[off] = 0.95 * ((levels[2] * profiles[2][section.hour.to_numpy()[off]]
                                     + levels[3] * profiles[3][section.hour.to_numpy()[off]]) / 2)
            result["prediction"] = prediction
            pieces.append(result)
    return complete_raw(pd.concat(pieces, ignore_index=True), cutoff, end)
