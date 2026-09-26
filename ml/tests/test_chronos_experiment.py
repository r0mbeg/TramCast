"""Lightweight adapter checks without downloading or running Chronos."""
import numpy as np
import pandas as pd

from experiments.chronos_experiment import distribute_daily, forecast, inputs
from pipeline import KEYS, full_grid, predict


def check():
    data = full_grid("2025-01-01", "2025-04-30").to_frame(index=False)
    data["date"] = pd.to_datetime(data.date)
    data["boardings"] = data.route * 10 + data.hour + data.date.dt.dayofweek
    data.loc[data.route.eq(5) | data.hour.between(1, 4), "boardings"] = 0
    cutoff, end = "2025-02-28", "2025-04-30"
    seen = []

    def fake(matrix, horizon):
        seen.append(matrix.copy())
        return np.full((len(matrix), horizon), 50.5)

    poisoned = data.copy()
    poisoned.loc[poisoned.date.gt(cutoff), "boardings"] = 99999999
    for representation, count in [("daily_total", 9), ("route_hour", 180)]:
        a = forecast(data, cutoff, end, representation, fake)
        b = forecast(poisoned, cutoff, end, representation, fake)
        pd.testing.assert_frame_equal(a, b)
        np.testing.assert_array_equal(seen[-1], seen[-2])
        assert seen[-1].shape == (count, 59)
        assert len(a) == 61 * 24 * 10 and not a.duplicated(KEYS).any()
        assert a.loc[a.route.eq(5) | a.hour.between(1, 4), "prediction"].eq(0).all()
        if representation == "route_hour":
            assert a.loc[a.route.ne(5) & ~a.hour.between(1, 4), "prediction"].eq(51).all()
    train, series = inputs(data, cutoff, "daily_total")
    daily = series.stack().rename("prediction").reset_index()
    distributed = distribute_daily(train, daily)
    sums = distributed.groupby(["date", "route"]).prediction.sum().sort_index()
    np.testing.assert_allclose(sums, daily.set_index(["date", "route"]).prediction.sort_index())
    # A weekday daily mean distributed with volume-weighted weekday shares equals mean_all.
    future = data.loc[data.date.gt(cutoff), KEYS].copy()
    profile = daily.assign(weekday=daily.date.dt.dayofweek).groupby(["route", "weekday"]).prediction.mean()
    dkeys = future[future.route.ne(5)][["route", "date"]].drop_duplicates()
    dkeys["weekday"] = dkeys.date.dt.dayofweek
    control = distribute_daily(train, dkeys.merge(profile, on=["route", "weekday"], validate="many_to_one"))
    baseline = predict(data, future, cutoff, 0, "mean")
    joined = control.merge(baseline, on=KEYS, suffixes=("_control", "_baseline"))
    np.testing.assert_array_equal(np.floor(joined.prediction_control + 0.5), joined.prediction_baseline)
    print("Chronos adapter checks passed: cutoff isolation, grid, shares, control, zeros, rounding.")


if __name__ == "__main__":
    check()
