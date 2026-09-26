"""Checks the calendar adapter without model downloads."""
import numpy as np
import pandas as pd

from chronos_calendar_experiment import calendar_for, hourly_forecast, model_inputs
from pipeline import KEYS, full_grid


def check():
    history = full_grid("2025-01-01", "2025-10-31").to_frame(index=False)
    history["date"] = pd.to_datetime(history.date)
    history["boardings"] = history.route * (history.hour + 1) + history.date.dt.dayofweek
    history.loc[history.route.eq(5) | history.hour.between(1, 4), "boardings"] = 0
    poisoned = history.copy()
    poisoned.loc[poisoned.date.gt("2025-04-30"), "boardings"] = 9999999
    for cov in [False, True]:
        a = model_inputs(history, "2025-04-30", "2025-06-30", cov)
        b = model_inputs(poisoned, "2025-04-30", "2025-06-30", cov)
        pd.testing.assert_frame_equal(a[0], b[0])
        for x, y in zip(a[3], b[3]):
            if not cov:
                np.testing.assert_array_equal(x, y)
                continue
            assert set(x) == {"target", "past_covariates", "future_covariates"}
            np.testing.assert_array_equal(x["target"], y["target"])
            assert x["target"].shape == (120,)
            for side, size in [("past_covariates", 120), ("future_covariates", 61)]:
                for name in x[side]:
                    np.testing.assert_array_equal(x[side][name], y[side][name])
                    assert x[side][name].shape == (size,)
        train, routes, dates, _ = a
        daily = pd.concat([pd.DataFrame(dict(route=r, date=dates, prediction=2000.5)) for r in routes])
        for shares in [False, True]:
            actual = hourly_forecast(train, daily, "2025-04-30", shares)
            assert len(actual) == 14640 and not actual.duplicated(KEYS).any()
            assert actual.loc[actual.route.eq(5) | actual.hour.between(1, 4), "prediction"].eq(0).all()
            assert np.isfinite(actual.prediction).all() and actual.prediction.ge(0).all()
    # November includes an unseen working-weekend type and December a transferred day off.
    train, routes, dates, _ = model_inputs(history, "2025-10-31", "2025-12-31", True)
    daily = pd.concat([pd.DataFrame(dict(route=r, date=dates, prediction=2000.5)) for r in routes])
    assert len(hourly_forecast(train, daily, "2025-10-31", True)) == 14640
    cal = calendar_for(pd.to_datetime(["2025-05-01", "2025-05-02", "2025-11-01"]), "2025-04-30")
    assert cal.is_holiday.tolist() == [True, False, False]
    assert cal.is_workday.tolist() == [False, False, True]
    for date, cutoff in [("2026-01-01", "2025-10-31"), ("2025-01-01", "2024-10-03")]:
        try:
            calendar_for(pd.to_datetime([date]), cutoff)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid calendar availability/coverage accepted")
    print("Calendar Chronos checks passed: known future inputs, cutoff, conservation, fallback, grid and zeros.")


if __name__ == "__main__":
    check()
