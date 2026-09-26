"""Run: python test_calendar_experiment.py."""
import numpy as np
import pandas as pd

from calendar_experiment import calendar_predict, calendar_table
from pipeline import KEYS, full_grid


def check():
    calendar = calendar_table().set_index("date")
    assert len(calendar) == 365 and calendar.index.is_unique
    assert calendar.is_workday.sum() == 247
    expected = {
        "2025-01-04": ("holiday", False), "2025-01-05": ("holiday", False),
        "2025-02-23": ("holiday", False), "2025-02-24": ("weekday_0", True),
        "2025-03-08": ("holiday", False), "2025-03-10": ("weekday_0", True),
        "2025-05-02": ("transferred_off", False), "2025-05-08": ("transferred_off", False),
        "2025-06-13": ("transferred_off", False), "2025-11-01": ("transferred_work", True),
        "2025-11-03": ("transferred_off", False), "2025-11-04": ("holiday", False),
        "2025-12-31": ("transferred_off", False),
    }
    for date, pair in expected.items():
        row = calendar.loc[date]
        assert (row.day_type, row.is_workday) == pair
    # Three regular Thursdays; rare holiday and unseen transfer types use broad groups.
    train = pd.DataFrame(dict(route=1, hour=5,
        date=pd.to_datetime(["2025-01-09", "2025-01-16", "2025-01-23", "2025-01-11", "2025-01-12", "2025-01-01"]),
        boardings=[10, 20, 30, 100, 200, 900]))
    keys = pd.DataFrame(dict(route=1, hour=5,
        date=pd.to_datetime(["2025-01-30", "2025-05-01", "2025-05-02", "2025-11-01"])))
    actual = calendar_predict(train, keys, "2025-01-23")
    assert actual.prediction.tolist() == [20, 400, 400, 20]
    fourth_thursday = pd.DataFrame(dict(route=[1], hour=[5], date=pd.to_datetime(["2025-01-30"]), boardings=[22]))
    rounded = calendar_predict(pd.concat([train, fourth_thursday]),
        keys.iloc[:1].assign(date=pd.Timestamp("2025-02-06")), "2025-01-30")
    assert rounded.prediction.tolist() == [21]  # (10+20+30+22)/4 = 20.5.
    extra = pd.DataFrame(dict(route=1, hour=5, date=pd.to_datetime(["2025-01-02", "2025-01-03"]), boardings=[901, 902]))
    assert calendar_predict(pd.concat([train, extra]), keys, "2025-01-23").prediction.tolist() == [20, 901, 601, 20]
    grid = full_grid("2025-01-01", "2025-10-31").to_frame(index=False)
    grid["date"] = pd.to_datetime(grid.date)
    grid["boardings"] = np.where(grid.date.le("2025-01-19"), 2, 3)
    # Ordinary Mondays before cutoff: Jan 13, 20, 27; mean 8/3 rounds to 3.
    future = full_grid("2025-11-01", "2025-12-31").to_frame(index=False)
    future["date"] = pd.to_datetime(future.date)
    forecast = calendar_predict(grid, future, "2025-01-31")
    active = forecast.route.ne(5) & ~forecast.hour.between(1, 4)
    assert forecast.loc[~active, "prediction"].eq(0).all()
    assert len(forecast) == 14640 and not forecast.duplicated(KEYS).any()
    assert forecast.prediction.dtype == np.dtype("int64") and forecast.prediction.ge(0).all()
    pd.testing.assert_frame_equal(forecast[KEYS], future)
    assert forecast.loc[active & forecast.date.eq("2025-11-10"), "prediction"].eq(3).all()
    poisoned = grid.copy()
    poisoned.loc[poisoned.date.gt("2025-01-31"), "boardings"] = 999999
    pd.testing.assert_frame_equal(forecast, calendar_predict(poisoned, future, "2025-01-31"))
    for bad_keys, cutoff in [(future.assign(date=pd.Timestamp("2026-01-01")), "2025-01-31"),
                             (keys.assign(date=pd.Timestamp("2025-01-01")), "2025-01-23")]:
        try:
            calendar_predict(train, bad_keys.iloc[:1], cutoff)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid forecast dates accepted")
    try:
        calendar_predict(train.assign(date=pd.Timestamp("2024-10-01")).iloc[:1], keys, "2024-10-02")
    except ValueError as error:
        assert "available" in str(error)
    else:
        raise AssertionError("Unavailable calendar accepted")
    print("Calendar checks passed: dates, profiles, fallback, cutoff, rounding, grid and zeros.")


if __name__ == "__main__":
    check()
