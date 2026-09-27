"""Run: python -m tests.test_pipeline."""
import numpy as np
import pandas as pd

from pipeline import aggregate, full_grid, metrics, predict, repeat_week


def check():
    frame = pd.DataFrame({
        "tran_date_time": ["2025-08-31 " + t for t in
                           ["00:59:59", "01:00:00", "05:29:59", "05:30:00", "05:31:00", "06:00:00"]]
                          + ["2025-09-01 00:00:00"],
        "ngpt_route": ["50 трамвай"] * 5 + ["5 трамвай", "50 трамвай"],
        "validation_result": [1, 1, 1, 1, 31, 1, 1],
    })
    raw, clean, present, counts = aggregate(frame, "2025-01-01", "2025-09-01")
    assert raw.sum() == 5 and clean.sum() == 2 and present.sum() == 4
    assert clean.to_dict() == {(50, "2025-08-31", 0): 1, (50, "2025-08-31", 5): 1}
    assert counts["outside_period"] == 1 and counts["removed_nonworking"] == 2
    assert counts["removed_route5_working"] == 1 and counts["unsuccessful_in_period"] == 1
    combined_raw, combined_clean, _, _ = aggregate(frame, "2025-01-01", "2025-11-01")
    assert combined_raw.loc[(50, "2025-09-01", 0)] == 1
    assert combined_clean.loc[(50, "2025-09-01", 0)] == 1
    november = frame.iloc[[-1]].assign(tran_date_time="2025-11-01 00:00:00")
    assert aggregate(november, "2025-01-01", "2025-11-01")[0].empty
    try:
        aggregate(frame.assign(ngpt_route="unknown"), "2025-01-01", "2025-09-01")
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid route accepted")
    assert metrics([10, 0], [8, 2])["wape_score"] == 0.6
    assert metrics([0], [1])["wape_score"] is None
    grid = full_grid("2025-01-01", "2025-02-28").to_frame(index=False)
    grid["date"] = pd.to_datetime(grid.date)
    grid["boardings"] = np.where(grid.date.lt("2025-01-08"), 2, 3)
    future = full_grid("2025-03-01", "2025-03-07").to_frame(index=False)
    future["date"] = pd.to_datetime(future.date)
    # First two weeks have 2 and 3 for each weekday: mean 2.5 must become 3.
    forecast = predict(grid, future, "2025-01-14", 0, "mean")
    operating = forecast.route.ne(5) & ~forecast.hour.between(1, 4)
    assert forecast.loc[operating, "prediction"].eq(3).all()
    assert forecast.loc[~operating, "prediction"].eq(0).all()
    poisoned = grid.copy()
    poisoned.loc[poisoned.date.gt("2025-01-14"), "boardings"] = 1000000
    pd.testing.assert_frame_equal(forecast, predict(poisoned, future, "2025-01-14", 0, "mean"))
    assert len(forecast) == 10 * 7 * 24 and not forecast.duplicated(["route", "date", "hour"]).any()
    history = full_grid("2025-06-01", "2025-10-31").to_frame(index=False)
    history["date"] = pd.to_datetime(history.date)
    history["boardings"] = history.route * 10000 + history.date.dt.dayofyear * 24 + history.hour
    for cutoff, end, source_end in [("2025-06-30", "2025-08-31", "2025-06-29"),
                                   ("2025-08-31", "2025-10-31", "2025-08-31")]:
        keys = history[history.date.gt(cutoff) & history.date.le(end)]
        repeated = repeat_week(history, keys, cutoff)
        source_dates = pd.Timestamp(source_end) - pd.to_timedelta(6 - repeated.date.dt.dayofweek, unit="D")
        expected = repeated.route * 10000 + source_dates.dt.dayofyear * 24 + repeated.hour
        expected.loc[repeated.route.eq(5) | repeated.hour.between(1, 4)] = 0
        np.testing.assert_array_equal(repeated.prediction, expected)
        poisoned = history.copy()
        poisoned.loc[~poisoned.date.between(pd.Timestamp(source_end) - pd.Timedelta(days=6), source_end), "boardings"] = 9999999
        pd.testing.assert_frame_equal(repeated, repeat_week(poisoned, keys, cutoff))
        try:
            repeat_week(history.drop(history[history.date.eq(source_end)].index[0]), keys, cutoff)
        except ValueError:
            pass
        else:
            raise AssertionError("Incomplete source week accepted")
    print("Checks passed: boundaries, success filter, route 5, rounding, WAPE, cutoff isolation.")


if __name__ == "__main__":
    check()
