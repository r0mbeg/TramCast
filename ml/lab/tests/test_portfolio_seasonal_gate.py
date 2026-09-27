"""Small check for past-only seasonal coverage and repeated forecast targets."""
import pandas as pd
from experiments.portfolio_seasonal_gate import support
from experiments.portfolio_windows import season


def check():
    dates = pd.Series(pd.date_range("2025-01-01", periods=12, freq="MS"))
    assert season(dates).tolist() == [0,0,1,1,1,2,2,2,3,3,3,0]
    past = pd.DataFrame(dict(route=[1,1,1], date=pd.to_datetime(["2025-03-03","2025-03-03","2025-03-10"]),
        effective_weekday=[0,0,0], origin=pd.to_datetime(["2025-01-01"]*3), base=[100.]*3, boardings=[80.]*3))
    future = pd.DataFrame(dict(route=[1,1,7,1],
        date=pd.to_datetime(["2025-05-05","2025-06-02","2025-05-05","2025-05-06"]), effective_weekday=[0,0,0,1]))
    result = support(past, future, "2025-04-30")
    assert result.past_dates.tolist() == [2,0,0,0]
    assert result.supported.tolist() == [True,False,False,False]
    assert not support(past.iloc[:2], future, "2025-04-30").supported.any()
    pd.testing.assert_frame_equal(result, support(past.assign(boardings=999999.), future, "2025-04-30"))
    try:
        support(past.assign(date=pd.Timestamp("2025-05-01")), future, "2025-04-30")
    except ValueError:
        pass
    else:
        raise AssertionError("Seasonal coverage used future facts")
    print("Seasonal gate checks passed: four seasons, distinct past dates, route/weekday fallback, future guard.")


if __name__ == "__main__":
    check()
