"""Run: python -m tests.test_model."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from model import ABSOLUTE_CALENDAR, FEATURES, RESIDUAL_LIMIT, features, forecast, training_examples
from pipeline import full_grid, postprocess, predict


def check():
    data = full_grid("2025-01-01", "2025-05-31").to_frame(index=False)
    data["date"] = pd.to_datetime(data.date)
    data["boardings"] = data.route * 10 + data.hour
    future = data[data.date.gt("2025-04-30")]
    original = features(data, future, "2025-04-30")
    poisoned = data.copy()
    poisoned.loc[poisoned.date.gt("2025-04-30"), "boardings"] = 999999
    pd.testing.assert_frame_equal(original, features(poisoned, future, "2025-04-30"))
    x, y, origins = training_examples(data, "2025-04-30")
    xp, yp, op = training_examples(poisoned, "2025-04-30")
    pd.testing.assert_frame_equal(x, xp)
    pd.testing.assert_series_equal(y, yp)
    assert origins == op and x.horizon.min() == 1 and x.horizon.max() == 62
    assert not x.route.eq(5).any() and not x.hour.between(1, 4).any()
    full_x, full_y, full_origins = training_examples(data, "2025-05-31")
    reduced_x, reduced_y, reduced_origins = training_examples(data, "2025-05-31", True)
    assert full_origins[0] == "2025-01-31" and reduced_origins == full_origins[1:]
    removed_rows = 9 * 20 * 62
    pd.testing.assert_frame_equal(reduced_x, full_x.iloc[removed_rows:].reset_index(drop=True))
    pd.testing.assert_series_equal(reduced_y, full_y.iloc[removed_rows:].reset_index(drop=True))
    # A profile must stop at its own origin, not merely at the outer training cutoff.
    jan = features(data, data[data.date.eq("2025-02-01")], "2025-01-31")
    poisoned.loc[poisoned.date.gt("2025-01-31"), "boardings"] = 888888
    pd.testing.assert_frame_equal(jan, features(poisoned, data[data.date.eq("2025-02-01")], "2025-01-31"))
    for selected in [FEATURES, [name for name in FEATURES if name not in ABSOLUTE_CALENDAR]]:
        model = HistGradientBoostingRegressor(max_iter=2, early_stopping=False, random_state=42).fit(x[selected], y)
        assert list(model.feature_names_in_) == selected
        prediction = forecast(model, data, future, "2025-04-30")
        assert len(prediction) == len(future) and prediction.prediction.ge(0).all()
        assert prediction.prediction.dtype.kind == "i"
        assert prediction.loc[prediction.route.eq(5) | prediction.hour.between(1, 4), "prediction"].eq(0).all()
    assert list(x.columns) == FEATURES
    selected = [name for name in FEATURES if name not in ABSOLUTE_CALENDAR]
    residual = HistGradientBoostingRegressor(max_iter=2, early_stopping=False, random_state=42).fit(
        x[selected], y - x.mean_all)
    residual.residual_limit_ = RESIDUAL_LIMIT
    corrected = forecast(residual, data, future, "2025-04-30")
    profile = features(data, future, "2025-04-30")
    for value, multiplier in [(0, 1), (1e9, 1 + RESIDUAL_LIMIT), (-1e9, 1 - RESIDUAL_LIMIT)]:
        stub = SimpleNamespace(feature_names_in_=selected, residual_limit_=RESIDUAL_LIMIT,
                               predict=lambda frame, v=value: np.full(len(frame), v))
        result = forecast(stub, data, future, "2025-04-30")
        expected = postprocess(profile.assign(prediction=profile.mean_all * multiplier))
        pd.testing.assert_frame_equal(result, expected)
        if value == 0:
            pd.testing.assert_frame_equal(result, predict(data, future, "2025-04-30", 0, "mean"))
    lower = postprocess(profile.assign(prediction=profile.mean_all * (1 - RESIDUAL_LIMIT)))
    upper = postprocess(profile.assign(prediction=profile.mean_all * (1 + RESIDUAL_LIMIT)))
    assert corrected.prediction.between(lower.prediction, upper.prediction).all()
    zero = data.assign(boardings=0)
    assert forecast(stub, zero, future, "2025-04-30").prediction.eq(0).all()
    print("ML checks passed: cutoff isolation, origin-safe profiles, training grid, integer zeros.")


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        check()
