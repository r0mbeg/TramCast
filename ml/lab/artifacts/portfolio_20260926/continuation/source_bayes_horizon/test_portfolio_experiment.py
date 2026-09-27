"""Small runnable leakage/rounding checks for the portfolio mechanisms."""
import numpy as np
import pandas as pd
from io import StringIO

from experiments.portfolio_experiment import cpu_forecast, impute_history, forecast_keys, WINDOWS
from experiments.portfolio_chronos import gpu_forecast, METHODS
from experiments.portfolio_movement import movement_forecast
from experiments.portfolio_combine import mix, transplant, route_weights
from experiments.portfolio_windows import analogue_forecast, inner_windows, season
from experiments.portfolio_bayes import MODELS, infer_weights
from experiments.portfolio_weather_volume import weather_forecast
from experiments.portfolio_chronos_weather import tasks_with_weather
from experiments.portfolio_operations import operations_forecast, normal_history
from experiments.portfolio_bayes_volume import calibrate
from experiments.portfolio_fares import cohort
from experiments.portfolio_particle import filter_level
from experiments.portfolio_direct_daily import daily_history, context, training_examples, FEATURES
from experiments.portfolio_finetune import task_inputs
from experiments.portfolio_conditional_shape import shape_forecast
from pipeline import KEYS, full_grid, postprocess


def check():
    assert cohort(pd.Series(["30дн ММ+МГТ СКС ", "СКМ МГТ", "КОШЕЛЕК", None, "СКМ студент"])).tolist() == [
        "education", "social", "other", "other", "education"]
    level = np.r_[np.zeros(40), np.full(30, -0.25)]
    tracked = filter_level(level, np.ones(len(level)), 1e-3, 0.02)
    assert abs(tracked["level"] + 0.25) < 0.05 and tracked["sd"] > 0
    isolated = np.zeros(70)
    isolated[-1] = -1
    robust = filter_level(isolated, np.ones(len(isolated)), 1e-4, 0.0)
    assert abs(robust["level"]) < 0.15
    for cutoff, end in WINDOWS:
        assert len(pd.date_range(pd.Timestamp(cutoff) + pd.Timedelta(days=1), end)) == 61
    data = full_grid("2025-01-01", "2025-05-01").to_frame(index=False)
    data["date"] = pd.to_datetime(data.date)
    data["boardings"] = data.route * 10 + data.hour + data.date.dt.dayofweek
    active = data.route.ne(5) & ~data.hour.between(1, 4)
    data.loc[~active, "boardings"] = 0
    data["working_events_observed"] = active
    cutoff, end = "2025-03-31", "2025-04-03"
    poisoned = data.copy()
    poisoned.loc[poisoned.date.gt(cutoff), "boardings"] = 99999999
    cases = [("seasonal", dict(weeks=0, statistic="mean")),
             ("adaptive", dict(half_life=28, trend=0.5, damping=28)),
             ("daily", dict(weeks=8, statistic="median")),
             ("pooled", dict(days=28, shared=0.5)),
             ("direct", dict(days=56, half_life=28, leaves=7, l2=1)),
             ("calendar", {}), ("coverage", {}),
             ("ridge", dict(correction_days=28, shape_mix=0.8, route_season=True)),
             ("regime", dict(occurrences=12, statistic="trimmed", regime=True)),
             ("structure", dict(days=28, half_life=7, shape_weeks=4, regime=True))]
    for family, params in cases:
        a = cpu_forecast(data, cutoff, end, family, params)
        b = cpu_forecast(poisoned, cutoff, end, family, params)
        pd.testing.assert_frame_equal(a, b)
        pd.testing.assert_frame_equal(a[KEYS], forecast_keys(cutoff, end))
        rounded = postprocess(a)
        assert rounded.prediction.ge(0).all()
        assert rounded.loc[rounded.route.eq(5) | rounded.hour.between(1, 4), "prediction"].eq(0).all()
    weather = pd.DataFrame({"date": pd.date_range("2025-01-01", "2025-12-31"),
        "temperature_2m_mean": 10., "precipitation_sum": 0., "daylight_duration": 40000.})
    from experiments.portfolio_factor import factor_inputs, factor_forecast
    factor_data = data.assign(boardings=data.boardings*3)
    factor_poisoned = factor_data.copy()
    factor_poisoned.loc[factor_poisoned.date.gt(cutoff), "boardings"] = 99999999
    a, scale, _ = factor_inputs(factor_data, cutoff)
    b, other_scale, _ = factor_inputs(factor_poisoned, cutoff)
    pd.testing.assert_frame_equal(a, b)
    np.testing.assert_array_equal(scale, other_scale)
    assert a.shape == (90, 180) and (scale > 0).all()
    from experiments.portfolio_timesfm import series_inputs
    for representation in ["daily", "normal_daily", "normal_ratio"]:
        a, restoration=series_inputs(factor_data,cutoff,end,representation)
        b, other_restoration=series_inputs(factor_poisoned,cutoff,end,representation)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(restoration,other_restoration)
        assert a.shape==(9,90) and restoration.shape==(9,3)
    shape = forecast_keys(cutoff, end).assign(prediction=100.)
    shape.loc[shape.route.eq(5) | shape.hour.between(1, 4), "prediction"] = 0.
    raw = factor_forecast(factor_data, cutoff, end, dict(rank=3, alpha=10, mode="rain"), weather, shape=shape)
    assert np.isfinite(raw.prediction).all() and raw.prediction.ge(0).all()
    assert raw.loc[raw.route.eq(5) | raw.hour.between(1, 4), "prediction"].eq(0).all()
    for representation in ["daily", "daily_weather"]:
        a, _ = task_inputs(data, cutoff, end, representation, weather)
        b, _ = task_inputs(poisoned, cutoff, end, representation, weather)
        for left, right in zip(a, b):
            if representation == "daily":
                np.testing.assert_array_equal(left, right)
            else:
                np.testing.assert_array_equal(left["target"], right["target"])
                assert all(v is None for v in left["future_covariates"].values())
                assert all(len(v) == 90 for v in left["past_covariates"].values())
    for representation in ["normal_daily", "normal_joint_weather"]:
        fit, _ = task_inputs(data, "2025-04-30", "2025-06-30", representation, weather)
        altered = data.copy()
        altered.loc[altered.date.gt("2025-04-30"), "boardings"] = 9999999
        other, _ = task_inputs(altered, "2025-04-30", "2025-06-30", representation, weather)
        matrix = fit[0]["target"] if representation == "normal_joint_weather" else np.stack(fit)
        comparison = other[0]["target"] if representation == "normal_joint_weather" else np.stack(other)
        np.testing.assert_array_equal(matrix, comparison)
        assert matrix.shape == (9, 120) and np.isnan(matrix[4]).sum() == 8
    normal = normal_history(data, "2025-05-01", extended=True)
    shortened = data.route.eq(17) & data.date.between("2025-04-05", "2025-04-30") & data.date.dt.dayofweek.ge(5)
    assert len(normal) == len(data) - int(shortened.sum())
    assert normal.loc[normal.route.eq(17) & normal.date.eq("2025-05-01")].shape[0] == 24
    from experiments.portfolio_movement import disrupted
    keys = forecast_keys("2025-08-15", "2025-09-06")
    changed = disrupted(keys, "august7")
    assert changed.sum() == 6*24 and keys.loc[changed, "route"].eq(7).all()
    assert not changed[keys.date.eq("2025-09-06")].any()
    without_august = normal_history(keys.rename(columns={"prediction": "boardings"}), "2025-09-06", august7=True)
    assert not disrupted(without_august, "august7").any()
    from experiments.portfolio_operations import apply_operations
    prior_train = data.loc[data.date.le("2025-07-31")]
    result = apply_operations(keys.assign(prediction=100.), prior_train,
        normal_history(prior_train, "2025-07-31", august7=True),
        dict(july7=0.45, july50=1.25, august7=0.8))
    assert result.loc[changed, "prediction"].eq(80.).all()
    assert result.loc[result.route.eq(50) & result.date.lt("2025-09-06"), "prediction"].eq(100.).all()
    dates = pd.date_range("2025-02-01", "2025-02-03")
    after_origin = data.copy()
    after_origin.loc[after_origin.date.gt("2025-01-31"), "boardings"] = 99999999
    a = context(daily_history(data, cutoff), "2025-01-31", dates, weather)
    b = context(daily_history(after_origin, cutoff), "2025-01-31", dates, weather)
    pd.testing.assert_frame_equal(a[FEATURES], b[FEATURES])
    a = training_examples(data, cutoff, weather)
    b = training_examples(poisoned, cutoff, weather)
    pd.testing.assert_frame_equal(a, b)
    assert a.date.max() <= pd.Timestamp(cutoff) and a.horizon.between(1, 61).all()
    from experiments.portfolio_bayes_horizon import features as horizon_features, allocate_volume
    examples=training_examples(data,cutoff,weather,august7=True)
    pd.testing.assert_frame_equal(examples,training_examples(poisoned,cutoff,weather,august7=True))
    for route_specific in [False,True]:
        x=horizon_features(examples,route_specific)
        np.testing.assert_array_equal(x,horizon_features(examples.assign(boardings=99999999),route_specific))
        assert len(x)==len(examples) and np.isfinite(x).all()
        factors,info=calibrate(examples.assign(base=examples.reference),
            examples.drop(columns="boardings").assign(base=examples.reference),cutoff,
            dict(strength=1,uncertainty=1,history_days=224),
            feature_fn=lambda frame:horizon_features(frame,route_specific))
        assert np.isfinite(factors).all() and (factors>0).all() and info["latest_target_date"]==cutoff
    shape=forecast_keys("2025-09-05","2025-09-07").assign(prediction=1.)
    closed=shape.route.eq(5)|(shape.route.eq(50)&disrupted(shape,"autumn"))
    shape.loc[closed,"prediction"]=0
    volumes=shape[["route","date"]].drop_duplicates().assign(volume=240.)
    restored=allocate_volume(shape,volumes)
    assert restored.loc[closed,"prediction"].eq(0).all() and np.isfinite(restored.prediction).all()
    np.testing.assert_allclose(restored.loc[~closed].groupby(["route","date"]).prediction.sum(),240.)
    try:
        allocate_volume(shape.assign(prediction=0.),volumes)
    except ValueError:
        pass
    else:
        raise AssertionError("Missing active hourly shape accepted")
    for rule in [("recent", 28), ("season", 56), ("weather", 7)]:
        a = analogue_forecast(data, weather, cutoff, end, rule)
        b = analogue_forecast(poisoned, weather, cutoff, end, rule)
        pd.testing.assert_frame_equal(a, b)
    params_weather = dict(alpha=10, mode="climate", residual_strength=1)
    base = cpu_forecast(data, cutoff, end, "seasonal", dict(weeks=0, statistic="mean"))
    for weather_enabled in [False, True]:
        params_shape = dict(leaves=31, weather=weather_enabled, mix=1.)
        a = shape_forecast(data, cutoff, params_shape, weather, base)
        b = shape_forecast(poisoned, cutoff, params_shape, weather, base)
        pd.testing.assert_frame_equal(a, b)
        np.testing.assert_allclose(a.groupby(["route", "date"]).prediction.sum(),
            base.groupby(["route", "date"]).prediction.sum())
        params_shape["normalize"] = False
        a = shape_forecast(data, cutoff, params_shape, weather, base)
        b = shape_forecast(poisoned, cutoff, params_shape, weather, base)
        pd.testing.assert_frame_equal(a, b)
        assert a.prediction.ge(0).all() and a.loc[a.route.eq(5) | a.hour.between(1, 4), "prediction"].eq(0).all()
    pd.testing.assert_frame_equal(weather_forecast(data, cutoff, end, params_weather, weather),
        weather_forecast(poisoned, cutoff, end, params_weather, weather))
    params_weather = dict(alpha=10, mode="spline", residual_strength=0, route_climate=True, fit_days=56)
    pd.testing.assert_frame_equal(weather_forecast(data, cutoff, end, params_weather, weather),
        weather_forecast(poisoned, cutoff, end, params_weather, weather))
    for shape in ["ridge", "season"]:
        operations_params = dict(july7=0.45, july50=1.25, route_season=True, shape=shape)
        a = operations_forecast(data, cutoff, "2025-11-30", operations_params, weather)
        b = operations_forecast(poisoned, cutoff, "2025-11-30", operations_params, weather)
        pd.testing.assert_frame_equal(a, b)
        closed = a.date.between("2025-11-01", "2025-11-14") & a.date.dt.dayofweek.ge(5)
        restored = a.date.ge("2025-11-15") & a.date.dt.dayofweek.ge(5)
        assert a.loc[a.route.eq(50) & closed, "prediction"].eq(0).all()
        assert a.loc[a.route.eq(50) & restored, "prediction"].sum() > 0
    past_dates = pd.date_range("2025-01-01", cutoff)
    future_dates = pd.date_range(pd.Timestamp(cutoff) + pd.Timedelta(days=1), end)
    task = tasks_with_weather([np.ones(len(past_dates))], past_dates, future_dates, cutoff, weather)[0]
    assert all(len(x) == len(past_dates) for x in task["past_covariates"].values())
    assert all(len(x) == len(future_dates) for x in task["future_covariates"].values())
    for outer, _ in WINDOWS:
        for start, stop in inner_windows(outer):
            assert pd.Timestamp(stop) <= pd.Timestamp(outer)
            assert (pd.Timestamp(stop) - pd.Timestamp(start)).days == 61
    assert season(pd.Series(pd.to_datetime(["2025-12-01", "2025-01-01", "2025-03-01", "2025-06-01", "2025-09-01"]))).tolist() == [0, 0, 1, 2, 3]
    panel = full_grid("2025-02-01", "2025-02-14").to_frame(index=False)
    panel = panel.loc[panel.route.isin([1, 7])].copy()
    panel["date"] = pd.to_datetime(panel.date)
    panel["boardings"] = 100
    for i, name in enumerate(MODELS):
        panel[name] = 100 + 30 * i
    summaries = infer_weights(panel, "2025-02-14")
    weights = summaries["1:0:0"]["weights"]
    assert np.isclose(sum(weights), 1) and min(weights) >= 0
    assert weights[0] > max(weights[1:])
    calibration = panel[["route", "date", "boardings"]].drop_duplicates().copy()
    calibration["base"] = 125.
    calibration["ridge"] = 120.
    calibration["regime"] = 130.
    calibration["daytype"] = 0
    calibration["effective_weekday"] = calibration.date.dt.dayofweek
    calibration["horizon"] = 20
    calibration["temperature_2m_mean"] = 0.
    calibration["precipitation_sum"] = 0.
    calibration["daylight_duration"] = 30000.
    calibration["off"] = False
    correction, info = calibrate(calibration, calibration.drop(columns="boardings"), "2025-02-14",
        dict(strength=1, uncertainty=0, history_days=0))
    assert (correction < 1).all() and np.isfinite(correction).all()
    assert info["latest_target_date"] == "2025-02-14"
    from experiments.portfolio_bayes_direct import direct_features, annual_features
    calibration["direct"] = 100.
    for days in [7, 14, 28]:
        calibration[f"ratio{days}"] = 1.
    correction, _ = calibrate(calibration, calibration.drop(columns="boardings"), "2025-02-14",
        dict(strength=1, uncertainty=0, history_days=0), feature_fn=direct_features)
    assert (correction < 1).all() and np.isfinite(correction).all()
    control, _ = calibrate(calibration, calibration.drop(columns="boardings"), "2025-02-14",
        dict(strength=0, uncertainty=1, history_days=0), feature_fn=direct_features)
    np.testing.assert_array_equal(control, np.ones(len(control)))
    for route_specific in [False, True]:
        cycle = annual_features(calibration, route_specific)
        assert cycle.shape[0] == len(calibration) and np.isfinite(cycle).all()
        poisoned_truth = calibration.assign(boardings=99999999)
        np.testing.assert_array_equal(cycle, annual_features(poisoned_truth, route_specific))
    try:
        calibrate(calibration, calibration, "2025-02-13", dict(strength=1, uncertainty=0, history_days=0))
    except ValueError:
        pass
    else:
        raise AssertionError("Future target entered daily Bayesian correction")
    try:
        infer_weights(panel, "2025-02-13")
    except ValueError:
        pass
    else:
        raise AssertionError("Bayesian weights used a future observed error")
    def fake(tasks, horizon, cross):
        return np.full((len(tasks), horizon), 1.5)
    for method in METHODS:
        a = gpu_forecast(data, cutoff, end, method, fake)
        b = gpu_forecast(poisoned, cutoff, end, method, fake)
        pd.testing.assert_frame_equal(a, b)
        pd.testing.assert_frame_equal(a[KEYS], forecast_keys(cutoff, end))
    params = dict(correction_days=28, shape_mix=0.8, route_season=True)
    pd.testing.assert_frame_equal(movement_forecast(data, cutoff, end, params),
                                  movement_forecast(poisoned, cutoff, end, params))
    movement = movement_forecast(data, cutoff, "2025-11-30", params)
    before = movement.date.between("2025-11-01", "2025-11-14") & movement.date.dt.dayofweek.ge(5)
    after = movement.date.ge("2025-11-15") & movement.date.dt.dayofweek.ge(5)
    assert movement.loc[movement.route.eq(50) & before, "prediction"].eq(0).all()
    assert movement.loc[movement.route.eq(50) & after, "prediction"].sum() > 0
    keys = forecast_keys("2025-03-01", "2025-03-03")
    a, b = keys.assign(prediction=10.0), keys.assign(prediction=20.0)
    assert mix(a, b, 0.5).prediction.eq(15).all()
    np.testing.assert_allclose(transplant(a, b, a).prediction, a.prediction)
    weights, global_weight = route_weights(data, "2025-03-03", a, b)
    assert set(weights) == set(data.route) and global_weight in [0, 0.5, 1]
    try:
        route_weights(data, "2025-03-02", a, b)
    except ValueError:
        pass
    else:
        raise AssertionError("Future route coefficient accepted")
    missing = data.route.eq(50) & data.date.eq("2025-03-01") & data.hour.eq(8)
    data.loc[missing, "boardings"] = 0
    data.loc[missing, "working_events_observed"] = False
    restored = impute_history(data, cutoff)
    assert restored.loc[missing[missing.index.isin(restored.index)], "boardings"].eq(513).all()
    assert data.loc[missing, "boardings"].eq(0).all()
    raw = forecast_keys(cutoff, end).assign(prediction=2.5)
    rounded = postprocess(raw)
    assert rounded.loc[rounded.route.ne(5) & ~rounded.hour.between(1, 4), "prediction"].eq(3).all()
    value = 460.49999999999994
    encoded = pd.DataFrame({"prediction": [value]}).to_csv(index=False)
    restored = pd.read_csv(StringIO(encoded), float_precision="round_trip")
    assert restored.prediction.iloc[0] == value
    print("Portfolio checks passed: 61 days, all CPU paths, cutoff isolation, training-only imputation, half-up.")


if __name__ == "__main__":
    check()
