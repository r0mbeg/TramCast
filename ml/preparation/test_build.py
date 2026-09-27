"""python ml/preparation/test_build.py — fast checks, no GPU or model substitution."""
import json
import hashlib
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

from build import LAB, ROOT, digest, publish, validate_history, validate_movement, validate_weather

sys.path.insert(0, str(LAB))
import pipeline
from experiments import calendar_experiment, portfolio_school_fraction, portfolio_movement
from experiments.portfolio_ridge import add_calendar
from stages import cached_quantiles, TIMESFM_CONFIG


def rejects(call):
    try:
        call()
    except (ValueError, AssertionError, FileExistsError):
        return
    raise AssertionError("Invalid input was accepted")


def check():
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        # Raw monthly exports: clean 05:30 before aggregation; reject overlapping duplicates.
        first, second = folder / "first.csv", folder / "second.csv"
        columns = ["tran_no", "device_no", "tran_date_time", "ngpt_route", "validation_result"]
        pd.DataFrame([["1", "a", "2025-01-01 06:00:00", "1 трамвай", 1]], columns=columns).to_csv(first, sep=";", index=False)
        events = pd.DataFrame([["2", "a", "2025-11-30 05:29:00", "1 трамвай", 1],
                               ["3", "a", "2025-11-30 05:30:00", "1 трамвай", 1]], columns=columns)
        events.to_csv(second, sep=";", index=False)
        raw_out = folder / "aggregates"
        raw_out.mkdir()
        with patch.object(pipeline, "OUT", raw_out):
            pipeline.prepare(end="2025-12-01", reconcile=False, files=[first, second])
            raw = validate_history(raw_out / "hourly_clean.csv", "2025-11-30")
            assert raw.loc[raw.date.eq("2025-11-30") & raw.route.eq(1) & raw.hour.eq(5), "boardings"].item() == 1
            duplicate = folder / "duplicate.csv"
            events.iloc[1:].to_csv(duplicate, sep=";", index=False)
            rejects(lambda: pipeline.prepare(end="2025-12-01", reconcile=False, files=[first, second, duplicate]))
        # New November facts are required; merely changing a JSON date is rejected.
        history = pd.read_csv(LAB / "artifacts/hourly_clean.csv", sep=";")
        path = folder / "history.csv"
        history.to_csv(path, sep=";", index=False)
        rejects(lambda: validate_history(path, "2025-11-30"))
        november = history.loc[history.date.between("2025-10-01", "2025-10-30")].copy()
        november["date"] = november.date.str.replace("2025-10", "2025-11")
        pd.concat([history, november]).sort_values(pipeline.KEYS).to_csv(path, sep=";", index=False)
        extended = validate_history(path, "2025-11-30")
        assert extended.date.max() == pd.Timestamp("2025-11-30")
        extended.loc[0, "working_events_observed"] = not extended.loc[0, "working_events_observed"]
        extended.to_csv(path, sep=";", index=False)
        rejects(lambda: validate_history(path, "2025-11-30"))

        weather = folder / "weather.csv"
        pd.DataFrame(dict(date=pd.date_range("2025-01-01", "2026-01-30"), temperature_2m_mean=0,
                          precipitation_sum=0, daylight_duration=30000)).to_csv(weather, sep=";", index=False)
        validate_weather(weather, "2026-01-30")
        rejects(lambda: validate_weather(weather, "2026-01-31"))

        calendars = json.loads((LAB / "artifacts/calendar_sources.json").read_text())
        future = dict(calendars, coverage=["2026-01-01", "2026-12-31"], known_at="2025-09-01",
                      holidays=["2026-01-01"], transfers={}, working_weekends=[])
        path = folder / "calendar.json"
        path.write_text(json.dumps(dict(calendars=[calendars, future])))
        with patch.object(calendar_experiment, "SOURCES", path):
            # A new year's publication must not invalidate earlier calendar-only examples.
            add_calendar(pd.DataFrame(dict(date=pd.to_datetime(["2025-05-01"]))), "2025-04-30")
            result = add_calendar(pd.DataFrame(dict(date=pd.to_datetime(["2026-01-01"]))), "2025-11-30")
            assert result.off.all()
            rejects(lambda: add_calendar(result[["date"]], "2025-08-31"))

        school = json.loads(portfolio_school_fraction.CALENDAR.read_text()) if portfolio_school_fraction.CALENDAR.exists() else json.loads((LAB / ROOT / "external/school_calendar/sources.json").read_text())
        school["coverage"][1] = "2026-01-30"
        path = folder / "school.json"
        path.write_text(json.dumps(school))
        with patch.object(portfolio_school_fraction, "CALENDAR", path):
            portfolio_school_fraction.calendar.cache_clear()
            features = portfolio_school_fraction.calendar_features(pd.DataFrame(dict(date=pd.to_datetime(["2026-01-01"]), horizon=[32])))
            assert features.shape == (1, 6) and np.isfinite(features).all()
        portfolio_school_fraction.calendar.cache_clear()

        events = dict(coverage=["2025-01-01", "2026-01-30"], sources=["test fixture"],
                      events={k: [] for k in ("april17", "july", "july_verified", "autumn", "august7")})
        events["events"]["autumn"] = [dict(routes=[7, 50], start="2025-12-01", end="2025-12-07", weekdays=[5, 6])]
        validate_movement(events, "2026-01-30")
        with patch.object(portfolio_movement, "movement_calendar", return_value=events):
            flags = portfolio_movement.disrupted(pd.DataFrame(dict(route=[7, 7, 12], date=pd.to_datetime(["2025-12-01", "2025-12-06", "2025-12-06"]))), "autumn")
            assert flags.tolist() == [False, True, False]

        # A cached teacher is reusable only for the exact input matrix and pinned metadata.
        from experiments.portfolio_timesfm import WEIGHTS_SHA256, WHEEL_SHA256
        cache = folder / "teacher-cache"
        cache.mkdir()
        matrix = np.ones((9, 304), dtype=np.float32)
        path = cache / "normal_ratio_2025-10-31_july_verified.npz"
        np.savez(path, inputs=matrix, quantiles=np.ones((9, 61, 10)))
        metadata = dict(cutoff="2025-10-31", end="2025-12-31", representation="normal_ratio", july_verified=True,
                        model_sha256=WEIGHTS_SHA256, wheel_sha256=WHEEL_SHA256, config=TIMESFM_CONFIG,
                        sha256=digest(path), input_sha256=hashlib.sha256(matrix.tobytes()).hexdigest())
        path.with_suffix(".json").write_text(json.dumps(metadata))
        assert cached_quantiles(cache, "2025-10-31", "2025-12-31", matrix).shape == (9, 61, 10)
        matrix[0, -1] += 1
        assert cached_quantiles(cache, "2025-10-31", "2025-12-31", matrix) is None

        staging, final = folder / "staging", folder / "release"
        staging.mkdir()
        (staging / "recipe.json").write_text("complete")
        publish(staging, final)
        staging.mkdir()
        rejects(lambda: publish(staging, final))
        assert (final / "recipe.json").read_text() == "complete"
    print("Preparation checks passed: new dates, masks, source coverage, teacher invalidation and immutable publication.")


if __name__ == "__main__":
    check()
