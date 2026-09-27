"""End-to-end date rollover on synthetic November facts, with an explicit teacher stub.

No GPU inference or quality measurement: checks orchestration and the resulting contract.
Run separately from test_build.py to preserve the builder's subprocess import isolation.
"""
import argparse
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

import build
import stages


def fixture_teachers(history, weights, cache, threads):
    from experiments.portfolio_timesfm import teacher_daily, WEIGHTS_SHA256
    from experiments.portfolio_windows import inner_windows
    for cutoff, end in inner_windows(stages.CUTOFF) + [(stages.CUTOFF, stages.END)]:
        values = np.full((9, 61, 10), 10000.)
        frame = teacher_daily(history, cutoff, end, values, np.ones((9, 61)))
        path = build.ROOT / "timesfm_teachers/daily" / cutoff / "daily.csv"
        stages.save(frame, path)
        build.write_json(path.parent / "source.json", dict(origin=cutoff, end=end, fit_latest_date=cutoff,
                         model_sha256=WEIGHTS_SHA256, quantile_index=3, july_verified=True,
                         sha256=build.digest(path), test_fixture=True))
    return {"test_fixture": True, "real_model_called": False}


def check():
    original = Path.cwd()
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        source = folder / "sources"
        build.init_sources(source)
        history = pd.read_csv(build.LAB / "artifacts/hourly_clean.csv", sep=";")
        november = history.loc[history.date.between("2025-10-01", "2025-10-30")].copy()
        november["date"] = november.date.str.replace("2025-10", "2025-11")
        history_path = folder / "synthetic.csv"
        pd.concat([history, november]).sort_values(["route", "date", "hour"]).to_csv(history_path, sep=";", index=False)

        external = source / build.ROOT / "external"
        weather_path = external / "weather_2025.csv"
        weather = pd.read_csv(weather_path, sep=";")
        extra = pd.concat([weather.tail(1)] * 30, ignore_index=True)
        extra["date"] = pd.date_range("2026-01-01", "2026-01-30").strftime("%Y-%m-%d")
        pd.concat([weather, extra]).to_csv(weather_path, sep=";", index=False)
        build.write_json(external / "weather_source.json", dict(sha256=build.digest(weather_path),
                         regime="synthetic test scenario", url="test-fixture", retrieved_at="2026-09-27"))
        calendar_path = source / "artifacts/calendar_sources.json"
        calendar = json.loads(calendar_path.read_text())
        future = dict(calendar, coverage=["2026-01-01", "2026-12-31"], known_at="2025-09-01",
                      holidays=["2026-01-01"], transfers={}, working_weekends=[])
        build.write_json(calendar_path, dict(calendars=[calendar, future]))
        for path in (external / "school_calendar/sources.json", external / "movement_calendar.json"):
            data = json.loads(path.read_text())
            data["coverage"][1] = "2026-01-30"
            build.write_json(path, data)
        args = argparse.Namespace(history_end="2025-11-30", output=folder / "release", sources=source,
                                  history=history_path, dataset=None, events=None, no_labels=False,
                                  timesfm_model=None, timesfm_cache=folder, threads=2,
                                  tabpfn_model=build.ML / "models/tabpfn-v2-regressor.ckpt")
        try:
            with patch.object(stages, "teachers", side_effect=fixture_teachers):
                build.run(args)
        finally:
            os.chdir(original)
        spec = json.loads((args.output / "recipe.json").read_text())
        assert spec["history_end"] == spec["forecast_from"] == "2025-12-01T00:00:00+03:00"
        assert spec["forecast_to"] == "2026-01-31T00:00:00+03:00"
        from model_030 import load_inputs, compose
        arrays, base = load_inputs(args.output, spec)
        assert len(base) == 14640 and base.date.max() == pd.Timestamp("2026-01-30")
        # Structural postprocessing only, not a substitute for a real TabPFN call.
        payload = compose(arrays, base, np.zeros((arrays["active"].sum(), len(arrays["components"]))), .5)
        assert len(payload.splitlines()) == 14641
    print("Synthetic rollover passed: November history -> December/January package. TimesFM stubbed, TabPFN not run.")


if __name__ == "__main__":
    check()
