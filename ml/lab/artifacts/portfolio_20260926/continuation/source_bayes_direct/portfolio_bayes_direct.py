"""P32: causal Bayesian volume correction using disagreement with a direct model."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.portfolio_bayes_volume import ROOT, features, calibrate, bayes_volume_forecast
from experiments.portfolio_combine import raw_frame, transplant
from experiments.portfolio_direct_daily import direct_daily_forecast, daily_history, context
from experiments.portfolio_experiment import load_history, run_study, save_candidate, write_json
from experiments.portfolio_ridge import ACTIVE_ROUTES
from experiments.portfolio_weather_volume import WEATHER_PATH
from experiments.portfolio_windows import inner_windows


def direct_features(frame):
    delta = np.log1p(frame.direct.to_numpy()) - np.log1p(frame.base.to_numpy())
    columns = [delta] + [delta * frame.route.eq(r).to_numpy(float) for r in ACTIVE_ROUTES]
    columns += [np.log(frame[f"ratio{d}"].to_numpy().clip(0.1, 10)) for d in [7, 14, 28]]
    result = np.column_stack([features(frame)] + columns)
    if not np.isfinite(result).all():
        raise ValueError("Invalid Bayesian disagreement features")
    return result


def forecast(history, cutoff, end, params, out):
    bayes_params = json.loads((ROOT / "bayes_volume/bayes_volume/selection.json").read_text())["params"]
    direct_params = json.loads((ROOT / "direct_daily/direct_daily/selection.json").read_text())["params"]
    # Existing P21 caches contain forecasts made at each origin, without future targets.
    bayes_volume_forecast(history, cutoff, end, bayes_params)
    weather = pd.read_csv(WEATHER_PATH, sep=";", parse_dates=["date"])

    def daily(origin, stop, observed):
        table = out / "inner" / origin / "daily.csv"
        table.parent.mkdir(parents=True, exist_ok=True)
        if table.exists():
            frame = pd.read_csv(table, sep=";", parse_dates=["date"], float_precision="round_trip")
        else:
            frame = pd.read_csv(ROOT / "bayes_volume/inner" / origin / "daily.csv",
                sep=";", parse_dates=["date"], float_precision="round_trip")
            if origin == "2025-01-31":
                # ponytail: no direct training targets at first origin; use base until past targets exist.
                direct = frame[["route", "date", "base"]].rename(columns={"base": "direct"})
            else:
                folder = out / "direct_inner" / origin
                source = ROOT / "direct_daily/direct_daily/selected" / f"raw_{origin}.csv"
                raw = (raw_frame(source, origin, stop) if source.exists() else
                    direct_daily_forecast(history, origin, stop, direct_params))
                if observed:
                    save_candidate(history, folder, "direct_inner", lambda c, e: raw,
                        windows=[(origin, stop)], final=False)
                direct = raw.groupby(["route", "date"], as_index=False).prediction.sum().rename(columns={"prediction": "direct"})
            ratios = context(daily_history(history, origin), origin,
                pd.date_range(pd.Timestamp(origin)+pd.Timedelta(days=1), stop), weather)
            frame = frame.merge(direct, on=["route", "date"], validate="one_to_one")
            frame = frame.merge(ratios[["route", "date", "ratio7", "ratio14", "ratio28"]],
                on=["route", "date"], how="left", validate="one_to_one")
            frame.loc[frame.route.eq(5), ["ratio7", "ratio14", "ratio28"]] = 1.
            frame.to_csv(table, sep=";", index=False, date_format="%Y-%m-%d")
        if frame.date.min() != pd.Timestamp(origin)+pd.Timedelta(days=1) or frame.date.max() != pd.Timestamp(stop):
            raise ValueError("Cached Bayesian direct input has wrong period")
        if observed:
            if pd.Timestamp(stop) > pd.Timestamp(cutoff):
                raise ValueError("Incomplete inner forecasts cannot train Bayesian correction")
            truth = history.loc[history.date.le(cutoff)].groupby(["route", "date"], as_index=False).boardings.sum()
            frame = frame.merge(truth, on=["route", "date"], validate="one_to_one")
        return frame

    earlier = inner_windows(cutoff)
    past = pd.concat([daily(a, b, True) for a, b in earlier], ignore_index=True)
    future = daily(cutoff, end, False)
    factor, summary = calibrate(past, future, cutoff, params, feature_fn=direct_features)
    name = f"{params['strength']}_{params['history_days']}"
    write_json(out / f"posterior_{cutoff}_{name}.json", dict(**summary, params=params, cutoff=cutoff,
        inner_windows=earlier, direct_params=direct_params, parent_bayes_params=bayes_params,
        uncertainty="conditional model variance; not confidence in hidden score"))
    future["factor"] = factor
    future[["route", "date", "factor"]].to_csv(out / f"factors_{cutoff}_{name}.csv", sep=";", index=False)
    raw = raw_frame(ROOT / "bayes_volume/inner" / cutoff / f"raw_{cutoff}.csv", cutoff, end)
    raw = raw.merge(future[["route", "date", "factor"]], on=["route", "date"], validate="many_to_one")
    raw["prediction"] *= raw.factor
    shape = raw_frame(ROOT / "conditional_shape/conditional_shape/selected" / f"raw_{cutoff}.csv", cutoff, end)
    return transplant(raw.drop(columns="factor"), shape, raw.drop(columns="factor"))


def run(args):
    if os.environ.get("SLURM_JOB_PARTITION") != "ais-cpu":
        raise RuntimeError("Expected ais-cpu")
    cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
    if not 1 <= cpus <= 4 or len(os.sched_getaffinity(0)) > cpus:
        raise RuntimeError("Unexpected allocation")
    if datetime.now(timezone.utc) >= datetime.fromisoformat("2026-09-27T15:40:40+00:00"):
        raise RuntimeError("Research reserve reached")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    paths = [Path(args.history), Path(__file__), Path(WEATHER_PATH)]
    paths += [Path("experiments") / f"portfolio_{name}.py" for name in
        ["experiment", "bayes_volume", "direct_daily", "combine", "operations", "windows", "ridge", "weather_volume", "structure"]]
    paths += [ROOT / f"{family}/{family}/selection.json" for family in ["direct_daily", "bayes_volume", "operations"]]
    paths += list((ROOT / "conditional_shape/conditional_shape/selected").glob("raw_*.csv"))
    write_json(out / "run_started.json", dict(job_id=os.environ["SLURM_JOB_ID"], command=os.sys.argv,
        sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))
    history = load_history(args.history)
    run_study(history, out, "bayes_direct", lambda c, e, p: forecast(
        history, c, e, p, out), dict(sampler="grid", space={
            "strength": [0., 0.5, 1.], "uncertainty": [1], "history_days": [112, 224]}, trials=6))
    write_json(out / "completed.json", dict(job_id=os.environ["SLURM_JOB_ID"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
