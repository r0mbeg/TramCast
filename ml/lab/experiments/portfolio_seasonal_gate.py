"""P55: use P53 correction only where completed seasonal error examples exist."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import resource
import time

import pandas as pd

from experiments.portfolio_experiment import load_history, run_study, write_json
from experiments.portfolio_nonlinear_errors import forecast as correction, prepare
from experiments.portfolio_timesfm_errors import examples, SOURCE, TEACHERS, SHAPE, VOLUME
from experiments.portfolio_windows import season


def support(past, future, cutoff):
    past = prepare(past, cutoff).drop_duplicates(["route", "date"])
    past["season"] = season(past.date)
    keys = ["route", "season", "effective_weekday"]
    counts = past.groupby(keys).size().rename("past_dates")
    result = future[["route", "date", "effective_weekday"]].copy()
    if result.duplicated(["route", "date"]).any() or not result.date.gt(pd.Timestamp(cutoff)).all():
        raise ValueError("Invalid seasonal gate forecast dates")
    result["season"] = season(result.date)
    result = result.join(counts, on=keys)
    result["past_dates"] = result.past_dates.fillna(0).astype(int)
    # ponytail: seasonal coverage is conservative; reconsider with several years of completed errors.
    result["supported"] = result.past_dates.ge(2)
    return result


def forecast(history, cutoff, end, recipe, out):
    base = correction(history, cutoff, end, "control", out/"models")
    if recipe == "control":
        return base
    if recipe != "gated":
        raise ValueError("Unknown seasonal gate recipe")
    learned = correction(history, cutoff, end, "half_15", out/"models")
    past, future = examples(history, cutoff, end)
    gate = support(past, future, cutoff)
    folder = out/"gates"
    folder.mkdir(exist_ok=True)
    gate.to_csv(folder/f"{cutoff}.csv", sep=";", index=False, date_format="%Y-%m-%d")
    result = learned.merge(gate[["route", "date", "supported"]], on=["route", "date"], validate="many_to_one")
    result.loc[~result.supported, "prediction"] = base.loc[~result.supported, "prediction"].to_numpy()
    return result.drop(columns="supported")


def run(args):
    if (os.environ.get("SLURM_JOB_PARTITION") != "ais-cpu"
        or int(os.environ["SLURM_CPUS_PER_TASK"]) != 1 or len(os.sched_getaffinity(0)) > 1):
        raise RuntimeError("Expected one bound ais-cpu core")
    if datetime.now(timezone.utc) >= datetime.fromisoformat("2026-09-27T15:40:40+00:00"):
        raise RuntimeError("Research reserve reached")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    paths = [Path(args.history), Path(__file__), Path("pipeline.py")]
    paths += [Path("experiments")/f"portfolio_{n}.py" for n in
        ["nonlinear_errors", "timesfm_errors", "timesfm", "bayes_volume", "combine", "experiment", "windows"]]
    paths += list(SOURCE.glob("*/daily.csv"))+list(TEACHERS.glob("*/*.csv"))+list(TEACHERS.glob("*/*.json"))
    paths += list(SHAPE.glob("raw_*.csv"))+list(VOLUME.glob("*/raw_*.csv"))
    write_json(out/"run_started.json", dict(job_id=os.environ["SLURM_JOB_ID"], command=os.sys.argv,
        versions=dict(python=platform.python_version(), **{n:importlib.metadata.version(n)
            for n in ["numpy", "pandas", "scikit-learn", "optuna"]}),
        sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))
    started = time.monotonic()
    history = load_history(args.history)
    run_study(history, out, "seasonal_gate",
        lambda c,e,p:forecast(history,c,e,p["recipe"],out),
        dict(sampler="grid", space={"recipe":["control", "gated"]}, trials=2))
    write_json(out/"completed.json", dict(job_id=os.environ["SLURM_JOB_ID"], elapsed_seconds=time.monotonic()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
