"""P11: retrospective permitted movement notices, coefficients from cutoff history only."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.portfolio_experiment import load_history, save_candidate, write_json
from experiments.portfolio_ridge import ridge_forecast


def disrupted(frame, event):
    if event == "july":
        return frame.route.isin([7, 50]) & frame.date.between("2025-07-10", "2025-08-06")
    if event == "july_verified":
        return frame.route.isin([7, 50]) & frame.date.between("2025-07-10", "2025-08-10")
    if event == "autumn":
        return (frame.route.isin([7, 50]) & frame.date.between("2025-09-06", "2025-11-14")
                & frame.date.dt.dayofweek.ge(5))
    if event == "august7":
        return frame.route.eq(7) & frame.date.between("2025-08-16", "2025-09-05") & frame.date.dt.dayofweek.ge(5)
    raise ValueError("Unknown event")


def movement_forecast(history, cutoff, end, params):
    train = history.loc[history.date.le(cutoff)].copy()
    normal = train.loc[~(disrupted(train, "july") | disrupted(train, "autumn"))].copy()
    raw = ridge_forecast(normal, cutoff, end, params)
    daily = train.groupby(["route", "date"], as_index=False).boardings.sum()
    ordinary = normal.groupby(["route", "date"], as_index=False).boardings.sum()
    ordinary["weekday"] = ordinary.date.dt.dayofweek
    profile = ordinary.groupby(["route", "weekday"]).boardings.median().clip(lower=1)
    for event in ["july", "autumn"]:
        observed = daily.loc[disrupted(daily, event)].copy()
        observed["weekday"] = observed.date.dt.dayofweek
        observed = observed.merge(profile.rename("expected"), on=["route", "weekday"], validate="many_to_one")
        observed["ratio"] = observed.boardings / observed.expected
        for route in [7, 50]:
            group = observed.loc[observed.route.eq(route)]
            factor = float(np.clip(group.ratio.median(), 0, 1)) if len(group) >= 3 else 1.0
            if event == "autumn" and route == 50:
                factor = 0.0
            raw.loc[disrupted(raw, event) & raw.route.eq(route), "prediction"] *= factor
    return raw


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--history", required=True)
    p.add_argument("--selection", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if os.environ.get("SLURM_JOB_PARTITION") != "ais-cpu":
        raise RuntimeError("Run movement research in ais-cpu")
    data = load_history(args.history)
    params = json.loads(Path(args.selection).read_text())["params"]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "spec.json", dict(params=params, job_id=os.environ["SLURM_JOB_ID"],
        regime="retrospective external announcements; learned coefficients cutoff-only", trials=1,
        event_windows=[['2025-07-10','2025-08-06'],['2025-09-06','2025-11-14']],
        sources=['https://transport.mos.ru/mostrans/all_news/125272',
                 'https://t.me/DtOperativno/22624','https://t.me/DtOperativno/23565']))
    rows = save_candidate(data, out, "ridge_movement",
        lambda cutoff, end: movement_forecast(data, cutoff, end, params))
    write_json(out / "completed.json", dict(scores=[r["wape_score"] for r in rows]))
    print([r["wape_score"] for r in rows], flush=True)
