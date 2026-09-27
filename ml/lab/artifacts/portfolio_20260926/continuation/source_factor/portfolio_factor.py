"""P35: shared nonnegative route-hour factors, forecast from cutoff history only."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.linear_model import Ridge

from experiments.portfolio_bayes_volume import ROOT
from experiments.portfolio_combine import raw_frame, transplant
from experiments.portfolio_experiment import load_history, run_study, save_candidate, write_json, impute_history, complete_raw
from experiments.portfolio_operations import normal_history, apply_operations
from experiments.portfolio_ridge import add_calendar
from experiments.portfolio_weather_volume import WEATHER_PATH
from pipeline import KEYS


def factor_inputs(history, cutoff):
    normal = normal_history(history, cutoff, extended=True, august7=True)
    daily = add_calendar(normal.groupby(["route", "date"], as_index=False).boardings.sum(), cutoff)
    typical = daily.groupby(["route", "daytype"]).boardings.transform("median")
    bad = daily.loc[daily.boardings.lt(np.maximum(500, 0.35*typical)), ["route", "date"]]
    normal = normal.merge(bad.assign(bad=True), on=["route", "date"], how="left", validate="many_to_one")
    normal.loc[normal.bad.eq(True), "working_events_observed"] = False
    keys = history.loc[history.date.le(cutoff), KEYS]
    filled = keys.merge(normal[KEYS+["boardings", "working_events_observed"]], on=KEYS,
        how="left", validate="one_to_one")
    filled["boardings"] = filled.boardings.fillna(0.)
    filled["working_events_observed"] = filled.working_events_observed.eq(True)
    active = filled.route.ne(5) & ~filled.hour.between(1, 4)
    count = int((active & ~filled.working_events_observed).sum())
    filled = impute_history(filled, cutoff)
    matrix = filled.loc[active].pivot(index="date", columns=["route", "hour"], values="boardings").sort_index(axis=1)
    if not np.isfinite(matrix.to_numpy()).all():
        raise ValueError("Incomplete normal factor input")
    scale = matrix.T.groupby(level="route").mean().T.median().clip(lower=1.)
    scales = scale.reindex(matrix.columns.get_level_values("route")).to_numpy()
    return matrix, scales, count


def factor_design(calendar, weather, mode):
    frame = calendar.merge(weather, on="date", validate="one_to_one")
    columns = [frame.effective_weekday.eq(d).to_numpy(float) for d in range(1, 7)]
    columns += [frame.summer.to_numpy(float), np.log1p(frame.precipitation_sum.to_numpy())]
    if mode == "climate":
        columns += [frame.temperature_2m_mean.to_numpy()/10, frame.daylight_duration.to_numpy()/21600]
    result = np.column_stack(columns)
    if not np.isfinite(result).all():
        raise ValueError("Incomplete factor exogenous input")
    return result


def factor_forecast(history, cutoff, end, params, weather, out=None, shape=None, seed=42):
    matrix, scales, imputed = factor_inputs(history, cutoff)
    model = NMF(n_components=params["rank"], init="nndsvda", max_iter=500, tol=1e-4, random_state=seed)
    factors = model.fit_transform(matrix.to_numpy(float)/scales)
    calendar = add_calendar(pd.DataFrame(dict(date=matrix.index)), cutoff)
    fit = ~calendar.off.to_numpy()
    regressor = Ridge(alpha=params["alpha"]).fit(factor_design(calendar, weather, params["mode"])[fit],
        np.log1p(factors[fit]))
    future = add_calendar(pd.DataFrame(dict(date=pd.date_range(pd.Timestamp(cutoff)+pd.Timedelta(days=1), end))), cutoff)
    projected = np.maximum(0., np.expm1(regressor.predict(factor_design(future, weather, params["mode"]))))
    projected[future.off.to_numpy()] *= 0.95
    values = projected @ model.components_ * scales
    frame = pd.DataFrame(values, index=future.date, columns=matrix.columns)
    raw = frame.stack(["route", "hour"], future_stack=True).rename("prediction").reset_index()
    raw = complete_raw(raw, cutoff, end)
    base = json.loads((ROOT / "operations/operations/selection.json").read_text())["params"]
    august = json.loads((ROOT / "august_operations/august_operations/selection.json").read_text())["params"]["august7"]
    raw = apply_operations(raw, history.loc[history.date.le(cutoff)],
        normal_history(history, cutoff, extended=True, august7=True), {**base, "august7": august})
    if shape is None:
        shape = raw_frame(ROOT / "conditional_shape/conditional_shape/selected" / f"raw_{cutoff}.csv", cutoff, end)
    raw = transplant(raw, shape, raw)
    if out is not None:
        name = f"{cutoff}_{params['rank']}_{params['alpha']}_{params['mode']}_seed{seed}"
        write_json(out / f"fit_{name}.json", dict(params=params, seed=seed, cutoff=cutoff,
            latest_target=str(matrix.index.max().date()), imputed_training_cells=imputed,
            iterations=int(model.n_iter_), reconstruction_error=float(model.reconstruction_err_),
            scales=scales.tolist(), basis=model.components_.tolist(),
            regression_coefficients=regressor.coef_.tolist(), regression_intercept=regressor.intercept_.tolist(),
            exogenous_mode="retrospective ERA5 and official operation notices",
            hourly_shape="fixed P29; factors determine daily volume only"))
    return raw


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
        ["experiment", "operations", "movement", "ridge", "combine"]]
    paths += [ROOT / f"{name}/{name}/selection.json" for name in ["operations", "august_operations"]]
    paths += list((ROOT / "conditional_shape/conditional_shape/selected").glob("raw_*.csv"))
    write_json(out / "run_started.json", dict(job_id=os.environ["SLURM_JOB_ID"], command=os.sys.argv,
        versions={p:importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "optuna"]},
        sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))
    history = load_history(args.history)
    weather = pd.read_csv(WEATHER_PATH, sep=";", parse_dates=["date"])
    run_study(history, out, "factor", lambda c,e,p:factor_forecast(history,c,e,p,weather,out),
        dict(sampler="grid", space={"rank":[3,6], "alpha":[1,10], "mode":["rain","climate"]},trials=8))
    selected = json.loads((out / "factor/selection.json").read_text())["params"]
    save_candidate(history,out / "seed73","factor_seed73",lambda c,e:
        factor_forecast(history,c,e,selected,weather,out,seed=73),final=False)
    write_json(out / "completed.json",dict(job_id=os.environ["SLURM_JOB_ID"]))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history",required=True)
    parser.add_argument("--output",required=True)
    run(parser.parse_args())
