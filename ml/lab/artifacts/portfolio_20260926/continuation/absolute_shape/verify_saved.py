"""Run from ml with PYTHONPATH=. : audit saved P51 artifacts without model refitting."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.portfolio_ridge import add_calendar
from experiments.portfolio_verify import read
from pipeline import KEYS

ROOT = Path("artifacts/portfolio_20260926/continuation")
OUT = ROOT/"absolute_shape"
SOURCE = ROOT/"verified_july/verified_july/selected"
INNER = ROOT/"bayes_hourly_target/inner_shape"
HOURS = [0]+list(range(5,24))
history = read("artifacts/hourly_clean.csv")
hash_counts = {}
for phase in ["pilot", "study"]:
    started = json.loads((OUT/phase/"run_started.json").read_text())
    for name, sha in started["sha256"].items():
        path = Path(name.removeprefix("/beegfs/home/m.persiyanov/codex_runs/tram-portfolio-20260926/"))
        assert hashlib.sha256(path.read_bytes()).hexdigest()==sha, name
    hash_counts[phase] = len(started["sha256"])

fits = []
for path in sorted(OUT.rglob("training.csv")):
    folder = path.parent
    cutoff = folder.name
    training = read(path)
    training["origin"] = pd.to_datetime(training.origin)
    pieces = []
    origins = pd.date_range("2025-01-31", pd.Timestamp(cutoff)-pd.Timedelta(days=61), freq="ME")
    for origin in origins:
        info = json.loads((INNER/str(origin.date())/"source.json").read_text())
        raw_path = INNER/str(origin.date())/f"raw_{origin.date()}.csv"
        assert info["fit_latest_date"] == info["origin"] == str(origin.date())
        assert pd.Timestamp(info["end"]) <= pd.Timestamp(cutoff)
        assert hashlib.sha256(raw_path.read_bytes()).hexdigest()==info["sha256"]
        raw = read(raw_path).merge(history[KEYS+["boardings"]], on=KEYS, validate="one_to_one")
        assert len(raw)==14640
        pieces.append(raw.assign(origin=origin))
    past = pd.concat(pieces, ignore_index=True)
    past = past.loc[past.date.gt(pd.Timestamp(cutoff)-pd.Timedelta(days=224)) & past.date.le(cutoff)].copy()
    weekend = past.date.dt.dayofweek.ge(5)
    changed = (past.route.isin([7,50]) & past.date.between("2025-07-10","2025-08-10"))
    changed |= past.route.isin([7,50]) & past.date.between("2025-09-06","2025-11-14") & weekend
    changed |= past.route.eq(7) & past.date.between("2025-08-16","2025-09-05") & weekend
    changed |= past.route.eq(17) & past.date.between("2025-04-05","2025-04-30") & weekend
    past = add_calendar(past.loc[~changed & past.route.ne(5) & past.hour.isin(HOURS)].copy(), cutoff)
    grouping = ["origin","route","date"]
    past["actual_day"] = past.groupby(grouping).boardings.transform("sum")
    past["base_day"] = past.groupby(grouping).prediction.transform("sum")
    typical = past.drop_duplicates(["route","date"]).groupby(["route","daytype"]).actual_day.median()
    threshold = typical.reindex(pd.MultiIndex.from_frame(past[["route","daytype"]])).to_numpy()
    past = past.loc[past.actual_day.gt(np.maximum(500, 0.35*threshold)) & past.base_day.gt(0)].reset_index(drop=True)
    pd.testing.assert_frame_equal(training[KEYS+["origin"]],past[KEYS+["origin"]])
    np.testing.assert_allclose(training[["prediction","boardings","actual_day","base_day"]],
        past[["prediction","boardings","actual_day","base_day"]], rtol=0, atol=1e-8)
    target = past.boardings/past.actual_day-past.prediction/past.base_day
    weights = past.actual_day/past.groupby(KEYS).prediction.transform("size")
    weights /= weights.mean()
    np.testing.assert_allclose(training.target,target,rtol=0,atol=1e-12)
    np.testing.assert_allclose(training.weight,weights,rtol=0,atol=1e-12)
    info = json.loads((folder/"fit.json").read_text())
    assert info["rows"]==len(past) and info["latest_target_date"] <= cutoff
    assert hashlib.sha256((folder/"model.pkl").read_bytes()).hexdigest()==info["model_sha256"]
    assert info["model_reload_exact"] and info["iterations"]==100
    base = read(SOURCE/f"raw_{cutoff}.csv")
    delta = read(folder/"delta.csv")
    active = base.route.ne(5) & base.hour.isin(HOURS)
    pd.testing.assert_frame_equal(delta[KEYS],base.loc[active,KEYS].reset_index(drop=True))
    np.testing.assert_array_equal(delta.prediction,base.loc[active,"prediction"].to_numpy())
    volume = base.groupby(["route","date"]).prediction.transform("sum")
    np.testing.assert_allclose(delta.base_day,volume.loc[active],rtol=0,atol=1e-8)
    shares = base.prediction.div(volume.where(volume.gt(0))).fillna(0)
    proposed = base[KEYS].assign(prediction=0.)
    proposed.loc[active,"prediction"] = np.maximum(0,shares.loc[active].to_numpy()+delta.delta.to_numpy())
    sums = proposed.groupby(["route","date"]).prediction.transform("sum")
    normalized = proposed.prediction.div(sums.where(sums.gt(0))).fillna(shares)
    raw = read(folder/f"raw_{cutoff}.csv")
    np.testing.assert_allclose(raw.prediction,volume*normalized,rtol=1e-12,atol=1e-8)
    np.testing.assert_allclose(raw.groupby(["route","date"]).prediction.sum(),base.groupby(["route","date"]).prediction.sum(),rtol=1e-12,atol=1e-8)
    assert raw.loc[~active,"prediction"].eq(0).all()
    fits.append(dict(cutoff=cutoff,leaves=info["leaves"],rows=len(past),origins=len(origins),phase=folder.parts[-3]))

count = 0
study = OUT/"study/absolute_shape"
for folder in sorted(study.glob("trial_*"))+[study/"selected"]:
    params = json.loads((folder/"parameters.json" if folder.name.startswith("trial_") else study/"selection.json").read_text())
    recipe = params.get("params",params)["recipe"]
    for path in folder.glob("raw_*.csv"):
        base = read(SOURCE/path.name)
        got = read(path)
        if recipe=="control":
            np.testing.assert_array_equal(got.prediction,base.prediction)
        else:
            strength,leaves = recipe.split("_")
            learned = read(OUT/f"study/corrected_{leaves}"/path.stem[4:]/path.name)
            weight = 0.5 if strength=="half" else 1.
            np.testing.assert_allclose(got.prediction,weight*learned.prediction+(1-weight)*base.prediction,rtol=1e-12,atol=1e-8)
        count += 1
final = read(study/"selected/submission.csv")
differences = []
for name in ["025_timesfm_volume_bayesian_shape","024_bayesian_hourly_share_errors","021_bayesian_verified_operations","023_gaussian_process_errors","026_timesfm_verified_operations","004_ridge_movement","005_movement_chronos"]:
    other = read(Path("submissions")/f"{name}.csv")
    error = np.abs(final.prediction-other.prediction)
    differences.append(dict(candidate=name,relative_l1=float(error.sum()/((final.prediction.sum()+other.prediction.sum())/2)),changed_rows=int(error.gt(0).sum())))
result = dict(checked_at=datetime.now(timezone.utc).isoformat(),input_code_sha256=hash_counts,
    training_fits=fits,raw_forecasts_checked=count,exact_021_control=True,
    cutoff_safe_targets=True,weighted_absolute_share_targets=True,dayvolume_preserved=True,
    model_reloads_exact_on_zhores=True,local_model_inference_skipped="sklearn1.9.1 differs from saved1.8.0; transforms audited locally",
    final_total=int(final.prediction.sum()),differences=differences,
    audit_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT/"independent_verification.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result))
