"""Rolling-origin, observed-target CPU training. No teacher outputs enter fit or selection."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import signal
import sys
import time

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd

from ml.lab.cpu_student.model import CATEGORICAL, HISTORY_FEATURES, features, read_history, rounded
from ml.lab.cpu_student.train import digest, measure, save_json
from ml.lab.cpu_forecast.model import predict, publish, reference
from ml.runtime.constants import KEYS

LAB = Path(__file__).resolve().parents[1]
CANDIDATES = [dict(depth=5, iterations=350), dict(depth=7, iterations=500)]
DEV = [("2025-04-30","2025-05-01"), ("2025-06-30","2025-07-01")]


def examples(history, cutoff, calendar, movement):
    cutoff = pd.Timestamp(cutoff)
    # Entire 61-day targets must be available before the fitting cutoff.
    last_start = cutoff-pd.Timedelta(days=60)
    starts = pd.date_range("2025-01-29", last_start, freq="10D")
    if len(starts) and starts[-1] != last_start:
        starts = starts.union(pd.DatetimeIndex([last_start]))
    truth = history.loc[history.date.le(cutoff) & history.working_events_observed & history.route.ne(5) & ~history.hour.between(1,4)]
    parts = []
    for start in starts:
        keys, matrix = features(history, str(start.date()), calendar, movement)
        frame = keys.merge(truth[KEYS+["boardings"]], on=KEYS, how="left", validate="one_to_one")
        usable = frame.boardings.notna()
        part = matrix.loc[usable].copy()
        part["target"] = frame.loc[usable,"boardings"].to_numpy()
        part["target_date"] = frame.loc[usable,"date"].to_numpy()
        part["start"] = start
        parts.append(part)
    if not parts:
        raise ValueError("Insufficient complete historical horizons")
    result = pd.concat(parts,ignore_index=True)
    if result.target_date.max() > cutoff or not result.target_date.ge(result.start).all():
        raise ValueError("Future target in training")
    return result


def fit(data, params):
    unique = data.drop_duplicates(["route","target_date","hour"])
    scales = {**unique.groupby("route").target.median().clip(lower=1).to_dict(), "5":1.}
    matrix = data.drop(columns=["target","target_date","start"])
    base = reference(matrix,scales)
    repetitions = data.groupby(["route","target_date","hour"]).target.transform("size").to_numpy()
    weights = base/repetitions
    weights /= weights.mean()
    # ponytail: 10% deterministic history-free examples support requests near January 1; verify with older history before production use.
    cold = matrix.iloc[::10].copy()
    cold[HISTORY_FEATURES] = np.nan
    cold["history_days"] = 0.
    cold_base = reference(cold,scales)
    x = pd.concat([matrix,cold],ignore_index=True)
    y = np.concatenate([data.target.to_numpy()/base,data.target.to_numpy()[::10]/cold_base])
    w = np.concatenate([weights,cold_base/repetitions[::10]/(base/repetitions).mean()])
    model = CatBoostRegressor(**params, loss_function="MAE", learning_rate=.06, random_seed=42,
        task_type="CPU", thread_count=2, allow_writing_files=False, verbose=False)
    model.fit(x,y,cat_features=CATEGORICAL,sample_weight=w)
    return model,scales


def evaluated(history,predicted):
    observed = history.loc[history.working_events_observed & history.route.ne(5) & ~history.hour.between(1,4)]
    paired = observed.merge(predicted,on=KEYS,validate="one_to_one")
    if not len(paired):
        raise ValueError("No observed evaluation targets")
    return paired, measure(paired.boardings,paired.prediction)


def run(out):
    begun=time.perf_counter()
    out.mkdir(parents=True,exist_ok=False)
    history_path=LAB/"artifacts/hourly_clean.csv"
    calendar_path=LAB/"artifacts/calendar_sources.json"
    movement_path=LAB.parent/"preparation/movement_calendar.json"
    sources=[history_path,calendar_path,movement_path,Path(__file__),Path(__file__).with_name("model.py"),
        LAB/"cpu_student/model.py",LAB/"cpu_student/train.py",LAB.parent/"runtime/constants.py"]
    hashes={str(p.relative_to(LAB.parent.parent)):digest(p) for p in sources}
    save_json(out/"protocol.json",dict(
        hypothesis="CPU direct multi-horizon model trained on observed targets generalizes to held-out 61-day periods",
        origin_step_days=10,first_start="2025-01-29",training_windows="Complete 61-day horizons; extra last start ensures latest target reaches fit cutoff",
        development=DEV,test=dict(fit_end="2025-08-31",start="2025-09-01",end="2025-10-31"),
        final_fit_end="2025-10-31",candidates=CANDIDATES,model_weights=[.5,1.],
        selection="Mean observed-target WAPE across two development windows; no September/October tuning",
        target="Real successful validation counts only where working_events_observed; hour 5 already raw-cleaned",
        loss="MAE of target/reference, weighted reference/repetitions(route,date,hour)",
        reference="Past observed route-weekday-hour mean; route-hour then fitted route median fallback",
        cold_start="One history-free copy per ten examples, with inverse-repetition weights",
        calendar_regime="Retrospective external movement; known route50 autumn weekend closure forced to zero",
        budget_seconds=1800,threads=2,seed=42,teacher_used_in_training=False,
        limitations=["Previously viewed history, not independent new data", "No prior-year history, unvalidated early January",
            "One annual partial cycle, 2026 extrapolation", "Observed events do not prove complete counting",
            "Final October-trained weights make earlier-date requests retrospective, not causal backtests"],
        versions={k:importlib.metadata.version(k) for k in ["catboost","numpy","pandas"]},
        platform=platform.platform(),python=platform.python_version(),sources_sha256=hashes))
    history=read_history(history_path)
    if digest(history_path)!="7031c686c149fcb552711df49d82bab23540d8c80ffa6f140c6d0444138384ae":
        raise ValueError("Frozen cleaned-history hash differs")
    calendar=json.loads(calendar_path.read_text()); movement=json.loads(movement_path.read_text())
    common=dict(calendar=calendar,movement=movement,model_version="evaluation-only")
    trials=[]; audit=[]
    for cutoff,start in DEV:
        data=examples(history,cutoff,calendar,movement)
        audit.append(dict(cutoff=cutoff,rows=len(data),origins=int(data.start.nunique()),latest_target=str(data.target_date.max().date())))
        print(f"Development {cutoff}: {len(data)} examples, {data.start.nunique()} origins",flush=True)
        for candidate,params in enumerate(CANDIDATES):
            before=time.perf_counter(); model,scales=fit(data,params)
            for weight in [.5,1.]:
                meta=dict(common,features=model.feature_names_,route_scales=scales,model_weight=weight,training_target_end=cutoff)
                result,_=predict(model,meta,history,start)
                _,metric=evaluated(history,result)
                trials.append(dict(candidate=candidate,weight=weight,cutoff=cutoff,fit_and_evaluation_seconds=time.perf_counter()-before,**metric))
                print(f"  candidate {candidate}, weight {weight}: score {metric['score']:.5f}",flush=True)
    trial_frame=pd.DataFrame(trials)
    ranking=trial_frame.groupby(["candidate","weight"],as_index=False).wape.mean().sort_values(["wape","candidate","weight"])
    selected=ranking.iloc[0]; candidate=int(selected.candidate); weight=float(selected.weight)
    trial_frame.to_csv(out/"development.csv",sep=";",index=False)
    save_json(out/"selection.json",dict(candidate=candidate,params=CANDIDATES[candidate],model_weight=weight,mean_dev_wape=float(selected.wape)))
    print(f"Selected {candidate}, model weight {weight}",flush=True)
    test_data=examples(history,"2025-08-31",calendar,movement)
    test_rows=len(test_data)
    audit.append(dict(cutoff="2025-08-31",rows=test_rows,origins=int(test_data.start.nunique()),latest_target=str(test_data.target_date.max().date())))
    before=time.perf_counter(); model,scales=fit(test_data,CANDIDATES[candidate])
    test_fit_seconds=time.perf_counter()-before
    meta=dict(common,features=model.feature_names_,route_scales=scales,model_weight=weight,training_target_end="2025-08-31")
    test,_=predict(model,meta,history,"2025-09-01")
    test.to_csv(out/"test_september_october.csv",sep=";",index=False,date_format="%Y-%m-%d")
    keys,matrix=features(history,"2025-09-01",calendar,movement)
    baseline=publish(keys,matrix,reference(matrix,scales))
    teacher_path=LAB/"artifacts/portfolio_20260926/continuation/tabular_shape/study/tabular_shape/selected/raw_2025-08-31.csv"
    raw=pd.read_csv(teacher_path,sep=";",parse_dates=["date"],float_precision="round_trip")
    pd.testing.assert_frame_equal(keys,raw[KEYS])
    teacher=rounded(keys,raw.prediction)
    metrics=[];breakdowns=[]
    for name,prediction in [("cpu_forecast",test),("mean_same_closure",baseline),("030",teacher)]:
        paired,metric=evaluated(history,prediction)
        metrics.append(dict(model=name,**metric))
        paired["horizon_week"]=(paired.date-pd.Timestamp("2025-09-01")).dt.days//7+1
        for dimension in ["route","hour","horizon_week"]:
            for value,part in paired.groupby(dimension):
                breakdowns.append(dict(model=name,dimension=dimension,value=int(value),**measure(part.boardings,part.prediction)))
        print(f"Test {name}: score {metric['score']:.5f}",flush=True)
    pd.DataFrame(metrics).to_csv(out/"test_metrics.csv",sep=";",index=False)
    pd.DataFrame(breakdowns).to_csv(out/"breakdown.csv",sep=";",index=False)
    evaluation=out/"evaluation";evaluation.mkdir()
    model.save_model(str(evaluation/"forecast.cbm"))
    meta.update(model_sha256=digest(evaluation/"forecast.cbm"))
    save_json(evaluation/"metadata.json",meta)
    del test_data,model,data
    final_data=examples(history,"2025-10-31",calendar,movement)
    audit.append(dict(cutoff="2025-10-31",rows=len(final_data),origins=int(final_data.start.nunique()),latest_target=str(final_data.target_date.max().date())))
    before=time.perf_counter();model,scales=fit(final_data,CANDIDATES[candidate]);final_fit_seconds=time.perf_counter()-before
    bundle=out/"bundle";bundle.mkdir()
    model.save_model(str(bundle/"forecast.cbm"))
    sha=digest(bundle/"forecast.cbm")
    identity=hashlib.sha256(json.dumps(dict(model=sha,sources=hashes,weight=weight),sort_keys=True).encode()).hexdigest()
    metadata=dict(common,features=model.feature_names_,route_scales=scales,model_weight=weight,
        training_target_end="2025-10-31",model_sha256=sha,model_version="cpu-observed-2025-"+identity[:16],
        dataset_version="observed-"+hashlib.sha256(json.dumps({str(p.relative_to(LAB.parent.parent)):digest(p)
            for p in [history_path,calendar_path,movement_path]},sort_keys=True).encode()).hexdigest()[:16],
        training_rows=len(final_data),training_origins=int(final_data.start.nunique()),params=CANDIDATES[candidate],
        role="direct_observed_target_forecaster",teacher_used=False,sources_sha256=hashes)
    save_json(bundle/"metadata.json",metadata)
    final,info=predict(model,metadata,history,"2025-11-01")
    final.to_csv(out/"forecast_november_december.csv",sep=";",index=False,date_format="%Y-%m-%d")
    save_json(out/"forecast_november_december.json",info)
    for p in sources:
        if digest(p)!=hashes[str(p.relative_to(LAB.parent.parent))]: raise ValueError("Source changed during training")
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    save_json(out/"completed.json",dict(elapsed_seconds=time.perf_counter()-begun,
        final_fit_seconds=final_fit_seconds,test_fit_seconds=test_fit_seconds,
        peak_rss_mib=rss/(1024**2 if sys.platform=="darwin" else 1024),model_bytes=(bundle/"forecast.cbm").stat().st_size,
        training_audit=audit,test_training_rows=test_rows,teacher_comparison_sha256=digest(teacher_path),sources_unchanged=True))
    print(f"Saved {bundle}; total {time.perf_counter()-begun:.1f}s",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();signal.alarm(1800)
    run(args.output.resolve())
