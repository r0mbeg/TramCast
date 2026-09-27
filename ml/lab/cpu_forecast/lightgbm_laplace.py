"""CPU LightGBM regression_l1: Laplace location with fixed scale, not distributional regression."""
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

import lightgbm as lgb
import numpy as np
import pandas as pd

from ml.lab.cpu_forecast.train import DEV, LAB, evaluated, examples
from ml.lab.cpu_forecast.model import publish, reference
from ml.lab.cpu_student.model import HISTORY_FEATURES, features, grid, read_history
from ml.lab.cpu_student.train import digest, measure, save_json
from ml.runtime.constants import KEYS, ROUTES

CANDIDATES = [dict(num_leaves=15,n_estimators=350),dict(num_leaves=31,n_estimators=500)]


def encoded(matrix):
    matrix=matrix.copy()
    matrix["route"]=pd.Categorical(matrix.route,categories=[str(r) for r in ROUTES])
    matrix["weekday"]=pd.Categorical(matrix.weekday,categories=[str(d) for d in range(7)])
    if matrix[["route","weekday"]].isna().any().any():
        raise ValueError("Unknown categorical value")
    return matrix


def fit(data,params):
    unique=data.drop_duplicates(["route","target_date","hour"])
    scales={**unique.groupby("route").target.median().clip(lower=1).to_dict(),"5":1.}
    matrix=data.drop(columns=["target","target_date","start"])
    base=reference(matrix,scales)
    repetitions=data.groupby(["route","target_date","hour"]).target.transform("size").to_numpy()
    weights=base/repetitions
    normalization=weights.mean();weights/=normalization
    # Preserve the previous CatBoost protocol exactly; early-year cold-start quality remains unvalidated.
    cold=matrix.iloc[::10].copy();cold[HISTORY_FEATURES]=np.nan;cold["history_days"]=0.
    cold_base=reference(cold,scales)
    x=encoded(pd.concat([matrix,cold],ignore_index=True))
    y=np.concatenate([data.target.to_numpy()/base,data.target.to_numpy()[::10]/cold_base])
    w=np.concatenate([weights,cold_base/repetitions[::10]/normalization])
    model=lgb.LGBMRegressor(**params,objective="regression_l1",learning_rate=.06,
        min_child_samples=40,reg_lambda=1.,random_state=42,n_jobs=2,
        deterministic=True,force_col_wise=True,verbosity=-1)
    model.fit(x,y,sample_weight=w,categorical_feature=["route","weekday"])
    return model.booster_,scales


def predict(model,metadata,history,start):
    keys,matrix=features(history,start,metadata["calendar"],metadata["movement"])
    if matrix.columns.tolist()!=model.feature_name():
        raise ValueError("Feature contract mismatch")
    base=reference(matrix,metadata["route_scales"])
    ratio=model.predict(encoded(matrix),num_threads=2)
    result=publish(keys,matrix,base*(metadata["model_weight"]*ratio+1-metadata["model_weight"]))
    pd.testing.assert_frame_equal(result[KEYS],grid(start))
    if len(result)!=14640 or not result.prediction.ge(0).all():
        raise ValueError("Invalid published grid")
    if not result.loc[result.route.eq(5)|result.hour.between(1,4),"prediction"].eq(0).all():
        raise ValueError("Structural zeros differ")
    return result


def run(out):
    begun=time.perf_counter();out.mkdir(parents=True,exist_ok=False)
    history_path=LAB/"artifacts/hourly_clean.csv"
    calendar_path=LAB/"artifacts/calendar_sources.json"
    movement_path=LAB.parent/"preparation/movement_calendar.json"
    catroot=LAB/"artifacts/cpu_forecast_20260927_v2"
    sources=[history_path,calendar_path,movement_path,Path(__file__),LAB/"cpu_forecast/train.py",
        LAB/"cpu_forecast/model.py",LAB/"cpu_student/model.py",LAB/"cpu_student/train.py",LAB.parent/"runtime/constants.py",
        catroot/"submission.csv",catroot/"test_september_october.csv"]
    hashes={str(p.relative_to(LAB.parent.parent)):digest(p) for p in sources}
    save_json(out/"protocol.json",dict(
        hypothesis="LightGBM fixed-scale Laplace location regression improves the observed-target CPU recipe",
        objective="regression_l1",interpretation="Laplace location / MAE; no learned scale or prediction intervals",
        docs="https://lightgbm.readthedocs.io/en/latest/Parameters.html#objective",
        candidates=CANDIDATES,model_weights=[.5,1.],learning_rate=.06,min_child_samples=40,reg_lambda=1.,
        development=DEV,test=["2025-09-01","2025-10-31"],test_fit_end="2025-08-31",final_fit_end="2025-10-31",
        selection="Mean WAPE on two development windows only; September/October already viewed, no tuning there",
        features_and_training_examples="Identical to cpu_forecast_20260927_v2 including cold-start sampling and weights",
        baseline_closed_score=dict(value=.85630,source="user report in conversation",independently_verified=False,
            submission_sha256=digest(catroot/"submission.csv")),
        threads=2,seed=42,budget_seconds=1800,teacher_used_in_training=False,
        versions={n:importlib.metadata.version(n) for n in ["lightgbm","numpy","pandas"]},
        platform=platform.platform(),sources_sha256=hashes))
    history=read_history(history_path)
    if digest(history_path)!="7031c686c149fcb552711df49d82bab23540d8c80ffa6f140c6d0444138384ae":
        raise ValueError("Frozen history hash differs")
    common=dict(calendar=json.loads(calendar_path.read_text()),movement=json.loads(movement_path.read_text()))
    trials=[]
    for cutoff,start in DEV:
        data=examples(history,cutoff,**common)
        for candidate,params in enumerate(CANDIDATES):
            before=time.perf_counter();model,scales=fit(data,params)
            for weight in [.5,1.]:
                meta=dict(common,route_scales=scales,model_weight=weight)
                result=predict(model,meta,history,start)
                _,metric=evaluated(history,result)
                trials.append(dict(candidate=candidate,weight=weight,cutoff=cutoff,seconds=time.perf_counter()-before,**metric))
                print(f"Dev {cutoff} candidate {candidate} weight {weight}: {metric['score']:.5f}",flush=True)
    trial_frame=pd.DataFrame(trials);trial_frame.to_csv(out/"development.csv",sep=";",index=False)
    selected=trial_frame.groupby(["candidate","weight"],as_index=False).wape.mean().sort_values(["wape","candidate","weight"]).iloc[0]
    candidate=int(selected.candidate);weight=float(selected.weight)
    save_json(out/"selection.json",dict(candidate=candidate,params=CANDIDATES[candidate],model_weight=weight,mean_dev_wape=float(selected.wape)))
    data=examples(history,"2025-08-31",**common)
    model,scales=fit(data,CANDIDATES[candidate]);meta=dict(common,route_scales=scales,model_weight=weight)
    predicted=predict(model,meta,history,"2025-09-01")
    predicted.to_csv(out/"test_september_october.csv",sep=";",index=False,date_format="%Y-%m-%d")
    model.save_model(str(out/"evaluation.lgb"));save_json(out/"evaluation_metadata.json",meta)
    comparisons=[("lightgbm_laplace",predicted),("cpu_catboost",pd.read_csv(catroot/"test_september_october.csv",sep=";",parse_dates=["date"]))]
    keys,matrix=features(history,"2025-09-01",**common)
    comparisons.append(("mean_same_closure",publish(keys,matrix,reference(matrix,scales))))
    rows=[];details=[]
    for name,prediction in comparisons:
        paired,metric=evaluated(history,prediction);rows.append(dict(model=name,**metric))
        paired["horizon_week"]=(paired.date-pd.Timestamp("2025-09-01")).dt.days//7+1
        for dimension in ["route","hour","horizon_week"]:
            for value,part in paired.groupby(dimension):
                details.append(dict(model=name,dimension=dimension,value=int(value),**measure(part.boardings,part.prediction)))
        print(f"Test {name}: {metric['score']:.5f}",flush=True)
    pd.DataFrame(rows).to_csv(out/"test_metrics.csv",sep=";",index=False)
    pd.DataFrame(details).to_csv(out/"breakdown.csv",sep=";",index=False)
    data=examples(history,"2025-10-31",**common)
    before=time.perf_counter();model,scales=fit(data,CANDIDATES[candidate]);fit_seconds=time.perf_counter()-before
    model.save_model(str(out/"forecast.lgb"))
    identity=hashlib.sha256(json.dumps(dict(model=digest(out/"forecast.lgb"),sources=hashes,weight=weight),sort_keys=True).encode()).hexdigest()
    meta=dict(common,route_scales=scales,model_weight=weight,model_version="lightgbm-laplace-2025-"+identity[:16],
        training_target_end="2025-10-31",model_sha256=digest(out/"forecast.lgb"),objective="regression_l1",
        interpretation="fixed_scale_laplace_location",params=CANDIDATES[candidate],training_examples=len(data),
        caveats=["Earlier-date requests use retrospective October-trained weights","Early January and 2026 unvalidated",
                 "Retrospective calendar/movement snapshots","Closed score of this LightGBM submission unknown"])
    save_json(out/"metadata.json",meta)
    before=time.perf_counter();final=predict(model,meta,history,"2025-11-01");inference_seconds=time.perf_counter()-before
    final.to_csv(out/"submission.csv",sep=";",index=False,date_format="%Y-%m-%d")
    reloaded=lgb.Booster(model_file=str(out/"forecast.lgb"))
    pd.testing.assert_frame_equal(final,predict(reloaded,meta,history,"2025-11-01"))
    # Verify arbitrary-start reload, year boundaries and future-target exclusion without a GPU.
    for start in ["2025-01-01","2025-09-15","2025-12-31"]:
        result=predict(reloaded,meta,history,start)
        poisoned=history.copy();mask=poisoned.date.ge(start)
        poisoned.loc[mask|~poisoned.working_events_observed,"boardings"]=99999999
        poisoned.loc[mask,"working_events_observed"]=False
        pd.testing.assert_frame_equal(result,predict(reloaded,meta,poisoned,start))
    if any(name in sys.modules for name in ["torch","timesfm","tabpfn"]):raise ValueError("Unexpected GPU dependency")
    for path in sources:
        if digest(path)!=hashes[str(path.relative_to(LAB.parent.parent))]:raise ValueError("Source changed")
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    save_json(out/"completed.json",dict(elapsed_seconds=time.perf_counter()-begun,final_fit_seconds=fit_seconds,
        features_predict_seconds=inference_seconds,peak_rss_mib=rss/(1024**2 if sys.platform=="darwin" else 1024),
        model_bytes=(out/"forecast.lgb").stat().st_size,submission_sha256=digest(out/"submission.csv"),
        rows=len(final),prediction_total=int(final.prediction.sum()),checks="grid, zeros, int64, reload, future poison passed",
        gpu_used=False,sources_unchanged=True))
    print(f"Saved submission: {out/'submission.csv'}",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();signal.alarm(1800);run(args.output.resolve())
