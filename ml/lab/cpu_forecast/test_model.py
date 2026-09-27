"""Runnable checks for observed-target training and CPU inference."""
import argparse
import json
from pathlib import Path
import resource
import statistics
import sys
import time

import numpy as np
import pandas as pd

from ml.lab.cpu_student.test_model import rejects
from ml.lab.cpu_student.model import date_start
from ml.lab.cpu_forecast.model import grid,load_bundle,predict,read_history
from ml.lab.cpu_forecast.train import examples


def run(bundle,history_path):
    before=time.perf_counter()
    model,meta=load_bundle(bundle);history=read_history(history_path)
    load_seconds=time.perf_counter()-before
    calendar,movement=meta["calendar"],meta["movement"]
    expected=examples(history,"2025-04-30",calendar,movement)
    poisoned=history.copy()
    future=poisoned.date.gt("2025-04-30")
    poisoned.loc[future | ~poisoned.working_events_observed,"boardings"]=99999999
    poisoned.loc[future,"working_events_observed"]=False
    pd.testing.assert_frame_equal(expected,examples(poisoned,"2025-04-30",calendar,movement))
    assert expected.target_date.max()==pd.Timestamp("2025-04-30")
    assert (expected.target_date>=expected.start).all()
    assert ((expected.target_date-expected.start).dt.days<61).all()
    timings=[]
    for start in ["2025-01-01","2025-01-29","2025-05-17","2025-09-15","2025-11-01","2025-12-31"]:
        before=time.perf_counter();result,info=predict(model,meta,history,start)
        timings.append(time.perf_counter()-before)
        pd.testing.assert_frame_equal(result.drop(columns="prediction"),grid(start))
        assert result.prediction.dtype==np.dtype("int64") and result.prediction.ge(0).all()
        assert result.loc[result.route.eq(5)|result.hour.between(1,4),"prediction"].eq(0).all()
        if start=="2025-01-01":
            assert info["history_latest_used"] is None and "no_past_history_unvalidated_cold_start" in info["warnings"]
        if start=="2025-12-31":
            assert result.date.max()==pd.Timestamp("2026-03-01") and "2026_extrapolation_calendar_and_movement_unknown" in info["warnings"]
    start="2025-09-15"
    expected,_=predict(model,meta,history,start)
    poisoned=history.copy()
    future=poisoned.date.ge(start)
    poisoned.loc[future|~poisoned.working_events_observed,"boardings"]=99999999
    poisoned.loc[future,"working_events_observed"]=False
    pd.testing.assert_frame_equal(expected,predict(model,meta,poisoned,start)[0])
    reloaded,other=load_bundle(bundle)
    pd.testing.assert_frame_equal(expected,predict(reloaded,other,history,start)[0])
    for start in ["2024-12-31","2026-01-01","2025-02-30","2025-01-01T00:00:00"]:
        rejects(lambda:date_start(start))
    assert not any(n in sys.modules for n in ["torch","timesfm","tabpfn"])
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps(dict(checks="passed",training_future_poison="passed",prediction_future_poison="passed",
        exact_training_cutoff="passed",boundary_dates=6,load_model_history_seconds=load_seconds,
        median_features_predict_seconds=statistics.median(timings),
        peak_rss_mib=rss/(1024**2 if sys.platform=="darwin" else 1024),gpu_libraries_loaded=False),indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--history",type=Path,default=Path("ml/lab/artifacts/hourly_clean.csv"))
    args=parser.parse_args();run(args.bundle,args.history)
