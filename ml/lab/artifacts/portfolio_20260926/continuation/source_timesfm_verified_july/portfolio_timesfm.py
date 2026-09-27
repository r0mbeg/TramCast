"""P36: pinned TimesFM 2.5 zero-shot alternative, with causal normal-demand inputs."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd

from experiments.chronos_experiment import inputs
from experiments.portfolio_bayes_volume import ROOT
from experiments.portfolio_chronos import gpu_forecast
from experiments.portfolio_combine import raw_frame, transplant
from experiments.portfolio_experiment import load_history, run_study, write_json, save_candidate, HISTORY_SHA256
from experiments.portfolio_factor import factor_inputs
from experiments.portfolio_operations import normal_history, apply_operations
from experiments.portfolio_ridge import add_calendar

REVISION = "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"
WEIGHTS_SHA256 = "2f776efe6245e42b24bc4153ffdf61810140210e4bd3b01fb21f7aa779ab6ce8"
WHEEL_SHA256 = "c7bde94beb1651e1251cdf1e9d09cf6f015e0218d038a4a836a170dc70b08071"


def download(args):
    from huggingface_hub import hf_hub_download
    if os.environ.get("SLURM_JOB_PARTITION")!="ais-cpu":
        raise RuntimeError("Download runs in ais-cpu")
    if datetime.now(timezone.utc)>=datetime.fromisoformat("2026-09-27T15:40:40+00:00"):
        raise RuntimeError("Research reserve reached")
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    write_json(out/"download_started.json",dict(job_id=os.environ["SLURM_JOB_ID"],
        model="google/timesfm-2.5-200m-pytorch",revision=REVISION,weights_sha256=WEIGHTS_SHA256,
        command=os.sys.argv,authentication="public download, token=False"))
    for name in ["model.safetensors","config.json"]:
        path=hf_hub_download("google/timesfm-2.5-200m-pytorch",filename=name,revision=REVISION,
            local_dir=args.model_path,token=False)
        if name=="model.safetensors" and hashlib.sha256(Path(path).read_bytes()).hexdigest()!=WEIGHTS_SHA256:
            raise ValueError("Downloaded TimesFM checksum differs")
    write_json(out/"download_completed.json",dict(job_id=os.environ["SLURM_JOB_ID"],
        revision=REVISION,weights_sha256=WEIGHTS_SHA256,bytes=(Path(args.model_path)/"model.safetensors").stat().st_size,
        huggingface_hub=importlib.metadata.version("huggingface-hub")))


def series_inputs(history, cutoff, end, representation, july_verified=False):
    if representation == "daily":
        _, series = inputs(history.loc[history.date.le(cutoff)], cutoff, "daily_total")
    elif representation in ["normal_daily", "normal_ratio"]:
        matrix, _, _ = factor_inputs(history, cutoff, july_verified=july_verified)
        series = matrix.T.groupby(level="route").sum().T
    else:
        raise ValueError(representation)
    future = pd.date_range(pd.Timestamp(cutoff)+pd.Timedelta(days=1), end)
    matrix = series.to_numpy(np.float32).T
    restoration = np.ones((len(series.columns), len(future)))
    if representation == "normal_ratio":
        past = add_calendar(pd.DataFrame(dict(date=series.index)), cutoff)
        selected = ~past.off.to_numpy()
        profile = series.loc[selected].groupby(past.loc[selected, "effective_weekday"].to_numpy()).median().clip(lower=1)
        denominator = profile.reindex(past.effective_weekday).to_numpy().T
        calendar = add_calendar(pd.DataFrame(dict(date=future)), cutoff)
        restoration = profile.reindex(calendar.effective_weekday).to_numpy().T
        restoration[:, calendar.off.to_numpy()] *= 0.95
        matrix = matrix/denominator
    if matrix.shape[0] != 9 or not np.isfinite(matrix).all() or not np.isfinite(restoration).all():
        raise ValueError("Invalid TimesFM cutoff input")
    return matrix, restoration


def run(args):
    import torch
    import timesfm
    if os.environ.get("SLURM_JOB_PARTITION") != "gpu_devel" or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one GPU in gpu_devel")
    cpus=int(os.environ["SLURM_CPUS_PER_TASK"])
    if not 1<=cpus<=4 or len(os.sched_getaffinity(0))>cpus:
        raise RuntimeError("Unexpected allocation")
    if datetime.now(timezone.utc)>=datetime.fromisoformat("2026-09-27T15:40:40+00:00"):
        raise RuntimeError("Research reserve reached")
    if importlib.metadata.version("timesfm") != "2.0.2":
        raise RuntimeError("TimesFM version differs from pinned wheel")
    weights=Path(args.model_path)/"model.safetensors"
    if hashlib.sha256(weights.read_bytes()).hexdigest()!=WEIGHTS_SHA256:
        raise ValueError("TimesFM weights differ from pinned revision")
    torch.set_num_threads(cpus);torch.set_num_interop_threads(1);torch.manual_seed(42)
    data=load_history(args.history)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    config=timesfm.ForecastConfig(max_context=512,max_horizon=128,per_core_batch_size=8,
        normalize_inputs=True,use_continuous_quantile_head=True,force_flip_invariance=True,
        infer_is_positive=True,fix_quantile_crossing=True)
    model=timesfm.TimesFM_2p5_200M_torch(torch_compile=False)
    model.load_checkpoint(weights,torch_compile=False)
    model.compile(config)
    paths=[Path(args.history),Path(__file__)] + [Path("experiments")/f"portfolio_{n}.py" for n in
        ["factor","chronos","combine","experiment","operations","movement","ridge"]]
    paths += list((ROOT/"conditional_shape/conditional_shape/selected").glob("raw_*.csv"))
    paths += list(Path("artifacts/portfolio_20260926/gpu/daily").glob("raw_*.csv"))
    paths += [ROOT/f"{n}/{n}/selection.json" for n in ["operations","august_operations"]]
    if args.verified_july:
        paths += list((ROOT/"bayes_shape/study/bayes_shape/selected").glob("raw_*.csv"))
        paths += list((ROOT/"timesfm/timesfm/selected").glob("raw_*.csv"))
        paths += list((ROOT/"timesfm/quantiles").glob("normal_ratio_*.*"))
        paths += [ROOT/"external/july_restoration_sources.json"]
    spec=dict(revision=REVISION,weights_sha256=WEIGHTS_SHA256,wheel_sha256=WHEEL_SHA256,
        config=vars(config),dtype="float32",torch_compile=False,seed=42,quantile_indices=[3] if args.verified_july else [3,5,7],
        verified_july_study=args.verified_july,
        pretrained_availability="modern weights and package; not available at historical 2025 cutoffs")
    write_json(out/("pilot_started.json" if args.pilot else "run_started.json"),dict(**spec,
        job_id=os.environ["SLURM_JOB_ID"],command=os.sys.argv,gpu=torch.cuda.get_device_name(),
        versions={p:importlib.metadata.version(p) for p in ["torch","timesfm","numpy","pandas","optuna"]},
        sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))
    started=time.monotonic()

    def quantiles(cutoff,end,representation,july_verified=False):
        matrix, restoration=series_inputs(data,cutoff,end,representation,july_verified=july_verified)
        cache=out/"quantiles";cache.mkdir(exist_ok=True)
        path=cache/f"{representation}_{cutoff}{'_july_verified' if july_verified else ''}.npz"
        signature=hashlib.sha256(matrix.tobytes()).hexdigest()
        metadata_path=path.with_suffix(".json")
        if path.exists() and metadata_path.exists():
            with np.load(path,allow_pickle=False) as saved:
                values=saved["quantiles"]
        else:
            previous=ROOT/"timesfm/quantiles"/f"{representation}_{cutoff}.npz"
            reused=False
            if args.verified_july and previous.exists():
                old=json.loads(previous.with_suffix(".json").read_text())
                if (old["input_sha256"]==signature and old["config"]==vars(config) and old["end"]==end
                    and old["model_sha256"]==WEIGHTS_SHA256 and old["wheel_sha256"]==WHEEL_SHA256):
                    if old["sha256"]!=hashlib.sha256(previous.read_bytes()).hexdigest():
                        raise ValueError("Changed original TimesFM quantiles")
                    with np.load(previous,allow_pickle=False) as saved:
                        values=saved["quantiles"]
                    reused=True
            # Upstream forecast pads/mutates the list; keep our route inputs unchanged.
            if not reused:
                with torch.inference_mode():
                    _,values=model.forecast(horizon=61,inputs=[r.copy() for r in matrix])
            temporary=path.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                if args.verified_july:
                    np.savez_compressed(stream,quantiles=values,inputs=matrix,restoration=restoration)
                else:
                    np.savez_compressed(stream,quantiles=values)
            temporary.replace(path)
            write_json(path.with_suffix(".json"),dict(cutoff=cutoff,end=end,representation=representation,
                shape=list(values.shape),history_sha256=HISTORY_SHA256,model_sha256=WEIGHTS_SHA256,
                input_sha256=signature,config=vars(config),wheel_sha256=WHEEL_SHA256,
                july_verified=july_verified,reused_original_quantiles=reused,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        if values.shape!=(9,61,10) or not np.isfinite(values).all():
            raise ValueError("Invalid TimesFM quantiles")
        metadata=json.loads(path.with_suffix(".json").read_text())
        if (metadata["sha256"]!=hashlib.sha256(path.read_bytes()).hexdigest() or metadata["end"]!=end
            or metadata["input_sha256"]!=signature or metadata["model_sha256"]!=WEIGHTS_SHA256
            or metadata["config"]!=vars(config) or metadata["wheel_sha256"]!=WHEEL_SHA256
            or metadata.get("july_verified",False)!=july_verified):
            raise ValueError("Changed TimesFM cache")
        return values,restoration

    if args.pilot:
        if (out/"pilot_completed.json").exists():
            raise ValueError("TimesFM pilot already completed; do not repeat")
        q,_=quantiles("2025-04-30","2025-06-30","daily")
        write_json(out/"pilot_completed.json",dict(job_id=os.environ["SLURM_JOB_ID"],
            seconds=time.monotonic()-started,shape=list(q.shape),peak_gpu_bytes=torch.cuda.max_memory_allocated()))
        return

    def forecast(cutoff,end,params):
        representation=params.get("representation","normal_ratio")
        july_verified=params.get("july_verified",False)
        if args.verified_july and not july_verified:
            return control(cutoff,end)
        q,restoration=quantiles(cutoff,end,representation,july_verified=july_verified)
        def predictor(tasks,horizon,cross):
            if horizon!=61:
                raise ValueError("Unexpected TimesFM horizon")
            return q[:,:,params.get("quantile_index",3)]*restoration
        raw=gpu_forecast(data,cutoff,end,"daily",predictor)
        shape_source=ROOT/("bayes_shape/study/bayes_shape/selected" if args.verified_july else "conditional_shape/conditional_shape/selected")
        shape=raw_frame(shape_source/f"raw_{cutoff}.csv",cutoff,end)
        raw=transplant(raw,shape,raw)
        if representation!="daily":
            base=json.loads((ROOT/"operations/operations/selection.json").read_text())["params"]
            august=json.loads((ROOT/"august_operations/august_operations/selection.json").read_text())["params"]["august7"]
            raw=apply_operations(raw,data.loc[data.date.le(cutoff)],
                normal_history(data,cutoff,extended=True,august7=True,july_verified=july_verified),
                {**base,"august7":august,**({"july_verified":True} if july_verified else {})})
        return raw
    def control(cutoff,end):
        if args.verified_july:
            raw=raw_frame(ROOT/"timesfm/timesfm/selected"/f"raw_{cutoff}.csv",cutoff,end)
            shape=raw_frame(ROOT/"bayes_shape/study/bayes_shape/selected"/f"raw_{cutoff}.csv",cutoff,end)
            return transplant(raw,shape,raw)
        raw=raw_frame(Path("artifacts/portfolio_20260926/gpu/daily")/f"raw_{cutoff}.csv",cutoff,end)
        shape=raw_frame(ROOT/"conditional_shape/conditional_shape/selected"/f"raw_{cutoff}.csv",cutoff,end)
        return transplant(raw,shape,raw)
    if args.verified_july:
        save_candidate(data,out/"timesfm_shape_control","timesfm_shape_control",control)
    run_study(data,out,"timesfm_verified_july" if args.verified_july else "timesfm",forecast,
        dict(sampler="grid",space={"july_verified":[False,True]},trials=2) if args.verified_july else
        dict(sampler="grid",space={"representation":["daily","normal_daily","normal_ratio"],"quantile_index":[3,5,7]},trials=9))
    if not args.verified_july:
        save_candidate(data,out/"chronos_shape_control","chronos_shape_control",control)
    write_json(out/"completed.json",dict(job_id=os.environ["SLURM_JOB_ID"],
        seconds=time.monotonic()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history",required=True);parser.add_argument("--output",required=True)
    parser.add_argument("--model-path",required=True);parser.add_argument("--pilot",action="store_true")
    parser.add_argument("--download",action="store_true")
    parser.add_argument("--verified-july",action="store_true")
    args=parser.parse_args()
    if args.verified_july and (args.download or args.pilot):
        parser.error("--verified-july is a separate study")
    (download if args.download else run)(args)
