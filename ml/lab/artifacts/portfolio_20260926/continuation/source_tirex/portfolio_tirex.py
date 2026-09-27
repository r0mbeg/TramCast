"""P56: pinned TiRex recurrent volumes with fixed Bayesian hourly shares."""
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

from experiments.portfolio_experiment import FINAL, load_history, run_study, save_candidate, write_json
from experiments.portfolio_timesfm import series_inputs, teacher_daily
from experiments.portfolio_timesfm_errors import SHAPE
from experiments.portfolio_combine import raw_frame

REVISION = "63c740922493f5fbe60b277609ec62babfba2762"
WEIGHTS_SHA256 = "b8c3f5a036c63272ce4b91c00187e26922a394cb6cb49d4e16db070ad0422314"
WHEEL_SHA256 = "e470b2e1a4ad2fe6ab95012da6b92ffa20f69ff3a5d44b5b37ba3cd86ff2b2e0"


def hourly(history, cutoff, end, values, restoration):
    # Reuse the fixed operations and route-grid transform; its q3 slot receives our selected quantile.
    daily = teacher_daily(history, cutoff, end, np.repeat(values[:,:,None],10,axis=2), restoration)
    shape = raw_frame(SHAPE/f"raw_{cutoff}.csv", cutoff, end)
    total = shape.groupby(["route", "date"]).prediction.transform("sum")
    result = shape.merge(daily.rename(columns={"prediction":"volume"}), on=["route", "date"], validate="many_to_one")
    result["prediction"] = shape.prediction.div(total.where(total.gt(0))).fillna(0)*result.volume
    if not np.isfinite(result.prediction).all() or result.prediction.lt(0).any():
        raise ValueError("Invalid TiRex hourly forecast")
    return result.drop(columns="volume")


def download(args):
    from huggingface_hub import hf_hub_download
    if os.environ.get("SLURM_JOB_PARTITION") != "ais-cpu":
        raise RuntimeError("Download requires ais-cpu")
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    write_json(out/"download_started.json",dict(job_id=os.environ["SLURM_JOB_ID"],revision=REVISION,
        authentication="public token=False",command=os.sys.argv))
    for name in ["model.ckpt","README.md","LICENSE"]:
        path=hf_hub_download("NX-AI/TiRex",filename=name,revision=REVISION,local_dir=args.model_path,token=False)
        if name=="model.ckpt" and hashlib.sha256(Path(path).read_bytes()).hexdigest()!=WEIGHTS_SHA256:
            raise ValueError("Wrong pinned TiRex weights")
    write_json(out/"download_completed.json",dict(job_id=os.environ["SLURM_JOB_ID"],revision=REVISION,
        weights_sha256=WEIGHTS_SHA256,bytes=(Path(args.model_path)/"model.ckpt").stat().st_size))


def run(args):
    if datetime.now(timezone.utc)>=datetime.fromisoformat("2026-09-27T15:40:40+00:00"):
        raise RuntimeError("Research reserve reached")
    if args.download:
        return download(args)
    import torch
    from tirex import TiRexZero
    if os.environ.get("SLURM_JOB_PARTITION")!="gpu_devel" or torch.cuda.device_count()!=1:
        raise RuntimeError("Expected one gpu_devel GPU")
    cpus=int(os.environ["SLURM_CPUS_PER_TASK"])
    if not 1<=cpus<=2 or len(os.sched_getaffinity(0))>cpus:
        raise RuntimeError("Unexpected CPU allocation")
    if importlib.metadata.version("tirex-ts")!="1.4.2":
        raise RuntimeError("Wrong TiRex package")
    weights=Path(args.model_path)/"model.ckpt"
    if hashlib.sha256(weights.read_bytes()).hexdigest()!=WEIGHTS_SHA256:
        raise ValueError("Wrong TiRex checkpoint")
    torch.set_num_threads(cpus);torch.set_num_interop_threads(1);torch.manual_seed(42)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    model=TiRexZero.from_pretrained(str(weights),backend="torch",device="cuda",compile=False).eval()
    np.testing.assert_allclose(model.config.quantiles,np.arange(1,10)/10,rtol=0,atol=1e-8)
    history=load_history(args.history)
    config=dict(backend="torch",compile=False,dtype="float32",batch_size=9,prediction_length=61,
        resample_strategy=None,seed=42,july_verified=True)
    paths=[Path(args.history),Path(__file__),Path("pipeline.py")]
    paths += [Path("experiments")/f"portfolio_{n}.py" for n in
        ["timesfm","timesfm_errors","factor","chronos","combine","experiment","operations","movement","ridge"]]
    paths += list(SHAPE.glob("raw_*.csv"))
    paths += [Path("artifacts/portfolio_20260926/continuation")/f"{n}/{n}/selection.json"
        for n in ["operations","august_operations"]]
    write_json(out/("pilot_started.json" if args.pilot else "run_started.json"),dict(job_id=os.environ["SLURM_JOB_ID"],
        config=config,revision=REVISION,weights_sha256=WEIGHTS_SHA256,wheel_sha256=WHEEL_SHA256,
        command=os.sys.argv,gpu=torch.cuda.get_device_name(),
        pretrained_availability="modern checkpoint, not available at historical2025cutoffs",
        versions={n:importlib.metadata.version(n) for n in ["torch","tirex-ts","numpy","pandas","optuna"]},
        sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))
    started=time.monotonic()

    def quantiles(cutoff,end,representation):
        matrix,restoration=series_inputs(history,cutoff,end,representation,july_verified=True)
        cache=out/"quantiles";cache.mkdir(exist_ok=True)
        path=cache/f"{representation}_{cutoff}.npz"
        signature=dict(input_sha256=hashlib.sha256(matrix.tobytes()).hexdigest(),config=config,end=end,
            model_sha256=WEIGHTS_SHA256,wheel_sha256=WHEEL_SHA256)
        if path.exists():
            info=json.loads(path.with_suffix(".json").read_text())
            if any(info.get(k)!=v for k,v in signature.items()) or info["sha256"]!=hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError("Changed TiRex quantile cache")
            with np.load(path,allow_pickle=False) as saved:
                values=saved["quantiles"]
                np.testing.assert_array_equal(saved["inputs"],matrix)
                np.testing.assert_array_equal(saved["restoration"],restoration)
        else:
            with torch.inference_mode():
                values,_=model.forecast(torch.tensor(matrix.copy()),prediction_length=61,batch_size=9,
                    output_type="numpy",resample_strategy=None)
            if values.shape!=(9,61,9) or not np.isfinite(values).all():
                raise ValueError("Unexpected TiRex API output")
            temporary=path.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream,quantiles=values,inputs=matrix,restoration=restoration)
            temporary.replace(path)
            write_json(path.with_suffix(".json"),dict(**signature,origin=cutoff,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        return values,restoration

    if args.pilot:
        values,_=quantiles(*FINAL,"normal_daily")
        with torch.inference_mode():
            matrix,_=series_inputs(history,*FINAL,"normal_daily",july_verified=True)
            repeat,_=model.forecast(torch.tensor(matrix),prediction_length=61,batch_size=9,output_type="numpy",resample_strategy=None)
        np.testing.assert_array_equal(values,repeat)
    else:
        def forecast(cutoff,end,recipe):
            if recipe=="control":
                return raw_frame(SHAPE/f"raw_{cutoff}.csv",cutoff,end)
            representation,index=recipe.rsplit("_",1)
            values,restoration=quantiles(cutoff,end,representation)
            return hourly(history,cutoff,end,values[:,:,int(index)],restoration)
        run_study(history,out,"tirex",lambda c,e,p:forecast(c,e,p["recipe"]),
            dict(sampler="grid",space={"recipe":["control"]+[f"{view}_{q}" for view in
                ["normal_daily","normal_ratio"] for q in [2,4,6]]},trials=7))
        if json.loads((out/"tirex/selection.json").read_text())["params"]["recipe"]=="control":
            import optuna
            study=optuna.load_study(study_name="tirex",storage=f"sqlite:///{out/'tirex/study.db'}")
            best=max((t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE
                and t.params['recipe']!='control'),key=lambda t:t.value)
            folder=out/"architecture_selected";folder.mkdir(exist_ok=True)
            write_json(folder/"parameters.json",dict(params=best.params,trial=best.number,value=best.value,
                selection_windows="W1/W2; separate architecture alternative, control remains winner"))
            save_candidate(history,folder,"tirex_architecture",lambda c,e:forecast(c,e,best.params['recipe']))
    write_json(out/("pilot_completed.json" if args.pilot else "completed.json"),dict(job_id=os.environ["SLURM_JOB_ID"],
        elapsed_seconds=time.monotonic()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,deterministic_repeat_exact=bool(args.pilot)))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history",required=True)
    parser.add_argument("--model-path",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--pilot",action="store_true")
    parser.add_argument("--download",action="store_true")
    run(parser.parse_args())
