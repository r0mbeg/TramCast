"""Small global blends of fixed saved candidates (P28 or P38)."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import resource
import time

from experiments.portfolio_combine import mix, raw_frame, transplant
from experiments.portfolio_experiment import load_history, run_study, write_json


def volume_blend(base, direct, weight):
    if weight == 1:
        return base.copy()
    return transplant(mix(base, direct, weight), base, base)


def run(args):
    if os.environ.get("SLURM_JOB_PARTITION") != "ais-cpu":
        raise RuntimeError("Expected ais-cpu")
    cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
    if not 1 <= cpus <= 4 or len(os.sched_getaffinity(0)) > cpus:
        raise RuntimeError("Unexpected allocation")
    if datetime.now(timezone.utc) >= datetime.fromisoformat("2026-09-27T15:40:40+00:00"):
        raise RuntimeError("Research reserve reached")
    root = Path("artifacts/portfolio_20260926/continuation")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    sources = dict(operations=root / "operations/operations/selected",
        bayes=root / "bayes_volume/bayes_volume/selected", direct=root / "direct_daily/direct_daily/selected")
    if args.timesfm:
        sources=dict(annual=root/("verified_july/verified_july/selected" if args.verified_july else "bayes_annual/bayes_annual/selected"),
            direct=root/"timesfm/timesfm/selected")
    if args.bayes_shape:
        sources["annual"] = root/"bayes_shape/study/bayes_shape/selected"
    paths = [Path(args.history), Path(__file__), Path("experiments/portfolio_combine.py"),
        Path("experiments/portfolio_experiment.py")] + [p for folder in sources.values() for p in folder.glob("raw_*.csv")]
    write_json(out / "run_started.json", dict(job_id=os.environ["SLURM_JOB_ID"], command=os.sys.argv,
        components={name: str(folder) for name, folder in sources.items()},
        hourly_shares="fixed 024" if args.bayes_shape else "raw blend",
        versions=dict(python=platform.python_version(), **{n:importlib.metadata.version(n) for n in ["numpy","pandas","scikit-learn","optuna"]}),
        sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))
    def forecast(cutoff, end, params):
        base = raw_frame(sources[params["base"]] / f"raw_{cutoff}.csv", cutoff, end)
        direct = raw_frame(sources["direct"] / f"raw_{cutoff}.csv", cutoff, end)
        return (volume_blend if args.bayes_shape else mix)(base, direct, params["base_weight"])
    family=("bayes_shape_timesfm_blend" if args.bayes_shape else "verified_timesfm_blend" if args.verified_july
        else "timesfm_blend" if args.timesfm else "direct_blend")
    run_study(load_history(args.history), out, family, forecast,
        dict(sampler="grid", space={"base": ["annual"] if args.timesfm else ["operations", "bayes"],
            "base_weight": [0., 0.25, 0.5, 0.75, 1.]}, trials=5 if args.timesfm else 10))
    write_json(out / "completed.json", dict(job_id=os.environ["SLURM_JOB_ID"],
        elapsed_seconds=time.monotonic()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timesfm", action="store_true")
    parser.add_argument("--verified-july", action="store_true")
    parser.add_argument("--bayes-shape", action="store_true")
    args=parser.parse_args()
    if args.verified_july and not args.timesfm:
        parser.error("--verified-july requires --timesfm")
    if args.bayes_shape and (not args.timesfm or args.verified_july):
        parser.error("--bayes-shape requires --timesfm as a separate study")
    run(args)
