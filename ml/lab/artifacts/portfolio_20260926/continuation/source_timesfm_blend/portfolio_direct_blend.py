"""Small global blends of fixed saved candidates (P28 or P38)."""
import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path

from experiments.portfolio_combine import mix, raw_frame
from experiments.portfolio_experiment import load_history, run_study, write_json


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
    sources = dict(operations=root / "operations/operations/selected",
        bayes=root / "bayes_volume/bayes_volume/selected", direct=root / "direct_daily/direct_daily/selected")
    if args.timesfm:
        sources=dict(annual=root/"bayes_annual/bayes_annual/selected",direct=root/"timesfm/timesfm/selected")
    paths = [Path(args.history), Path(__file__), Path("experiments/portfolio_combine.py"),
        Path("experiments/portfolio_experiment.py")] + [p for folder in sources.values() for p in folder.glob("raw_*.csv")]
    write_json(out / "run_started.json", dict(job_id=os.environ["SLURM_JOB_ID"], command=os.sys.argv,
        components={name: str(folder) for name, folder in sources.items()},
        sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))
    def forecast(cutoff, end, params):
        base = raw_frame(sources[params["base"]] / f"raw_{cutoff}.csv", cutoff, end)
        direct = raw_frame(sources["direct"] / f"raw_{cutoff}.csv", cutoff, end)
        return mix(base, direct, params["base_weight"])
    run_study(load_history(args.history), out, "timesfm_blend" if args.timesfm else "direct_blend", forecast,
        dict(sampler="grid", space={"base": ["annual"] if args.timesfm else ["operations", "bayes"],
            "base_weight": [0., 0.25, 0.5, 0.75, 1.]}, trials=5 if args.timesfm else 10))
    write_json(out / "completed.json", dict(job_id=os.environ["SLURM_JOB_ID"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timesfm", action="store_true")
    run(parser.parse_args())
