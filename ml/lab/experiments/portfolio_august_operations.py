"""P33: separately model the officially documented August weekend route 7 regime."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from experiments.portfolio_bayes_volume import ROOT
from experiments.portfolio_combine import raw_frame, transplant
from experiments.portfolio_experiment import load_history, run_study, write_json
from experiments.portfolio_operations import operations_forecast


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
    paths = [Path(args.history), Path(__file__), ROOT / "operations/operations/selection.json",
        ROOT / "external/route7_august_sources.json"]
    paths += [Path("experiments") / f"portfolio_{name}.py" for name in
        ["operations", "movement", "combine", "experiment", "weather_volume", "structure", "ridge"]]
    paths += list((ROOT / "conditional_shape/conditional_shape/selected").glob("raw_*.csv"))
    write_json(out / "run_started.json", dict(job_id=os.environ["SLURM_JOB_ID"], command=os.sys.argv,
        regime="retrospective official movement announcements, no future target observations",
        sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))
    history = load_history(args.history)
    base_params = json.loads((ROOT / "operations/operations/selection.json").read_text())["params"]
    def forecast(cutoff, end, params):
        settings = {**base_params, "extended_ops": True}
        if params["august7"] != "control":
            settings["august7"] = params["august7"]
        raw = operations_forecast(history, cutoff, end, settings)
        shape = raw_frame(ROOT / "conditional_shape/conditional_shape/selected" / f"raw_{cutoff}.csv", cutoff, end)
        return transplant(raw, shape, raw)
    run_study(history, out, "august_operations", forecast, dict(sampler="grid",
        space={"august7": ["control", 0.6, 0.8, 1.]}, trials=4))
    write_json(out / "completed.json", dict(job_id=os.environ["SLURM_JOB_ID"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
