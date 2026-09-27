"""Rebuild 030-family inputs for the 61 days after updated history, outside runtime."""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone

ML = Path(__file__).resolve().parents[1]
LAB = ML / "lab"
ROOT = Path("artifacts/portfolio_20260926/continuation")
FAMILIES = ("operations", "bayes_volume", "direct_daily", "conditional_shape", "august_operations")


def source_paths(sources):
    paths = [Path("artifacts/calendar_sources.json")]
    paths += [ROOT / f"{name}/{name}/selection.json" for name in FAMILIES]
    paths += [ROOT / "external" / name for name in
              ("weather_2025.csv", "weather_source.json", "school_calendar/sources.json", "july_restoration_sources.json")]
    paths += [ROOT / "external" / name for name in ("weather_2025_era5.json", "movement_calendar.json")
              if (sources / ROOT / "external" / name).exists()]
    return paths


def init_sources(destination):
    destination.mkdir(parents=True, exist_ok=False)
    for path in source_paths(LAB):
        snapshot(LAB / path, destination / path)
    snapshot(Path(__file__).with_name("movement_calendar.json"), destination / ROOT / "external/movement_calendar.json")
    print(destination.resolve())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def snapshot(source, destination):
    before = digest(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if digest(destination) != before or digest(source) != before:
        raise ValueError(f"Source changed while copying: {source}")
    return before


def validate_history(path, cutoff):
    import numpy as np
    import pandas as pd
    from pipeline import KEYS, full_grid
    data = pd.read_csv(path, sep=";", parse_dates=["date"])
    expected = full_grid(end=cutoff).to_frame(index=False)
    expected["date"] = pd.to_datetime(expected.date)
    pd.testing.assert_frame_equal(data[KEYS].reset_index(drop=True), expected)
    for name in ("boardings", "working_events"):
        values = data[name].to_numpy(float)
        if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
            raise ValueError(f"Invalid observed counts: {name}")
    if data.working_events_observed.dtype != bool or not data.working_events_observed.equals(data.working_events.gt(0)):
        raise ValueError("Missing or inconsistent observation mask")
    if (data.boardings > data.working_events).any():
        raise ValueError("Successful count exceeds all working events")
    if data.loc[data.working_events.gt(0), "date"].max() != pd.Timestamp(cutoff):
        raise ValueError("History cutoff exceeds the latest observed event date")
    if not data.loc[data.route.eq(5) | data.hour.between(1, 4), "boardings"].eq(0).all():
        raise ValueError("Invalid structural zeros")
    return data


def validate_weather(path, end):
    import numpy as np
    import pandas as pd
    data = pd.read_csv(path, sep=";", parse_dates=["date"])
    expected = pd.date_range("2025-01-01", end)
    if data.date.duplicated().any() or not expected.isin(data.date).all():
        raise ValueError(f"Weather must cover every day from 2025-01-01 through {end}")
    values = data[["temperature_2m_mean", "precipitation_sum", "daylight_duration"]]
    if not np.isfinite(values).all().all() or data.precipitation_sum.lt(0).any() or not data.daylight_duration.between(0, 86400).all():
        raise ValueError("Invalid weather values")
    return data


def publish(staging, destination):
    # An exclusive reservation prevents two builders publishing to the same name.
    destination.mkdir()
    try:
        os.replace(staging, destination)
    except BaseException:
        destination.rmdir()
        raise


def validate_movement(spec, end):
    coverage = spec["coverage"]
    if coverage[0] > "2025-01-01" or coverage[1] < end or not spec.get("sources"):
        raise ValueError("Movement calendar needs reviewed coverage and sources")
    expected = {"april17": {17}, "july": {7, 50}, "july_verified": {7, 50},
                "autumn": {7, 50}, "august7": {7}}
    if set(spec["events"]) != set(expected):
        raise ValueError("Unsupported operation categories; extend and validate the model first")
    for name, intervals in spec["events"].items():
        for item in intervals:
            if (date.fromisoformat(item["start"]) > date.fromisoformat(item["end"])
                    or not item["routes"] or not set(item["routes"]) <= expected[name]
                    or not set(item.get("weekdays", range(7))) <= set(range(7))):
                raise ValueError(f"Invalid movement interval: {name}")


def run(args):
    cutoff = date.fromisoformat(args.history_end)
    if cutoff < date(2025, 10, 31):
        raise ValueError("This release builder requires history starting 2025-01-01 through at least 2025-10-31")
    end = cutoff + timedelta(days=61)
    destination = args.output.resolve()
    if destination.exists():
        raise FileExistsError("Output already exists; choose a new release directory")
    sources = args.sources.resolve()
    source_files = source_paths(sources)
    weather_raw = ROOT / "external/weather_2025_era5.json"
    movement = ROOT / "external/movement_calendar.json"
    if movement not in source_files and end > date(2025, 12, 31):
        raise ValueError("New period requires external/movement_calendar.json with reviewed coverage and event intervals")
    for path in source_files:
        if not (sources / path).is_file():
            raise FileNotFoundError(sources / path)
    args.history = args.history.resolve() if args.history else None
    args.dataset = args.dataset.resolve() if args.dataset else None
    args.events = [p.resolve() for p in args.events] if args.events else None
    args.timesfm_cache = args.timesfm_cache.resolve() if args.timesfm_cache else None
    args.timesfm_model = args.timesfm_model.resolve() if args.timesfm_model else None
    args.tabpfn_model = args.tabpfn_model.resolve()
    template = json.loads((ML / "recipes/tabpfn-030/recipe.json").read_text())
    if digest(args.tabpfn_model) != template["model_sha256"]:
        raise ValueError("Unverified TabPFN checkpoint")
    if args.timesfm_model is None and args.timesfm_cache is None:
        raise ValueError("Provide --timesfm-model (GPU) or --timesfm-cache (verified reuse only)")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A new workspace invalidates all legacy caches. Historical scripts/artifacts stay untouched.
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=destination.parent) as temporary:
        work = Path(temporary)
        for path in [*LAB.glob("*.py"), *LAB.glob("experiments/*.py")]:
            snapshot(path, work / path.relative_to(LAB))
        lineage = {"sources": {}, "code": {}, "history_end": str(cutoff), "forecast_end": str(end),
                   "method": "030 family with fixed parameters; updated inputs are not the historical submission"}
        for path in source_files:
            lineage["sources"][str(path)] = snapshot(sources / path, work / path)
        for path in sorted(work.rglob("*.py")):
            lineage["code"][str(path.relative_to(work))] = digest(path)
        for path in sorted(Path(__file__).parent.glob("*.py")):
            lineage["code"]["preparation/" + path.name] = snapshot(path, work / "build_source" / path.name)
        # The source snapshot is imported only after all paths have been redirected.
        sys.path.insert(0, str(work))
        import pipeline
        pipeline.OUT = work / "artifacts"
        os.chdir(work)
        weather = validate_weather(ROOT / "external/weather_2025.csv", str(end))
        weather_spec = json.loads((ROOT / "external/weather_source.json").read_text())
        if weather_spec["sha256"] != digest(ROOT / "external/weather_2025.csv"):
            # Historical metadata hashes the API JSON, not the derived CSV.
            if not weather_raw.exists() or digest(weather_raw) != weather_spec["sha256"]:
                raise ValueError("Weather checksum does not match its provenance")
            import pandas as pd
            original_weather = pd.DataFrame(json.loads(weather_raw.read_text())["daily"]).rename(columns={"time": "date"})
            original_weather["date"] = pd.to_datetime(original_weather.date)
            pd.testing.assert_frame_equal(weather, original_weather, check_dtype=False)
        if not weather_spec.get("regime") or not weather_spec.get("retrieved_at") or not weather_spec.get("url"):
            raise ValueError("Weather requires regime, retrieved_at and source URL")
        from experiments.calendar_experiment import calendar_table
        import pandas as pd
        dates = pd.date_range("2025-01-01", end)
        calendar = calendar_table()
        if not dates.isin(calendar.date).all() or calendar.known_at.isna().any():
            raise ValueError("Production calendar lacks coverage or publication dates")
        school = json.loads((ROOT / "external/school_calendar/sources.json").read_text())
        if school["coverage"][0] > "2025-01-01" or school["coverage"][1] < str(end):
            raise ValueError("School calendar does not cover the new horizon")
        if movement in source_files:
            validate_movement(json.loads(movement.read_text()), str(end))
        lineage["weather_regime"] = weather_spec["regime"]
        print("1/5 Preparing history", flush=True)
        history_path = pipeline.OUT / "hourly_clean.csv"
        if args.history:
            lineage["history_input"] = snapshot(args.history, history_path)
        else:
            event_files = args.events or [args.dataset / n for n in ("train.csv", "test.csv")]
            labels = sorted((args.dataset / "labels").glob("*.csv")) if args.dataset else []
            raw_files = event_files + labels
            if not labels and not args.no_labels:
                raise ValueError("Supplied labels are required for raw reconciliation")
            before = {str(p): digest(p) for p in raw_files}
            if len({before[str(p)] for p in event_files}) != len(event_files):
                raise ValueError("Duplicate event file contents")
            if args.dataset:
                pipeline.DATA = args.dataset
            pipeline.prepare(end=str(cutoff + timedelta(days=1)), reconcile=not args.no_labels, files=event_files)
            if before != {str(p): digest(p) for p in raw_files}:
                raise ValueError("Raw inputs changed during preparation")
            lineage["raw"] = before
        history = validate_history(history_path, str(cutoff))
        active = history.route.ne(5) & ~history.hour.between(1, 4)
        lineage["quality"] = {"working_cells_without_events": int((active & ~history.working_events_observed).sum()),
                              "missing_policy": "zero-filled counts with observation masks; not confirmed zero demand"}
        import stages
        stages.CUTOFF, stages.END = str(cutoff), str(end)
        from stages import teachers, parents, package_inputs
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=args.threads):
            print("2/5 Rebuilding TimesFM teacher inputs", flush=True)
            lineage["teachers"] = teachers(history, args.timesfm_model, args.timesfm_cache, args.threads)
            print("3/5 Rebuilding 021, 024 and 029", flush=True)
            base = parents(history, weather)
            print("4/5 Preparing TabPFN examples and validating the package", flush=True)
            release = work / "release"
            release.mkdir()
            package_inputs(history, base, release)
        lineage["history_sha256"] = digest(history_path)
        lineage["libraries"] = {p: version(p) for p in ("numpy", "pandas", "scikit-learn", "scipy")}
        write_json(release / "sources.json", lineage)
        spec = dict(template, model_file=str(args.tabpfn_model), generation="prepared-" + digest(release / "sources.json")[:16],
                    history_end=str(cutoff + timedelta(days=1)) + "T00:00:00+03:00",
                    forecast_from=str(cutoff + timedelta(days=1)) + "T00:00:00+03:00",
                    forecast_to=str(end + timedelta(days=1)) + "T00:00:00+03:00",
                    history_sha256=digest(history_path), source_manifest_sha256=digest(release / "sources.json"),
                    files={n: digest(release / n) for n in ("base.csv", "inputs.npz")},
                    provenance="Fixed-parameter offline rebuild; see sources.json; quality requires separate evaluation")
        write_json(release / "recipe.json", spec)
        shutil.copyfile(ML / "recipes/tabpfn-030/LICENSE.txt", release / "LICENSE.txt")
        # Validate using the exact runtime loader without invoking expensive TabPFN inference.
        sys.path.insert(0, str(ML / "runtime"))
        from model_030 import load_inputs
        load_inputs(release, spec)
        write_json(release / "build.json", {"completed_at": datetime.now(timezone.utc).isoformat(),
                   "status": "prepared", "forecast_inference_run": False, "quality_evaluated": False,
                   "history_rows": len(history), "forecast_rows": len(base)})
        print("5/5 Publishing a new immutable input package", flush=True)
        # Keep preparation outputs, source snapshots and fit evidence with the release.
        (release / "preparation").mkdir()
        for name in ("artifacts", "build_source", "experiments", "pipeline.py", "constants.py"):
            shutil.move(str(work / name), release / "preparation" / name)
        publish(release, destination)
        print(destination / "recipe.json", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-sources", type=Path, help="Copy editable external snapshots and fixed parameters, then exit")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--dataset", type=Path, help="Directory with train.csv, test.csv and labels/")
    inputs.add_argument("--events", type=Path, nargs="+", help="Original exports plus new monthly event CSV files; use --no-labels")
    inputs.add_argument("--history", type=Path, help="Already aggregated hourly_clean.csv, with observation masks")
    parser.add_argument("--history-end", help="Last observed date, inclusive, e.g. 2025-11-30")
    parser.add_argument("--no-labels", action="store_true", help="Explicitly skip labels reconciliation when no updated labels exist")
    parser.add_argument("--sources", type=Path, default=LAB, help="Lab-shaped source snapshots and fixed selections")
    parser.add_argument("--output", type=Path, help="New release directory; never overwritten")
    parser.add_argument("--timesfm-model", type=Path, help="Pinned model.safetensors; requires CUDA and timesfm 2.0.2")
    parser.add_argument("--timesfm-cache", type=Path, help="Optional verified quantiles directory; mismatches require model")
    parser.add_argument("--tabpfn-model", type=Path, default=ML / "models/tabpfn-v2-regressor.ckpt")
    parser.add_argument("--threads", type=int, choices=range(1, 9), default=2)
    args = parser.parse_args()
    if args.init_sources:
        init_sources(args.init_sources)
        return
    if not args.output or not args.history_end or not (args.dataset or args.events or args.history):
        parser.error("Build requires --output, --history-end and one of --dataset / --events / --history")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    original = Path.cwd()
    try:
        run(args)
    finally:
        os.chdir(original)


if __name__ == "__main__":
    main()
