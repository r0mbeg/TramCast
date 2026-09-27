"""Run from any directory after final reports are ready; only stdlib."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def main():
    files = ["ml/service.py", "ml/constants.py", "ml/client.py", "ml/benchmark_service.py",
             "ml/requirements-service.txt", "ml/forecast_bundle.json", "ml/Dockerfile",
             "ml/INTEGRATION.md", "ml/submissions/002_chronos_daily_zero_shot.csv",
             "ml/tests/__init__.py", "ml/tests/test_service.py", "ml/tests/test_client.py",
             "proto/README.md", "proto/tramcast/forecast/v1/forecast.proto"]
    sources = {name: ROOT / name for name in files}
    sources.update({str(p.relative_to(ROOT)): p for p in (ROOT / "ml/generated").rglob("*.py")})
    for name in ["REPORT.md", "PROTOCOL.md", "PROTOCOL_v1.md", "PROTOCOL_v1_1.md", "DIRECTIONS.md",
                 "service_benchmark.json", "service_benchmark_source.py", "service_environment.json",
                 "build_package.py"]:
        sources[str((OUT / name).relative_to(ROOT))] = OUT / name
    for folder in ["data", "review"]:
        sources.update({str(p.relative_to(ROOT)): p for p in (OUT / folder).rglob("*")
                        if p.is_file() and p.suffix in (".md", ".json", ".csv", ".py")})
    for name in ["ROUTE_REPORT.md", "slurm_accounting.csv", "model_provenance.json",
                 "hardware_reproduction.csv", "m2_historical_reproduction.csv", "resource_accounting.json",
                 "model_config_provenance.txt", "model_weights_metadata.txt", "dependency_match.json",
                 "m2_dependencies.sha256", "m2_calendar.sha256", "executed_M1.py", "executed_M2.py",
                 "executed_evaluator.py", "evaluation/evaluator_run.json"]:
        sources[str((OUT / "route" / name).relative_to(ROOT))] = OUT / "route" / name
    sources.update({str(p.relative_to(ROOT)): p for p in (OUT / "route/evaluation").glob("*.csv")})
    sources["README.md"] = OUT / "PACKAGE_README.md"
    missing = [name for name, path in sources.items() if not path.is_file()]
    if missing:
        raise ValueError(f"Missing package evidence: {missing}")
    payload = {name: path.read_bytes() for name, path in sorted(sources.items())}
    manifest = {name: hashlib.sha256(content).hexdigest() for name, content in payload.items()}
    destination = OUT / "tramcast_ml_handoff.zip"
    if destination.exists():
        raise ValueError("Refusing to overwrite a package")
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in payload.items():
            archive.writestr(name, content)
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == set(manifest) | {"manifest.json"}
        assert all(hashlib.sha256(archive.read(name)).hexdigest() == digest
                   for name, digest in manifest.items())
    (OUT / "package.json").write_text(json.dumps(dict(file=destination.name, files=len(manifest),
        bytes=destination.stat().st_size, sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        all_member_hashes_verified=True), indent=2), encoding="utf-8")
    print(json.dumps(dict(file=str(destination), files=len(manifest), bytes=destination.stat().st_size)))


if __name__ == "__main__":
    main()
