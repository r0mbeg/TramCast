"""Run on the GPU server: check HTTP, database publication and service isolation."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import urllib.error
import urllib.request

root = Path.home() / "tramcast-app"
metadata = json.loads((Path.home() / "tramcast/run/metadata.json").read_text())
psql = [str(root / "tools/pgsql/bin/psql"), "-X", "-v", "ON_ERROR_STOP=1",
        "-h", str(root / "pgsocket"), "-p", "54329", "-d", "tramcast", "-At"]


def sql(query):
    return subprocess.check_output(psql + ["-c", query], text=True).strip()


def http(path):
    with urllib.request.urlopen("http://127.0.0.1:8080" + path, timeout=5) as response:
        assert response.status == 200
        return response.read()


http("/healthz")
http("/readyz")
index = http("/").decode()
assets = re.findall(r'(?:src|href)="(/assets/[^" ]+)"', index)
assert assets, "Missing compiled frontend assets"
for asset in assets:
    assert http(asset)
routes = json.loads(http("/api/routes"))["routes"]
assert len(routes) == 15
assert sorted(r["route_number"] for r in routes if r["forecast_enabled"]) == metadata["route_numbers"]
stops = json.loads(http("/api/stops"))["stops"]
assert len(stops) > 0
geometry = json.loads(http("/api/routes/geometry"))
assert geometry["type"] == "FeatureCollection" and geometry["features"]
for route in routes:
    result = json.loads(http(f'/api/routes/{route["id"]}/stops'))
    assert result["route_id"] == route["id"] and "patterns" in result
try:
    http("/api/does-not-exist")
    raise AssertionError("Unknown API route returned success")
except urllib.error.HTTPError as error:
    assert error.code == 404

version = json.loads(sql("SELECT row_to_json(v) FROM forecast_versions v WHERE is_active"))
assert version["model_version"] == metadata["model_version"]
assert version["dataset_version"] == metadata["dataset_version"]
jobs = json.loads(sql("SELECT json_agg(j ORDER BY route_id) FROM prediction_jobs j"))
assert len(jobs) == 10
assert all(j["status"] == "succeeded" and j["attempt_count"] == 1 for j in jobs)
actual = list(csv.DictReader(io.StringIO(sql("COPY (SELECT r.route_number AS route, p.date, p.hour, p.boardings AS prediction FROM validation_predictions p JOIN routes r ON r.id=p.route_id ORDER BY r.route_number,p.date,p.hour) TO STDOUT WITH (FORMAT CSV, HEADER, DELIMITER ';')")), delimiter=";"))
with sqlite3.connect(f"file:{Path.home()}/tramcast/ml/cache/forecasts.sqlite3?mode=ro", uri=True) as db:
    cached = db.execute("SELECT csv,sha256 FROM forecasts").fetchall()
assert len(cached) == 1
content, digest = cached[0]
assert hashlib.sha256(content).hexdigest() == digest
expected = list(csv.DictReader(io.StringIO(content.decode()), delimiter=";"))
assert len(actual) == 14640 and actual == expected, "Published grid differs from ML cache"

listeners = subprocess.check_output(["ss", "-ltnH"], text=True).splitlines()
for port in (8080, 54329):
    addresses = [line.split()[3] for line in listeners if line.split()[3].endswith(f":{port}")]
    assert addresses == [f"127.0.0.1:{port}"], addresses
assert not any(line.split()[3].endswith(":50051") for line in listeners)
for service in ("tramcast-app", "tramcast-postgres", "tramcast-ml"):
    subprocess.run(["systemctl", "--user", "is-active", "--quiet", service], check=True)
    subprocess.run(["systemctl", "--user", "is-enabled", "--quiet", service], check=True)
    pid = subprocess.check_output(["systemctl", "--user", "show", service, "-p", "MainPID", "--value"], text=True).strip()
    status = dict(line.split(":", 1) for line in Path(f"/proc/{pid}/status").read_text().splitlines() if ":" in line)
    assert status["NoNewPrivs"].strip() == "1" and int(status["CapEff"], 16) == 0
    assert len(os.sched_getaffinity(int(pid))) <= 2
assert (root / "app.env").stat().st_mode & 0o077 == 0
assert (root / "pgdata").stat().st_mode & 0o077 == 0
print(json.dumps(dict(commit="40e2adfefded24a22beb3047cd5a3085d9e637d0", deployment_patch="ml-unix-address.patch",
    forecast_version_id=version["id"], model_version=version["model_version"], dataset_version=version["dataset_version"],
    routes=len(routes), stops=len(stops), geometry_features=len(geometry["features"]),
    prediction_rows=len(actual), jobs_succeeded=len(jobs), job_ids=[j["id"] for j in jobs],
    prediction_sha256=digest, http_and_assets="passed", loopback_only="passed", service_checks="passed"), indent=2))
