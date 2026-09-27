"""Native CPU student on immutable, offline-prepared features; no lab imports."""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

from bundle import ROOT, MAX_HOURS, json_hour
from cache import cache_key
from constants import ROUTES
from recipe import Recipe

DEFAULT_CPU_RECIPE = ROOT.parent / "recipes/catboost-cpu/recipe.json"


class CPURecipe(Recipe):
    # Reuse cancellation, timeout, immutable input snapshots and manual refresh.
    worker = "model_cpu.py"

    def __init__(self, path):
        self.path = Path(path).resolve()
        self.spec = spec = json.loads(self.path.read_text())
        if (spec["recipe"] != "catboost-cpu" or spec["device"] != "cpu"
                or spec["timezone"] != "Europe/Moscow" or spec["route_numbers"] != ROUTES
                or type(spec["threads"]) is not int or not 1 <= spec["threads"] <= 8
                or type(spec["timeout_seconds"]) is not int or not 1 <= spec["timeout_seconds"] <= 3600
                or not isinstance(spec["generation"], str) or not spec["generation"].strip()
                or set(spec["files"]) != {"features.csv", "sources.json"}):
            raise ValueError("Invalid CPU recipe")
        self.start, self.end = json_hour(spec["forecast_from"]), json_hour(spec["forecast_to"])
        if (self.end-self.start != MAX_HOURS*3600 or (self.start+10800) % 86400
                or json_hour(spec["history_end"]) != self.start):
            raise ValueError("CPU package requires 61 complete days after history")
        directory = self.path.parent / spec.get("inputs_dir", ".")
        self.inputs = {}
        for name, digest in spec["files"].items():
            source = directory/name
            if source.stat().st_size > 32*1024*1024:
                raise ValueError("CPU input exceeds 32 MiB")
            content = source.read_bytes()
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError(f"Input checksum mismatch: {name}")
            self.inputs[name] = content
        self.model_file = (self.path.parent/spec["model_file"]).resolve()
        self.check_weights()
        source = json.loads(self.inputs["sources.json"])
        if (spec["model_sha256"] != source["model_sha256"]
                or spec["features"] != source["features"]
                or source["kind"] != "catboost"):
            raise ValueError("CPU model provenance mismatch")
        model = dict(recipe=spec["recipe"], weights=spec["model_sha256"],
                     features=spec["features"], generation=spec["generation"],
                     threads=spec["threads"], device="cpu", platform=platform.machine(),
                     libraries={p: version(p) for p in ("catboost", "numpy", "pandas")},
                     code={n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest()
                           for n in (self.worker, "constants.py")})
        data = {k: spec[k] for k in ("files", "history_sha256", "history_end", "forecast_from", "forecast_to")}
        self.metadata = {k: spec[k] for k in ("timezone", "route_numbers", "history_end", "forecast_from", "forecast_to")}
        self.metadata.update(model_version="catboost-cpu-"+cache_key(model),
                             dataset_version="prepared-cpu-"+cache_key(data))

    def check_weights(self):
        with self.model_file.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != self.spec["model_sha256"]:
                raise ValueError("CPU checkpoint checksum mismatch")
