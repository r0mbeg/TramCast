"""Pinned prepared inputs, cancellable model execution and explicit refresh versions."""
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid

from bundle import ROOT, MAX_HOURS, json_hour
from cache import cache_key
from constants import ROUTES

DEFAULT_RECIPE = ROOT.parent / "recipes" / "tabpfn-030" / "recipe.json"
MODEL_SHA = "2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736"


class Recipe:
    worker = "model_030.py"
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.spec = json.loads(self.path.read_text())
        spec = self.spec
        if spec["recipe"] != "tabpfn-profile-030" or spec["model_sha256"] != MODEL_SHA:
            raise ValueError("Unsupported recipe or unverified checkpoint")
        if (spec["timezone"] != "Europe/Moscow" or spec["route_numbers"] != ROUTES
                or spec["device"] not in ("cpu", "cuda")
                or type(spec["threads"]) is not int or not 1 <= spec["threads"] <= 8
                or type(spec["timeout_seconds"]) is not int or not 1 <= spec["timeout_seconds"] <= 3600
                or type(spec["seed"]) is not int or not 0 <= spec["seed"] < 2**32
                or spec["n_estimators"] != 4 or spec["shape_weight"] != .5
                or not isinstance(spec["generation"], str) or not spec["generation"].strip()):
            raise ValueError("Invalid fixed recipe configuration")
        self.start, self.end = json_hour(spec["forecast_from"]), json_hour(spec["forecast_to"])
        if (not 0 < self.end-self.start <= MAX_HOURS*3600
                or (self.start + 3*3600) % 86400 or (self.end + 3*3600) % 86400
                or json_hour(spec["history_end"]) != self.start):
            raise ValueError("Prepared recipe requires full days immediately after history")
        if set(spec["files"]) != {"inputs.npz", "base.csv"}:
            raise ValueError("Expected prepared features and parent forecast")
        directory = self.path.parent / spec.get("inputs_dir", ".")
        self.inputs = {}
        for name, digest in spec["files"].items():
            path = directory / name
            if path.stat().st_size > 32*1024*1024:
                raise ValueError("Prepared input exceeds 32 MiB")
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f"Input checksum mismatch: {name}")
            self.inputs[name] = data
        self.model_file = (self.path.parent / spec["model_file"]).resolve()
        self.check_weights()
        model = dict(recipe=spec["recipe"], weights=MODEL_SHA, seed=spec["seed"],
                     n_estimators=spec["n_estimators"], shape_weight=spec["shape_weight"],
                     generation=spec["generation"], device=spec["device"], threads=spec["threads"],
                     platform=platform.machine(),
                     libraries={p: version(p) for p in ("torch", "tabpfn", "numpy", "pandas", "scikit-learn", "scipy")},
                     code={n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest()
                           for n in ("model_030.py", "constants.py")})
        data = {k: spec[k] for k in ("files", "history_sha256", "history_end", "forecast_from", "forecast_to")}
        if "source_manifest_sha256" in spec:
            manifest = self.path.parent / spec.get("inputs_dir", ".") / "sources.json"
            if hashlib.sha256(manifest.read_bytes()).hexdigest() != spec["source_manifest_sha256"]:
                raise ValueError("Source manifest checksum mismatch")
            data["source_manifest_sha256"] = spec["source_manifest_sha256"]
        self.metadata = {k: spec[k] for k in ("timezone", "route_numbers", "history_end", "forecast_from", "forecast_to")}
        self.metadata.update(model_version="tabpfn030-"+cache_key(model),
                             dataset_version="prepared-"+cache_key(data))

    def check_weights(self):
        with self.model_file.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != MODEL_SHA:
                raise ValueError("Checkpoint checksum mismatch")

    def compute(self, check_active):
        check_active()
        self.check_weights()
        with tempfile.TemporaryDirectory(prefix="tramcast-inference-") as temporary:
            folder = Path(temporary)
            for name, data in self.inputs.items():
                (folder/name).write_bytes(data)
            spec = dict(self.spec, model_file=str(self.model_file))
            (folder/"spec.json").write_text(json.dumps(spec))
            env = dict(os.environ, HF_HUB_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
                       OMP_NUM_THREADS=str(spec["threads"]), MKL_NUM_THREADS=str(spec["threads"]),
                       OPENBLAS_NUM_THREADS=str(spec["threads"]))
            with (folder/"worker.log").open("w+") as log:
                process = subprocess.Popen([sys.executable, str(ROOT/self.worker),
                    "--spec", str(folder/"spec.json"), "--output", str(folder/"forecast.csv")],
                    env=env, stdout=log, stderr=log)
                try:
                    deadline = time.monotonic() + spec["timeout_seconds"]
                    while process.poll() is None:
                        check_active()
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Model calculation exceeded its time limit")
                        time.sleep(.05)
                    if process.returncode:
                        log.seek(0)
                        raise RuntimeError("Model worker failed: " + log.read()[-4000:])
                    check_active()
                    self.check_weights()
                    return (folder/"forecast.csv").read_bytes()
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()

    def refresh(self, destination):
        """A manual refresh creates new model identity, never overwrites old results."""
        spec = dict(self.spec, generation=uuid.uuid4().hex, model_file=str(self.model_file),
                    inputs_dir=str((self.path.parent/self.spec.get("inputs_dir", ".")).resolve()))
        with Path(destination).open("x", encoding="utf-8") as output:
            json.dump(spec, output, ensure_ascii=False, indent=2)
            output.write("\n")


def download_model(config):
    """Explicit installation step; no network access during Predict."""
    config = Path(config).resolve()
    spec = json.loads(config.read_text())
    if spec["model_sha256"] != MODEL_SHA:
        raise ValueError("Only the audited TabPFN checkpoint can be installed")
    destination = (config.parent / spec["model_file"]).resolve()
    if destination.exists():
        with destination.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != MODEL_SHA:
                raise ValueError("Existing model hash mismatch; refusing to overwrite")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = ("https://huggingface.co/Prior-Labs/TabPFN-v2-reg/resolve/"
           "4972a65a1b30806315c6f92499959ffbfc69a673/tabpfn-v2-regressor.ckpt")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as target:
            temporary = Path(target.name)
            digest = hashlib.sha256()
            with urllib.request.urlopen(url, timeout=120) as source:
                while block := source.read(1024*1024):
                    target.write(block)
                    digest.update(block)
        if digest.hexdigest() != MODEL_SHA:
            raise ValueError("Downloaded checkpoint hash mismatch")
        temporary.chmod(0o644)
        temporary.replace(destination)
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Install the pinned model before starting the service")
    parser.add_argument("--config", type=Path, default=DEFAULT_RECIPE)
    args = parser.parse_args()
    print(download_model(args.config))
