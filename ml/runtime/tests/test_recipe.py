"""Version changes and cancellation of the actual subprocess, without ML dependencies."""
import json
import hashlib
import io
import os
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

from recipe import Recipe, DEFAULT_RECIPE


def check():
    # Only model installation/version discovery are replaced for this unit check.
    with patch.object(Recipe, "check_weights"), patch("recipe.version", return_value="test"):
        original = Recipe(DEFAULT_RECIPE)
        with tempfile.TemporaryDirectory() as folder:
            new_path = Path(folder)/"refresh.json"
            original.refresh(new_path)
            refreshed = Recipe(new_path)
            assert original.metadata["model_version"] != refreshed.metadata["model_version"]
            assert original.metadata["dataset_version"] == refreshed.metadata["dataset_version"]
            assert Recipe(new_path).metadata == refreshed.metadata
            manifest = Path(folder) / "sources.json"
            manifest.write_text('{"source":"first"}')
            spec = dict(refreshed.spec, inputs_dir=folder)
            for name, content in original.inputs.items():
                (Path(folder) / name).write_bytes(content)
            spec["source_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
            manifest_config = Path(folder) / "manifest.json"
            manifest_config.write_text(json.dumps(spec))
            with_manifest = Recipe(manifest_config)
            assert with_manifest.metadata["dataset_version"] != refreshed.metadata["dataset_version"]
            manifest.write_text('{"source":"changed"}')
            try:
                Recipe(manifest_config)
            except ValueError:
                pass
            else:
                raise AssertionError("Changed source provenance accepted")
            from service import main
            output = io.StringIO()
            with patch.dict(os.environ, {"ML_CONFIG": str(new_path)}), \
                    patch("sys.argv", ["service.py", "--describe"]), \
                    patch("service.logging.basicConfig"), redirect_stdout(output):
                main()
            assert json.loads(output.getvalue()) == refreshed.metadata
            try:
                original.refresh(new_path)
            except FileExistsError:
                pass
            else:
                raise AssertionError("Refresh overwrote a version")
            bad = json.loads(new_path.read_text())
            bad["files"]["inputs.npz"] = "bad"
            new_path.write_text(json.dumps(bad))
            try:
                Recipe(new_path)
            except ValueError:
                pass
            else:
                raise AssertionError("Corrupt prepared inputs accepted")
        real_popen = subprocess.Popen
        children = []
        def start_slow(command, **kwargs):
            child = real_popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
            children.append(child)
            return child
        class Cancelled(Exception):
            pass
        def active():
            if children:
                raise Cancelled()
        with patch("recipe.subprocess.Popen", side_effect=start_slow):
            try:
                original.compute(active)
            except Cancelled:
                pass
            else:
                raise AssertionError("Cancelled worker completed")
        assert len(children) == 1 and children[0].poll() is not None
    print("Recipe: refresh IDs, snapshot integrity and child termination passed.")


if __name__ == "__main__":
    check()
