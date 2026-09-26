"""Run from ml/: python -m tests [--boost]."""
import argparse
import importlib
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--boost", action="store_true", help="Include optional boosting dependencies")
args = parser.parse_args()
for path in sorted(Path(__file__).parent.glob("test_*.py")):
    if path.stem == "test_boost_experiment" and not args.boost:
        continue
    print(path.stem, flush=True)
    importlib.import_module(f"tests.{path.stem}").check()
