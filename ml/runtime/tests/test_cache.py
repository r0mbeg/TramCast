"""Persistent cache, concurrent misses, failures, cancellation and refresh identity."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time

from bundle import DEFAULT_BUNDLE
from cache import ForecastCache, cache_key


def check():
    metadata = json.loads(DEFAULT_BUNDLE.read_text())
    content = (DEFAULT_BUNDLE.parent / metadata.pop("prediction_file")).read_bytes()
    metadata.pop("prediction_sha256")
    calls = []

    def compute(active):
        calls.append(1)
        time.sleep(.08)
        active()
        return content

    active = lambda: None
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)/"cache.sqlite3"
        cache = ForecastCache(path)
        # Separate connections as in different RPC workers/processes.
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: ForecastCache(path).get(metadata, compute, active), range(4)))
        assert len(calls) == 1
        assert all(result.points == results[0].points for result in results)
        ForecastCache(path).get(metadata, lambda _: (_ for _ in ()).throw(AssertionError("cache miss")), active)
        process_path = str(Path(folder)/"processes.sqlite3")
        ForecastCache(process_path)
        program = """
import json,sys,time
from bundle import DEFAULT_BUNDLE
from cache import ForecastCache
m=json.loads(DEFAULT_BUNDLE.read_text())
c=(DEFAULT_BUNDLE.parent/m.pop('prediction_file')).read_bytes(); m.pop('prediction_sha256')
def calculate(active):
    print('COMPUTED',flush=True); time.sleep(.1); return c
ForecastCache(sys.argv[1]).get(m,calculate,lambda:None)
"""
        children = [subprocess.Popen([sys.executable, "-c", program, process_path],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(3)]
        outputs = [child.communicate(timeout=15) for child in children]
        assert all(child.returncode == 0 for child in children), outputs
        assert sum(stdout.count("COMPUTED") for stdout, _ in outputs) == 1
        new = dict(metadata, model_version=metadata["model_version"]+"-manual-refresh")
        assert cache_key(new) != cache_key(metadata)
        cache.get(new, compute, active)
        assert len(calls) == 2
        cache.get(metadata, compute, active)
        assert len(calls) == 2  # Refresh preserved the previous generation.
        with sqlite3.connect(path) as db:
            db.execute("UPDATE forecasts SET csv=? WHERE key=?", (b"partial", cache_key(metadata)))
        cache.get(metadata, compute, active)
        assert len(calls) == 3
        broken = dict(metadata, dataset_version="bad-output")
        try:
            cache.get(broken, lambda _: b"invalid", active)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid model output cached")
        assert cache.get(broken, compute, active).points == results[0].points
        # Abort a waiter while another calculation holds the writer lock.
        started, release = threading.Event(), threading.Event()
        slow = dict(metadata, dataset_version="slow")
        def slow_compute(active):
            started.set()
            assert release.wait(3)
            return content
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(cache.get, slow, slow_compute, active)
            assert started.wait(3)
            class Cancelled(Exception):
                pass
            def cancelled():
                raise Cancelled()
            try:
                cache.get(slow, compute, cancelled)
            except Cancelled:
                pass
            else:
                raise AssertionError("Cancelled waiter was served")
            finally:
                release.set()
            pending.result(timeout=3)
        cancelled_meta = dict(metadata, dataset_version="cancelled-before-commit")
        checks = []
        def cancel_after_output():
            checks.append(1)
            if len(checks) == 3:
                raise Cancelled()
        try:
            cache.get(cancelled_meta, lambda _: content, cancel_after_output)
        except Cancelled:
            pass
        else:
            raise AssertionError("Cancelled output was committed")
        with sqlite3.connect(path) as db:
            assert db.execute("SELECT 1 FROM forecasts WHERE key=?", (cache_key(cancelled_meta),)).fetchone() is None
        assert cache.get(cancelled_meta, compute, active).points == results[0].points
    print("Cache: concurrent miss computed once, restart hit, refresh, corruption and cancellation passed.")


if __name__ == "__main__":
    check()
