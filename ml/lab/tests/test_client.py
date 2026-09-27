"""PYTHONPATH=generated python -m tests.test_client; exercise the example CLI over gRPC."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from client import validate_response
from service import ROOT, create_server
from tramcast.forecast.v1 import forecast_pb2 as pb


def check():
    request = pb.PredictRequest()
    request.forecast_from.seconds = 0
    request.forecast_to.seconds = 3600
    answer = pb.PredictResponse(model_version="m", dataset_version="d")
    answer.points.add(boardings=0).hour_start.seconds = 0
    validate_response(answer, request)
    for change in (lambda p: p.ClearField("boardings"), lambda p: setattr(p, "boardings", -1),
                   lambda p: setattr(p.hour_start, "nanos", 1),
                   lambda p: setattr(p.hour_start, "seconds", 3600)):
        bad = pb.PredictResponse(); bad.CopyFrom(answer)
        change(bad.points[0])
        try:
            validate_response(bad, request)
        except ValueError:
            pass
        else:
            raise AssertionError("Malformed response accepted")
    server, service = create_server(ROOT / "forecast_bundle.json")
    port = server.add_insecure_port("127.0.0.1:0")
    assert port
    server.start()
    base = [sys.executable, str(ROOT / "client.py"), "--address", f"127.0.0.1:{port}",
            "--from", "2025-11-01T00:00:00+03:00", "--to", "2025-11-02T00:00:00+03:00"]
    try:
        for route in (1, 5):
            result = subprocess.run(base + ["--route", str(route)], capture_output=True, text=True, timeout=10)
            assert result.returncode == 0, result.stderr
            answer = json.loads(result.stdout)
            assert answer["model_version"] == service.bundle.model_version
            assert answer["dataset_version"] == service.bundle.dataset_version
            assert len(answer["points"]) == 24
            for hour, point in enumerate(answer["points"]):
                assert point["boardings"] == service.bundle.points[route, service.bundle.start + hour * 3600]
            assert answer["points"][0]["hour_start"] == "2025-10-31T21:00:00Z"
        bad = subprocess.run(base + ["--route", "999"], capture_output=True, text=True, timeout=10)
        assert bad.returncode == 1 and "NOT_FOUND" in bad.stderr and not bad.stdout
        for options in (["--route", "0"], ["--route", "2147483648"],
                        ["--route", "1", "--timeout", "nan"], ["--route", "1", "--from", "invalid"]):
            result = subprocess.run(base + options, capture_output=True, text=True, timeout=10)
            assert result.returncode == 2 and "error:" in result.stderr
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "answer.json"
            result = subprocess.run(base + ["--route", "5", "--output", str(output)],
                                    capture_output=True, text=True, timeout=10)
            assert result.returncode == 0, result.stderr
            assert json.loads(result.stdout)["boardings"] == 0
            assert len(json.loads(output.read_text())["points"]) == 24
    finally:
        server.stop(0).wait()
    print("Client: exact values, timezone, zeros, JSON output and error exit codes passed.")


if __name__ == "__main__":
    check()
