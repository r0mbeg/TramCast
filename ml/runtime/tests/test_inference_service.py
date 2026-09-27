"""gRPC/cache wiring with a counted model double; real TabPFN is checked separately."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import grpc
from bundle import DEFAULT_BUNDLE, json_hour
from cache import ForecastCache
from service import create_server
from tests.test_service import request, expect_status
from tramcast.forecast.v1 import forecast_pb2_grpc as rpc


def check():
    metadata = json.loads(DEFAULT_BUNDLE.read_text())
    content = (DEFAULT_BUNDLE.parent/metadata.pop("prediction_file")).read_bytes()
    metadata.pop("prediction_sha256")
    calls = []
    def compute(active):
        active()
        calls.append(1)
        return content
    recipe = SimpleNamespace(metadata=metadata, compute=compute,
                             start=json_hour(metadata["forecast_from"]), end=json_hour(metadata["forecast_to"]))
    with tempfile.TemporaryDirectory() as folder:
        for restart in range(2):
            server, _ = create_server(recipe=recipe, cache=ForecastCache(Path(folder)/"cache.sqlite3"))
            port = server.add_insecure_port("127.0.0.1:0")
            server.start()
            try:
                with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
                    stub = rpc.ForecastServiceStub(channel)
                    expect_status(lambda: stub.Predict(request(999), timeout=5), grpc.StatusCode.NOT_FOUND)
                    expect_status(lambda: stub.Predict(request(5, "2027-10-28T00:00:00Z", "2027-10-29T00:00:00Z"), timeout=5),
                                  grpc.StatusCode.FAILED_PRECONDITION)
                    zeros = stub.Predict(request(5), timeout=5)
                    assert all(p.boardings == 0 and p.HasField("boardings") for p in zeros.points)
                    assert len(calls) == restart
                    for route in metadata["route_numbers"]:
                        response = stub.Predict(request(route), timeout=5)
                        assert len(response.points) == 1464
                        assert response.model_version == metadata["model_version"]
                    day = stub.Predict(request(1, "2025-11-01T00:00:00+03:00", "2025-11-02T00:00:00+03:00"), timeout=5)
                    assert len(day.points) == 24 and len(calls) == 1
            finally:
                server.stop(0).wait()
    print("Inference gRPC: validation before compute, all routes/slices reuse one calculation, restart hit passed.")


if __name__ == "__main__":
    check()
