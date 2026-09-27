"""Real CPU model: golden submission, gRPC, cache, stops, integrity and refresh."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import time

import grpc
from bundle import Bundle
from cache import ForecastCache
from client import validate_response, validate_stop_response
from recipe_cpu import CPURecipe, DEFAULT_CPU_RECIPE
from service import create_server
from tests.test_service import request, expect_status
from tramcast.forecast.v1 import forecast_pb2_grpc as rpc


def check():
    assert all(importlib.util.find_spec(name) is None for name in ('torch', 'tabpfn', 'timesfm'))
    recipe = CPURecipe(DEFAULT_CPU_RECIPE)
    started = time.perf_counter()
    content = recipe.compute(lambda: None)
    seconds = time.perf_counter()-started
    digest = hashlib.sha256(content).hexdigest()
    assert digest == '538872b074d8ced7f0997e2f1427db4f3dab39db3f3dba0e8be9f80a68a15695', digest
    expected = Bundle.from_csv(dict(recipe.metadata, prediction_sha256=digest), content)
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        cache = ForecastCache(folder/'cache.sqlite3')
        calls = []
        # Bind the class implementation: count actual subprocess calculations.
        def compute(active):
            calls.append(1)
            return CPURecipe.compute(recipe, active)
        recipe.compute = compute
        for _ in range(2):
            server, _ = create_server(recipe=recipe, cache=cache)
            port = server.add_insecure_port('127.0.0.1:0')
            server.start()
            try:
                with grpc.insecure_channel(f'127.0.0.1:{port}') as channel:
                    stub = rpc.ForecastServiceStub(channel)
                    for route in recipe.metadata['route_numbers']:
                        req = request(route)
                        response = stub.Predict(req, timeout=60)
                        validate_response(response, req)
                        assert (response.model_version, response.dataset_version) == (expected.model_version, expected.dataset_version)
                        assert all(p.boardings == expected.points[route, p.hour_start.seconds] for p in response.points)
                    req = request(12)
                    stops = stub.PredictStops(req, timeout=10)
                    validate_stop_response(stops, req)
                    assert stops.route_model_version == expected.model_version
                    expect_status(lambda: stub.Predict(request(1, '2025-10-01T00:00:00+03:00', '2025-10-02T00:00:00+03:00'), timeout=5), grpc.StatusCode.FAILED_PRECONDITION)
            finally:
                server.stop(0).wait()
        assert len(calls) == 1
        recipe.refresh(folder/'refresh.json')
        refreshed = CPURecipe(folder/'refresh.json')
        assert refreshed.metadata['model_version'] != recipe.metadata['model_version']
        assert refreshed.metadata['dataset_version'] == recipe.metadata['dataset_version']
        spec = json.loads((folder/'refresh.json').read_text())
        spec['model_sha256'] = '0'*64
        (folder/'bad.json').write_text(json.dumps(spec))
        try:
            CPURecipe(folder/'bad.json')
        except ValueError:
            pass
        else:
            raise AssertionError('Corrupt weights accepted')
    print(json.dumps(dict(check='CPU: golden CSV, full gRPC grid, stops, restart cache, versions, integrity',
                          rows=len(expected.points), cold_worker_seconds=seconds, sha256=digest,
                          metadata=recipe.metadata), indent=2))


if __name__ == '__main__':
    check()
