import hashlib, json, sys, tempfile, time
from pathlib import Path
import grpc
from bundle import Bundle
from cache import ForecastCache
from recipe import Recipe, DEFAULT_RECIPE
from service import create_server
from tramcast.forecast.v1 import forecast_pb2 as pb, forecast_pb2_grpc as rpc
recipe = Recipe(DEFAULT_RECIPE)
assert recipe.spec["device"] == "cuda"
assert "torch" not in sys.modules
content = Path("/verification/forecast.csv").read_bytes()
expected = Bundle.from_csv(dict(recipe.metadata, prediction_sha256=hashlib.sha256(content).hexdigest()), content)
def forbidden(*args):
    raise AssertionError("Cache hit attempted a model calculation")
recipe.compute = forbidden
with tempfile.TemporaryDirectory() as folder:
    db_path = Path(folder)/"cache.sqlite3"
    # GPU-verified CSV is a fixture here, not a local GPU computation.
    ForecastCache(db_path).get(recipe.metadata, lambda _: content, lambda: None)
    cache = ForecastCache(db_path)
    server, _ = create_server(recipe=recipe, cache=cache)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    timings=[]
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            client=rpc.ForecastServiceStub(channel)
            for route in recipe.metadata["route_numbers"]:
                request=pb.PredictRequest(route_number=route)
                request.forecast_from.seconds=recipe.start
                request.forecast_to.seconds=recipe.end
                started=time.monotonic()
                response=client.Predict(request,timeout=5)
                timings.append(time.monotonic()-started)
                assert len(response.points)==1464
                assert response.model_version==recipe.metadata["model_version"]
                for point in response.points:
                    assert point.boardings==expected.points[route,point.hour_start.seconds]
    finally:
        server.stop(0).wait()
assert "torch" not in sys.modules
print(json.dumps(dict(points=14640,model_called=False,torch_imported=False,
    seconds_min=min(timings),seconds_max=max(timings),
    scope="CPU cache and gRPC using GPU-verified CSV fixture; CUDA recipe kept unchanged"),indent=2))
