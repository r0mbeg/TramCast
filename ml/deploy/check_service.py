"""Check the deployed full horizon, pinned IDs, fallback, and gRPC errors."""
import argparse
import json
from pathlib import Path
import sys
import time

runtime = Path(__file__).resolve().parents[1] / "runtime"
sys.path[:0] = [str(runtime), str(runtime / "generated")]
import grpc
from client import validate_response
from tramcast.forecast.v1 import forecast_pb2 as pb, forecast_pb2_grpc as rpc

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--address", required=True)
parser.add_argument("--metadata", type=Path, required=True)
parser.add_argument("--timeout", type=float, default=900)
args = parser.parse_args()
metadata = json.loads(args.metadata.read_text())
report = {"model_version": metadata["model_version"], "dataset_version": metadata["dataset_version"], "routes": []}
with grpc.insecure_channel(args.address) as channel:
    grpc.channel_ready_future(channel).result(timeout=15)
    client = rpc.ForecastServiceStub(channel)
    first = None
    for route in metadata["route_numbers"]:
        request = pb.PredictRequest(route_number=route)
        request.forecast_from.FromJsonString(metadata["forecast_from"])
        request.forecast_to.FromJsonString(metadata["forecast_to"])
        started = time.monotonic()
        answer = client.Predict(request, timeout=args.timeout)
        elapsed = time.monotonic() - started
        validate_response(answer, request)
        assert answer.model_version == metadata["model_version"]
        assert answer.dataset_version == metadata["dataset_version"]
        for point in answer.points:
            hour = (point.hour_start.seconds // 3600 + 3) % 24
            if route == 5 or 1 <= hour <= 4:
                assert point.boardings == 0
        report["routes"].append(dict(route=route, hours=len(answer.points), boardings=sum(p.boardings for p in answer.points), seconds=elapsed))
        if first is None:
            first = request, answer
    assert client.Predict(first[0], timeout=5) == first[1], "Repeated result changed"
    cases = [(pb.PredictRequest(), grpc.StatusCode.INVALID_ARGUMENT)]
    unknown = pb.PredictRequest()
    unknown.CopyFrom(first[0])
    unknown.route_number = 999
    cases.append((unknown, grpc.StatusCode.NOT_FOUND))
    outside = pb.PredictRequest()
    outside.CopyFrom(first[0])
    outside.forecast_from.seconds -= 3600
    outside.forecast_to.seconds -= 3600
    cases.append((outside, grpc.StatusCode.FAILED_PRECONDITION))
    for request, expected in cases:
        try:
            client.Predict(request, timeout=5)
            raise AssertionError(f"Accepted invalid request: {expected}")
        except grpc.RpcError as error:
            assert error.code() == expected, error
report["repeat_and_error_checks"] = "passed"
print(json.dumps(report, indent=2))
