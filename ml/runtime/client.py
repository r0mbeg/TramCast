"""Example: PYTHONPATH=ml/runtime/generated python ml/runtime/client.py --help."""
import argparse
import json
from pathlib import Path
import sys

import grpc
from tramcast.forecast.v1 import forecast_pb2 as pb
from tramcast.forecast.v1 import forecast_pb2_grpc as rpc


def validate_response(answer, request):
    start, end = request.forecast_from.seconds, request.forecast_to.seconds
    if (not answer.model_version.strip() or not answer.dataset_version.strip()
            or end <= start or (end - start) % 3600
            or len(answer.points) != (end - start) // 3600):
        raise ValueError("Invalid response versions or coverage")
    for i, point in enumerate(answer.points):
        if (not point.HasField("boardings") or not point.HasField("hour_start")
                or point.boardings < 0 or point.hour_start.nanos
                or point.hour_start.seconds != start + i * 3600):
            raise ValueError("Invalid hourly response point")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", default="127.0.0.1:50051")
    parser.add_argument("--route", type=int, required=True)
    parser.add_argument("--from", dest="start", required=True, help="RFC3339, inclusive")
    parser.add_argument("--to", dest="end", required=True, help="RFC3339, exclusive")
    parser.add_argument("--timeout", type=float, default=5)
    parser.add_argument("--output", type=Path, help="Write the full JSON response to this file")
    args = parser.parse_args()
    if not 0 < args.route <= 2**31 - 1:
        parser.error("route must be a positive int32")
    request = pb.PredictRequest(route_number=args.route)
    try:
        request.forecast_from.FromJsonString(args.start)
        request.forecast_to.FromJsonString(args.end)
    except ValueError as error:
        parser.error(str(error))
    if not 0 < args.timeout < float("inf"):
        parser.error("timeout must be finite and positive")
    try:
        with grpc.insecure_channel(args.address) as channel:
            answer = rpc.ForecastServiceStub(channel).Predict(request, timeout=args.timeout)
    except grpc.RpcError as error:
        print(f"{error.code().name}: {error.details()}", file=sys.stderr)
        return 1
    validate_response(answer, request)
    result = dict(model_version=answer.model_version, dataset_version=answer.dataset_version,
                  points=[dict(hour_start=p.hour_start.ToJsonString(), boardings=p.boardings)
                          for p in answer.points])
    content = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(content, encoding="utf-8")
        print(json.dumps(dict(output=str(args.output), hours=len(answer.points),
                              boardings=sum(p.boardings for p in answer.points))))
    else:
        print(content, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
