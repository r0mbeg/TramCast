"""Example: PYTHONPATH=ml/runtime/generated python ml/runtime/client.py --help."""
import argparse
import json
from pathlib import Path
import sys

import grpc
from google.protobuf.json_format import MessageToDict
from bundle import MOSCOW
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


def validate_stop_response(answer, request):
    start, end = request.forecast_from.seconds, request.forecast_to.seconds
    versions = ('model_version', 'dataset_version', 'route_model_version', 'route_dataset_version',
                'allocation_model_version', 'allocation_dataset_version', 'network_version', 'allocation_package_version')
    if (any(not getattr(answer, key).strip() for key in versions) or
        answer.estimate_status != 'scenario_estimate_unvalidated' or not answer.geography_status.strip() or
        end <= start or (end-start) % 3600 or len(answer.hours) != (end-start)//3600 or
        not 1 <= len(answer.occurrences) <= 128):
        raise ValueError('Invalid stop response identity/status/coverage')
    keys, ids, source_routes = [], set(), set()
    for item in answer.occurrences:
        if (any(not getattr(item, k).strip() for k in ['occurrence_id', 'source_route_id', 'pattern_key', 'source_stop_id', 'stop_name']) or
            item.occurrence_id in ids or item.direction_id not in (0, 1) or item.stop_sequence <= 0 or
            not item.HasField('boarding_allowed') or
            item.boarding_role not in {'assumed_unverified', 'stop', 'stop_entry_only', 'stop_exit_only'} or
            item.boarding_allowed != (item.boarding_role != 'stop_exit_only')):
            raise ValueError('Invalid stop response occurrence')
        keys.append((item.direction_id, item.pattern_key, item.stop_sequence, item.source_stop_id))
        ids.add(item.occurrence_id); source_routes.add(item.source_route_id)
    if keys != sorted(set(keys)) or len(source_routes) != 1:
        raise ValueError('Unordered/duplicate/mixed occurrence catalog')
    for direction in (0, 1):
        positions = [o for o in answer.occurrences if o.direction_id == direction]
        if (not positions or len({o.pattern_key for o in positions}) != 1 or
            [o.stop_sequence for o in positions] != list(range(1, len(positions)+1))):
            raise ValueError('Incomplete stop direction')
    for index, hour in enumerate(answer.hours):
        if (not hour.HasField('hour_start') or hour.hour_start.nanos or hour.hour_start.seconds != start+3600*index or
            not hour.HasField('route_boardings') or hour.route_boardings < 0 or
            len(hour.estimated_boardings) != len(keys) or any(v < 0 for v in hour.estimated_boardings) or
            sum(hour.estimated_boardings) != hour.route_boardings):
            raise ValueError('Invalid stop hourly grid/vector/balance')
        local = hour.hour_start.ToDatetime(tzinfo=MOSCOW)
        expected_status = (pb.STOP_HOUR_STATUS_ROUTE5_FALLBACK if request.route_number == 5 else
            pb.STOP_HOUR_STATUS_NONWORKING_ZERO if 1 <= local.hour <= 4 else pb.STOP_HOUR_STATUS_SCENARIO)
        if hour.status != expected_status or (expected_status != pb.STOP_HOUR_STATUS_SCENARIO and hour.route_boardings != 0):
            raise ValueError('Invalid zero semantics')
        if any(v != 0 and not o.boarding_allowed for o, v in zip(answer.occurrences, hour.estimated_boardings)):
            raise ValueError('Exit-only boarding estimate')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", default="127.0.0.1:50051")
    parser.add_argument("--route", type=int, required=True)
    parser.add_argument("--from", dest="start", required=True, help="RFC3339, inclusive")
    parser.add_argument("--to", dest="end", required=True, help="RFC3339, exclusive")
    parser.add_argument("--timeout", type=float, default=5)
    parser.add_argument("--output", type=Path, help="Write the full JSON response to this file")
    parser.add_argument("--stops", action="store_true", help="Call PredictStops and validate occurrence vectors")
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
            stub = rpc.ForecastServiceStub(channel)
            answer = (stub.PredictStops if args.stops else stub.Predict)(request, timeout=args.timeout)
    except grpc.RpcError as error:
        print(f"{error.code().name}: {error.details()}", file=sys.stderr)
        return 1
    if args.stops:
        validate_stop_response(answer, request)
        result = MessageToDict(answer, preserving_proto_field_name=True, always_print_fields_with_no_presence=True)
        hours, total = len(answer.hours), sum(h.route_boardings for h in answer.hours)
    else:
        validate_response(answer, request)
        result = dict(model_version=answer.model_version, dataset_version=answer.dataset_version,
                  points=[dict(hour_start=p.hour_start.ToJsonString(), boardings=p.boardings)
                          for p in answer.points])
        hours, total = len(answer.points), sum(p.boardings for p in answer.points)
    content = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(content, encoding="utf-8")
        print(json.dumps(dict(output=str(args.output), hours=hours, boardings=total)))
    else:
        print(content, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
