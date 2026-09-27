"""Serve a verified, precomputed Chronos forecast through the shared gRPC contract."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import logging
from pathlib import Path
import signal
import threading
import time
from types import MappingProxyType
from typing import Mapping
from zoneinfo import ZoneInfo

import grpc
from constants import ROUTES
from google.protobuf.timestamp_pb2 import Timestamp
from tramcast.forecast.v1 import forecast_pb2 as pb
from tramcast.forecast.v1 import forecast_pb2_grpc as rpc

ROOT = Path(__file__).resolve().parent
MOSCOW = ZoneInfo("Europe/Moscow")
MAX_HOURS = 1464
INT64_MAX = 2**63 - 1
LOG = logging.getLogger("tramcast.ml")


def hour_seconds(stamp):
    if stamp.nanos != 0:
        raise ValueError("Timestamp nanos must be zero")
    local = stamp.ToDatetime(tzinfo=MOSCOW)  # Also checks the Protobuf range.
    if local.minute or local.second:
        raise ValueError("Timestamp must start a Moscow hour")
    return stamp.seconds


def json_hour(value):
    stamp = Timestamp()
    stamp.FromJsonString(value)
    return hour_seconds(stamp)


@dataclass(frozen=True)
class Bundle:
    model_version: str
    dataset_version: str
    start: int
    end: int
    points: Mapping[tuple[int, int], int]

    @classmethod
    def load(cls, path):
        path = Path(path)
        meta = json.loads(path.read_text(encoding="utf-8"))
        for key in ("model_version", "dataset_version"):
            if not isinstance(meta[key], str) or not meta[key].strip():
                raise ValueError(f"Missing {key}")
        if meta["timezone"] != "Europe/Moscow" or meta["route_numbers"] != list(ROUTES):
            raise ValueError("Unsupported timezone or routes")
        start, end = json_hour(meta["forecast_from"]), json_hour(meta["forecast_to"])
        if not 0 < end - start <= MAX_HOURS * 3600 or json_hour(meta["history_end"]) > start:
            raise ValueError("Invalid bundle horizon")
        source = path.parent / meta["prediction_file"]
        if source.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("Forecast file exceeds 16 MiB")
        content = source.read_bytes()
        if hashlib.sha256(content).hexdigest() != meta["prediction_sha256"]:
            raise ValueError("Forecast checksum mismatch")
        reader = csv.DictReader(io.StringIO(content.decode("utf-8")), delimiter=";")
        if reader.fieldnames != ["route", "date", "hour", "prediction"]:
            raise ValueError("Invalid forecast CSV schema")
        points = {}
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Invalid CSV row")
            route, hour, value = int(row["route"]), int(row["hour"]), int(row["prediction"])
            if route not in ROUTES or not 0 <= hour <= 23 or not 0 <= value <= INT64_MAX:
                raise ValueError("Invalid route, hour or int64 boardings")
            second = json_hour(f'{row["date"]}T{hour:02}:00:00+03:00')
            key = (route, second)
            if key in points or not start <= second < end:
                raise ValueError("Duplicate or out-of-range forecast point")
            if (route == 5 or 1 <= hour <= 4) and value != 0:
                raise ValueError("Structural zeros violated")
            points[key] = value
        expected = {(route, second) for route in ROUTES for second in range(start, end, 3600)}
        if points.keys() != expected:
            raise ValueError("Forecast does not cover the exact hourly grid")
        # Already rounded at export; serving must never round these integers again.
        return cls(meta["model_version"], meta["dataset_version"], start, end,
                   MappingProxyType(points))


class ForecastService(rpc.ForecastServiceServicer):
    def __init__(self, bundle_path):
        self.slot = threading.BoundedSemaphore(1)
        try:
            self.bundle = Bundle.load(bundle_path)
        except (OSError, ValueError, KeyError, TypeError, OverflowError, csv.Error):
            LOG.exception("Invalid forecast bundle")
            self.bundle = None

    @staticmethod
    def check_active(context):
        remaining = context.time_remaining()
        if remaining is not None and remaining <= 0:
            context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, "RPC deadline exceeded")
        if not context.is_active():
            context.abort(grpc.StatusCode.CANCELLED, "RPC cancelled")

    def Predict(self, request, context):
        started = time.monotonic()
        if request.route_number <= 0:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "route_number must be positive")
        if not request.HasField("forecast_from") or not request.HasField("forecast_to"):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Both timestamps are required")
        try:
            start, end = hour_seconds(request.forecast_from), hour_seconds(request.forecast_to)
        except (ValueError, OverflowError):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Invalid or unaligned timestamp")
        if not 0 < end - start <= MAX_HOURS * 3600:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Expected 1..1464 hours in [from, to)")
        if request.route_number not in ROUTES:
            context.abort(grpc.StatusCode.NOT_FOUND, "Unsupported route number")
        bundle = self.bundle
        if bundle is None:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Forecast bundle unavailable; see server logs")
        if start < bundle.start or end > bundle.end:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Period outside the loaded forecast bundle")
        self.check_active(context)
        # ponytail: one response builder; raise concurrency only after measuring memory/latency.
        if not self.slot.acquire(blocking=False):
            context.abort(grpc.StatusCode.UNAVAILABLE, "Forecast worker is busy")
        try:
            response = pb.PredictResponse(model_version=bundle.model_version,
                                          dataset_version=bundle.dataset_version)
            for second in range(start, end, 3600):
                self.check_active(context)
                point = response.points.add(boardings=bundle.points[request.route_number, second])
                point.hour_start.seconds = second
            LOG.info(json.dumps(dict(event="predict", route_number=request.route_number,
                hours=len(response.points), seconds=time.monotonic() - started,
                model_version=bundle.model_version, dataset_version=bundle.dataset_version)))
            return response
        finally:
            self.slot.release()


def create_server(bundle_path):
    # Additional threads reject busy calls promptly; they do not queue inference work.
    server = grpc.server(ThreadPoolExecutor(max_workers=4), maximum_concurrent_rpcs=8, options=[
        ("grpc.max_receive_message_length", 64 * 1024),
        ("grpc.max_send_message_length", 4 * 1024 * 1024),
    ])
    service = ForecastService(bundle_path)
    rpc.add_ForecastServiceServicer_to_server(service, server)
    return server, service


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="127.0.0.1:50051")
    parser.add_argument("--bundle", type=Path, default=ROOT / "forecast_bundle.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    server, service = create_server(args.bundle)
    if not server.add_insecure_port(args.bind):
        raise RuntimeError(f"Cannot bind {args.bind}")
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    server.start()
    LOG.info(json.dumps(dict(event="started", address=args.bind, bundle_ready=service.bundle is not None)))
    try:
        stop.wait()
    finally:
        server.stop(grace=5).wait()


if __name__ == "__main__":
    main()
