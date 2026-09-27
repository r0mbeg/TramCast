"""Serve a pinned forecast recipe through gRPC, computing only on cache misses."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import signal
import sqlite3
import threading
import time

import grpc
from constants import ROUTES
from tramcast.forecast.v1 import forecast_pb2 as pb
from tramcast.forecast.v1 import forecast_pb2_grpc as rpc

from bundle import Bundle, DEFAULT_BUNDLE, INT64_MAX, MOSCOW, ROOT, MAX_HOURS, hour_seconds
from datetime import datetime
from stops import StopBundle, DEFAULT_STOPS, allocate

LOG = logging.getLogger("tramcast.ml")

class ForecastService(rpc.ForecastServiceServicer):
    def __init__(self, bundle_path=None, *, recipe=None, cache=None, stop_bundle_path=DEFAULT_STOPS):
        self.slot = threading.BoundedSemaphore(1)
        self.recipe, self.cache = recipe, cache
        self.bundle = None
        self.stop_bundle = None
        try:
            self.stop_bundle = StopBundle.load(stop_bundle_path)
        except (OSError, ValueError, KeyError, TypeError, OverflowError, EOFError, csv.Error):
            LOG.exception("Invalid stop allocation bundle; route prediction remains available")
        if recipe is not None:
            return
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

    def forecast(self, request, context, *, with_stops=False):
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
        if bundle is None and self.recipe is None:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Forecast bundle unavailable; see server logs")
        period = self.recipe if self.recipe is not None else bundle
        if start < period.start or end > period.end:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Period outside the loaded forecast bundle")
        if with_stops and (self.stop_bundle is None or start < self.stop_bundle.start or end > self.stop_bundle.end):
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Stop allocation unavailable for requested period")
        self.check_active(context)
        if self.recipe is not None and request.route_number == 5:
            metadata = self.recipe.metadata
            bundle = Bundle(metadata["model_version"], metadata["dataset_version"], start, end,
                            {(5, second): 0 for second in range(start, end, 3600)})
        elif self.recipe is not None:
            try:
                bundle = self.cache.get(self.recipe.metadata, self.recipe.compute,
                                        lambda: self.check_active(context))
            except TimeoutError:
                LOG.exception("Model time limit exceeded")
                context.abort(grpc.StatusCode.UNAVAILABLE, "Model time limit exceeded")
            except (OSError, sqlite3.Error):
                LOG.exception("Model/cache storage unavailable")
                context.abort(grpc.StatusCode.UNAVAILABLE, "Model/cache storage unavailable")
            except (ValueError, RuntimeError):
                LOG.exception("Forecast calculation failed")
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Calculation failed; see server logs")
        return bundle, start, end

    def Predict(self, request, context):
        started = time.monotonic()
        bundle, start, end = self.forecast(request, context)
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

    def PredictStops(self, request, context):
        started = time.monotonic()
        bundle, start, end = self.forecast(request, context, with_stops=True)
        stops = self.stop_bundle
        if not self.slot.acquire(blocking=False):
            context.abort(grpc.StatusCode.UNAVAILABLE, "Forecast worker is busy")
        try:
            response = pb.PredictStopsResponse(**stops.versions(bundle.model_version, bundle.dataset_version))
            occurrences = stops.occurrences[request.route_number]
            for item in occurrences:
                response.occurrences.add(**asdict(item))
            for second in range(start, end, 3600):
                self.check_active(context)
                local = datetime.fromtimestamp(second, MOSCOW)
                total = bundle.points[request.route_number, second]
                if request.route_number == 5:
                    values, status = [0]*len(occurrences), pb.STOP_HOUR_STATUS_ROUTE5_FALLBACK
                elif 1 <= local.hour <= 4:
                    values, status = [0]*len(occurrences), pb.STOP_HOUR_STATUS_NONWORKING_ZERO
                else:
                    weights = stops.weights[request.route_number, int(local.weekday() >= 5), local.hour]
                    values = allocate(total, weights)
                    status = pb.STOP_HOUR_STATUS_SCENARIO
                if sum(values) != total:
                    context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Stop allocation lost route total")
                hour = response.hours.add(route_boardings=total, estimated_boardings=values, status=status)
                hour.hour_start.seconds = second
            self.check_active(context)
            if response.ByteSize() > 4*1024*1024:
                context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "Stop response exceeds 4 MiB")
            LOG.info(json.dumps(dict(event="predict_stops", route_number=request.route_number,
                hours=len(response.hours), occurrences=len(occurrences), seconds=time.monotonic()-started,
                model_version=response.model_version, dataset_version=response.dataset_version)))
            return response
        finally:
            self.slot.release()


def create_server(bundle_path=None, *, recipe=None, cache=None, stop_bundle_path=DEFAULT_STOPS):
    # Four bounded RPC workers; cache waiters respect their individual deadlines.
    server = grpc.server(ThreadPoolExecutor(max_workers=4), maximum_concurrent_rpcs=8, options=[
        ("grpc.max_receive_message_length", 64 * 1024),
        ("grpc.max_send_message_length", 4 * 1024 * 1024),
    ])
    service = ForecastService(bundle_path, recipe=recipe, cache=cache, stop_bundle_path=stop_bundle_path)
    rpc.add_ForecastServiceServicer_to_server(service, server)
    return server, service


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="127.0.0.1:50051")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--bundle", type=Path, help="Explicit replay of a precomputed package")
    mode.add_argument("--config", type=Path, default=os.environ.get("ML_CONFIG"),
                      help="Prepared recipe; defaults to ML_CONFIG or tabpfn-030")
    parser.add_argument("--cache", type=Path, default=ROOT.parent / "cache" / "forecasts.sqlite3")
    parser.add_argument("--stop-bundle", type=Path, default=os.environ.get("ML_STOP_BUNDLE", DEFAULT_STOPS),
                        help="Pinned stop profiles; independent of route model/cache")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--warm-cache", action="store_true", help="Compute/cache the full recipe now, then exit")
    actions.add_argument("--describe", action="store_true", help="Print model/data IDs for the Go forecast version")
    actions.add_argument("--refresh", type=Path, help="Create a NEW recipe config for manual recalculation, then exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    recipe = cache = None
    if args.bundle is None:
        from recipe import Recipe, DEFAULT_RECIPE
        from cache import ForecastCache
        recipe = Recipe(args.config or DEFAULT_RECIPE)
        if args.refresh:
            recipe.refresh(args.refresh)
            print(json.dumps(dict(config=str(args.refresh.resolve()), next="Start service with --config and this path")))
            return
        if args.describe:
            print(json.dumps(describe(recipe.metadata, args.stop_bundle), indent=2))
            return
        cache = ForecastCache(args.cache)
        if args.warm_cache:
            result = cache.get(recipe.metadata, recipe.compute, lambda: None)
            print(json.dumps(dict(recipe.metadata, points=len(result.points))))
            return
    elif args.describe:
        bundle = Bundle.load(args.bundle)
        print(json.dumps(describe(json.loads(args.bundle.read_text(encoding="utf-8")), args.stop_bundle), indent=2))
        return
    elif args.refresh or args.warm_cache:
        parser.error("--refresh/--warm-cache require a recipe, not --bundle replay")
    server, service = create_server(args.bundle, recipe=recipe, cache=cache, stop_bundle_path=args.stop_bundle)
    if not server.add_insecure_port(args.bind):
        raise RuntimeError(f"Cannot bind {args.bind}")
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    server.start()
    LOG.info(json.dumps(dict(event="started", address=args.bind, bundle_ready=service.bundle is not None,
        stops_ready=service.stop_bundle is not None, recipe=recipe.metadata if recipe else None)))
    try:
        stop.wait()
    finally:
        server.stop(grace=5).wait()


def describe(route_metadata, stop_path):
    try:
        stops = StopBundle.load(stop_path)
        info = dict(available=True, **stops.versions(route_metadata['model_version'], route_metadata['dataset_version']),
            forecast_from=datetime.fromtimestamp(stops.start, MOSCOW).isoformat(),
            forecast_to=datetime.fromtimestamp(stops.end, MOSCOW).isoformat(),
            occurrence_counts={str(r): len(items) for r, items in stops.occurrences.items()})
    except (OSError, ValueError, KeyError, TypeError, OverflowError, EOFError, csv.Error):
        LOG.exception("Stop allocation bundle unavailable")
        info = dict(available=False)
    return dict(route_metadata, stop_forecast=info)


if __name__ == "__main__":
    main()
