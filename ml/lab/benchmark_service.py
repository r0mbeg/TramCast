"""Measure the production CSV-serving gRPC implementation in a separate process."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
import math
import multiprocessing
from pathlib import Path
import platform
import resource
import sys
import time

import grpc
from service import ROOT, create_server
from tramcast.forecast.v1 import forecast_pb2 as pb
from tramcast.forecast.v1 import forecast_pb2_grpc as rpc


def usage():
    value = resource.getrusage(resource.RUSAGE_SELF)
    return dict(cpu_seconds=value.ru_utime + value.ru_stime,
                peak_rss_bytes=value.ru_maxrss * (1 if sys.platform == "darwin" else 1024))


def worker(pipe, bundle, logfile):
    logging.basicConfig(level=logging.INFO, filename=logfile, format="%(message)s")
    started = time.perf_counter()
    server, service = create_server(bundle)
    if service.bundle is None:
        raise ValueError("Invalid benchmark bundle")
    port = server.add_insecure_port("127.0.0.1:0")
    if not port:
        raise RuntimeError("Could not bind benchmark port")
    server.start()
    pipe.send(dict(port=port, startup_seconds=time.perf_counter() - started, **usage()))
    try:
        while pipe.recv() == "usage":
            pipe.send(usage())
    finally:
        server.stop(grace=5).wait()


def percentile(values, proportion):
    values = sorted(values)
    return values[max(0, math.ceil(proportion * len(values)) - 1)] if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "forecast_bundle.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=200)
    args = parser.parse_args()
    if args.samples < 20:
        parser.error("At least 20 samples are required")
    if args.output.exists():
        parser.error("Refusing to overwrite a benchmark result")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta = json.loads(args.bundle.read_text())
    parent, child = multiprocessing.get_context("spawn").Pipe()
    process = multiprocessing.get_context("spawn").Process(
        target=worker, args=(child, str(args.bundle.resolve()), str(args.output.with_suffix(".log"))))
    process.start()
    rows = []
    try:
        if not parent.poll(15):
            raise TimeoutError("Benchmark server did not start")
        initial = parent.recv()
        with grpc.insecure_channel(f"127.0.0.1:{initial['port']}") as channel:
            grpc.channel_ready_future(channel).result(timeout=5)
            stub = rpc.ForecastServiceStub(channel)
            for hours in (1, 24, 720, 1464):
                request = pb.PredictRequest(route_number=1)
                request.forecast_from.FromJsonString(meta["forecast_from"])
                request.forecast_to.seconds = request.forecast_from.seconds + hours * 3600
                for _ in range(10):
                    stub.Predict(request, timeout=5)
                for concurrency in (1, 4):
                    parent.send("usage"); before = parent.recv()

                    def call(_):
                        start = time.perf_counter()
                        try:
                            answer = stub.Predict(request, timeout=5)
                        except grpc.RpcError as error:
                            return time.perf_counter() - start, error.code().name
                        elapsed = time.perf_counter() - start
                        assert len(answer.points) == hours
                        assert answer.model_version == meta["model_version"]
                        assert answer.dataset_version == meta["dataset_version"]
                        assert all(p.HasField("boardings") for p in answer.points)
                        return elapsed, "OK"

                    started = time.perf_counter()
                    if concurrency == 1:
                        results = [call(i) for i in range(args.samples)]
                    else:
                        with ThreadPoolExecutor(max_workers=concurrency) as executor:
                            results = list(executor.map(call, range(args.samples)))
                    wall = time.perf_counter() - started
                    parent.send("usage"); after = parent.recv()
                    successful = [elapsed for elapsed, status in results if status == "OK"]
                    statuses = Counter(status for _, status in results)
                    rows.append(dict(hours=hours, concurrency=concurrency, requests=args.samples,
                        statuses=dict(statuses), success_rps=len(successful) / wall,
                        completed_rps=len(results) / wall, wall_seconds=wall,
                        successful_p50_ms=1000 * percentile(successful, .5) if successful else None,
                        successful_p95_ms=1000 * percentile(successful, .95) if successful else None,
                        server_cpu_seconds=after["cpu_seconds"] - before["cpu_seconds"],
                        server_cpu_core_equivalents=(after["cpu_seconds"] - before["cpu_seconds"]) / wall,
                        server_peak_rss_bytes=after["peak_rss_bytes"],
                        samples=[dict(seconds=elapsed, status=status) for elapsed, status in results]))
        parent.send("usage"); final = parent.recv()
        result = dict(serving_mode="precomputed CSV; no model inference", transport="loopback gRPC",
            platform=platform.platform(), python=platform.python_version(), grpc=grpc.__version__,
            production_logging="INFO to a local file", startup=initial, final_server_usage=final,
            warmup_per_request_size=10, client_validation="counts, versions and optional boardings",
            workload="route1; repeated fixed sizes; concurrency1 and4; 5second RPC deadline",
            throughput_includes="client scheduling and response presence checks",
            excluded="network distance, TLS, Go, DB, training, Chronos inference, cold OS cache",
            model_version=meta["model_version"], dataset_version=meta["dataset_version"],
            sha256={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (ROOT / "service.py", Path(__file__), args.bundle,
                              args.bundle.parent / meta["prediction_file"])}, measurements=rows)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps([{k: v for k, v in row.items() if k != "samples"} for row in rows], indent=2))
    finally:
        if process.is_alive():
            parent.send("stop")
            process.join(timeout=10)
        if process.is_alive():
            process.terminate(); process.join(timeout=5)
        if process.exitcode not in (0, None):
            raise RuntimeError(f"Benchmark server exit code: {process.exitcode}")


if __name__ == "__main__":
    main()
