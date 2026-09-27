"""From ml/runtime/: PYTHONPATH=generated python -m tests.test_service. No GPU or database."""
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import grpc

from service import DEFAULT_BUNDLE, Bundle, ForecastService, INT64_MAX, MOSCOW, ROOT, ROUTES, create_server
from tramcast.forecast.v1 import forecast_pb2 as pb
from tramcast.forecast.v1 import forecast_pb2_grpc as rpc


def request(route=50, start="2025-10-31T21:00:00Z", end="2025-12-31T21:00:00Z"):
    result = pb.PredictRequest(route_number=route)
    result.forecast_from.FromJsonString(start)
    result.forecast_to.FromJsonString(end)
    return result


def expect_status(call, code):
    try:
        call()
    except grpc.RpcError as error:
        assert error.code() == code, (error.code(), error.details())
    else:
        raise AssertionError(f"Expected {code}")


def check(source=DEFAULT_BUNDLE):
    meta = json.loads(source.read_text())
    bundle = Bundle.load(source)
    original = (source.parent / meta["prediction_file"]).read_text()
    rows = list(csv.reader(io.StringIO(original), delimiter=";"))
    # Read the CSV independently of the server's loader, including Moscow conversion.
    expected = {(int(r), d, int(h)): int(v) for r, d, h, v in rows[1:]}
    assert len(expected) == 14640
    server, service = create_server(source)
    port = server.add_insecure_port("127.0.0.1:0")
    assert port
    server.start()
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            grpc.channel_ready_future(channel).result(timeout=5)
            stub = rpc.ForecastServiceStub(channel)
            for route in ROUTES:
                answer = stub.Predict(request(route), timeout=5)
                assert answer.model_version == meta["model_version"]
                assert answer.dataset_version == meta["dataset_version"]
                assert len(answer.points) == 1464
                assert [p.hour_start.seconds for p in answer.points] == list(range(bundle.start, bundle.end, 3600))
                for point in answer.points:
                    assert point.HasField("boardings") and point.HasField("hour_start")
                    assert point.hour_start.nanos == 0
                    local = point.hour_start.ToDatetime(tzinfo=MOSCOW)
                    assert point.boardings == expected[route, local.date().isoformat(), local.hour]
            single = request(start="2025-12-31T23:00:00+03:00", end="2026-01-01T00:00:00+03:00")
            assert len(stub.Predict(single, timeout=5).points) == 1
            assert not pb.PredictionPoint().HasField("boardings")
            assert pb.PredictionPoint(boardings=0).HasField("boardings")
            bad_requests = [pb.PredictRequest(route_number=50), request(0), request(-1),
                request(end="2025-10-31T21:00:00Z"), request(end="2025-10-31T20:00:00Z"),
                request(start="2025-10-31T21:30:00Z"), request(end="2026-01-01T21:00:00Z")]
            nanos = request(); nanos.forecast_from.nanos = 1; bad_requests.append(nanos)
            too_early = request(); too_early.forecast_from.seconds = -62135596801; bad_requests.append(too_early)
            too_late = request(); too_late.forecast_to.seconds = 253402300800; bad_requests.append(too_late)
            for bad in bad_requests:
                expect_status(lambda: stub.Predict(bad, timeout=5), grpc.StatusCode.INVALID_ARGUMENT)
            expect_status(lambda: stub.Predict(request(999), timeout=5), grpc.StatusCode.NOT_FOUND)
            for route in (5, 50):
                expect_status(lambda: stub.Predict(request(route, "2027-10-28T00:00:00Z", "2027-10-29T00:00:00Z"), timeout=5),
                              grpc.StatusCode.FAILED_PRECONDITION)
            service.slot.acquire()
            try:
                expect_status(lambda: stub.Predict(request(), timeout=5), grpc.StatusCode.UNAVAILABLE)
            finally:
                service.slot.release()
            service.bundle = None
            expect_status(lambda: stub.Predict(request(), timeout=5), grpc.StatusCode.FAILED_PRECONDITION)
            service.bundle = bundle
    finally:
        server.stop(0).wait()

    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        manifest = directory / "bundle.json"
        predictions = directory / "forecast.csv"

        def write_bundle(content, change=None):
            current = copy.deepcopy(meta)
            current.update(prediction_file="forecast.csv", prediction_sha256=hashlib.sha256(content.encode()).hexdigest())
            current.update(change or {})
            predictions.write_text(content)
            manifest.write_text(json.dumps(current))

        invalid = [original.replace("route;date;hour;prediction", "route;date;hour;boardings"),
                   "\n".join(original.splitlines()[:-1]), original + original.splitlines()[1] + "\n",
                   original.replace("1;2025-11-01;0;", "1;2025-10-31;0;", 1)]
        for value in ("-1", "nan", "inf", "1.5", str(INT64_MAX + 1)):
            edited = original.splitlines()
            edited[1] = ";".join(edited[1].split(";")[:3] + [value])
            invalid.append("\n".join(edited))
        for route, date, hour in ((5, "2025-11-01", 0), (50, "2025-11-01", 1)):
            invalid.append(original.replace(f"{route};{date};{hour};0\n", f"{route};{date};{hour};1\n"))
        for content in invalid:
            write_bundle(content)
            try:
                Bundle.load(manifest)
            except ValueError:
                pass
            else:
                raise AssertionError("Corrupt grid or value accepted")
        for change in ({"model_version": " "}, {"dataset_version": ""}, {"prediction_sha256": "bad"},
                       {"history_end": "2026-01-01T00:00:00+03:00"}):
            write_bundle(original, change)
            try:
                Bundle.load(manifest)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid metadata accepted")
        edited = original.splitlines()
        edited[1] = ";".join(edited[1].split(";")[:3] + [str(INT64_MAX)])
        write_bundle("\n".join(edited))
        assert Bundle.load(manifest).points[1, bundle.start] == INT64_MAX
        # Snapshot is immutable for the life of a process, even if the CSV is replaced.
        write_bundle(original)
        loaded = Bundle.load(manifest)
        predictions.write_text("broken")
        assert loaded.points == bundle.points
        try:
            loaded.points[50, bundle.start] = 0
        except TypeError:
            pass
        else:
            raise AssertionError("Mutable loaded bundle")
        with patch("service.LOG.exception"):
            assert ForecastService(manifest).bundle is None
            assert ForecastService(directory / "missing.json").bundle is None

    # Deterministic cancellation during response construction, not a race with a fast RPC.
    class Aborted(Exception):
        pass

    class Context:
        calls = 0
        def time_remaining(self):
            return 10
        def is_active(self):
            self.calls += 1
            return self.calls < 5
        def abort(self, code, details):
            raise Aborted(code)

    try:
        service.Predict(request(), Context())
    except Aborted as error:
        assert error.args == (grpc.StatusCode.CANCELLED,)
    else:
        raise AssertionError("Cancellation ignored")
    assert service.slot.acquire(blocking=False)
    service.slot.release()
    context = Context()
    context.time_remaining = lambda: 0
    try:
        service.Predict(request(), context)
    except Aborted as error:
        assert error.args == (grpc.StatusCode.DEADLINE_EXCEEDED,)
    else:
        raise AssertionError("Expired deadline ignored")
    print(f"gRPC bundle {source.parent.name}: all 14640 values, grid, zeros, errors, integrity and cancellation passed.")


if __name__ == "__main__":
    check()
