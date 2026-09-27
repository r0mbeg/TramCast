"""Real gRPC stop checks; optional frozen lab CSV verifies every exported integer."""
import copy
import csv
from fractions import Fraction
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import grpc

from bundle import ROOT, INT64_MAX, Bundle, json_hour
from cache import ForecastCache
from client import validate_stop_response
from service import create_server, ForecastService
from stops import DEFAULT_STOPS, StopBundle, allocate, digest
from tests.test_service import request, expect_status
from tramcast.forecast.v1 import forecast_pb2 as pb
from tramcast.forecast.v1 import forecast_pb2_grpc as rpc

REPLAY = ROOT.parent/'bundles/030/forecast_bundle.json'


def rejected(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('Invalid stop data accepted')


def check(reference=None):
    q, remainder = divmod(INT64_MAX, 3)
    assert allocate(INT64_MAX, (Fraction(1, 3),)*3) == [q+int(i < remainder) for i in range(3)]
    assert allocate(2, (Fraction(1, 3),)*3) == [1, 1, 0]
    assert allocate(5, (Fraction(0), Fraction(1))) == [0, 5]
    for value in [-1, True, 1.5, INT64_MAX+1]:
        rejected(lambda: allocate(value, (Fraction(1),)))
    original_spec = DEFAULT_STOPS.read_bytes()
    spec = json.loads(original_spec)
    original_profiles = (DEFAULT_STOPS.parent/'profiles.csv.gz').read_bytes()
    original_positions = (DEFAULT_STOPS.parent/'occurrences.csv').read_bytes()
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        def write(profiles=original_profiles, positions=original_positions, edit=None):
            current = copy.deepcopy(spec)
            (folder/'profiles.csv.gz').write_bytes(profiles)
            (folder/'occurrences.csv').write_bytes(positions)
            current['files'] = {'profiles.csv.gz': digest(profiles), 'occurrences.csv': digest(positions)}
            if edit:
                edit(current)
            path = folder/'allocation_bundle.json'
            path.write_text(json.dumps(current), encoding='utf-8')
            return path
        text = gzip.decompress(original_profiles).decode().splitlines(True)
        header = text[0].strip().split(';')
        share_index = header.index('share')
        row = text[1].strip().split(';')
        variants = [''.join(text[:-1]), ''.join(text+[text[1]])]
        for value in ['nan', '-1', '0', '1']:
            changed = row.copy(); changed[share_index] = value
            variants.append(text[0]+';'.join(changed)+'\n'+''.join(text[2:]))
        for invalid in variants:
            path = write(profiles=gzip.compress(invalid.encode()))
            rejected(lambda: StopBundle.load(path))
        path = write(profiles=b'not gzip')
        rejected(lambda: StopBundle.load(path))
        path = write(positions=b'\n'.join(original_positions.splitlines()[:-1])+b'\n')
        rejected(lambda: StopBundle.load(path))
        path = write(edit=lambda m: m['source_metadata'].update(stop_wape=.1))
        rejected(lambda: StopBundle.load(path))
        path = write(edit=lambda m: m['files'].update({'profiles.csv.gz': 'bad'}))
        rejected(lambda: StopBundle.load(path))
        path = write()
        loaded = StopBundle.load(path)
        weights = loaded.weights[12, 0, 0]
        (folder/'profiles.csv.gz').write_bytes(b'broken')
        assert loaded.weights[12, 0, 0] == weights
        try:
            loaded.weights[12, 0, 0] = ()
        except TypeError:
            pass
        else:
            raise AssertionError('Mutable profile snapshot')

    server, service = create_server(REPLAY)
    port = server.add_insecure_port('127.0.0.1:0'); server.start()
    golden_file = gzip.open(reference, 'rt', encoding='utf-8') if reference else None
    golden = csv.DictReader(golden_file, delimiter=';') if golden_file else None
    checked, max_bytes = 0, 0
    try:
        with grpc.insecure_channel(f'127.0.0.1:{port}') as channel:
            stub = rpc.ForecastServiceStub(channel)
            for route in spec['source_metadata']['route_numbers']:
                req = request(route)
                response = stub.PredictStops(req, timeout=30)
                validate_stop_response(response, req)
                route_response = stub.Predict(req, timeout=5)
                assert response.route_model_version == route_response.model_version
                assert response.route_dataset_version == route_response.dataset_version
                assert response.model_version != route_response.model_version
                assert response.ByteSize() < 4*1024*1024
                max_bytes = max(max_bytes, response.ByteSize())
                for hour, point in zip(response.hours, route_response.points):
                    assert hour.route_boardings == point.boardings
                    local = hour.hour_start.ToDatetime(tzinfo=__import__('bundle').MOSCOW)
                    for occurrence, value in zip(response.occurrences, hour.estimated_boardings):
                        checked += 1
                        if golden is not None:
                            row = next(golden)
                            assert (int(row['route']), row['date'], int(row['hour']), row['occurrence_id'], int(row['estimated_validations'])) == (
                                route, local.date().isoformat(), local.hour, occurrence.occurrence_id, value)
            if golden is not None:
                assert next(golden, None) is None
            single = request(12, '2025-12-31T23:00:00+03:00', '2026-01-01T00:00:00+03:00')
            answer = stub.PredictStops(single, timeout=5)
            validate_stop_response(answer, single)
            for mutate in [lambda a: a.hours[0].ClearField('route_boardings'),
                           lambda a: a.hours[0].estimated_boardings.pop(),
                           lambda a: setattr(a.hours[0], 'status', 0),
                           lambda a: a.occurrences[0].ClearField('boarding_allowed'),
                           lambda a: setattr(a, 'estimate_status', 'observed')]:
                bad = pb.PredictStopsResponse(); bad.CopyFrom(answer); mutate(bad)
                rejected(lambda: validate_stop_response(bad, single))
            for invalid in [request(0), pb.PredictRequest(route_number=1), request(start='2025-11-01T00:30:00+03:00')]:
                expect_status(lambda: stub.PredictStops(invalid, timeout=5), grpc.StatusCode.INVALID_ARGUMENT)
            expect_status(lambda: stub.PredictStops(request(999), timeout=5), grpc.StatusCode.NOT_FOUND)
            expect_status(lambda: stub.PredictStops(request(5, '2026-02-01T00:00:00+03:00', '2026-02-02T00:00:00+03:00'), timeout=5),
                          grpc.StatusCode.FAILED_PRECONDITION)
            service.slot.acquire()
            try:
                expect_status(lambda: stub.PredictStops(single, timeout=5), grpc.StatusCode.UNAVAILABLE)
            finally:
                service.slot.release()
            loaded = service.stop_bundle; service.stop_bundle = None
            expect_status(lambda: stub.PredictStops(single, timeout=5), grpc.StatusCode.FAILED_PRECONDITION)
            assert stub.Predict(single, timeout=5).points
            service.stop_bundle = loaded
            cli = subprocess.run([sys.executable, str(ROOT/'client.py'), '--address', f'127.0.0.1:{port}', '--stops',
                '--route', '12', '--from', '2025-11-01T00:00:00+03:00', '--to', '2025-11-02T00:00:00+03:00'],
                capture_output=True, text=True, timeout=15)
            assert cli.returncode == 0, cli.stderr
            assert len(json.loads(cli.stdout)['hours']) == 24
    finally:
        if golden_file:
            golden_file.close()
        server.stop(0).wait()

    # A different calculated route total must change stop counts: never replay fixed stop CSVs.
    metadata = json.loads(REPLAY.read_text())
    content = (REPLAY.parent/metadata.pop('prediction_file')).read_text()
    lines = content.splitlines(); first = lines[1].split(';'); first[3] = '7'; lines[1] = ';'.join(first)
    content = ('\n'.join(lines)+'\n').encode()
    metadata.pop('prediction_sha256'); metadata['model_version'] = 'counted-model-double-not-tabpfn'
    calls = []
    def compute(active):
        active(); calls.append(1); return content
    recipe = SimpleNamespace(metadata=metadata, compute=compute, start=json_hour(metadata['forecast_from']), end=json_hour(metadata['forecast_to']))
    with tempfile.TemporaryDirectory() as directory:
        cache = ForecastCache(Path(directory)/'cache.sqlite3')
        for restart in range(2):
            server, current = create_server(recipe=recipe, cache=cache)
            port = server.add_insecure_port('127.0.0.1:0'); server.start()
            try:
                with grpc.insecure_channel(f'127.0.0.1:{port}') as channel:
                    stub = rpc.ForecastServiceStub(channel)
                    assert all(h.route_boardings == 0 for h in stub.PredictStops(request(5), timeout=5).hours)
                    assert len(calls) == restart
                    stops_before = current.stop_bundle; current.stop_bundle = None
                    expect_status(lambda: stub.PredictStops(request(1), timeout=5), grpc.StatusCode.FAILED_PRECONDITION)
                    assert len(calls) == restart
                    current.stop_bundle = stops_before
                    stop_answer = stub.PredictStops(request(1), timeout=30)
                    assert stop_answer.hours[0].route_boardings == sum(stop_answer.hours[0].estimated_boardings) == 7
                    assert stop_answer.route_model_version == metadata['model_version']
                    assert stub.Predict(request(1), timeout=5).points[0].boardings == 7 and len(calls) == 1
            finally:
                server.stop(0).wait()

    class Aborted(Exception):
        pass
    class Context:
        calls = 0
        def time_remaining(self): return 10
        def is_active(self):
            self.calls += 1
            return self.calls < 7
        def abort(self, code, details): raise Aborted(code)
    for deadline, expected in [(10, grpc.StatusCode.CANCELLED), (0, grpc.StatusCode.DEADLINE_EXCEEDED)]:
        context = Context(); context.time_remaining = lambda: deadline
        try:
            service.PredictStops(request(12), context)
        except Aborted as error:
            assert error.args == (expected,)
        else:
            raise AssertionError('Stop cancellation ignored')
        assert service.slot.acquire(blocking=False); service.slot.release()
    with patch('service.LOG.exception'):
        broken = ForecastService(REPLAY, stop_bundle_path=Path('/missing/stops.json'))
        assert broken.stop_bundle is None and broken.bundle is not None
    assert checked == 884256
    print(f'Stop gRPC: {checked} values, largest response {max_bytes} bytes; versions, cache reuse/restart, zeros, malformed input, cancellation and CLI passed. Frozen reference: {bool(reference)}')


if __name__ == '__main__':
    check(os.environ.get('STOP_REFERENCE_FILE'))
