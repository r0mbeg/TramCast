"""Check the CPU release through HTTP; optionally measure reads of ready forecasts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'ml/lab/artifacts/dense_student_20260927_v3/submission.csv'
REFERENCE_SHA256 = '538872b074d8ced7f0997e2f1427db4f3dab39db3f3dba0e8be9f80a68a15695'


def request(base, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = Request(base + path, data=data, headers={'Content-Type': 'application/json'})
    with urlopen(req, timeout=15) as response:
        return response.status, json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://localhost:8080')
    parser.add_argument('--requests', type=int, default=0, help='Optional warm HTTP read count')
    parser.add_argument('--concurrency', type=int, default=4)
    args = parser.parse_args()
    if not 0 <= args.requests <= 10000 or not 1 <= args.concurrency <= 16:
        parser.error('requests must be 0..10000; concurrency must be 1..16')
    base = args.url.rstrip('/')
    if hashlib.sha256(REFERENCE.read_bytes()).hexdigest() != REFERENCE_SHA256:
        raise ValueError('Reference CPU submission checksum mismatch')
    with REFERENCE.open() as stream:
        expected = {}
        for row in csv.DictReader(stream, delimiter=';'):
            expected.setdefault(int(row['route']), []).append(
                (row['date'], int(row['hour']), int(row['prediction'])))

    request(base, '/readyz')
    _, version = request(base, '/api/forecast-versions/active')
    if not version['model_version'].startswith('catboost-cpu-'):
        raise ValueError('Active version is not CPU CatBoost')
    _, catalog = request(base, '/api/routes')
    routes = {int(r['route_number']): r['id'] for r in catalog['routes'] if r['forecast_enabled']}
    if routes.keys() != expected.keys():
        raise ValueError('Forecast route set differs from CPU submission')

    paths = []
    slices = []
    started = time.perf_counter()
    for number, route_id in sorted(routes.items()):
        query = dict(forecast_version_id=version['id'], route_id=route_id,
                     **{'from': version['forecast_from'], 'to': version['forecast_to']})
        deadline = time.monotonic() + 120
        while True:
            status, result = request(base, '/api/predictions/query', query)
            if status == 200:
                break
            if status != 202 or result['status'] == 'failed':
                raise ValueError(f'Route {number}: unexpected job response {result}')
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Route {number}: forecast not ready after 120 seconds')
            time.sleep(1)
        points = result['points']
        actual = [(p['date'], p['hour'], p['boardings']) for p in points]
        if (result['forecast_version_id'] != version['id'] or result['route_id'] != route_id
                or actual != expected[number]
                or any(type(p['boardings']) is not int or p['boardings'] < 0 for p in points)):
            raise ValueError(f'Route {number}: version, grid or values differ from CPU submission')
        paths.append('/api/predictions?' + urlencode(query))
        slices.append(result)

    report = dict(checked_at=datetime.now(timezone.utc).isoformat(), version=version,
                  routes=len(routes), rows=sum(map(len, expected.values())),
                  reference_sha256=REFERENCE_SHA256,
                  full_grid_matches_cpu_submission=True,
                  check_seconds=time.perf_counter() - started)
    if args.requests:
        def sample(index):
            start = time.perf_counter()
            try:
                i = index % len(paths)
                status, result = request(base, paths[i])
                if result != slices[i]:
                    return dict(seconds=time.perf_counter() - start, status='CONTENT_MISMATCH')
                return dict(seconds=time.perf_counter() - start, status=status)
            except HTTPError as error:
                return dict(seconds=time.perf_counter() - start, status=error.code)
            except (OSError, ValueError) as error:
                return dict(seconds=time.perf_counter() - start, status=type(error).__name__)

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            samples = list(pool.map(sample, range(args.requests)))
        elapsed = time.perf_counter() - start
        times = sorted(s['seconds'] for s in samples if s['status'] == 200)
        report['warm_http_reads'] = dict(requests=args.requests, concurrency=args.concurrency,
            errors=len(samples)-len(times), seconds=elapsed, successful_rps=len(times)/elapsed,
            median_ms=statistics.median(times)*1000 if times else None,
            p95_ms=times[math.ceil(len(times)*0.95)-1]*1000 if times else None,
            samples=samples)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.requests and report['warm_http_reads']['errors']:
        raise SystemExit('HTTP read check failed; see report')


if __name__ == '__main__':
    main()
