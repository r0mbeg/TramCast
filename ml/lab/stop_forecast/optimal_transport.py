"""Partial OT of registration peaks to hypothetical stop visits; never stop truth."""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks
from scipy.special import softmax

from audit import ROOT, sha, write_json
from build_metro_access import distances

HERE = Path(__file__).resolve().parent
HISTORY = HERE / 'runs/historical-geometry-20260927-v1'
COUNTS = HERE / 'runs/device-holdout-20260927-v1'
SEED = 20260927


def partial_cost(events, visits):
    """Unit-capacity transport, with unmatched costs 4 (event) and .25 (visit)."""
    a, b = np.asarray(events, float), np.asarray(visits, float)
    if a.ndim != 1 or b.ndim != 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Expected finite time vectors')
    if not len(a):
        return .25 * len(b), 0
    costs = np.concatenate((((a[:, None] - b) / 30) ** 2 - .25,
                            np.full((len(a), len(a)), 4.)), axis=1)
    row, col = linear_sum_assignment(costs)
    return float(costs[row, col].sum() + .25 * len(b)), int((col < len(b)).sum())


def prefix_peaks(reference):
    counts = np.asarray(reference, float)
    if counts.shape != (360,) or not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError('Expected 360 nonnegative bins')
    # Slice BEFORE smoothing: the future cannot leak through the filter boundary.
    prefix = counts[:240]
    smoothed = gaussian_filter1d(prefix, 2)
    peaks, _ = find_peaks(smoothed, distance=3, prominence=.5 * prefix.mean())
    times = (peaks + .5) * 10
    return times[(times >= 30) & (times <= 2370)]


def templates(lengths, turns, boarding):
    lengths, turns, boarding = np.asarray(lengths, float), np.asarray(turns, bool), np.asarray(boarding, bool)
    if (lengths.ndim != 1 or not len(lengths) or turns.shape != lengths.shape
            or boarding.shape != lengths.shape or not boarding.any()
            or not np.isfinite(lengths).all() or (lengths < 0).any()):
        raise ValueError('Invalid cycle')
    result = []
    n = len(lengths)
    for speed in (10, 15, 20):
        for dwell in (0, 15):
            duration = np.maximum(1., lengths / (speed / 3.6) + dwell + 60 * turns)
            cycle = np.r_[0., np.cumsum(duration)]
            # ponytail: fixed one-hour pilot; extend periodic support for longer horizons.
            cycles = int(np.ceil(4000 / cycle[-1])) + 1
            for start in range(n):
                index = np.arange(-cycles * n, (cycles + 1) * n)
                times = (index // n) * cycle[-1] + cycle[index % n] - cycle[start]
                result.append((times[boarding[index % n]], start, speed, dwell))
    return result


def predict(events, candidates):
    events = np.asarray(events, float)
    if (events.ndim != 1 or not len(events) or not np.isfinite(events).all()
            or (np.diff(events) <= 0).any() or (events < 30).any() or (events > 2370).any()):
        raise ValueError('Invalid prefix peaks')
    grid = (np.arange(240, 360) + .5) * 10
    costs, matched, densities = [], [], []
    for times, _, _, _ in candidates:
        visits = times + events[0]
        fit_visits = visits[(visits >= 30) & (visits <= 2370)]
        cost, count = partial_cost(events, fit_visits)
        costs.append(cost)
        matched.append(count)
        future = visits[(visits >= 2400 - 180) & (visits <= 3600 + 180)]
        density = np.exp(-.5 * ((grid[:, None] - future) / 45) ** 2).sum(axis=1)
        density = density / density.sum() if density.sum() else np.ones(120) / 120
        densities.append(.9 * density + .1 / 120)
    weights = softmax(-np.asarray(costs))
    density = weights @ np.asarray(densities)
    return density, dict(min_cost=float(min(costs)), max_hypothesis_weight=float(weights.max()),
                         expected_matched_fraction=float(weights @ matched / len(events)))


def checked(folder, name, inputs):
    manifest = folder / 'manifest.json'
    path = folder / name
    if sha(path) != json.loads(manifest.read_text())['outputs'][name]:
        raise ValueError(f'Changed input: {path}')
    inputs.extend([manifest, path])
    return path


def summarize(rows):
    frame = pd.DataFrame(rows)
    total = int(frame.held_events.sum())
    if total <= 0:
        raise ValueError('No held events to score')
    columns = ['historical', 'equal_spacing', 'uniform'] + [f'permuted_{i:02}' for i in range(19)]
    scores = {c: float(frame[c].sum() / total) for c in columns}
    null = columns[3:]
    frame['null_mean'] = frame[null].mean(axis=1)
    groups = frame.groupby('vehicle_day')[['historical', 'null_mean', 'held_events']].sum().to_numpy()
    rng = np.random.default_rng(SEED)
    samples = groups[rng.integers(len(groups), size=(1000, len(groups)))].sum(axis=1)
    valid = samples[:, 2] > 0
    interval = np.quantile((samples[valid, 0] - samples[valid, 1]) / samples[valid, 2], [.025, .975]).tolist()
    daily = frame.groupby('date')[columns + ['held_events']].sum()
    days = {str(day): {c: float(row[c] / row.held_events) if row.held_events else None
                      for c in ['historical', 'uniform', 'equal_spacing']}
            for day, row in daily.iterrows()}
    above = sum(scores['historical'] > scores[c] for c in null)
    return dict(selected_hours=len(frame), selected_vehicle_days=int(frame.vehicle_day.nunique()),
                held_future_events=total, zero_held_future_hours=int(frame.held_events.eq(0).sum()),
                log_score_per_event=scores, daily_log_score=days,
                gain_vs_mean_permutation=float(scores['historical'] - np.mean([scores[c] for c in null])),
                descriptive_vehicle_day_bootstrap_95=interval,
                historical_beats_permutations=above,
                promotion_gate_passed=bool(above == 19 and interval[0] > 0
                    and scores['historical'] > max(scores['uniform'], scores['equal_spacing'])),
                stop_mae=None, stop_wape=None, verified_stop_assignments=0,
                forecast_artifact=None, status='temporal_geometry_diagnostic_not_stop_forecast')


def run(out):
    started = time.monotonic()
    if out.exists():
        raise ValueError('Output must be new')
    inputs = [Path(__file__), HERE / 'OT_PROTOCOL.md', HERE / 'test_optimal_transport.py',
              HERE / 'audit.py', HERE / 'build_metro_access.py']
    geography = pd.read_csv(checked(HISTORY, 'historical_occurrences.csv', inputs), sep=';')
    hours = pd.read_csv(checked(COUNTS, 'device_hour_scores.csv', inputs), sep=';')
    with np.load(checked(COUNTS, 'binned_counts.npz', inputs), allow_pickle=False) as data:
        reference, held = data['reference'], data['held']
    if (reference.shape != held.shape or reference.shape != (len(hours), 360)
            or not np.isfinite(reference).all() or not np.isfinite(held).all()
            or (reference < 0).any() or (held < 0).any()
            or not np.array_equal(reference.sum(axis=1), hours.reference_events)
            or not np.array_equal(held.sum(axis=1), hours.held_events)):
        raise ValueError('Unaligned or invalid hourly counts')
    geography = geography.sort_values(['direction', 'stop_sequence']).reset_index(drop=True)
    if (len(geography) != 91 or geography.occurrence_id.duplicated().any()
            or geography.groupby('direction').size().to_dict() != {0: 46, 1: 45}
            or not geography.boarding_permitted_by_osm.isin([True, False]).all()):
        raise ValueError('Changed historical support')
    nxt = np.roll(np.arange(len(geography)), -1)
    lengths = np.diag(distances(geography.stop_lat, geography.stop_lon,
                               geography.stop_lat.iloc[nxt], geography.stop_lon.iloc[nxt])).copy()
    direction = geography.direction.to_numpy()
    turns = direction != direction[nxt]
    boarding = geography.boarding_permitted_by_osm.to_numpy(bool)
    weekday = pd.to_datetime(hours.date).dt.dayofweek < 5
    peaks = {i: prefix_peaks(reference[i]) for i in np.flatnonzero(weekday)}
    eligible = [i for i, values in peaks.items() if len(values) >= 5]
    if not eligible:
        raise ValueError('No eligible prefix')
    chosen = np.sort(np.random.default_rng(SEED).choice(eligible, min(96, len(eligible)), replace=False))
    equal = lengths.copy()
    equal[~turns] = lengths[~turns].mean()
    networks = {'historical': lengths, 'equal_spacing': equal}
    rng = np.random.default_rng(SEED + 1)
    for i in range(19):
        shuffled = lengths.copy()
        for d in (0, 1):
            mask = (direction == d) & ~turns
            shuffled[mask] = rng.permutation(lengths[mask])
        networks[f'permuted_{i:02}'] = shuffled
    rows = [dict(hour_row=int(i), vehicle_day=int(hours.iloc[i].vehicle_day), date=hours.iloc[i].date,
                 hour=int(hours.iloc[i].hour), prefix_peaks=len(peaks[i]),
                 held_events=int(held[i, 240:].sum()), uniform=float(-held[i, 240:].sum() * np.log(120)))
            for i in chosen]
    out.mkdir(parents=True, exist_ok=False)
    for name, edges in networks.items():
        candidates = templates(edges, turns, boarding)
        for row, index in zip(rows, chosen):
            density, diagnostic = predict(peaks[index], candidates)
            row[name] = float(held[index, 240:] @ np.log(density))
            if name == 'historical':
                row.update(diagnostic)
        print(json.dumps(dict(completed=name, elapsed_seconds=round(time.monotonic() - started, 1))), flush=True)
        if time.monotonic() - started > 900:
            write_json(out / 'aborted.json', dict(reason='15 minute budget exceeded', completed=name))
            raise RuntimeError('Budget exceeded; preserve aborted run')
    summary = summarize(rows)
    summary.update(version='OT20260927-v1', source_hours=len(hours), weekday_hours=int(weekday.sum()),
                   eligible_hours=len(eligible), source_events=int(reference.sum() + held.sum()),
                   selected_prefix_reference_events=int(reference[chosen, :240].sum()),
                   selected_future_reference_events_unused=int(reference[chosen, 240:].sum()),
                   selected_hour_events=int(reference[chosen].sum() + held[chosen].sum()),
                   cycle_distance_m=float(lengths.sum()), historical_positions=len(geography),
                   mean_max_hypothesis_weight=float(np.mean([r['max_hypothesis_weight'] for r in rows])),
                   mean_matched_peak_fraction=float(np.mean([r['expected_matched_fraction'] for r in rows])),
                   elapsed_seconds=time.monotonic() - started)
    write_json(out / 'hour_scores.json', rows)
    write_json(out / 'summary.json', summary)
    write_json(out / 'manifest.json', dict(version=summary['version'],
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__),
        inputs={str(p.relative_to(ROOT)): sha(p) for p in inputs},
        outputs={p.name: sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(summary, ensure_ascii=False))


def verify(out):
    manifest = json.loads((out / 'manifest.json').read_text())
    for name, digest in manifest['inputs'].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f'Changed input: {name}')
    for name, digest in manifest['outputs'].items():
        if sha(out / name) != digest:
            raise ValueError(f'Changed output: {name}')
    computed = summarize(json.loads((out / 'hour_scores.json').read_text()))
    saved = json.loads((out / 'summary.json').read_text())
    if any(saved[key] != value for key, value in computed.items()):
        raise ValueError('Summary differs from hourly scores')
    print('Verified hashes and independently reaggregated hourly scores; not a full inference rerun.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if bool(args.out) == bool(args.verify):
        parser.error('Choose --out or --verify')
    verify(args.verify) if args.verify else run(args.out)
