"""Pre-model feasibility checks: aggregate rank and raw vehicle-time sequences."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from audit import ROOT, sha, write_json, valid_key
from bayes_experiment import design, moments
from metro_experiment import augment
from proximity_experiment import features, PARENT, ACCESS

HERE = Path(__file__).resolve().parent


def gap_stats(frame, keys):
    f = frame.sort_values(keys+['time'])
    gaps = f.groupby(keys, sort=False).time.diff().dt.total_seconds().dropna()
    if gaps.empty:
        return dict(pairs=0, quantiles_seconds=None)
    return dict(pairs=len(gaps), zero_gap_fraction=float(gaps.eq(0).mean()),
                quantiles_seconds={str(q):float(gaps.quantile(q)) for q in [.1,.25,.5,.75,.9,.99]},
                fraction_at_most_60s=float(gaps.le(60).mean()),
                fraction_over_30min=float(gaps.gt(1800).mean()))


def rank_check():
    stops = pd.read_csv(PARENT/'occurrences.csv', sep=';')
    base, edges, groups, occurrence = design(stops, np.load(PARENT/'spatial_features.npy', allow_pickle=False))
    values, _ = features(stops, pd.read_csv(ACCESS/'stop_features.csv', sep=';'))
    x = augment(base, occurrence, groups, edges, values, 'flow')
    b = json.loads((PARENT/'results.json').read_text())
    p = json.loads((HERE/'runs/proximity-20260927-v1/results.json').read_text())
    result = {}
    for cutoff in ['2025-04-30', '2025-06-30']:
        for name, matrix, beta in [('base', base, b['diagnostics'][f'{cutoff}/poisson_sd1.0']['beta']),
                                    ('proximity', x, p['models'][cutoff]['beta'])]:
            _, _, jac = moments(np.asarray(beta), matrix, edges)
            singular = np.linalg.svd(jac, compute_uv=False)
            tol = singular[0]*max(jac.shape)*np.finfo(float).eps
            rank = int((singular > tol).sum())
            result[f'{cutoff}/{name}'] = dict(shape=list(jac.shape), rank=rank,
                nullity=jac.shape[1]-rank, tolerance=float(tol), singular_values=singular.tolist())
    return result


def run(out):
    path = ROOT/'dataset/test.csv'
    parts, scanned = [], 0
    columns = ['tran_date_time','device_no','garage_number','ngpt_route','validation_result']
    before = path.stat()
    for f in pd.read_csv(path, sep=';', dtype=str, usecols=columns, keep_default_na=False, chunksize=250000):
        scanned += len(f)
        # Fixed discovery week, all routes for device/vehicle consistency checks.
        mask = f.tran_date_time.ge('2025-09-08') & f.tran_date_time.lt('2025-09-15')
        if mask.any():
            parts.append(f.loc[mask].copy())
    f = pd.concat(parts, ignore_index=True)
    f['time'] = pd.to_datetime(f.tran_date_time, format='%Y-%m-%d %H:%M:%S', errors='raise')
    f['date'] = f.time.dt.strftime('%Y-%m-%d')
    minute = f.time.dt.hour*60+f.time.dt.minute
    f['eligible'] = f.validation_result.eq('1') & (minute.lt(60) | minute.ge(330))
    valid = valid_key(f.device_no) & valid_key(f.garage_number)
    pairs = f.loc[valid].groupby(['date','device_no']).garage_number.nunique()
    conflict_index = pairs.loc[pairs.gt(1)].index
    f['device_day_conflict'] = pd.MultiIndex.from_frame(f[['date','device_no']]).isin(conflict_index)
    pilot = f.loc[f.ngpt_route.eq('12 трамвай') & f.eligible].copy()
    if pilot.empty:
        raise ValueError('No pilot events; inspect route encoding')
    usable = pilot.loc[valid.loc[pilot.index] & ~pilot.device_day_conflict].copy()
    ledger_path = HERE/'runs/20260927-v2/route_hour_balance.csv'
    ledger = pd.read_csv(ledger_path, sep=';')
    expected = int(ledger.loc[ledger.route.eq(12) & ledger.date.ge('2025-09-08') & ledger.date.lt('2025-09-15'), 'boardings'].sum())
    if len(pilot) != expected:
        raise ValueError('Sample does not match audited route ledger')
    vehicles = usable.groupby(['date','garage_number']).size()
    all_routes = f.loc[valid].groupby(['date','garage_number']).ngpt_route.nunique()
    devices = usable.groupby(['date','garage_number']).device_no.nunique()
    # No stop names, vehicle IDs or passenger hashes in saved outputs.
    summary = dict(version='Q20260927-v1', discovery_period=['2025-09-08','2025-09-14'],
        raw_rows_scanned=scanned, all_route_week_rows=len(f), pilot_clean_successes=len(pilot),
        pilot_ledger_successes=expected, pilot_invalid_keys=int((~valid.loc[pilot.index]).sum()),
        pilot_conflicting_device_day_events=int(pilot.device_day_conflict.sum()),
        key_consistent_pilot_events=len(usable), vehicle_days=len(vehicles),
        vehicle_days_with_multiple_devices=int(devices.gt(1).sum()),
        vehicle_days_with_multiple_routes=int(all_routes.reindex(vehicles.index).gt(1).sum()),
        events_per_vehicle_day_quantiles={str(q):float(vehicles.quantile(q)) for q in [.1,.5,.9]},
        timestamp_second_counts={str(k):int(v) for k,v in usable.time.dt.second.value_counts().sort_index().items()},
        vehicle_gaps=gap_stats(usable,['date','garage_number']),
        device_gaps=gap_stats(usable,['date','garage_number','device_no']),
        verified_stop_labels=0, direction_or_trip_anchors_verified=False,
        scope='Discovery only; key consistency does not prove vehicle identity or stop linkage')
    after = path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise ValueError('Raw input changed during reading')
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'sequence_quality.json', summary)
    write_json(out/'aggregate_rank.json', rank_check())
    inputs = [path, ledger_path, PARENT/'occurrences.csv', PARENT/'spatial_features.npy', PARENT/'results.json',
              ACCESS/'stop_features.csv', HERE/'runs/proximity-20260927-v1/results.json']
    inputs += [HERE/n for n in ['sequence_feasibility.py','bayes_experiment.py','metro_experiment.py',
                               'proximity_experiment.py','audit.py']]
    write_json(out/'manifest.json', dict(version='Q20260927-v1',
        inputs={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(summary, ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    with threadpool_limits(limits=1):
        run(parser.parse_args().out)
