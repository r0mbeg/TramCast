"""Three-hour device traces, new time split; no raw identities in outputs."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit import ROOT, sha, valid_key, write_json
from pipeline import working_time

HERE = Path(__file__).resolve().parent


def blocks(reference_seconds):
    seconds = np.asarray(reference_seconds, int)
    if seconds.ndim != 1 or (np.diff(seconds) < 0).any() or (seconds < 0).any() or (seconds >= 86400).any():
        raise ValueError('Expected sorted times inside one day')
    result = []
    for part in np.split(seconds, np.flatnonzero(np.diff(seconds) > 600) + 1):
        if not len(part):
            continue
        first, end = part[0] // 10, part[-1] // 10 + 1
        for start in range(first, end - 1080 + 1, 1080):
            prefix = ((part >= start * 10) & (part < (start + 360) * 10)).sum()
            if prefix >= 20:
                result.append(int(start))
    return result


def select(frame):
    rng = np.random.default_rng(20260928)
    chosen = []
    for day, group in frame.groupby('date', sort=True):
        representatives = [int(rng.choice(g.index)) for _, g in group.groupby('vehicle_day', sort=True)]
        cap = 4 if day <= '2025-09-19' else 12
        chosen.extend(rng.choice(representatives, min(cap, len(representatives)), replace=False).tolist())
    return np.asarray(sorted(chosen), int)


def run(out):
    if out.exists():
        raise ValueError('Output must be new')
    raw = ROOT / 'dataset/test.csv'
    before = raw.stat()
    columns = ['tran_date_time', 'device_no', 'garage_number', 'ngpt_route', 'validation_result']
    parts = []
    for frame in pd.read_csv(raw, sep=';', dtype=str, usecols=columns, keep_default_na=False, chunksize=250000):
        mask = frame.tran_date_time.ge('2025-09-15') & frame.tran_date_time.lt('2025-09-27')
        if mask.any():
            parts.append(frame.loc[mask].copy())
    if not parts:
        raise ValueError('No source events')
    f = pd.concat(parts, ignore_index=True)
    del parts
    f['time'] = pd.to_datetime(f.tran_date_time, format='%Y-%m-%d %H:%M:%S', errors='raise')
    f['date'] = f.time.dt.strftime('%Y-%m-%d')
    valid = valid_key(f.device_no) & valid_key(f.garage_number)
    dv = f.loc[valid].groupby(['date', 'device_no']).garage_number.nunique()
    vr = f.loc[valid].groupby(['date', 'garage_number']).ngpt_route.nunique()
    dc = pd.MultiIndex.from_frame(f[['date', 'device_no']]).isin(dv.loc[dv.gt(1)].index)
    rc = pd.MultiIndex.from_frame(f[['date', 'garage_number']]).isin(vr.loc[vr.gt(1)].index)
    keep = f.ngpt_route.eq('12 трамвай') & f.validation_result.eq('1') & working_time(f.time)
    total = int(keep.sum())
    ledger_path = HERE / 'runs/20260927-v2/route_hour_balance.csv'
    ledger = pd.read_csv(ledger_path, sep=';')
    expected = int(ledger.loc[ledger.route.eq(12) & ledger.date.between('2025-09-15', '2025-09-26'), 'boardings'].sum())
    if total != expected:
        raise ValueError('Raw count disagrees with audited ledger')
    excluded = {}
    for name, rejected in [('invalid_keys', ~valid), ('conflicting_device_day', dc),
                           ('multiple_route_vehicle_day', rc), ('weekend', f.time.dt.dayofweek.ge(5)),
                           ('partial_hour5', f.time.dt.hour.eq(5))]:
        excluded[name] = int((keep & rejected).sum())
        keep &= ~rejected
    f = f.loc[keep].copy()
    f['second'] = f.time.dt.hour * 3600 + f.time.dt.minute * 60 + f.time.dt.second
    rows, reference, held = [], [], []
    for vehicle_day, ((day, _), g) in enumerate(f.groupby(['date', 'garage_number'], sort=True)):
        device = sorted(g.device_no.unique())[0]
        r = np.sort(g.loc[g.device_no.ne(device), 'second'].to_numpy())
        h = g.loc[g.device_no.eq(device), 'second'].to_numpy()
        for start in blocks(r):
            rb = np.bincount((r[(r >= start * 10) & (r < (start + 1080) * 10)] // 10 - start), minlength=1080)
            hb = np.bincount((h[(h >= start * 10) & (h < (start + 1080) * 10)] // 10 - start), minlength=1080)
            rows.append(dict(vehicle_day=vehicle_day, date=day, start_bin=start,
                             split='train' if day <= '2025-09-19' else 'validation' if day == '2025-09-22' else 'test'))
            reference.append(rb); held.append(hb)
    if not rows:
        raise ValueError('No continuous three-hour blocks')
    meta = pd.DataFrame(rows)
    reference, held = np.asarray(reference), np.asarray(held)
    selected = select(meta)
    meta['selected'] = meta.index.isin(selected)
    candidate_count = int(reference.sum() + held.sum())
    excluded['outside_eligible_three_hour_blocks'] = len(f) - candidate_count
    excluded['eligible_but_not_selected'] = candidate_count - int(reference[selected].sum() + held[selected].sum())
    used = int(reference[selected].sum() + held[selected].sum())
    if min(excluded.values()) < 0 or used + sum(excluded.values()) != total:
        raise ValueError('Broken disjoint event ledger')
    source_manifest = HERE / 'runs/device-holdout-20260927-v1/manifest.json'
    raw_digest = sha(raw)
    if raw_digest != json.loads(source_manifest.read_text())['inputs']['dataset/test.csv']:
        raise ValueError('Changed raw source')
    after = raw.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Raw changed during scan')
    if set(meta.loc[selected, 'split']) != {'train', 'validation', 'test'}:
        raise ValueError('A required time split is empty')
    out.mkdir(parents=True, exist_ok=False)
    meta.to_csv(out / 'blocks.csv', sep=';', index=False)
    np.savez_compressed(out / 'counts.npz', reference=reference, held=held)
    write_json(out / 'summary.json', dict(version='ELD20260927-v1', source_events=total, selected_events=used,
        exclusions=excluded, eligible_blocks=len(meta), selected_blocks=len(selected),
        blocks_by_date=meta.loc[selected].groupby('date').size().to_dict(),
        selected_reference_events=int(reference[selected].sum()), selected_held_events=int(held[selected].sum()),
        completeness_verified=False, verified_stop_labels=0))
    inputs = [Path(__file__), HERE / 'ELASTIC_PROTOCOL.md', HERE / 'audit.py', HERE.parent / 'pipeline.py', ledger_path, source_manifest]
    write_json(out / 'manifest.json', dict(inputs={**{str(p.relative_to(ROOT)): sha(p) for p in inputs}, 'dataset/test.csv': raw_digest},
        outputs={p.name: sha(p) for p in out.iterdir() if p.is_file()}))
    print((out / 'summary.json').read_text(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    run(parser.parse_args().out)
