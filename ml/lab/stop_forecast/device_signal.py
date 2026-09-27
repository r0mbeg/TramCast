"""Held-device temporal signal, conditional on counts; not a stop validation."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from audit import ROOT, sha, write_json, valid_key

HERE = Path(__file__).resolve().parent


def density(counts, sigma):
    x = np.asarray(counts, float)
    if x.shape != (360,) or not np.isfinite(x).all() or (x < 0).any() or x.sum() <= 0 or sigma <= 0:
        raise ValueError('Expected positive total in 360 valid time bins')
    smooth = gaussian_filter1d(x, sigma, mode='reflect')
    return .9*smooth/smooth.sum()+.1/len(x)


def score(held, reference):
    held = np.asarray(held, float)
    if held.shape != (360,) or not np.isfinite(held).all() or (held < 0).any() or held.sum() <= 0:
        raise ValueError('Invalid held counts')
    fast = float(held@np.log(density(reference, 2)))
    slow = float(held@np.log(density(reference, 30)))
    uniform = float(-held.sum()*np.log(360))
    placebo = [float(held@np.log(density(np.roll(reference, shift), 2))) for shift in range(60,360,60)]
    return dict(held_events=int(held.sum()), reference_events=int(np.sum(reference)),
                fast_log_score=fast, slow_log_score=slow, uniform_log_score=uniform,
                mean_placebo_log_score=float(np.mean(placebo)),
                **{f'placebo_{i+1}_log_score':v for i,v in enumerate(placebo)})


def summarize(scores):
    contrasts = ['slow', 'uniform', 'mean_placebo']
    groups = scores.groupby('vehicle_day')
    counts = groups.held_events.sum().to_numpy()
    delta = np.column_stack([groups.apply(lambda f:(f.fast_log_score-f[c+'_log_score']).sum(),
                                          include_groups=False).to_numpy() for c in contrasts])
    rng = np.random.default_rng(20260927)
    boot = []
    for _ in range(1000):
        indices = rng.integers(0, len(counts), len(counts))
        boot.append(delta[indices].sum(0)/counts[indices].sum())
    boot = np.asarray(boot)
    return {name:dict(nats_per_held_event=float(delta[:,j].sum()/counts.sum()),
                cluster_bootstrap_95=[float(v) for v in np.quantile(boot[:,j], [.025,.975])])
            for j,name in enumerate(contrasts)}


def run(out):
    path = ROOT/'dataset/test.csv'
    before = path.stat()
    parts = []
    columns = ['tran_date_time','device_no','garage_number','ngpt_route','validation_result']
    for f in pd.read_csv(path, sep=';', dtype=str, usecols=columns, keep_default_na=False, chunksize=250000):
        mask = f.tran_date_time.ge('2025-09-08') & f.tran_date_time.lt('2025-09-15')
        if mask.any():
            parts.append(f.loc[mask].copy())
    f = pd.concat(parts, ignore_index=True)
    f['time'] = pd.to_datetime(f.tran_date_time, format='%Y-%m-%d %H:%M:%S', errors='raise')
    f['date'] = f.time.dt.strftime('%Y-%m-%d')
    valid = valid_key(f.device_no) & valid_key(f.garage_number)
    mapping = f.loc[valid].groupby(['date','device_no']).garage_number.nunique()
    device_conflict = pd.MultiIndex.from_frame(f[['date','device_no']]).isin(mapping.loc[mapping.gt(1)].index)
    routes = f.loc[valid].groupby(['date','garage_number']).ngpt_route.nunique()
    route_conflict = pd.MultiIndex.from_frame(f[['date','garage_number']]).isin(routes.loc[routes.gt(1)].index)
    minute = f.time.dt.hour*60+f.time.dt.minute
    pilot_mask = f.ngpt_route.eq('12 трамвай') & f.validation_result.eq('1') & (minute.lt(60) | minute.ge(330))
    qpath = HERE/'runs/sequence-20260927-v1/sequence_quality.json'
    q = json.loads(qpath.read_text())
    if int(pilot_mask.sum()) != q['pilot_clean_successes']:
        raise ValueError('Pilot count changed')
    reasons = {}
    keep = pilot_mask.copy()
    for reason, reject in [('invalid_keys',~valid), ('conflicting_device_day',device_conflict),
                            ('multiple_route_vehicle_day',route_conflict), ('partial_hour5',f.time.dt.hour.eq(5))]:
        reasons[reason] = int((keep & reject).sum())
        keep &= ~reject
    sample = f.loc[keep].copy()
    sample['hour'] = sample.time.dt.hour
    sample['bin'] = (sample.time.dt.minute*60+sample.time.dt.second)//10
    rows, held_bins, reference_bins = [], [], []
    for group_id, ((day, _), day_frame) in enumerate(sample.groupby(['date','garage_number'], sort=True)):
        held_device = sorted(day_frame.device_no.unique())[0]
        for hour, h in day_frame.groupby('hour'):
            held = h.loc[h.device_no.eq(held_device), 'bin'].to_numpy()
            reference = h.loc[h.device_no.ne(held_device), 'bin'].to_numpy()
            if len(held)<10 or len(reference)<20:
                continue
            hb, rb = np.bincount(held, minlength=360), np.bincount(reference, minlength=360)
            rows.append(dict(vehicle_day=group_id, date=day, hour=int(hour), **score(hb, rb)))
            held_bins.append(hb); reference_bins.append(rb)
    scores = pd.DataFrame(rows)
    if scores.empty:
        raise ValueError('No eligible device-hour comparisons')
    reasons['insufficient_device_hour_events'] = len(sample)-int(scores.held_events.sum()+scores.reference_events.sum())
    used = int(scores.held_events.sum()+scores.reference_events.sum())
    assert used+sum(reasons.values())==int(pilot_mask.sum())
    summary = dict(version='V20260927-v1', pilot_clean_successes=int(pilot_mask.sum()),
        used_events=used, held_events=int(scores.held_events.sum()), reference_events=int(scores.reference_events.sum()),
        compared_vehicle_days=int(scores.vehicle_day.nunique()), compared_hours=len(scores),
        exclusions=reasons, contrasts=summarize(scores),
        per_day={day:summarize(g) for day,g in scores.groupby('date')},
        seed=20260927, sigma_seconds=[20,300], bin_seconds=10, uniform_mixture=.1,
        stop_accuracy=None, status='contemporaneous_device_signal_not_stop_truth')
    raw_sha = sha(path)
    qmanifest = HERE/'runs/sequence-20260927-v1/manifest.json'
    if raw_sha != json.loads(qmanifest.read_text())['inputs'][str(path.relative_to(ROOT))]:
        raise ValueError('Raw source differs from Q-v1')
    after = path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise ValueError('Raw source changed during scan')
    out.mkdir(parents=True, exist_ok=False)
    scores.to_csv(out/'device_hour_scores.csv', sep=';', index=False)
    np.savez_compressed(out/'binned_counts.npz', held=np.asarray(held_bins), reference=np.asarray(reference_bins))
    write_json(out/'summary.json', summary)
    inputs = [qpath,qmanifest,Path(__file__),HERE/'DEVICE_SIGNAL_PROTOCOL.md',HERE/'audit.py',HERE/'duration_model.py']
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in inputs}; hashes[str(path.relative_to(ROOT))]=raw_sha
    write_json(out/'manifest.json', dict(version='V20260927-v1', inputs=hashes,
        outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps({k:v for k,v in summary.items() if k!='per_day'}, ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args().out)
