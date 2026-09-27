"""Exact route-12 quarter-hour targets and frozen 61-day forecasting benchmark."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from audit import LAB, ROOT, sha, write_json
from pipeline import working_time

HERE = Path(__file__).resolve().parent
AUDIT = HERE / 'runs/20260927-v2'
CONTROL = LAB / 'artifacts/portfolio_20260926/continuation/tabular_shape/study/tabular_shape/selected'
PROTOCOL = HERE / 'KNOWN_COUNTS_PROTOCOL.md'
VERSION = 'K20260927-v1'
KEYS = ['date', 'hour', 'quarter']
MODELS = ['uniform_030', 'historical_030', 'daytype_030', 'bayes_030', 'direct_calendar']
CUTOFFS = ['2025-04-30', '2025-06-30', '2025-08-31', '2025-10-31']


def grid(start, end):
    f = pd.MultiIndex.from_product([pd.date_range(start, end).strftime('%Y-%m-%d'),
        range(24), range(4)], names=KEYS).to_frame(index=False)
    f['route'] = 12
    f['working_minutes'] = np.where(f.hour.between(1, 4) | (f.hour.eq(5) & f.quarter.lt(2)), 0, 15)
    f['weekday'] = pd.to_datetime(f.date).dt.dayofweek
    f['weekend'] = f.weekday.ge(5).astype(int)
    return f


def count_chunk(f):
    f = f.loc[f.ngpt_route.eq('12 трамвай')].copy()
    t = pd.to_datetime(f.tran_date_time, format='%Y-%m-%d %H:%M:%S', errors='raise')
    if t.isna().any() or f.validation_result.isna().any():
        raise ValueError('Missing raw time or result')
    keep = t.ge('2025-01-01') & t.lt('2025-11-01') & working_time(t)
    z = pd.DataFrame({'date': t.dt.strftime('%Y-%m-%d'), 'hour': t.dt.hour,
                      'quarter': t.dt.minute // 15, 'success': f.validation_result.eq(1)})
    return (z.loc[keep & z.success].groupby(KEYS).size().to_dict(),
            z.loc[keep].groupby(KEYS).size().to_dict())


def attach_counts(f, success, events):
    f = f.set_index(KEYS)
    for name, counts in [('registered_successes', success), ('working_events', events)]:
        f[name] = pd.Series(counts, dtype='int64').reindex(f.index, fill_value=0).astype('int64')
    f['counter_status'] = np.where(f.working_events.gt(0), 'observed_unknown_completeness', 'missing')
    f.loc[f.working_minutes.eq(0), 'counter_status'] = 'structural_zero'
    f['observed_counter'] = f.registered_successes.astype('Int64')
    f.loc[f.counter_status.eq('missing'), 'observed_counter'] = pd.NA
    f = f.reset_index()
    f['complete_hour'] = f.counter_status.ne('missing').groupby([f.date, f.hour]).transform('all')
    return f


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    expected = {x['path']: x['sha256'] for x in json.loads((AUDIT/'run.json').read_text())['sources']}
    success, events, inputs = Counter(), Counter(), {}
    for name in ['train.csv', 'test.csv']:
        path = ROOT/'dataset'/name
        before = path.stat()
        rows = 0
        for f in pd.read_csv(path, sep=';', usecols=['tran_date_time', 'ngpt_route', 'validation_result'],
                             dtype={'validation_result': 'int64'}, chunksize=250000):
            a, b = count_chunk(f)
            success.update(a); events.update(b)
            rows += len(f)
            if rows % 5000000 == 0:
                print(name, rows, 'rows scanned', flush=True)
        digest = sha(path)
        after = path.stat()
        if digest != expected[str(path.relative_to(ROOT))] or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Raw source changed versus frozen audit')
        inputs[str(path.relative_to(ROOT))] = digest
    f = attach_counts(grid('2025-01-01', '2025-10-31'), success, events)
    ledger = pd.read_csv(AUDIT/'route_hour_balance.csv', sep=';')
    ledger = ledger.loc[ledger.route.eq(12)].set_index(['date', 'hour']).sort_index()
    sums = f.groupby(['date', 'hour'])[['registered_successes', 'working_events']].sum()
    np.testing.assert_array_equal(sums.to_numpy(), ledger[['boardings', 'working_events']].to_numpy())
    if list(sums.index) != list(ledger.index):
        raise ValueError('Hourly keys differ')
    f.to_csv(out/'observations.csv', sep=';', index=False)
    hourly = ledger[['boardings', 'working_events', 'observed_counter', 'counter_status']].reset_index()
    hourly.insert(0, 'route', 12)
    hourly.to_csv(out/'hourly_observations.csv', sep=';', index=False)
    for path in [Path(__file__), HERE/'audit.py', LAB/'pipeline.py', PROTOCOL,
                 AUDIT/'run.json', AUDIT/'route_hour_balance.csv']:
        inputs[str(path.relative_to(ROOT))] = sha(path)
    summary = dict(version=VERSION, route=12, rows=len(f), reconciled_hours=len(sums),
        successes=int(f.registered_successes.sum()), status_counts=f.counter_status.value_counts().to_dict(),
        complete_working_hours=int(f.loc[f.complete_hour & f.working_minutes.gt(0), ['date', 'hour']].drop_duplicates().shape[0]),
        stop_labels=0, wall_seconds=time.monotonic()-started)
    write_json(out/'summary.json', summary)
    write_json(out/'manifest.json', dict(version=VERSION, inputs=inputs,
        outputs={p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}))
    print(json.dumps(summary), flush=True)


def fit(f, cutoff):
    start = (pd.Timestamp(cutoff)-pd.Timedelta(days=56)).strftime('%Y-%m-%d')
    t = f.loc[f.date.gt(start) & f.date.le(cutoff) & f.complete_hour].copy()
    if t.empty:
        raise ValueError('No training observations')
    profiles = {}
    for hour in range(24):
        h = t.loc[t.hour.eq(hour)]
        active = grid(cutoff, cutoff).loc[lambda z: z.hour.eq(hour), 'working_minutes'].to_numpy() > 0
        if not active.any():
            profiles[hour] = dict(uniform=np.zeros(4), historical=np.zeros(4),
                daytype=np.zeros((2, 4)), bayes=np.zeros((7, 4)), direct=np.zeros((7, 4)))
            continue
        if h.empty:
            raise ValueError(f'No complete training hours for {hour}')
        uniform = active/active.sum()
        counts = h.groupby('quarter').registered_successes.sum().reindex(range(4), fill_value=0).to_numpy()
        historical = counts/counts.sum() if counts.sum() else uniform
        prior = (counts+uniform)/(counts.sum()+1)
        alpha = max(1, 7*counts.sum()/h.date.nunique())
        avg = h.groupby('quarter').registered_successes.mean().reindex(range(4), fill_value=0).to_numpy()
        daytype, bayes, direct = [], [], []
        for weekday in range(7):
            s = h.loc[h.weekday.eq(weekday)]
            n = s.groupby('quarter').registered_successes.sum().reindex(range(4), fill_value=0).to_numpy()
            bayes.append((n+alpha*prior)/(n.sum()+alpha))
            direct.append(n/s.date.nunique() if len(s) else avg)
        for weekend in range(2):
            n = h.loc[h.weekend.eq(weekend)].groupby('quarter').registered_successes.sum().reindex(range(4), fill_value=0).to_numpy()
            daytype.append(n/n.sum() if n.sum() else historical)
        profiles[hour] = dict(uniform=uniform, historical=historical,
            daytype=np.array(daytype), bayes=np.array(bayes), direct=np.array(direct))
    return profiles


def apportion(total, p):
    p = np.asarray(p, dtype=float)
    if p.shape != (4,) or not np.isfinite(p).all() or (p < 0).any() or not np.isfinite(total) or total < 0:
        raise ValueError('Invalid count or shares')
    if total == 0 and p.sum() == 0:
        return np.zeros(4, dtype='int64')
    if not np.isclose(p.sum(), 1):
        raise ValueError('Shares do not sum to one')
    target = int(np.floor(total+0.5))
    raw = target*p/p.sum()
    result = np.floor(raw).astype('int64')
    left = target-int(result.sum())
    result[np.argsort(-(raw-result), kind='stable')[:left]] += 1
    return result


def predict(keys, profiles, hourly, model):
    if model not in MODELS or keys.duplicated(KEYS).any() or hourly.duplicated(['date', 'hour']).any():
        raise ValueError('Invalid model or duplicate keys')
    output = []
    totals = hourly.set_index(['date', 'hour']).prediction
    for (date, hour), f in keys.groupby(['date', 'hour'], sort=True):
        if sorted(f.quarter) != list(range(4)):
            raise ValueError('Incomplete quarter support')
        profile = profiles[hour]
        weekday = int(f.weekday.iloc[0])
        if model == 'direct_calendar':
            values = np.floor(profile['direct'][weekday]+0.5).astype('int64')
        else:
            kind = model.removesuffix('_030')
            p = profile[kind]
            if kind == 'bayes':
                p = p[weekday]
            elif kind == 'daytype':
                p = p[int(weekday >= 5)]
            values = apportion(float(totals.loc[(date, hour)]), p)
        output.extend(values.tolist())
    return np.array(output, dtype='int64')


def metric(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Invalid scoring inputs')
    err = np.abs(y-p)
    return dict(rows=len(y), wape=float(err.sum()/y.sum()) if y.sum() else None,
        mae=float(err.mean()) if len(y) else None, total=float(y.sum()), bias=float((p-y).sum()))


def evaluate(data, out):
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((data/'manifest.json').read_text())
    for name, digest in manifest['outputs'].items():
        if sha(data/name) != digest:
            raise ValueError('Prepared data hash mismatch')
    if manifest['inputs'].get(str(PROTOCOL.relative_to(ROOT))) != sha(PROTOCOL):
        raise ValueError('Protocol changed since preparation')
    f = pd.read_csv(data/'observations.csv', sep=';')
    if f.duplicated(KEYS).any() or len(f) != 304*96:
        raise ValueError('Invalid dataset grid')
    rows, coverage, breakdown, input_paths, selected = [], [], [], [], None
    for cutoff in CUTOFFS:
        start = pd.Timestamp(cutoff)+pd.Timedelta(days=1)
        end = pd.Timestamp(cutoff)+pd.Timedelta(days=61)
        keys = grid(start, end)
        profiles = fit(f, cutoff)
        write_json(out/f'profiles_{cutoff}.json', {h: {k: v.tolist() for k, v in p.items()} for h, p in profiles.items()})
        path = CONTROL/f'raw_{cutoff}.csv'
        base = pd.read_csv(path, sep=';')
        base = base.loc[base.route.eq(12)]
        input_paths.append(path)
        if set(base[['date', 'hour']].itertuples(index=False, name=None)) != set(keys[['date', 'hour']].itertuples(index=False, name=None)):
            raise ValueError('Incomplete 030 horizon')
        pred = keys.copy()
        for model in MODELS:
            pred[model] = predict(keys, profiles, base, model)
            if model != 'direct_calendar':
                sums = pred.groupby(['date', 'hour'])[model].sum()
                expected = base.set_index(['date', 'hour']).prediction.reindex(sums.index)
                np.testing.assert_array_equal(sums, np.floor(expected+0.5).astype('int64'))
        pred['cutoff'] = cutoff
        pred['model_version'] = VERSION
        pred['dataset_version'] = VERSION+'-'+sha(data/'observations.csv')[:12]
        pred['selected_model'] = selected if selected else 'selection_window'
        pred.to_csv(out/f'predictions_{cutoff}.csv', sep=';', index=False)
        if cutoff == '2025-10-31':
            final = pred[['route', 'date', 'hour', 'quarter', 'working_minutes', 'cutoff', 'model_version', 'dataset_version', 'selected_model']].copy()
            final['prediction'] = pred[selected]
            final.to_csv(out/'forecast.csv', sep=';', index=False)
            continue
        scored = pred.merge(f[KEYS+['observed_counter', 'complete_hour', 'working_events']], on=KEYS, validate='one_to_one')
        full = scored.working_minutes.gt(0) & scored.complete_hour
        single = scored.working_minutes.gt(0) & scored.observed_counter.notna()
        coverage.append(dict(cutoff=cutoff, working_quarters=int(scored.working_minutes.gt(0).sum()),
            observed_quarters=int(single.sum()), complete_hour_quarters=int(full.sum()),
            complete_hours=len(scored.loc[full, ['date', 'hour']].drop_duplicates()),
            scored_event_fraction=float(scored.loc[full, 'observed_counter'].sum()/scored.observed_counter.sum())))
        actual_hour = scored.groupby(['date', 'hour']).observed_counter.sum(min_count=1).fillna(0).rename('prediction').reset_index()
        for model in MODELS:
            row = dict(cutoff=cutoff, model=model, quarter=metric(scored.loc[full, 'observed_counter'], scored.loc[full, model]),
                observed_quarters=metric(scored.loc[single, 'observed_counter'], scored.loc[single, model]))
            hourly = scored.loc[full].groupby(['date', 'hour'])[['observed_counter', model]].sum()
            row['hour'] = metric(hourly.observed_counter, hourly[model])
            if model != 'direct_calendar':
                oracle = predict(keys, profiles, actual_hour, model)
                row['oracle_distribution_only'] = metric(scored.loc[full, 'observed_counter'], oracle[full])
            rows.append(row)
            distance = (pd.to_datetime(scored.date)-pd.Timestamp(cutoff)).dt.days
            groups = dict(hour=scored.hour, daytype=scored.weekend,
                          horizon=pd.cut(distance, [0, 14, 35, 61], labels=['1-14', '15-35', '36-61']))
            for dimension, labels in groups.items():
                for label in labels.unique():
                    m = full & labels.eq(label)
                    if m.any():
                        breakdown.append(dict(cutoff=cutoff, model=model, dimension=dimension, value=str(label),
                            **metric(scored.loc[m, 'observed_counter'], scored.loc[m, model])))
        if selected is None:
            selected = min(rows, key=lambda r: r['quarter']['wape'])['model']
        print(cutoff, 'selected:', selected, [(r['model'], round(r['quarter']['wape'], 4)) for r in rows if r['cutoff'] == cutoff], flush=True)
    result = dict(version=VERSION, selected_model=selected, selection_cutoff=CUTOFFS[0],
        evaluation_status='previously_seen_historical_windows', target='registered_successful_validations_per_quarter_hour',
        stop_accuracy=None, metrics=rows, coverage=coverage)
    write_json(out/'results.json', result)
    write_json(out/'breakdown.json', breakdown)
    input_paths += [Path(__file__), PROTOCOL, HERE/'audit.py', LAB/'pipeline.py', data/'manifest.json', data/'observations.csv']
    write_json(out/'manifest.json', dict(version=VERSION,
        inputs={str(p.resolve().relative_to(ROOT)) if p.resolve().is_relative_to(ROOT) else str(p.resolve()): sha(p) for p in input_paths},
        outputs={p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'evaluate'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--data', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.out)
    elif args.data is None:
        parser.error('evaluate requires --data')
    else:
        evaluate(args.data, args.out)
