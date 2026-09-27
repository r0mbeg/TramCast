"""Frozen train/validation/test experiment for long, variable-tempo traces."""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from activity_model import fit
from audit import ROOT, sha, write_json
from build_metro_access import distances
from elastic_model import activity_chain, geographic_chain, predict, scores, WINDOWS

HERE = Path(__file__).resolve().parent
HISTORY = HERE / 'runs/historical-geometry-20260927-v1'


def checked(folder, name, inputs):
    manifest = folder / 'manifest.json'
    path = folder / name
    if sha(path) != json.loads(manifest.read_text())['outputs'][name]:
        raise ValueError(f'Changed artifact: {path}')
    inputs.extend([manifest, path])
    return path


def load(data):
    inputs = []
    meta = pd.read_csv(checked(data, 'blocks.csv', inputs), sep=';')
    with np.load(checked(data, 'counts.npz', inputs), allow_pickle=False) as arrays:
        reference, held = arrays['reference'], arrays['held']
    if (reference.shape != held.shape or reference.shape != (len(meta), 1080)
            or not np.isfinite(reference).all() or not np.isfinite(held).all()
            or (reference < 0).any() or (held < 0).any()
            or (reference != np.floor(reference)).any() or (held != np.floor(held)).any()
            or not meta.selected.isin([True, False]).all()):
        raise ValueError('Invalid aligned data')
    expected = np.where(meta.date.le('2025-09-19'), 'train', np.where(meta.date.eq('2025-09-22'), 'validation', 'test'))
    if not np.array_equal(meta.split, expected) or not meta.date.between('2025-09-15', '2025-09-26').all():
        raise ValueError('Changed temporal split')
    return meta, reference, held, inputs


def networks(inputs):
    g = pd.read_csv(checked(HISTORY, 'historical_occurrences.csv', inputs), sep=';').sort_values(['direction', 'stop_sequence'])
    if (len(g) != 91 or g.occurrence_id.duplicated().any()
            or g.groupby('direction').size().to_dict() != {0: 46, 1: 45}
            or not g.boarding_permitted_by_osm.isin([True, False]).all()):
        raise ValueError('Changed geographical support')
    nxt = np.roll(np.arange(len(g)), -1)
    edges = np.diag(distances(g.stop_lat, g.stop_lon, g.stop_lat.iloc[nxt], g.stop_lon.iloc[nxt])).copy()
    direction = g.direction.to_numpy()
    turn = direction != direction[nxt]
    equal = edges.copy(); equal[~turn] = edges[~turn].mean()
    result = {'historical_dynamic': edges, 'equal_spacing': equal}
    rng = np.random.default_rng(20260929)
    for i in range(19):
        value = edges.copy()
        for d in (0, 1):
            mask = (direction == d) & ~turn
            value[mask] = rng.permutation(edges[mask])
        result[f'permuted_{i:02}'] = value
    return result, turn, g.boarding_permitted_by_osm.to_numpy(bool)


def finish_manifest(out, inputs):
    inputs += [HERE / n for n in ('elastic_experiment.py', 'elastic_model.py', 'ELASTIC_PROTOCOL.md',
        'test_elastic.py', 'build_elastic_data.py', 'activity_model.py', 'duration_model.py', 'audit.py', 'build_metro_access.py')]
    write_json(out / 'manifest.json', dict(environment=dict(python=platform.python_version(), numpy=np.__version__,
        scipy=scipy.__version__, pandas=pd.__version__), inputs={str(p.relative_to(ROOT)): sha(p) for p in inputs},
        outputs={p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name != 'manifest.json'}))


def train(data, out):
    meta, reference, _, inputs = load(data)
    layouts, turn, board = networks(inputs)
    train_ids = meta.index[meta.selected & meta.split.eq('train')].to_numpy()
    val_ids = meta.index[meta.selected & meta.split.eq('validation')].to_numpy()
    if not len(train_ids) or not len(val_ids):
        raise ValueError('Empty fit or validation split')
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    theta, diagnostics = fit(reference[train_ids])
    write_json(out / 'parameters.json', dict(theta=theta.tolist(), diagnostics=diagnostics,
        training_blocks=train_ids.tolist(), training_reference_events=int(reference[train_ids].sum()),
        training_dates=sorted(meta.loc[train_ids, 'date'].unique()), elapsed_seconds=time.monotonic() - started))
    candidates = []
    for persistence in (0., .9):
        chain, _ = geographic_chain(layouts['equal_spacing'], turn, board, theta, persistence)
        values = [scores(reference[i], predict(reference[i], chain)) for i in val_ids]
        candidates.append(dict(persistence=persistence, short_reference_score=sum(v['short_score'] for v in values),
                               short_reference_count=sum(v['short_count'] for v in values)))
    if not candidates[0]['short_reference_count']:
        raise ValueError('Validation has no hidden reference events')
    chosen = max(candidates, key=lambda x: x['short_reference_score'])['persistence']
    write_json(out / 'selection.json', dict(persistence=chosen, candidates=candidates,
        criterion='masked reference short log score on equal-spacing, no held/test targets',
        validation_date='2025-09-22', validation_blocks=val_ids.tolist(), frozen_before_test=True))
    finish_manifest(out, inputs)
    print('Frozen selection:', json.dumps(candidates), 'selected', chosen, flush=True)


def summarize(rows):
    frame = pd.DataFrame(rows)
    results = {}
    for model, g in frame.groupby('model'):
        results[model] = {}
        for device in ('held', 'reference'):
            for horizon in ('short', 'tail'):
                key = f'{device}_{horizon}'
                count = int(g[f'{key}_count'].sum())
                results[model][key] = dict(count=count,
                    log_score_per_event=float(g[f'{key}_score'].sum() / count) if count else None)
    pivot = frame.pivot(index='block', columns='model', values='held_short_score')
    nulls = [f'permuted_{i:02}' for i in range(19)]
    actual = frame.loc[frame.model.eq('historical_dynamic')].set_index('block').loc[pivot.index]
    count = actual.held_short_count.to_numpy()
    if count.sum() <= 0:
        raise ValueError('No held test events in primary windows')
    delta = pivot.historical_dynamic.to_numpy() - pivot[nulls].mean(axis=1).to_numpy()
    clustered = pd.DataFrame({'vehicle_day': actual.vehicle_day.to_numpy(), 'delta': delta, 'count': count}).groupby('vehicle_day')[['delta', 'count']].sum().to_numpy()
    rng = np.random.default_rng(20260928)
    samples = clustered[rng.integers(len(clustered), size=(1000, len(clustered)))].sum(axis=1)
    nonzero = samples[:, 1] > 0
    interval = np.quantile(samples[nonzero, 0] / samples[nonzero, 1], [.025, .975]).tolist()
    score = results['historical_dynamic']['held_short']['log_score_per_event']
    comparisons = {m: float(score - v['held_short']['log_score_per_event']) for m, v in results.items()}
    beaten = sum(comparisons[m] > 0 for m in nulls)
    daily = {}
    for day, day_frame in frame.groupby('date'):
        daily[day] = {model: float(g.held_short_score.sum() / g.held_short_count.sum()) if g.held_short_count.sum() else None
                      for model, g in day_frame.groupby('model')}
    return dict(version='EOT20260927-v1', models=results, primary_held_short_gain=comparisons,
        held_short_gain_vs_mean_permutation=float(delta.sum() / count.sum()),
        descriptive_vehicle_day_bootstrap_95=interval, historical_beats_permutations=beaten,
        promotion_gate_passed=bool(beaten == 19 and interval[0] > 0
            and all(comparisons[m] > 0 for m in ('uniform', 'activity', 'equal_spacing'))),
        test_blocks=len(actual), test_vehicle_days=int(actual.vehicle_day.nunique()),
        daily_primary_log_scores=daily, stop_mae=None, stop_wape=None, verified_stop_assignments=0,
        status='causal_temporal_validation_not_stop_forecast')


def evaluate(data, fitted, out):
    meta, reference, held, inputs = load(data)
    layouts, turn, board = networks(inputs)
    params = json.loads(checked(fitted, 'parameters.json', inputs).read_text())
    selection = json.loads(checked(fitted, 'selection.json', inputs).read_text())
    theta, persistence = np.asarray(params['theta']), selection['persistence']
    for name, digest in json.loads((fitted / 'manifest.json').read_text())['inputs'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Training provenance changed: ' + name)
    test = meta.index[meta.selected & meta.split.eq('test')].to_numpy()
    if not len(test) or not meta.loc[test, 'date'].between('2025-09-23', '2025-09-26').all():
        raise ValueError('Invalid held time split')
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    chains = {'activity': activity_chain(theta), 'uniform': None}
    diagnostics = {}
    for name, layout in layouts.items():
        chains[name], diagnostics[name] = geographic_chain(layout, turn, board, theta, persistence)
    chains['historical_fixed'], diagnostics['historical_fixed'] = geographic_chain(layouts['historical_dynamic'], turn, board, theta, persistence, fixed=True)
    rows = []
    for name, chain in chains.items():
        for i in test:
            density = [np.ones(end - start) / (end - start) for start, end in WINDOWS] if chain is None else predict(reference[i], chain)
            row = dict(block=int(i), vehicle_day=int(meta.iloc[i].vehicle_day), date=meta.iloc[i].date, model=name)
            for device, counts in (('held', held[i]), ('reference', reference[i])):
                row.update({f'{device}_{k}': v for k, v in scores(counts, density).items()})
            rows.append(row)
        print(json.dumps(dict(completed=name, seconds=round(time.monotonic() - started, 2))), flush=True)
        if time.monotonic() - started > 1800:
            write_json(out / 'aborted.json', dict(reason='30 minute evaluation budget exceeded', completed=name))
            raise RuntimeError('Budget exceeded')
    summary = summarize(rows)
    summary.update(persistence=persistence, elapsed_seconds=time.monotonic() - started,
        test_reference_events=int(reference[test].sum()), test_held_events=int(held[test].sum()),
        fitted_parameters_sha256=sha(fitted / 'parameters.json'), frozen_selection_sha256=sha(fitted / 'selection.json'))
    write_json(out / 'per_block_scores.json', rows)
    write_json(out / 'summary.json', summary)
    write_json(out / 'model_diagnostics.json', diagnostics)
    finish_manifest(out, inputs)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('models', 'daily_primary_log_scores')}, ensure_ascii=False))


def verify(out):
    manifest = json.loads((out / 'manifest.json').read_text())
    for name, digest in manifest['inputs'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Changed input: ' + name)
    for name, digest in manifest['outputs'].items():
        if sha(out / name) != digest:
            raise ValueError('Changed output: ' + name)
    calculated = summarize(json.loads((out / 'per_block_scores.json').read_text()))
    saved = json.loads((out / 'summary.json').read_text())
    if any(saved[k] != v for k, v in calculated.items()):
        raise ValueError('Summary differs from scores')
    print('Verified hashes and reaggregated all model scores and intervals.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('fit', 'evaluate', 'verify'))
    parser.add_argument('--data', type=Path, default=HERE / 'runs/elastic-data-20260927-v1')
    parser.add_argument('--fitted', type=Path, default=HERE / 'runs/elastic-fit-20260927-v1')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.stage == 'fit':
        train(args.data, args.out)
    elif args.stage == 'evaluate':
        evaluate(args.data, args.fitted, args.out)
    else:
        verify(args.out)
