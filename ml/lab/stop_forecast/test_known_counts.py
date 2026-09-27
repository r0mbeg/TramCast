"""Small checks of real-target preparation, rounding and cutoff isolation."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from known_counts import CONTROL, MODELS, ROOT, attach_counts, apportion, count_chunk, fit, grid, predict, sha, write_json


def check():
    raw = pd.DataFrame({
        'tran_date_time': ['2025-01-01 '+s for s in ['00:59:59', '01:00:00', '05:29:59',
            '05:30:00', '05:44:59', '05:45:00', '06:01:00']],
        'ngpt_route': ['12 трамвай']*7, 'validation_result': [1, 1, 1, 1, 1, 1, 0]})
    a, b = count_chunk(raw)
    f = attach_counts(grid('2025-01-01', '2025-01-01'), a, b)
    assert f.registered_successes.sum() == 4 and f.working_events.sum() == 5
    assert f.loc[f.hour.eq(5), 'registered_successes'].tolist() == [0, 0, 2, 1]
    assert f.loc[f.hour.eq(5), 'complete_hour'].all()
    zero = f.loc[f.hour.eq(6) & f.quarter.eq(0)].iloc[0]
    missing = f.loc[f.hour.eq(6) & f.quarter.eq(1)].iloc[0]
    assert zero.observed_counter == 0 and zero.counter_status == 'observed_unknown_completeness'
    assert pd.isna(missing.observed_counter) and missing.counter_status == 'missing'
    assert not zero.complete_hour
    assert apportion(3.5, [0, 0, .5, .5]).tolist() == [0, 0, 2, 2]
    assert apportion(5, [.25]*4).tolist() == [2, 1, 1, 1]
    rng = np.random.default_rng(27)
    for _ in range(100):
        p = rng.dirichlet(np.ones(4))
        total = rng.uniform(0, 1000)
        out = apportion(total, p)
        assert (out >= 0).all() and out.sum() == int(np.floor(total+.5))
    for total, p in [(-1, [.25]*4), (1, [.1]*4), (1, [np.nan]*4)]:
        try:
            apportion(total, p)
            raise AssertionError('Invalid allocation accepted')
        except ValueError:
            pass
    train = grid('2025-01-01', '2025-03-05')
    train['complete_hour'] = True
    train['registered_successes'] = np.where(train.working_minutes.gt(0), 10+train.quarter*10, 0)
    first = fit(train, '2025-02-28')
    train.loc[train.date.gt('2025-02-28'), 'registered_successes'] = 999999
    train.loc[train.date.gt('2025-02-28'), 'complete_hour'] = False
    second = fit(train, '2025-02-28')
    for hour in first:
        for key in first[hour]:
            np.testing.assert_array_equal(first[hour][key], second[hour][key])
    assert np.allclose(first[6]['bayes'].sum(axis=1), 1)
    assert np.all(first[5]['bayes'][:, :2] == 0)
    keys = grid('2025-03-01', '2025-03-01')
    hourly = keys[['date', 'hour']].drop_duplicates().copy()
    hourly['prediction'] = np.where(hourly.hour.between(1, 4), 0, 103.5)
    values = predict(keys, first, hourly, 'bayes_030')
    np.testing.assert_array_equal(values.reshape(-1, 4).sum(axis=1), np.floor(hourly.prediction+.5))
    assert (values[keys.working_minutes.eq(0)] == 0).all()
    print('Known-count checks passed: boundaries, missing/zero, balance, prior and cutoff isolation')


def verify(data, run):
    for folder in [data, run]:
        manifest = json.loads((folder/'manifest.json').read_text())
        for name, digest in manifest['inputs'].items():
            assert sha(ROOT/name) == digest, name
        for name, digest in manifest['outputs'].items():
            assert sha(folder/name) == digest, name
    f = pd.read_csv(data/'observations.csv', sep=';')
    hourly = pd.read_csv(data/'hourly_observations.csv', sep=';').set_index(['date', 'hour'])
    totals = f.groupby(['date', 'hour']).registered_successes.sum()
    np.testing.assert_array_equal(totals, hourly.boardings.reindex(totals.index))
    assert f.observed_counter.isna().equals(f.counter_status.eq('missing'))
    assert (f.loc[f.counter_status.eq('structural_zero'), 'registered_successes'] == 0).all()
    result = json.loads((run/'results.json').read_text())
    first = [r for r in result['metrics'] if r['cutoff'] == result['selection_cutoff']]
    assert result['selected_model'] == min(first, key=lambda r: r['quarter']['wape'])['model']
    for path in sorted(run.glob('predictions_*.csv')):
        p = pd.read_csv(path, sep=';')
        cutoff = p.cutoff.iloc[0]
        assert len(p) == 61*96 and not p.duplicated(['date', 'hour', 'quarter']).any()
        assert pd.to_datetime(p.date).min() == pd.Timestamp(cutoff)+pd.Timedelta(days=1)
        assert pd.to_datetime(p.date).max() == pd.Timestamp(cutoff)+pd.Timedelta(days=61)
        for model in MODELS:
            assert np.isfinite(p[model]).all() and p[model].ge(0).all() and p[model].mod(1).eq(0).all()
            assert p.loc[p.working_minutes.eq(0), model].eq(0).all()
        base = pd.read_csv(CONTROL/f'raw_{cutoff}.csv', sep=';')
        base = base.loc[base.route.eq(12)].set_index(['date', 'hour']).prediction
        for model in MODELS[:-1]:
            sums = p.groupby(['date', 'hour'])[model].sum()
            np.testing.assert_array_equal(sums, np.floor(base.reindex(sums.index)+.5))
        # Independent replay of the selected daytype profile directly from train counts.
        start = (pd.Timestamp(cutoff)-pd.Timedelta(days=56)).strftime('%Y-%m-%d')
        train = f.loc[f.date.gt(start) & f.date.le(cutoff) & f.complete_hour]
        saved = json.loads((run/f'profiles_{cutoff}.json').read_text())
        for (hour, weekend), group in train.loc[train.working_minutes.gt(0)].groupby(['hour', 'weekend']):
            counts = group.groupby('quarter').registered_successes.sum().reindex(range(4), fill_value=0)
            np.testing.assert_allclose(saved[str(hour)]['daytype'][weekend], counts/counts.sum(), atol=1e-14)
        if cutoff != '2025-10-31':
            z = p.merge(f[['date', 'hour', 'quarter', 'observed_counter', 'complete_hour']],
                on=['date', 'hour', 'quarter'], validate='one_to_one')
            z = z.loc[z.complete_hour & z.working_minutes.gt(0)]
            for row in [r for r in result['metrics'] if r['cutoff'] == cutoff]:
                error = (z[row['model']]-z.observed_counter).abs()
                assert np.isclose(error.sum()/z.observed_counter.sum(), row['quarter']['wape'])
                assert np.isclose(error.mean(), row['quarter']['mae'])
    final = pd.read_csv(run/'forecast.csv', sep=';')
    pred = pd.read_csv(run/'predictions_2025-10-31.csv', sep=';')
    np.testing.assert_array_equal(final.prediction, pred[result['selected_model']])
    assert final.selected_model.eq(result['selected_model']).all()
    write_json(run/'verification.json', dict(manifest_sha256=sha(run/'manifest.json'),
        verifier_sha256=sha(Path(__file__)), input_and_output_hashes='passed',
        hourly_balance='passed_7296_hours', forecast_grid='passed_5856_rows',
        frozen_selection='passed', independent_profile_replay='passed', metrics_recomputed='passed'))
    print('Artifact verification passed: hashes, targets, horizon, profiles, metrics and forecast')


if __name__ == '__main__':
    check()
    if len(sys.argv) == 3:
        verify(Path(sys.argv[1]), Path(sys.argv[2]))
    elif len(sys.argv) != 1:
        raise SystemExit('Usage: test_known_counts.py [DATA_DIR RUN_DIR]')
