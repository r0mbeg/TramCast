"""Checks for the scenario export and portable verification of its data files."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from build_app_package import HOURS, KEY, allocate, profile_table, validate_route_forecast
from audit import sha


def rejected(function, *args):
    try:
        function(*args)
    except (ValueError, AssertionError):
        return
    raise AssertionError('Invalid input accepted')


def check():
    keys = [(12, 0, 'a', 1, 'same'), (12, 0, 'a', 2, 'same'), (12, 1, 'b', 1, 'other')]
    assert allocate(2, [1, 1, 1], keys) == [1, 1, 0]  # repeated stop ID, distinct positions
    assert allocate(0, [1, 1, 1], keys) == [0, 0, 0]
    assert allocate(1, [1, 0, 0], keys) == [1, 0, 0]
    total = 2**63-1
    q, r = divmod(total, 3)
    assert allocate(total, [1, 1, 1], keys) == [q+int(i < r) for i in range(3)]
    for value in [-1, True, 3.5, 2**63]:
        rejected(allocate, value, [1, 1, 1], keys)
    for weights in [[0, 0, 0], [1, -1, 1], [1, float('nan'), 1], [1, float('inf'), 1]]:
        rejected(allocate, 3, weights, keys)
    rejected(allocate, 1, [1, 1, 1], [keys[0]]*3)
    rng = np.random.default_rng(27)
    for _ in range(100):
        weights = rng.dirichlet(np.ones(3))
        n = int(rng.integers(0, 100000))
        counts = allocate(n, weights, keys)
        assert sum(counts) == n and min(counts) >= 0
        order = rng.permutation(3)
        reordered = allocate(n, weights[order], [keys[i] for i in order])
        assert dict(zip(keys, counts)) == dict(zip([keys[i] for i in order], reordered))
    rows = pd.MultiIndex.from_product([pd.date_range('2025-11-01', '2025-12-31').strftime('%Y-%m-%d'), range(24)],
        names=['date', 'hour']).to_frame(index=False).assign(route=12, prediction=0)
    validate_route_forecast(rows)
    rejected(validate_route_forecast, rows.iloc[:-1])
    rejected(validate_route_forecast, pd.concat([rows, rows.iloc[:1]]))
    wrong = rows.copy(); wrong.loc[wrong.hour.eq(1), 'prediction'] = 1
    rejected(validate_route_forecast, wrong)
    positions = pd.DataFrame(keys, columns=KEY).assign(occurrence_id=['p1', 'p2', 'p3'],
        stop_lat=55., stop_lon=37., start_date='2025-12-20', actual_date='2026-02-05', stop_mode='1')
    source = pd.concat([positions.assign(weekend=w, hour=h, prior='uniform', strength=s, share=1/3)
        for w in [0, 1] for h in HOURS for s in [0, 1]], ignore_index=True)
    assert len(profile_table(source, positions)) == 3*40
    rejected(profile_table, source.iloc[1:], positions)
    rejected(profile_table, source, positions.iloc[:-1])
    changed = positions.copy(); changed.loc[0, 'stop_lat'] += .01
    rejected(profile_table, source, changed)
    print('Passed allocation, int64, repeated occurrences, stable ties and forecast-grid checks')


def verify(folder):
    manifest = json.loads((folder/'manifest.json').read_text())
    expected_files = {'occurrences.csv', 'profiles.csv', 'route_forecast.csv', 'stop_estimates.csv.gz',
                      'route_balance.csv', 'metadata.json', 'sensitivity.json'}
    assert set(manifest['outputs']) == expected_files
    for name, digest in manifest['outputs'].items():
        assert sha(folder/name) == digest, name
    metadata = json.loads((folder/'metadata.json').read_text())
    assert metadata['package_version'] == manifest['package_version']
    assert metadata['status'] == 'scenario_estimate_unvalidated'
    assert metadata['stop_wape'] is None and metadata['stop_mae'] is None
    assert metadata['real_service_coverage'] is None and metadata['verified_stop_observation_coverage'] == 0
    assert not metadata['occupancy_available'] and metadata['parameter_selection'] == 'fixed_scenario_not_accuracy_selected'
    dtype = {'pattern_key': str, 'source_stop_id': str, 'source_route_id': str}
    positions = pd.read_csv(folder/'occurrences.csv', sep=';', dtype=dtype)
    profiles = pd.read_csv(folder/'profiles.csv', sep=';', dtype=dtype, float_precision='round_trip')
    route = pd.read_csv(folder/'route_forecast.csv', sep=';')
    estimates = pd.read_csv(folder/'stop_estimates.csv.gz', sep=';')
    balance = pd.read_csv(folder/'route_balance.csv', sep=';')
    assert len(positions) == 97 and not positions.duplicated(['route', 'pattern_key', 'stop_sequence']).any()
    assert not positions.occurrence_id.duplicated().any()
    assert positions.groupby('direction_id').size().to_dict() == {0: 50, 1: 47}
    assert positions.boarding_eligibility.eq('assumed_unverified').all()
    assert positions.historical_validity_verified.eq(False).all()
    assert len(profiles) == 97*40 and not profiles.duplicated(['weekend', 'hour', 'occurrence_id']).any()
    assert profiles[['source_share', 'prior_share', 'share']].gt(0).all().all()
    np.testing.assert_allclose(profiles.groupby(['weekend', 'hour'])[['source_share', 'prior_share', 'share']].sum(), 1, atol=1e-12)
    np.testing.assert_allclose(profiles.prior_share, 1/97)
    # Independent closed-form solution of min KL(p||s)+KL(p||q).
    root = np.sqrt(profiles.source_share*profiles.prior_share)
    expected = root/root.groupby([profiles.weekend, profiles.hour]).transform('sum')
    np.testing.assert_allclose(profiles.share, expected, atol=1e-14)
    validate_route_forecast(route)
    assert len(estimates) == 97*1464 and not estimates.duplicated(['date', 'hour', 'occurrence_id']).any()
    assert set(estimates.occurrence_id) == set(positions.occurrence_id)
    assert np.isfinite(estimates[['share', 'expected_validations', 'estimated_validations']]).all().all()
    assert estimates[['share', 'expected_validations', 'estimated_validations']].ge(0).all().all()
    assert pd.api.types.is_integer_dtype(estimates.estimated_validations.dtype)
    joined = estimates.merge(route, on=['route', 'date', 'hour'], validate='many_to_one', how='left')
    assert joined.prediction.notna().all()
    assert np.allclose(joined.expected_validations, joined.prediction*joined.share, atol=1e-10)
    assert (joined.estimated_validations-joined.expected_validations).abs().max() < 1+1e-10
    sums = estimates.groupby(['route', 'date', 'hour']).estimated_validations.agg(['size', 'sum'])
    assert sums['size'].eq(97).all()
    np.testing.assert_array_equal(sums['sum'], route.set_index(['route', 'date', 'hour']).prediction.reindex(sums.index))
    night = estimates.hour.between(1, 4)
    assert estimates.loc[night, ['share', 'expected_validations', 'estimated_validations']].eq(0).all().all()
    assert estimates.loc[night, 'estimate_status'].eq('structural_zero').all()
    assert estimates.loc[~night, 'estimate_status'].eq('scenario_estimate').all()
    active = estimates.loc[~night].copy()
    active['weekend'] = pd.to_datetime(active.date).dt.dayofweek.ge(5).astype(int)
    active = active.merge(profiles[['weekend', 'hour', 'occurrence_id', 'share']],
        on=['weekend', 'hour', 'occurrence_id'], validate='many_to_one', suffixes=('', '_profile'), how='left')
    np.testing.assert_allclose(active.share, active.share_profile, atol=1e-14)
    contexts = {(int(w), int(h)): p.sort_values(KEY) for (w, h), p in profiles.groupby(['weekend', 'hour'])}
    for (date, hour), group in estimates.loc[~night].groupby(['date', 'hour']):
        profile = contexts[(int(pd.Timestamp(date).dayofweek >= 5), hour)]
        total = int(route.loc[route.date.eq(date) & route.hour.eq(hour), 'prediction'].iloc[0])
        counts = allocate(total, profile.share.to_numpy(), list(profile[KEY].itertuples(index=False, name=None)))
        saved_counts = group.set_index('occurrence_id').estimated_validations.reindex(profile.occurrence_id)
        np.testing.assert_array_equal(counts, saved_counts)
    assert len(balance) == 1464 and balance.unassigned_prediction.eq(0).all()
    assert (balance.allocated_prediction+balance.unassigned_prediction).equals(balance.route_prediction)
    # A missing mapping retains its allocated integer mass as unassigned, not as new stop demand.
    missing = set(positions.occurrence_id.iloc[::10])
    kept = estimates.loc[~estimates.occurrence_id.isin(missing)].groupby(['date', 'hour']).estimated_validations.sum()
    rest = estimates.loc[estimates.occurrence_id.isin(missing)].groupby(['date', 'hour']).estimated_validations.sum()
    np.testing.assert_array_equal(kept+rest, route.set_index(['date', 'hour']).prediction.reindex(kept.index))
    print('Verified package hashes, closed-form entropy solution, 142008 estimates, 1464 exact balances, masks and partial mapping')
    return dict(package_version=metadata['package_version'], package_manifest_sha256=sha(folder/'manifest.json'),
        estimates=len(estimates), route_hours=len(route), total=int(estimates.estimated_validations.sum()),
        source_checksums='verified_by_builder', output_checksums='passed', analytic_entropy_solution='passed',
        exact_balance='passed', structural_zeros='passed', partial_mapping_balance='passed',
        exported_profile_integer_replay='passed',
        empirical_stop_accuracy='unavailable')


if __name__ == '__main__':
    check()
    if len(sys.argv) == 2:
        print(json.dumps(verify(Path(sys.argv[1])), indent=2))
    elif len(sys.argv) != 1:
        raise SystemExit('Usage: test_app_package.py [PACKAGE_DIRECTORY]')
