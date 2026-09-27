"""Run directly, optionally verify a complete released network package."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from audit import sha
from network_package import (HERE, HOURS, KEY, ROUTES, FITTED, allocate, checksums, geography, make_profiles,
                             metro_prior, project, read_csv, validate_positions, validate_routes)


def rejects(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except (ValueError, KeyError):
        return
    raise AssertionError('Invalid input accepted')


def check():
    assert checksums([HERE/'..'/'stop_forecast'/'network_package.py']) == checksums([HERE/'network_package.py'])
    active = np.array([True, False, True])
    q = metro_prior([0, 0, 1000], active)
    assert q[1] == 0 and .25 <= q[2] < q[0] <= .75 and np.isclose(q.sum(), 1)
    np.testing.assert_allclose(metro_prior([0, 0, 1000], active, alpha=0), [.5, 0, .5])
    np.testing.assert_allclose(metro_prior([500, 0, 500], active), [.5, 0, .5])
    p = project([.8, 0, .2], q, active)
    expected = np.sqrt(np.array([.8, 0, .2])*q)
    np.testing.assert_allclose(p, expected/expected.sum())
    assert p[1] == 0
    for distance in [[0, 0, np.nan], [-1, 0, 0], [0, 0, np.inf]]:
        rejects(metro_prior, distance, active)
    rejects(metro_prior, [0, 1, 2], np.array([False]*3))
    rejects(metro_prior, [0, 1, 2], active, alpha=1)
    rejects(metro_prior, [0, 1, 2], active, scale=0)
    p = geography()
    validate_positions(p)
    assert p.groupby('route').size().to_dict() == {1: 44, 5: 32, 7: 89, 11: 85, 12: 97,
        17: 52, 25: 41, 26: 51, 28: 30, 50: 83}
    assert p.loc[p.route.eq(28) & p.direction_id.eq(0), 'pattern_key'].unique().tolist() == ['osm:relation/3184023']
    assert p.loc[p.geography_source.eq('openstreetmap'), 'source_stop_id'].str.startswith('osm:node/').all()
    rejects(validate_positions, p.iloc[:-1])
    invalid = p.copy(); invalid.loc[0, 'stop_lat'] = np.nan
    rejects(validate_positions, invalid)
    profiles = []
    for route, pos in p.loc[p.route.isin(FITTED)].groupby('route'):
        for w in [0, 1]:
            for h in HOURS:
                profiles.append(pos[['route', 'occurrence_id']].assign(weekend=w, hour=h, source_share=1/len(pos)))
    source = pd.concat(profiles, ignore_index=True)
    p['nearest_metro_entrance_distance_m'] = np.arange(len(p), dtype=float)
    fitted, _ = make_profiles(p, source)
    assert len(fitted) == len(p)*40
    rejects(make_profiles, p, source.iloc[1:])
    bad = source.copy(); bad.loc[0, 'source_share'] = -1
    rejects(make_profiles, p, bad)
    routes = pd.MultiIndex.from_product([ROUTES, pd.date_range('2025-11-01', '2025-12-31').strftime('%Y-%m-%d'), range(24)],
        names=['route', 'date', 'hour']).to_frame(index=False).assign(prediction=0)
    validate_routes(routes)
    rejects(validate_routes, routes.iloc[1:])
    wrong = routes.copy(); wrong.loc[wrong.route.eq(5), 'prediction'] = 1
    rejects(validate_routes, wrong)
    wrong = routes.copy(); wrong.loc[wrong.hour.eq(1), 'prediction'] = 1
    rejects(validate_routes, wrong)
    print('Network checks passed: 10-route keys, direction mapping, prior, exit-only and invalid inputs')


def verify(folder):
    manifest = json.loads((folder/'manifest.json').read_text())
    expected = {'occurrences.csv', 'profiles.csv.gz', 'route_forecast.csv', 'route_balance.csv',
                'coverage.csv', 'sensitivity.csv', 'stop_estimates.csv.gz', 'metadata.json'}
    assert set(manifest['outputs']) == expected
    for name, digest in manifest['outputs'].items():
        assert sha(folder/name) == digest, name
    meta = json.loads((folder/'metadata.json').read_text())
    assert meta['package_version'] == manifest['package_version']
    assert meta['route_numbers'] == ROUTES and meta['status'] == 'scenario_estimate_unvalidated'
    assert meta['stop_wape'] is None and meta['stop_mae'] is None and meta['real_service_coverage'] is None
    assert meta['verified_stop_observation_coverage'] == 0 and meta['occupancy_available'] is False
    pos, profiles = read_csv(folder/'occurrences.csv'), read_csv(folder/'profiles.csv.gz')
    routes, balance = read_csv(folder/'route_forecast.csv'), read_csv(folder/'route_balance.csv')
    estimates = read_csv(folder/'stop_estimates.csv.gz')
    validate_positions(pos); validate_routes(routes)
    assert len(pos) == 604 and len(profiles) == 604*40 and len(estimates) == 604*1464
    assert not profiles.duplicated(['route', 'weekend', 'hour', 'occurrence_id']).any()
    assert not estimates.duplicated(['route', 'date', 'hour', 'occurrence_id']).any()
    assert profiles.share.ge(0).all() and profiles.prior_share.ge(0).all()
    context = ['route', 'weekend', 'hour']
    np.testing.assert_allclose(profiles.groupby(context)[['share', 'prior_share', 'source_share']].sum(), 1, atol=1e-12)
    for route, g in pos.groupby('route'):
        profile = profiles.loc[profiles.route.eq(route)].merge(
            g[['occurrence_id', 'boarding_allowed', 'nearest_metro_entrance_distance_m']],
            on='occurrence_id', validate='many_to_one')
        for _, frame in profile.groupby(['weekend', 'hour']):
            active = frame.boarding_allowed.to_numpy(bool)
            d = frame.nearest_metro_entrance_distance_m.to_numpy()
            a = np.exp(-d[active]/500)
            q = np.zeros(len(frame)); q[active] = .5/active.sum()+.5*a/a.sum()
            np.testing.assert_allclose(frame.prior_share, q, rtol=1e-12, atol=1e-14)
            p = np.sqrt(frame.source_share.to_numpy()*q); p /= p.sum()
            np.testing.assert_allclose(frame.share, p, rtol=1e-12, atol=1e-14)
    linked = estimates.merge(pos[['route', 'occurrence_id', 'boarding_allowed']],
                            on=['route', 'occurrence_id'], how='left', validate='many_to_one')
    assert linked.boarding_allowed.notna().all()
    assert estimates.estimated_validations.ge(0).all() and pd.api.types.is_integer_dtype(estimates.estimated_validations)
    assert np.isfinite(estimates[['share', 'expected_validations']]).all().all()
    counts = estimates.groupby(['route', 'date', 'hour']).size()
    assert all(count == pos.route.eq(route).sum() for (route, _, _), count in counts.items())
    summed = estimates.groupby(['route', 'date', 'hour']).estimated_validations.sum()
    reference = routes.set_index(['route', 'date', 'hour']).prediction.sort_index()
    pd.testing.assert_series_equal(summed.sort_index(), reference, check_names=False)
    pd.testing.assert_series_equal(balance.set_index(['route', 'date', 'hour']).route_prediction.sort_index(), reference, check_names=False)
    pd.testing.assert_series_equal(balance.set_index(['route', 'date', 'hour']).allocated_prediction.sort_index(), reference, check_names=False)
    assert balance.unassigned_prediction.eq(0).all()
    zero = linked.route.eq(5) | linked.hour.between(1, 4) | ~linked.boarding_allowed
    assert linked.loc[zero, ['estimated_validations', 'expected_validations']].eq(0).all().all()
    assert linked.loc[linked.route.eq(5), 'estimate_status'].eq('route5_no_history_zero_fallback').all()
    assert linked.loc[~linked.route.eq(5) & linked.hour.between(1, 4), 'estimate_status'].eq('structural_zero_nonworking').all()
    assert linked.loc[~linked.boarding_allowed & ~linked.hour.between(1, 4), 'estimate_status'].eq('scenario_zero_exit_only').all()
    # Read exported profiles and reproduce every integer allocation, including tie ordering.
    cache = {}
    for (route, weekend, hour), f in profiles.groupby(context):
        f = f.sort_values(KEY)
        cache[(route, weekend, hour)] = (f, list(f[KEY].itertuples(index=False, name=None)))
    for (route, date, hour), f in estimates.groupby(['route', 'date', 'hour']):
        if hour in range(1, 5):
            assert f.share.eq(0).all()
            continue
        profile, keys = cache[(route, int(pd.Timestamp(date).dayofweek >= 5), hour)]
        f = f.set_index('occurrence_id').loc[profile.occurrence_id]
        np.testing.assert_allclose(f.share, profile.share, rtol=0, atol=0)
        np.testing.assert_allclose(f.expected_validations, float(reference.loc[(route, date, hour)])*profile.share, rtol=1e-14)
        assert f.estimated_validations.tolist() == allocate(int(reference.loc[(route, date, hour)]), profile.share, keys)
    # With a partial app mapping, retain the integer remainder; never redistribute it.
    mapped = estimates.occurrence_id.isin(pos.occurrence_id.iloc[::3])
    partial = estimates.loc[mapped].groupby(['route', 'date', 'hour']).estimated_validations.sum().reindex(reference.index, fill_value=0)
    unassigned = reference-partial
    assert unassigned.ge(0).all() and (partial+unassigned).equals(reference)
    old = read_csv(HERE/'releases/route12-entropy-v1/profiles.csv')
    current = profiles.loc[profiles.route.eq(12)].merge(old[KEY+['weekend', 'hour', 'share']],
        on=KEY+['weekend', 'hour'], validate='one_to_one', suffixes=('', '_old'))
    old_p = np.sqrt(current.source_share.to_numpy()/97); old_p /= np.repeat(
        np.sqrt(current.source_share.to_numpy()/97).reshape(40, 97).sum(axis=1), 97)
    np.testing.assert_allclose(old_p, current.share_old, rtol=1e-12, atol=1e-14)
    result = dict(package_version=meta['package_version'], positions=len(pos), estimates=len(estimates),
        route_hours=len(routes), total=int(routes.prediction.sum()), exit_only_positions=int((~pos.boarding_allowed).sum()),
        hashes='passed', exact_balances='passed', entropy_formula='passed', integer_replay='passed',
        route12_uniform_control='matches_preserved_release', empirical_stop_accuracy=None)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    if len(sys.argv) > 2:
        raise SystemExit('Usage: test_network_package.py [RELEASE_DIRECTORY]')
    check()
    if len(sys.argv) == 2:
        verify(Path(sys.argv[1]))
