"""Prepare and build the ten-route scenario; never infer observed stop labels."""
import argparse
import gzip
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit import ROOT, prior, sha, write_json
from build_app_package import HOURS, KEY, allocate, fingerprint
from build_metro_access import distances
from fractional_split import entropy_projection

HERE = Path(__file__).resolve().parent
ROUTES = [1, 5, 7, 11, 12, 17, 25, 26, 28, 50]
FITTED = [1, 7, 11, 12]
POSITION_COUNTS = {1: 44, 5: 32, 7: 89, 11: 85, 12: 97, 17: 52, 25: 41, 26: 51, 28: 30, 50: 83}
# Same order as catalog_service.OSMRoutes; directions are not sorted relation IDs.
OSM = {17: [540033, 540139], 25: [3186264, 3186265], 26: [1689026, 1689064],
       28: [3184023, 3184022], 50: [1538169, 1538170]}
PROTOCOL = HERE/'NETWORK_PROTOCOL.md'
BUNDLE = ROOT/'ml/bundles/030'
DTYPES = {c: str for c in ['source_route_id', 'pattern_key', 'source_stop_id', 'stop_mode']}


def read_csv(path):
    return pd.read_csv(path, sep=';', dtype=DTYPES, keep_default_na=False, float_precision='round_trip')


def save_csv(frame, path):
    if path.suffix == '.gz':
        with path.open('wb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding='utf-8', newline='') as text:
                frame.to_csv(text, sep=';', index=False, lineterminator='\n')
    else:
        frame.to_csv(path, sep=';', index=False, lineterminator='\n')


def checksums(paths):
    return {str(p.resolve().relative_to(ROOT)): sha(p) for p in paths}


def geography():
    book = ROOT/'data/catalog/catalog.xlsx'
    tables = {name: pd.DataFrame(rows) for name, _, _, rows in prior.sheets(book)}
    f = tables['Порядок_остановок GTFS_TRIPS_ST'].merge(
        tables['Остановки GTFS_STOPS'][['stop_id', 'stop_name', 'stop_lat', 'stop_lon']],
        on='stop_id', how='left', validate='many_to_one').rename(columns={
            'route_short_name': 'route', 'route_id': 'source_route_id',
            'trip_id': 'pattern_key', 'stop_id': 'source_stop_id'})
    f['route'] = pd.to_numeric(f.route, errors='raise')
    f = f.loc[f.route.isin(ROUTES)].copy()
    for col in ['direction_id', 'stop_sequence']:
        f[col] = pd.to_numeric(f[col], errors='raise')
    if f.end_date.ne('').any() or f.is_addpoint.ne('0').any():
        raise ValueError('Review changed workbook activity/support')
    f['occurrence_id'] = 'catalog:'+f.route.astype(str)+':'+f.pattern_key+':'+f.stop_sequence.astype(str)
    f['geography_source'] = 'organizer_catalog'
    f['boarding_role'] = 'assumed_unverified'
    f['boarding_allowed'] = True
    f['network_snapshot'] = f.actual_date
    cols = ['occurrence_id', 'source_route_id']+KEY+['stop_name', 'stop_lat', 'stop_lon',
        'start_date', 'actual_date', 'stop_mode', 'geography_source', 'network_snapshot',
        'boarding_role', 'boarding_allowed']
    doc = json.loads((ROOT/'data/osm/tram_routes.json').read_text())
    nodes = {e['id']: e for e in doc['elements'] if e['type'] == 'node'}
    relations = {e['id']: e for e in doc['elements'] if e['type'] == 'relation'}
    rows = []
    for route, ids in OSM.items():
        masters = [e for e in relations.values() if e.get('tags', {}).get('type') == 'route_master'
            and e['tags'].get('ref') == str(route) and set(ids) <= {
                m['ref'] for m in e['members'] if m['type'] == 'relation'}]
        if len(masters) != 1:
            raise ValueError('Ambiguous route master')
        for direction, relation_id in enumerate(ids):
            relation = relations[relation_id]
            if relation['tags'].get('ref') != str(route) or relation['tags'].get('route') != 'tram':
                raise ValueError('Wrong OSM relation')
            members = [m for m in relation['members'] if m['type'] == 'node' and
                       m['role'] in {'stop', 'stop_entry_only', 'stop_exit_only'}]
            for sequence, m in enumerate(members, 1):
                node = nodes[m['ref']]
                if node['tags'].get('public_transport') != 'stop_position' or not node['tags'].get('name'):
                    raise ValueError('Invalid OSM stop')
                rows.append(dict(route=route, direction_id=direction, stop_sequence=sequence,
                    occurrence_id=f'osm:{route}:{relation_id}:{sequence}',
                    source_route_id=f"osm:relation/{masters[0]['id']}", pattern_key=f'osm:relation/{relation_id}',
                    source_stop_id=f"osm:node/{node['id']}", stop_name=node['tags']['name'],
                    stop_lat=node['lat'], stop_lon=node['lon'], start_date='', actual_date='', stop_mode='',
                    geography_source='openstreetmap', network_snapshot=doc['osm3s']['timestamp_osm_base'],
                    boarding_role=m['role'], boarding_allowed=m['role'] != 'stop_exit_only'))
    positions = pd.concat([f[cols], pd.DataFrame(rows)[cols]], ignore_index=True)
    for col in ['stop_lat', 'stop_lon']:
        positions[col] = pd.to_numeric(positions[col], errors='raise')
    positions['historical_validity_verified'] = False
    positions = positions.sort_values(KEY).reset_index(drop=True)
    validate_positions(positions)
    return positions


def validate_positions(p):
    if (p.groupby('route').size().to_dict() != POSITION_COUNTS or p.occurrence_id.duplicated().any() or p.duplicated(KEY).any() or
        p[KEY+['occurrence_id', 'stop_name']].isna().any().any() or
        any(p[col].astype(str).str.strip().eq('').any() for col in ['source_route_id', 'pattern_key', 'source_stop_id', 'occurrence_id', 'stop_name']) or
        any(not pd.api.types.is_integer_dtype(p[col]) for col in ['route', 'direction_id', 'stop_sequence']) or
        p.boarding_allowed.dtype != bool or
        not np.isfinite(p[['stop_lat', 'stop_lon']].to_numpy(float)).all() or
        p.stop_lat.abs().gt(90).any() or p.stop_lon.abs().gt(180).any()):
        raise ValueError('Invalid occurrence support')
    for _, g in p.groupby('route'):
        if (set(g.direction_id) != {0, 1} or not g.boarding_allowed.any() or
            g.groupby('direction_id').pattern_key.nunique().to_dict() != {0: 1, 1: 1}):
            raise ValueError('New variants require explicit service weights')
    for _, g in p.groupby(['route', 'direction_id', 'pattern_key']):
        if sorted(g.stop_sequence) != list(range(1, len(g)+1)):
            raise ValueError('Incomplete stop sequence')
    exit_only = p.boarding_role.eq('stop_exit_only')
    if not p.loc[exit_only, 'boarding_allowed'].eq(False).all():
        raise ValueError('Exit-only boarding allowed')


def prepare(out):
    if out.exists():
        raise ValueError('Output exists')
    p = geography()
    access = HERE/'runs/access-20260927-v1'
    am = json.loads((access/'manifest.json').read_text())
    if sha(access/'entrances.csv') != am['outputs']['entrances.csv']:
        raise ValueError('Metro entrances changed')
    metro = read_csv(access/'entrances.csv')
    d = distances(p.stop_lat, p.stop_lon, metro.Latitude_WGS84, metro.Longitude_WGS84)
    p['nearest_metro_entrance_distance_m'] = d.min(axis=1)
    p['metro_snapshot_date'] = '2021-10-23'
    p['walking_access_verified'] = False
    parent = HERE/'runs/shares-20260927-v2'
    pm = json.loads((parent/'manifest.json').read_text())
    if sha(parent/'scenario_profiles.csv.gz') != pm['outputs']['scenario_profiles.csv.gz']:
        raise ValueError('Source shares changed')
    s = pd.read_csv(parent/'scenario_profiles.csv.gz', sep=';', dtype={'trip_id': str, 'stop_id': str})
    s = s.loc[s.prior.eq('uniform') & s.strength.eq(0)].rename(
        columns={'trip_id': 'pattern_key', 'stop_id': 'source_stop_id', 'share': 'source_share'})
    joined = s.merge(p[KEY+['occurrence_id', 'stop_lat', 'stop_lon']], on=KEY, how='left',
                     validate='many_to_one', suffixes=('', '_catalog'))
    if (set(s.route) != set(FITTED) or joined.occurrence_id.isna().any() or
        not np.allclose(joined[['stop_lat', 'stop_lon']], joined[['stop_lat_catalog', 'stop_lon_catalog']],
                        rtol=0, atol=1e-8)):
        raise ValueError('Source profile geography differs')
    source = joined[['route', 'occurrence_id', 'weekend', 'hour', 'source_share']].sort_values(
        ['route', 'weekend', 'hour', 'occurrence_id']).reset_index(drop=True)
    # Validate the complete 40-context support before freezing the inputs.
    make_profiles(p, source)
    out.mkdir(parents=True)
    save_csv(p, out/'positions.csv')
    save_csv(source, out/'source_profiles.csv.gz')
    write_json(out/'manifest.json', dict(version='network-inputs-v1',
        inputs=checksums([ROOT/'data/catalog/catalog.xlsx', ROOT/'data/osm/tram_routes.json',
            access/'entrances.csv', access/'manifest.json', parent/'scenario_profiles.csv.gz',
            parent/'manifest.json', Path(__file__), PROTOCOL,
            HERE/'build_metro_access.py', Path(prior.__file__)]),
        sources=dict(metro=am['source'], metro_usage='nearest entrance distance only, not metro passenger counts',
            osm=dict(snapshot='2026-09-26T22:39:54Z', attribution='© участники OpenStreetMap',
                license='ODbL-1.0', url='https://www.openstreetmap.org/copyright'),
            shares='B-v4 aggregate-only shares preserved in F-v2; not observed stop fractions'),
        outputs={name: sha(out/name) for name in ['positions.csv', 'source_profiles.csv.gz']}))
    print(f'Prepared {len(p)} positions, {len(source)} source-profile rows', flush=True)


def metro_prior(distance, allowed, alpha=.5, scale=500.):
    d, active = np.asarray(distance, float), np.asarray(allowed)
    if (d.ndim != 1 or active.shape != d.shape or active.dtype != bool or not active.any() or
        not np.isfinite(d).all() or (d < 0).any() or not 0 <= alpha < 1 or
        not np.isfinite(scale) or scale <= 0):
        raise ValueError('Invalid prior features or parameters')
    # Closest entrance, not a sum over entrances. The floor limits missing-inventory effects.
    a = np.exp(-(d[active]-d[active].min())/scale)
    q = np.zeros(len(d))
    q[active] = (1-alpha)/active.sum()+alpha*a/a.sum()
    return q


def project(source, q, allowed, strength=1.):
    s, active = np.asarray(source, float), np.asarray(allowed, bool)
    if s.shape != active.shape or not np.isfinite(s).all() or (s[active] <= 0).any():
        raise ValueError('Invalid source shares')
    p = np.zeros(len(s))
    p[active] = entropy_projection(np.log(s[active]), np.array([0, active.sum()]), q[active], strength)
    return p


def make_profiles(positions, source):
    validate_positions(positions)
    if (source.duplicated(['route', 'occurrence_id', 'weekend', 'hour']).any() or
        set(source.route) != set(FITTED) or set(source.weekend) != {0, 1} or set(source.hour) != set(HOURS)):
        raise ValueError('Invalid source profile keys')
    parts, comparisons = [], []
    for route, pos in positions.groupby('route', sort=True):
        pos = pos.sort_values(KEY)
        active = pos.boarding_allowed.to_numpy(bool)
        q = metro_prior(pos.nearest_metro_entrance_distance_m, active)
        uniform = active.astype(float)/active.sum()
        for weekend in [0, 1]:
            for hour in HOURS:
                basis = 'uniform_source_geographic_prior_only'
                s = uniform
                if route in FITTED:
                    rows = source.loc[source.route.eq(route) & source.weekend.eq(weekend) & source.hour.eq(hour)]
                    if set(rows.occurrence_id) != set(pos.occurrence_id):
                        raise ValueError('Incomplete source profile; no partial renormalization')
                    s = rows.set_index('occurrence_id').loc[pos.occurrence_id, 'source_share'].to_numpy(float)
                    if not np.isclose(s.sum(), 1, rtol=0, atol=1e-10):
                        raise ValueError('Source shares do not sum to one')
                    basis = 'aggregate_fit_B_v4_F_v2'
                if route == 5:
                    basis = 'route5_no_history_zero_fallback'
                p = project(s, q, active)
                part = pos[KEY+['occurrence_id']].copy()
                part['weekend'], part['hour'], part['source_share'] = weekend, hour, s
                part['prior_share'], part['share'], part['share_basis'] = q, p, basis
                parts.append(part)
                for label, alpha, scale, strength in [
                    ('uniform_prior', 0., 500., 1.), ('alpha_025', .25, 500., 1.),
                    ('alpha_075', .75, 500., 1.), ('distance_300m', .5, 300., 1.),
                    ('distance_800m', .5, 800., 1.), ('unregularized', .5, 500., 0.),
                    ('lambda_10', .5, 500., 10.)]:
                    other = project(s, metro_prior(pos.nearest_metro_entrance_distance_m, active, alpha, scale), active, strength)
                    comparisons.append((route, label, float(np.abs(p-other).sum()/2)))
                comparisons.append((route, 'equal_shares', float(np.abs(p-uniform).sum()/2)))
    sensitivity = pd.DataFrame(comparisons, columns=['route', 'comparison', 'tv']).groupby(
        ['route', 'comparison']).tv.agg(['mean', 'max']).reset_index()
    return pd.concat(parts, ignore_index=True), sensitivity


def validate_routes(routes):
    grid = pd.MultiIndex.from_product([ROUTES, pd.date_range('2025-11-01', '2025-12-31').strftime('%Y-%m-%d'), range(24)])
    keys = ['route', 'date', 'hour']
    if (routes.duplicated(keys).any() or set(routes[keys].itertuples(index=False, name=None)) != set(grid) or
        not pd.api.types.is_integer_dtype(routes.prediction) or routes.prediction.lt(0).any() or
        routes.prediction.gt(2**63-1).any() or
        not routes.loc[routes.route.eq(5) | routes.hour.between(1, 4), 'prediction'].eq(0).all()):
        raise ValueError('Invalid ten-route 61-day forecast')


def render_route(route, positions, profiles):
    pos = positions.sort_values(KEY)
    keys = list(pos[KEY].itertuples(index=False, name=None))
    cache = {(int(w), int(h)): f.set_index('occurrence_id').loc[pos.occurrence_id, 'share'].to_numpy()
             for (w, h), f in profiles.groupby(['weekend', 'hour'])}
    output = []
    for row in route.itertuples(index=False):
        night = row.hour in range(1, 5)
        p = np.zeros(len(pos)) if night else cache[(int(pd.Timestamp(row.date).dayofweek >= 5), row.hour)]
        counts = [0]*len(p) if night else allocate(row.prediction, p, keys)
        for occurrence, allowed, share, count in zip(pos.occurrence_id, pos.boarding_allowed, p, counts):
            status = ('route5_no_history_zero_fallback' if row.route == 5 else
                'structural_zero_nonworking' if night else
                'scenario_zero_exit_only' if not allowed else 'scenario_estimate')
            output.append((row.route, row.date, row.hour, occurrence, float(share),
                           float(row.prediction)*share, count, status))
    return pd.DataFrame(output, columns=['route', 'date', 'hour', 'occurrence_id', 'share',
        'expected_validations', 'estimated_validations', 'estimate_status'])


def build(inputs, out):
    if out.exists():
        raise ValueError('Output exists; releases are immutable')
    manifest = json.loads((inputs/'manifest.json').read_text())
    if set(manifest['outputs']) != {'positions.csv', 'source_profiles.csv.gz'}:
        raise ValueError('Unexpected prepared inputs')
    for name, digest in manifest['outputs'].items():
        if sha(inputs/name) != digest:
            raise ValueError(f'Changed input: {name}')
    positions, source = read_csv(inputs/'positions.csv'), read_csv(inputs/'source_profiles.csv.gz')
    profiles, sensitivity = make_profiles(positions, source)
    route_meta = json.loads((BUNDLE/'forecast_bundle.json').read_text())
    if (sha(BUNDLE/'forecast.csv') != route_meta['prediction_sha256'] or
        route_meta['timezone'] != 'Europe/Moscow' or route_meta['history_end'] != '2025-11-01T00:00:00+03:00' or
        route_meta['forecast_from'] != route_meta['history_end'] or
        route_meta['forecast_to'] != '2026-01-01T00:00:00+03:00'):
        raise ValueError('Changed route bundle')
    routes = read_csv(BUNDLE/'forecast.csv').sort_values(['route', 'date', 'hour']).reset_index(drop=True)
    validate_routes(routes)
    # Input folder location is not an identity: a copied Git checkout builds the same release.
    hashes = dict(prepared={name: sha(inputs/name) for name in [*manifest['outputs'], 'manifest.json']},
        code=checksums([Path(__file__), PROTOCOL, HERE/'fractional_split.py', HERE/'build_app_package.py',
                       HERE/'audit.py', HERE/'build_metro_access.py']),
        route_bundle=checksums([BUNDLE/'forecast.csv', BUNDLE/'forecast_bundle.json']))
    version = 'E-network-v1-'+fingerprint(hashes)[:16]
    out.mkdir(parents=True)
    balances, rows = [], 0
    with (out/'stop_estimates.csv.gz').open('wb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as zipped:
        with io.TextIOWrapper(zipped, encoding='utf-8', newline='') as stream:
            for number, route in routes.groupby('route', sort=True):
                frame = render_route(route, positions.loc[positions.route.eq(number)], profiles.loc[profiles.route.eq(number)])
                frame.to_csv(stream, sep=';', index=False, header=rows == 0, lineterminator='\n')
                rows += len(frame)
                summed = frame.groupby(['route', 'date', 'hour']).estimated_validations.sum().rename('allocated_prediction').reset_index()
                balance = route.merge(summed, on=['route', 'date', 'hour'], validate='one_to_one').rename(columns={'prediction': 'route_prediction'})
                balance['unassigned_prediction'] = balance.route_prediction-balance.allocated_prediction
                if not balance.unassigned_prediction.eq(0).all():
                    raise ValueError('Lost route total')
                balances.append(balance)
                print(f'Built route {number}: {len(frame)} estimates', flush=True)
    coverage = positions.groupby('route').agg(positions=('occurrence_id', 'size'), boarding_positions=('boarding_allowed', 'sum'),
        unique_stop_ids=('source_stop_id', 'nunique'), nearest_metro_min_m=('nearest_metro_entrance_distance_m', 'min'),
        nearest_metro_max_m=('nearest_metro_entrance_distance_m', 'max')).reset_index()
    near = positions.nearest_metro_entrance_distance_m.le(500).groupby(positions.route).sum()
    coverage['positions_with_metro_candidate_500m'] = coverage.route.map(near)
    coverage['share_basis'] = coverage.route.map(profiles.groupby('route').share_basis.first())
    coverage['verified_stop_observations'] = 0
    for name, frame in [('occurrences.csv', positions), ('profiles.csv.gz', profiles), ('route_forecast.csv', routes),
                        ('route_balance.csv', pd.concat(balances)), ('coverage.csv', coverage), ('sensitivity.csv', sensitivity)]:
        save_csv(frame, out/name)
    write_json(out/'metadata.json', dict(schema_version='tramcast.stop-estimate.v2', package_version=version,
        status='scenario_estimate_unvalidated', route_numbers=ROUTES, stop_estimates=rows,
        timezone='Europe/Moscow', history_end=route_meta['history_end'], forecast_from=route_meta['forecast_from'],
        forecast_to=route_meta['forecast_to'], route_model_version=route_meta['model_version'],
        route_forecast_mode=route_meta['serving_mode'], route_dataset_version=route_meta['dataset_version'],
        ui_label='Расчётное распределение посадок',
        ui_description='Оценка по маршрутному прогнозу и предположениям модели; точность по отдельным остановкам не проверена.',
        method='min_KL_p_to_s_plus_lambda_KL_p_to_q', strength=1., alpha=.5, distance_scale_m=500.,
        prior='half_uniform_half_exponential_nearest_metro_entrance_distance',
        parameter_selection='fixed_scenario_not_accuracy_selected', calendar='hour_and_Saturday_Sunday_indicator_only',
        allocation_model_version='network-entropy-'+fingerprint(dict(code=hashes['code'], prepared=hashes['prepared']))[:16],
        dataset_version='network-inputs-'+fingerprint(hashes['prepared'])[:16],
        source_share_policy={'aggregate_fit_routes': FITTED, 'uniform_source_routes': list(OSM), 'zero_fallback_routes': [5]},
        network_validity='late_snapshots_assumed_available_all_forecast_dates_not_verified',
        boarding_policy='OSM_exit_only_zero_other_OSM_roles_allowed_workbook_stop_mode_unverified',
        rounding='largest_remainder_exact_decimal_weights', rounding_tie_break=KEY,
        catalog_mapping_fraction=1., verified_stop_observation_coverage=0., real_service_coverage=None,
        stop_wape=None, stop_mae=None, occupancy_available=False,
        sensitivity_interpretation='assumption_sensitivity_not_accuracy_or_confidence_interval',
        mapping_failure_policy='keep_unmapped_integer_mass_unassigned_never_renormalize',
        unassigned_semantics='zero_under_complete_snapshot_support_assumption_only', sources=manifest['sources'],
        integrated=False, published=False))
    write_json(out/'manifest.json', dict(package_version=version, inputs=hashes,
        outputs={p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}))
    print(json.dumps(dict(package_version=version, rows=rows, route_hours=len(routes)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'build'])
    parser.add_argument('--inputs', type=Path, default=HERE/'inputs/network-v1')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.out) if args.command == 'prepare' else build(args.inputs, args.out)
