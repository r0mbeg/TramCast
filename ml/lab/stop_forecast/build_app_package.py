"""Build a local, unvalidated entropy-allocation package for route 12."""
import argparse
from fractions import Fraction
import hashlib
import json
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from audit import ROOT, sha, write_json
from catalog_sequences import catalog
from fractional_split import entropy_projection

HERE = Path(__file__).resolve().parent
PARENT = HERE/'runs/shares-20260927-v2'
ROUTE_BUNDLE = ROOT/'ml/bundles/030'
PROTOCOL = HERE/'APP_PACKAGE_PROTOCOL.md'
KEY = ['route', 'direction_id', 'pattern_key', 'stop_sequence', 'source_stop_id']
HOURS = [0]+list(range(5, 24))
VERSION = 'E20260927-v1'


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def allocate(total, weights, keys):
    """Integer total -> nonnegative counts; stable ties, no float overflow in rounding."""
    w = np.asarray(weights, dtype=float)
    if (isinstance(total, bool) or not isinstance(total, Integral) or not 0 <= total <= 2**63-1 or
        w.ndim != 1 or not len(w) or len(keys) != len(w) or len(set(keys)) != len(keys) or
        not np.isfinite(w).all() or (w < 0).any() or not (w > 0).any()):
        raise ValueError('Invalid int64 total, weights or occurrence keys')
    # Exact decimal representations of the exported weights also work at int64 maximum.
    masses = [Fraction(str(float(x))) for x in w]
    denominator = sum(masses)
    expected = [int(total)*x/denominator for x in masses]
    counts = [x.numerator//x.denominator for x in expected]
    order = sorted(range(len(w)), key=lambda i: (-(expected[i]-counts[i]), keys[i]))
    left = int(total)-sum(counts)
    if not 0 <= left < len(w):
        raise ValueError('Rounding invariant violated')
    for i in order[:left]:
        counts[i] += 1
    return counts


def read_profiles(path):
    f = pd.read_csv(path, sep=';', dtype={'trip_id': str, 'stop_id': str})
    return f.loc[f.route.eq(12)].rename(columns={'trip_id': 'pattern_key', 'stop_id': 'source_stop_id'})


def profile_table(source, positions):
    expected = set(positions[KEY].itertuples(index=False, name=None))
    output = []
    for weekend in [0, 1]:
        for hour in HOURS:
            f = source.loc[source.weekend.eq(weekend) & source.hour.eq(hour) &
                           source.prior.eq('uniform') & source.strength.eq(0)].sort_values(KEY)
            if f.duplicated(KEY).any() or set(f[KEY].itertuples(index=False, name=None)) != expected:
                raise ValueError('Missing or changed support: never renormalize a partial catalog')
            reference = positions.sort_values(KEY)
            np.testing.assert_allclose(f[['stop_lat', 'stop_lon']], reference[['stop_lat', 'stop_lon']], atol=1e-8, rtol=0)
            for column in ['start_date', 'actual_date', 'stop_mode']:
                if f[column].astype(str).tolist() != reference[column].astype(str).tolist():
                    raise ValueError('Catalog attributes differ from frozen scenario')
            s = f.share.to_numpy()
            if not np.isfinite(s).all() or (s <= 0).any() or not np.isclose(s.sum(), 1, atol=1e-12):
                raise ValueError('Invalid source shares')
            q = np.full(len(s), 1/len(s))
            p = entropy_projection(np.log(s), np.array([0, len(s)]), q, 1.)
            old = source.loc[source.weekend.eq(weekend) & source.hour.eq(hour) &
                             source.prior.eq('uniform') & source.strength.eq(1)].sort_values(KEY)
            if list(old[KEY].itertuples(index=False, name=None)) != list(f[KEY].itertuples(index=False, name=None)):
                raise ValueError('Frozen profile layout differs')
            np.testing.assert_allclose(p, old.share.to_numpy(), rtol=1e-12, atol=1e-14)
            part = positions[KEY+['occurrence_id']].merge(f[KEY], on=KEY, validate='one_to_one').sort_values(KEY)
            part['weekend'] = weekend
            part['hour'] = hour
            part['source_share'] = s
            part['prior_share'] = q
            part['share'] = p
            output.append(part)
    return pd.concat(output, ignore_index=True)


def validate_route_forecast(f):
    expected = pd.MultiIndex.from_product([pd.date_range('2025-11-01', '2025-12-31').strftime('%Y-%m-%d'), range(24)])
    if (f.duplicated(['date', 'hour']).any() or not f.route.eq(12).all() or
        set(f[['date', 'hour']].itertuples(index=False, name=None)) != set(expected) or
        not pd.api.types.is_integer_dtype(f.prediction.dtype) or f.prediction.lt(0).any() or
        f.prediction.gt(2**63-1).any() or not f.loc[f.hour.between(1, 4), 'prediction'].eq(0).all()):
        raise ValueError('Invalid route-12 integer forecast or incomplete 61-day grid')


def render_estimates(route, positions, profiles):
    validate_route_forecast(route)
    positions = positions.sort_values(KEY)
    keys = list(positions[KEY].itertuples(index=False, name=None))
    by_context = {(int(w), int(h)): f.sort_values(KEY) for (w, h), f in profiles.groupby(['weekend', 'hour'])}
    rows = []
    for item in route.sort_values(['date', 'hour']).itertuples(index=False):
        if item.hour in range(1, 5):
            p, counts, status = np.zeros(len(keys)), [0]*len(keys), 'structural_zero'
        else:
            profile = by_context[(int(pd.Timestamp(item.date).dayofweek >= 5), item.hour)]
            if list(profile[KEY].itertuples(index=False, name=None)) != keys:
                raise ValueError('Forecast support mismatch')
            p = profile.share.to_numpy()
            counts = allocate(item.prediction, p, keys)
            status = 'scenario_estimate'
        for occurrence, share, count in zip(positions.occurrence_id, p, counts):
            rows.append((item.route, item.date, item.hour, occurrence, float(share),
                         float(item.prediction)*share, count, status))
    return pd.DataFrame(rows, columns=['route', 'date', 'hour', 'occurrence_id', 'share',
        'expected_validations', 'estimated_validations', 'estimate_status'])


def sensitivity(source, profiles):
    selected = profiles[KEY+['weekend', 'hour', 'share']].rename(columns={'share': 'selected'})
    rows = []
    for name, prior, strength in [('unregularized', 'uniform', 0.), ('stronger_regularization', 'uniform', 10.),
                                 ('centre_prior', 'centre_scenario', 1.)]:
        other = source.loc[source.prior.eq(prior) & source.strength.eq(strength)]
        z = selected.merge(other[KEY+['weekend', 'hour', 'share']], on=KEY+['weekend', 'hour'],
                           how='left', validate='one_to_one')
        if z.share.isna().any():
            raise ValueError('Incomplete sensitivity reference')
        tv = (z.selected-z.share).abs().groupby([z.weekend, z.hour]).sum()/2
        rows.append(dict(comparison=name, mean_total_variation=float(tv.mean()), max_total_variation=float(tv.max())))
    tv = (profiles.share-profiles.prior_share).abs().groupby([profiles.weekend, profiles.hour]).sum()/2
    rows.append(dict(comparison='uniform_control', mean_total_variation=float(tv.mean()), max_total_variation=float(tv.max())))
    direction = profiles.groupby(['weekend', 'hour', 'direction_id']).share.sum()
    return dict(interpretation='sensitivity_to_assumptions_not_accuracy_or_confidence_interval', comparisons=rows,
        maximum_occurrence_share=float(profiles.share.max()),
        direction_share_range={str(d): [float(g.min()), float(g.max())] for d, g in direction.groupby(level='direction_id')})


def build(out):
    if out.exists():
        raise ValueError('Output exists; do not overwrite a release')
    parent_manifest = json.loads((PARENT/'manifest.json').read_text())
    profile_path = PARENT/'scenario_profiles.csv.gz'
    if sha(profile_path) != parent_manifest['outputs'][profile_path.name]:
        raise ValueError('Changed F-v2 source profiles')
    if sha(HERE/'fractional_split.py') != parent_manifest['inputs'][str((HERE/'fractional_split.py').relative_to(ROOT))]:
        raise ValueError('Changed entropy implementation')
    route_metadata = json.loads((ROUTE_BUNDLE/'forecast_bundle.json').read_text())
    if (route_metadata['timezone'] != 'Europe/Moscow' or
        route_metadata['forecast_from'] != '2025-11-01T00:00:00+03:00' or
        route_metadata['forecast_to'] != '2026-01-01T00:00:00+03:00' or
        route_metadata['history_end'] != route_metadata['forecast_from']):
        raise ValueError('Unsupported route bundle period')
    route_path = ROUTE_BUNDLE/'forecast.csv'
    if sha(route_path) != route_metadata['prediction_sha256']:
        raise ValueError('Route bundle checksum mismatch')
    routes = pd.read_csv(route_path, sep=';')
    route = routes.loc[routes.route.eq(12)].sort_values(['date', 'hour']).reset_index(drop=True)
    validate_route_forecast(route)
    catalog_frame, _, catalog_inputs = catalog()
    positions = catalog_frame.loc[catalog_frame.route.eq(12)].copy().rename(columns={'route_id': 'source_route_id', 'trip_id': 'pattern_key'})
    if (len(positions) != 97 or positions.groupby('direction_id').pattern_key.nunique().to_dict() != {0: 1, 1: 1} or
        positions.duplicated(['route', 'pattern_key', 'stop_sequence']).any() or
        positions.end_date.ne('').any() or positions.is_addpoint.ne('0').any()):
        raise ValueError('Unexpected pilot geography; review new version explicitly')
    positions = positions.sort_values(KEY).reset_index(drop=True)
    positions['boarding_eligibility'] = 'assumed_unverified'
    positions = positions[['occurrence_id', 'source_route_id']+KEY+['stop_name', 'stop_lat', 'stop_lon',
        'start_date', 'actual_date', 'stop_mode', 'boarding_eligibility', 'historical_validity_verified']]
    source = read_profiles(profile_path)
    profiles = profile_table(source, positions)
    estimates = render_estimates(route, positions, profiles)
    inputs = [profile_path, PARENT/'manifest.json', route_path, ROUTE_BUNDLE/'forecast_bundle.json',
              *catalog_inputs, PROTOCOL, HERE/'APP_ESTIMATE_DECISION.md', Path(__file__),
              HERE/'fractional_split.py', HERE/'catalog_sequences.py', HERE/'audit.py',
              LAB_READER]
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    model_version = VERSION+'-'+fingerprint(dict(code=inputs[str(Path(__file__).relative_to(ROOT))],
        entropy=inputs[str((HERE/'fractional_split.py').relative_to(ROOT))], source=sha(profile_path), strength=1., prior='uniform'))[:16]
    dataset_version = 'stop-scenario-'+fingerprint(dict(route=sha(route_path), catalog=sha(ROOT/'data/catalog/catalog.xlsx')))[:16]
    metadata = dict(schema_version='tramcast.stop-estimate.v1', package_version=VERSION+'-'+fingerprint(inputs)[:16],
        allocation_model_version=model_version, dataset_version=dataset_version, status='scenario_estimate_unvalidated',
        ui_label='Расчётное распределение посадок',
        ui_description='Оценка по маршрутному прогнозу и предположениям модели; точность по отдельным остановкам не проверена.',
        timezone='Europe/Moscow', route_numbers=[12], history_end=route_metadata['history_end'],
        forecast_from=route_metadata['forecast_from'], forecast_to=route_metadata['forecast_to'],
        route_model_version=route_metadata['model_version'], route_dataset_version=route_metadata['dataset_version'],
        route_forecast_mode=route_metadata['serving_mode'],
        network_version='catalog-'+sha(ROOT/'data/catalog/catalog.xlsx'),
        geography_status='late_snapshot_not_verified_for_forecast_dates',
        support_assumption='all_97_snapshot_positions_assumed_available_in_both_directions',
        stop_mode_interpretation='unknown_codes_retained_not_used_to_assert_boarding_eligibility',
        method='min_KL_p_to_s_plus_lambda_KL_p_to_q', strength=1., prior='uniform_over_all_97_positions',
        source_share_origin='B20260927-v4_aggregate_only_fit_frozen_in_F20260927-v2',
        parameter_selection='fixed_scenario_not_accuracy_selected',
        calendar='hour_and_Saturday_Sunday_indicator_only',
        rounding='largest_remainder_exact_decimal_weights', rounding_tie_break=KEY,
        numeric_type='nonnegative_int64', fractional_field='expected_validations_is_not_observed',
        catalog_mapping_fraction=1., verified_stop_observation_coverage=0., real_service_coverage=None,
        stop_wape=None, stop_mae=None, occupancy_available=False,
        unassigned_semantics='zero_only_under_full_snapshot_support_assumption_not_proof_of_stop_accuracy',
        mapping_failure_policy='keep_unmapped_integer_mass_unassigned_never_renormalize',
        pilot_only=True, published=False)
    sums = estimates.groupby(['route', 'date', 'hour']).estimated_validations.sum().rename('allocated_prediction').reset_index()
    balance = route.merge(sums, on=['route', 'date', 'hour'], validate='one_to_one').rename(columns={'prediction': 'route_prediction'})
    balance['unassigned_prediction'] = balance.route_prediction-balance.allocated_prediction
    if not balance.unassigned_prediction.eq(0).all():
        raise ValueError('Route totals lost')
    out.mkdir(parents=True)
    positions.to_csv(out/'occurrences.csv', sep=';', index=False)
    profiles.to_csv(out/'profiles.csv', sep=';', index=False)
    route.to_csv(out/'route_forecast.csv', sep=';', index=False)
    estimates.to_csv(out/'stop_estimates.csv.gz', sep=';', index=False, compression={'method': 'gzip', 'mtime': 0})
    balance.to_csv(out/'route_balance.csv', sep=';', index=False)
    write_json(out/'metadata.json', metadata)
    write_json(out/'sensitivity.json', sensitivity(source, profiles))
    write_json(out/'manifest.json', dict(package_version=metadata['package_version'], inputs=inputs,
        outputs={p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}))
    print(json.dumps(dict(package_version=metadata['package_version'], rows=len(estimates),
        route_hours=len(route), total=int(estimates.estimated_validations.sum()), status=metadata['status'])), flush=True)


# Frozen catalog reader is reused; no separate spreadsheet parser is introduced.
LAB_READER = ROOT/'ml/lab/artifacts/methodology_20260927/data/audit_data.py'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    build(parser.parse_args().out)
