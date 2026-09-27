"""Preliminary Moscow covariates + route observations; never creates stop labels."""
import argparse
import calendar
from datetime import date
import json
from pathlib import Path
import re
import unicodedata

import pandas as pd
from audit import ROOT, LAB, prior, sha, write_json
from bayes_experiment import geography, ROUTES

HERE = Path(__file__).resolve().parent
METRO = ROOT/'dataset/external/stop_history_20260927/metro_passenger_traffic.csv'


def normalize(value):
    return ' '.join(unicodedata.normalize('NFKC', str(value)).casefold().replace('ё', 'е').split())


def named_metro(value):
    match = re.fullmatch(r'\s*Метро\s+["«](.+?)["»]\s*', str(value), flags=re.IGNORECASE)
    return normalize(match[1]) if match else None


def metro_history(path, cutoff):
    f = pd.read_csv(path, sep=';', skiprows=[1])
    required = ['NameOfStation', 'Line', 'Year', 'Quarter', 'IncomingPassengers', 'OutgoingPassengers', 'global_id']
    f = f[required].copy()
    quarters = {'I квартал': 1, 'II квартал': 2, 'III квартал': 3, 'IV квартал': 4}
    f['quarter'] = f.Quarter.map(quarters)
    if f.quarter.isna().any():
        raise ValueError('Unknown metro quarter')
    for c in ['Year', 'IncomingPassengers', 'OutgoingPassengers']:
        f[c] = pd.to_numeric(f[c], errors='raise')
        if f[c].isna().any() or (f[c] < 0).any() or (f[c] != f[c].astype('int64')).any():
            raise ValueError('Invalid metro year/count')
    f['period_end'] = [date(int(y), int(q)*3, calendar.monthrange(int(y), int(q)*3)[1]).isoformat()
                       for y, q in zip(f.Year, f.quarter)]
    if f.duplicated(['NameOfStation', 'Line', 'Year', 'quarter']).any():
        raise ValueError('Duplicate metro observations')
    f['metro_name_key'] = f.NameOfStation.map(normalize)
    f['zero_counter_requires_review'] = f.IncomingPassengers.eq(0) | f.OutgoingPassengers.eq(0)
    f['source_status'] = 'public_mirror_original_hash_not_verified'
    f['published_at'] = None
    f['strict_asof_availability_verified'] = False
    return f.loc[f.period_end.le(cutoff.isoformat())].reset_index(drop=True)


def run(out, cutoff):
    if not date(2025, 1, 1) <= cutoff <= date(2025, 10, 31):
        raise ValueError('Supported historical cutoff: Jan–Oct 2025')
    out.mkdir(parents=True, exist_ok=False)
    source_manifest = HERE/'sources/historical_manifest_20260927.json'
    expected = next(a['sha256'] for a in json.loads(source_manifest.read_text())['artifacts']
                    if a['path'].endswith('metro_passenger_traffic.csv'))
    if sha(METRO) != expected:
        raise ValueError('Metro source changed; audit new version first')
    stops, _, book = geography()
    for name, _, _, records in prior.sheets(book):
        if name == 'Остановки GTFS_STOPS':
            attrs = pd.DataFrame(records)[['stop_id', 'street', 'region', 'district', 'pavilion']]
            stops = stops.merge(attrs, on='stop_id', how='left', validate='many_to_one')
            break
    stops['occurrence_id'] = 'workbook:'+stops.route.astype(str)+':'+stops.trip_id+':'+stops.stop_sequence.astype(str)
    stops['network_version'] = 'workbook:'+sha(book)[:16]
    stops['metro_name_key'] = stops.stop_name.map(named_metro)
    stops['metro_name_explicit_in_stop_name'] = stops.metro_name_key.notna()
    stops['pilot_catalog_routes_at_stop'] = stops.groupby('stop_id').route.transform('nunique')
    stops['historical_network_verified'] = False
    metro = metro_history(METRO, cutoff)
    latest = metro.loc[metro.period_end.eq(metro.period_end.max())].copy()
    links = stops.loc[stops.metro_name_key.notna(), ['occurrence_id', 'metro_name_key']].merge(
        latest, on='metro_name_key', how='inner', validate='many_to_many')
    links['link_status'] = 'exact_normalized_name_candidate_not_verified_transfer'
    counts = links.groupby('occurrence_id').size()
    stops['metro_candidate_count_latest_quarter'] = stops.occurrence_id.map(counts).fillna(0).astype(int)
    unique = links.loc[links.occurrence_id.map(counts).eq(1),
                       ['occurrence_id', 'NameOfStation', 'Line', 'IncomingPassengers', 'OutgoingPassengers',
                        'period_end', 'zero_counter_requires_review']]
    stops = stops.merge(unique, on='occurrence_id', how='left', validate='one_to_one')
    stops['metro_link_status'] = stops.metro_candidate_count_latest_quarter.map(
        lambda n: 'no_candidate_not_evidence_of_no_metro' if n == 0 else
        'unique_name_candidate_unverified' if n == 1 else 'ambiguous_multiple_station_line_candidates')
    stops['metro_strict_asof_availability_verified'] = False
    stops['stop_target_status'] = 'missing_no_confirmed_stop_labels'
    stops['stop_observed_boardings'] = pd.NA
    ledger = HERE/'runs/20260927-v2/route_hour_balance.csv'
    route = pd.read_csv(ledger, sep=';')
    route = route.loc[route.route.isin(ROUTES) & route.date.le(cutoff.isoformat())].copy()
    route['weekday'] = pd.to_datetime(route.date).dt.dayofweek
    route['weekend'] = route.weekday.ge(5)
    route = route.rename(columns={'boardings': 'route_clean_counter_storage',
                                  'observed_counter': 'route_observed_counter'})
    route['counter_scope'] = 'route_not_stop'
    if route.duplicated(['route', 'date', 'hour']).any() or stops.occurrence_id.duplicated().any():
        raise ValueError('Duplicate dataset keys')
    if not route.loc[route.counter_status.eq('missing'), 'route_observed_counter'].isna().all():
        raise ValueError('Missing counter became observation')
    if route.matched_boardings.ne(0).any() or not stops.stop_observed_boardings.isna().all():
        raise ValueError('Unexpected stop labels; audit new source first')
    for filename, frame in [('stop_features.csv', stops), ('route_hours.csv.gz', route),
                            ('metro_quarters.csv', metro), ('metro_links.csv', links)]:
        frame.to_csv(out/filename, sep=';', index=False)
    summary = dict(version='D20260927-v1', cutoff=cutoff.isoformat(), routes=ROUTES,
        stop_occurrences=len(stops), route_hours=len(route), metro_quarter_rows=len(metro),
        metro_period_from=metro.period_end.min(), metro_period_to=metro.period_end.max(),
        explicit_metro_name_occurrences=int(stops.metro_name_explicit_in_stop_name.sum()),
        unique_metro_candidates=int(stops.metro_candidate_count_latest_quarter.eq(1).sum()),
        ambiguous_metro_candidates=int(stops.metro_candidate_count_latest_quarter.gt(1).sum()),
        candidate_link_rows=len(links), confirmed_stop_labels=0,
        historical_network_verified=False, strict_asof_metro_availability_verified=False,
        counter_status_counts={str(k): int(v) for k, v in route.counter_status.value_counts().items()})
    write_json(out/'summary.json', summary)
    inputs = [METRO, source_manifest, book, ledger, Path(__file__), HERE/'audit.py',
              HERE/'bayes_experiment.py', LAB/'artifacts/methodology_20260927/data/audit_data.py']
    write_json(out/'manifest.json', dict(**summary,
        sources=dict(metro_original='https://data.mos.ru/opendata/62743',
            mirror_description='https://hsedesign.com/project/analiz-4e76e86f90634eeeb0d42c5de35e42d0',
            download='https://disk.360.yandex.ru/d/UUrUv2RA7TiPiA'),
        inputs={str(p.relative_to(ROOT)): sha(p) for p in inputs},
        outputs={p.name: sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cutoff', type=date.fromisoformat, default=date(2025, 10, 31))
    args = parser.parse_args()
    run(args.out, args.cutoff)
