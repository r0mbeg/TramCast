"""Historical entrance proximity covariates, not verified walking transfers or labels."""
import argparse
import gzip
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from audit import ROOT, sha, write_json
from build_preliminary import normalize

HERE = Path(__file__).resolve().parent
ARCHIVE = ROOT/'dataset/external/stop_history_20260927/metro_entrances_20211023.zip'
ARCHIVE_SHA = '3b998e85764885662fa62b1e6c1f4b05cc2cf70dbf20279b3ce414fbc612931b'
BASE = HERE/'runs/preliminary-20260927-v1'


def distances(lat, lon, other_lat, other_lon):
    """Pairwise great-circle metres; latitude/longitude arrays in degrees."""
    arrays = [np.asarray(a, dtype=float) for a in (lat, lon, other_lat, other_lon)]
    for a, bound in zip(arrays, (90, 180, 90, 180)):
        if a.ndim != 1 or not np.isfinite(a).all() or (np.abs(a) > bound).any():
            raise ValueError('Invalid coordinates')
    a, b, c, d = [np.deg2rad(v) for v in arrays]
    if len(a) != len(b) or len(c) != len(d):
        raise ValueError('Coordinate lengths differ')
    h = np.sin((a[:, None]-c)/2)**2 + np.cos(a[:, None])*np.cos(c)*np.sin((b[:, None]-d)/2)**2
    return 6371008.8*2*np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def run(out):
    # ponytail: this small 315×1020 matrix needs no spatial index; revisit for city-scale POIs.
    from bson import decode_all  # pymongo 4.16.0, isolated preparation dependency only
    if sha(ARCHIVE) != ARCHIVE_SHA:
        raise ValueError('Archive changed; audit a new version')
    base_manifest = BASE/'manifest.json'
    source = BASE/'stop_features.csv'
    if sha(source) != json.loads(base_manifest.read_text())['outputs'][source.name]:
        raise ValueError('Preliminary dataset changed')
    with zipfile.ZipFile(ARCHIVE) as z:
        metro = pd.DataFrame(decode_all(gzip.decompress(z.read('data.bson.gz'))))
        meta = json.loads(z.read('meta.json'))
        package = json.loads(z.read('package.json'))
    metro = metro[['global_id', 'NameOfStation', 'Line', 'Name', 'Latitude_WGS84',
                   'Longitude_WGS84', 'ObjectStatus']].copy()
    if len(metro) != 1020 or metro.global_id.isna().any() or metro.global_id.duplicated().any():
        raise ValueError('Unexpected entrance support')
    if not metro.ObjectStatus.eq('действует').all():
        raise ValueError('New entrance statuses require review')
    metro['metro_name_key'] = metro.NameOfStation.map(normalize)
    metro['snapshot_date'] = '2021-10-23'
    stops = pd.read_csv(source, sep=';')
    if stops.occurrence_id.duplicated().any() or not stops.stop_observed_boardings.isna().all():
        raise ValueError('Unexpected stop keys or labels')
    d = distances(stops.stop_lat, stops.stop_lon, metro.Latitude_WGS84, metro.Longitude_WGS84)
    nearest = d.argmin(axis=1)
    stops['nearest_2021_entrance_id'] = metro.global_id.to_numpy()[nearest]
    stops['nearest_2021_entrance_distance_m'] = d.min(axis=1)
    stops['nearest_2021_station_name'] = metro.NameOfStation.to_numpy()[nearest]
    stops['nearest_2021_station_line'] = metro.Line.to_numpy()[nearest]
    named_distance = []
    for i, name in enumerate(stops.metro_name_key):
        mask = metro.metro_name_key.eq(name).to_numpy()
        named_distance.append(float(d[i, mask].min()) if mask.any() else np.nan)
    stops['named_2021_station_nearest_distance_m'] = named_distance
    si, mi = np.nonzero(d <= 800)
    links = metro.iloc[mi].reset_index(drop=True)
    links.insert(0, 'occurrence_id', stops.occurrence_id.to_numpy()[si])
    links['straight_distance_m'] = d[si, mi]
    links['candidate_status'] = 'historical_proximity_not_verified_transfer'
    for radius in (300, 500, 800):
        within = links.loc[links.straight_distance_m.le(radius)]
        counts = within.drop_duplicates(['occurrence_id', 'NameOfStation', 'Line']).groupby('occurrence_id').size()
        stops[f'metro_2021_station_line_candidates_{radius}m'] = stops.occurrence_id.map(counts).fillna(0).astype(int)
    stops['metro_entrance_snapshot_date'] = '2021-10-23'
    stops['metro_entrance_valid_in_2025_verified'] = False
    stops['walking_access_verified'] = False
    stops['metro_proximity_status'] = 'historical_snapshot_candidate_only'
    summary = dict(version='G20260927-v1', stop_occurrences=len(stops), entrance_records=len(metro),
        candidate_pairs_within_800m=len(links), confirmed_stop_labels=0,
        occurrences_with_candidates={str(r): int(stops[f'metro_2021_station_line_candidates_{r}m'].gt(0).sum())
                                     for r in (300, 500, 800)},
        explicit_named_occurrences=int(stops.metro_name_key.notna().sum()),
        named_station_present_in_snapshot=int(stops.named_2021_station_nearest_distance_m.notna().sum()),
        named_station_more_than_800m=int(stops.named_2021_station_nearest_distance_m.gt(800).sum()),
        historical_validity_verified=False, walking_access_verified=False)
    out.mkdir(parents=True, exist_ok=False)
    for filename, f in [('stop_features.csv', stops), ('entrances.csv', metro), ('candidate_links.csv', links)]:
        f.to_csv(out/filename, sep=';', index=False)
    write_json(out/'summary.json', summary)
    inputs = [ARCHIVE, source, base_manifest, Path(__file__), HERE/'audit.py', HERE/'build_preliminary.py']
    write_json(out/'manifest.json', dict(**summary,
        source=dict(original='https://data.mos.ru/opendata/624',
            mirror='https://data.apicrafter.ru/packages/datamos-metro',
            download='https://data.apicrafter.ru/packages/datamos-metro/build/datamos-7704786030-metro-2021-10-23-7-35/get',
            retrieved_date='2026-09-27', snapshot_date='2021-10-23',
            original_modified=meta['Modified'], original_version=meta['VersionNumber'],
            licenses_in_archive=package['licenses'], mirror_page_license='CC-BY-SA (version unspecified)',
            license_discrepancy='Mirror page and archive differ; retained locally, not published'),
        decoder='pymongo==4.16.0', distance='great-circle metres, mean Earth radius 6371008.8',
        inputs={str(p.relative_to(ROOT)): sha(p) for p in inputs},
        outputs={p.name: sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
