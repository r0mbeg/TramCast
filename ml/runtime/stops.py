"""Immutable stop profiles and exact allocation; stdlib only, no research imports."""
import csv
from dataclasses import dataclass
from fractions import Fraction
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
from types import MappingProxyType
import zlib

from bundle import ROOT, INT64_MAX, MAX_HOURS, json_hour
from constants import ROUTES

DEFAULT_STOPS = ROOT.parent/'bundles/stops-network-v1/allocation_bundle.json'
HOURS = (0, *range(5, 24))


def digest(content):
    return hashlib.sha256(content).hexdigest()


def identity(value):
    return digest(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def allocate(total, weights):
    """Weights are validated, normalized Fractions in canonical occurrence order."""
    if type(total) is not int or not 0 <= total <= INT64_MAX:
        raise ValueError('Invalid route int64')
    expected = [total*w for w in weights]
    counts = [x.numerator//x.denominator for x in expected]
    left = total-sum(counts)
    if not 0 <= left < len(weights):
        raise ValueError('Invalid allocation weights')
    # Index tie break is the canonical route/direction/pattern/sequence/stop key.
    order = sorted(range(len(weights)), key=lambda i: (-(expected[i]-counts[i]), i))
    for i in order[:left]:
        counts[i] += 1
    return counts


def rows(content):
    reader = csv.DictReader(io.StringIO(content.decode('utf-8')), delimiter=';')
    if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
        raise ValueError('Invalid CSV header')
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError('Invalid CSV row')
        yield row


@dataclass(frozen=True)
class Occurrence:
    occurrence_id: str
    source_route_id: str
    pattern_key: str
    direction_id: int
    stop_sequence: int
    source_stop_id: str
    stop_name: str
    boarding_allowed: bool
    boarding_role: str

    @property
    def key(self):
        return self.direction_id, self.pattern_key, self.stop_sequence, self.source_stop_id


class StopBundle:
    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.stat().st_size > 128*1024:
            raise ValueError('Stop manifest exceeds limit')
        raw = path.read_bytes()
        spec = json.loads(raw)
        if (spec['schema_version'] != 'tramcast.stop-allocation.v1' or
            set(spec['files']) != {'occurrences.csv', 'profiles.csv.gz'} or
            set(spec['position_counts']) != {str(r) for r in ROUTES} or
            any(type(v) is not int or not 1 <= v <= 128 for v in spec['position_counts'].values())):
            raise ValueError('Unsupported stop bundle schema')
        meta = spec['source_metadata']
        if (meta['schema_version'] != 'tramcast.stop-estimate.v2' or meta['route_numbers'] != list(ROUTES) or
            meta['timezone'] != 'Europe/Moscow' or meta['status'] != 'scenario_estimate_unvalidated' or
            meta['stop_wape'] is not None or meta['stop_mae'] is not None or meta['occupancy_available'] is not False or
            meta['verified_stop_observation_coverage'] != 0 or meta['real_service_coverage'] is not None or
            any(not isinstance(meta[k], str) or not meta[k].strip() for k in
                ['package_version', 'allocation_model_version', 'dataset_version', 'network_validity'])):
            raise ValueError('Invalid stop metadata or unsupported accuracy claim')
        result = cls()
        result.start, result.end = json_hour(meta['forecast_from']), json_hour(meta['forecast_to'])
        if (not 0 < result.end-result.start <= MAX_HOURS*3600 or
            json_hour(meta['history_end']) != result.start):
            raise ValueError('Invalid stop profile period')
        data = {}
        for name, expected in spec['files'].items():
            source = path.parent/name
            if source.stat().st_size > 4*1024*1024:
                raise ValueError('Stop input exceeds limit')
            content = source.read_bytes()
            if digest(content) != expected:
                raise ValueError(f'Stop input checksum mismatch: {name}')
            if name.endswith('.gz'):
                try:
                    with gzip.GzipFile(fileobj=io.BytesIO(content)) as zipped:
                        content = zipped.read(8*1024*1024+1)
                except (OSError, EOFError, zlib.error) as error:
                    raise ValueError('Invalid compressed stop profiles') from error
                if len(content) > 8*1024*1024:
                    raise ValueError('Decompressed profiles exceed limit')
            data[name] = content
        occurrences = {r: [] for r in ROUTES}
        ids = {}
        for row in rows(data['occurrences.csv']):
            route = int(row['route'])
            if (route not in occurrences or row['boarding_allowed'] not in {'True', 'False'} or
                any(not row[k].strip() for k in ['occurrence_id', 'source_route_id', 'pattern_key',
                                               'source_stop_id', 'stop_name'])):
                raise ValueError('Invalid occurrence')
            item = Occurrence(row['occurrence_id'], row['source_route_id'], row['pattern_key'],
                int(row['direction_id']), int(row['stop_sequence']), row['source_stop_id'], row['stop_name'],
                row['boarding_allowed'] == 'True', row['boarding_role'])
            if (item.occurrence_id in ids or item.direction_id not in (0, 1) or item.stop_sequence < 1 or
                item.boarding_role not in {'assumed_unverified', 'stop', 'stop_entry_only', 'stop_exit_only'} or
                item.boarding_allowed != (item.boarding_role != 'stop_exit_only')):
                raise ValueError('Duplicate or invalid boarding occurrence')
            ids[item.occurrence_id] = (route, item)
            occurrences[route].append(item)
        for route, items in occurrences.items():
            items.sort(key=lambda o: o.key)
            if (len(items) != spec['position_counts'][str(route)] or not 1 <= len(items) <= 128 or
                len({o.key for o in items}) != len(items) or not any(o.boarding_allowed for o in items) or
                len({o.source_route_id for o in items}) != 1):
                raise ValueError('Incomplete occurrence support')
            for direction in (0, 1):
                group = [o for o in items if o.direction_id == direction]
                if (not group or len({o.pattern_key for o in group}) != 1 or
                    [o.stop_sequence for o in group] != list(range(1, len(group)+1))):
                    raise ValueError('Incomplete direction or unweighted variants')
        profiles = {}
        for row in rows(data['profiles.csv.gz']):
            key = (int(row['route']), int(row['weekend']), int(row['hour']))
            occurrence = row['occurrence_id']
            if (key[0] not in ROUTES or key[1] not in (0, 1) or key[2] not in HOURS or occurrence not in ids):
                raise ValueError('Invalid profile context')
            route, item = ids[occurrence]
            if (route != key[0] or (int(row['direction_id']), row['pattern_key'], int(row['stop_sequence']),
                                    row['source_stop_id']) != item.key):
                raise ValueError('Profile/catalog key mismatch')
            share = float(row['share'])
            if (not math.isfinite(share) or not 0 <= share <= 1 or
                (item.boarding_allowed and share <= 0) or (not item.boarding_allowed and share != 0)):
                raise ValueError('Invalid share or boarding eligibility')
            group = profiles.setdefault(key, {})
            if occurrence in group:
                raise ValueError('Duplicate profile occurrence')
            # Match the frozen laboratory exporter: round-trip float then decimal Fraction.
            group[occurrence] = Fraction(str(share))
        weights = {}
        for route, items in occurrences.items():
            expected = {o.occurrence_id for o in items}
            for weekend in (0, 1):
                for hour in HOURS:
                    key = route, weekend, hour
                    group = profiles.get(key, {})
                    if set(group) != expected or not math.isclose(float(sum(group.values())), 1., abs_tol=1e-12, rel_tol=0):
                        raise ValueError('Incomplete profile or shares not summing to one')
                    denominator = sum(group.values())
                    weights[key] = tuple(group[o.occurrence_id]/denominator for o in items)
        result.occurrences = MappingProxyType({route: tuple(items) for route, items in occurrences.items()})
        result.weights = MappingProxyType(weights)
        result.package_version = meta['package_version']
        result.geography_status = meta['network_validity']
        result.allocation_model_version = 'stop-runtime-'+identity(dict(
            source=meta['allocation_model_version'], code=digest(Path(__file__).read_bytes())))
        result.allocation_dataset_version = 'stop-data-'+digest(raw)
        result.network_version = 'stop-network-'+spec['files']['occurrences.csv']
        return result

    def versions(self, route_model, route_dataset):
        return dict(model_version='stops-'+identity([route_model, self.allocation_model_version]),
            dataset_version='stops-data-'+identity([route_dataset, self.allocation_dataset_version]),
            route_model_version=route_model, route_dataset_version=route_dataset,
            allocation_model_version=self.allocation_model_version,
            allocation_dataset_version=self.allocation_dataset_version, network_version=self.network_version,
            allocation_package_version=self.package_version, estimate_status='scenario_estimate_unvalidated',
            geography_status=self.geography_status)
