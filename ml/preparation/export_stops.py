"""Export compact serving inputs from an immutable stop scenario release; offline only."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'ml/runtime'))
from stops import StopBundle, digest


def export(source, out):
    if out.exists():
        raise ValueError('Output exists; do not overwrite a serving package')
    manifest_bytes = (source/'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    metadata = json.loads((source/'metadata.json').read_text(encoding='utf-8'))
    if manifest['package_version'] != metadata['package_version']:
        raise ValueError('Mismatched source versions')
    names = ('occurrences.csv', 'profiles.csv.gz')
    for name in (*names, 'metadata.json'):
        if digest((source/name).read_bytes()) != manifest['outputs'][name]:
            raise ValueError(f'Changed release source: {name}')
    import csv
    with (source/'occurrences.csv').open(encoding='utf-8', newline='') as stream:
        counts = {}
        for row in csv.DictReader(stream, delimiter=';'):
            counts[row['route']] = counts.get(row['route'], 0)+1
    spec = dict(schema_version='tramcast.stop-allocation.v1', source_manifest_sha256=digest(manifest_bytes),
        source_metadata=metadata, position_counts=counts, files={name: manifest['outputs'][name] for name in names})
    out.mkdir(parents=True)
    for name in names:
        shutil.copyfile(source/name, out/name)
    (out/'allocation_bundle.json').write_text(json.dumps(spec, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    bundle = StopBundle.load(out/'allocation_bundle.json')
    print(json.dumps(dict(output=str(out), source_package=bundle.package_version, positions=sum(counts.values())), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.out)
