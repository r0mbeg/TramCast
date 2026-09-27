"""Prepare the fixed competition CPU package offline; runtime never imports lab."""
import argparse
import hashlib
import json
from pathlib import Path

from ml.lab.dense_student.model import load_bundle, read_history, features, reference
from ml.runtime.constants import ROUTES


def export(bundle, history, output):
    _, meta = load_bundle(bundle)
    if meta['kind'] != 'catboost':
        raise ValueError('Only the native CatBoost CPU model is supported')
    keys, frame = features(read_history(history), '2025-11-01', meta)
    if frame.columns.tolist() != meta['features']:
        raise ValueError('Feature contract mismatch')
    frame['date'] = keys.date
    frame['reference'] = reference(frame, meta)
    history_sha = hashlib.sha256(history.read_bytes()).hexdigest()
    sources = dict(meta, runtime_history_sha256=history_sha,
        preparation_code={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__), Path('ml/lab/dense_student/model.py'),
            Path('ml/lab/cpu_student/model.py'))})
    output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(output/'features.csv', sep=';', index=False, date_format='%Y-%m-%d')
    (output/'sources.json').write_text(json.dumps(sources, indent=2)+'\n')
    spec = dict(recipe='catboost-cpu', device='cpu', threads=2, timeout_seconds=60,
        generation='dense-student-v3', timezone='Europe/Moscow', route_numbers=ROUTES,
        history_end='2025-11-01T00:00:00+03:00', forecast_from='2025-11-01T00:00:00+03:00',
        forecast_to='2026-01-01T00:00:00+03:00', history_sha256=history_sha,
        model_file='../../models/dense-student-v3.cbm', model_sha256=meta['model_sha256'],
        features=meta['features'], files={n: hashlib.sha256((output/n).read_bytes()).hexdigest()
                                       for n in ('features.csv', 'sources.json')})
    (output/'recipe.json').write_text(json.dumps(spec, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--history', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export(args.bundle, args.history, args.output)
