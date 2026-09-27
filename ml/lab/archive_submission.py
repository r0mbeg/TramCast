"""Archive a validated submission without overwriting an existing attempt."""
import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
from pipeline import DATA, KEYS, ROOT, full_grid


def archive(source, attempt, directory=ROOT / 'submissions'):
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]+', attempt):
        raise ValueError('Use a unique lowercase attempt ID')
    source = Path(source)
    frame = pd.read_csv(source, sep=';')
    expected = full_grid('2025-11-01', '2025-12-31').to_frame(index=False)
    if list(frame.columns) != KEYS + ['prediction']:
        raise ValueError('Invalid schema')
    pd.testing.assert_frame_equal(frame[KEYS].sort_values(KEYS).reset_index(drop=True), expected)
    template = pd.read_csv(DATA / 'test_submission.csv', sep=';')[KEYS]
    pd.testing.assert_frame_equal(template.sort_values(KEYS).reset_index(drop=True), expected)
    values = frame.prediction
    if values.dtype.kind not in 'iu' or not np.isfinite(values).all() or values.lt(0).any():
        raise ValueError('Predictions must be nonnegative integers')
    if not values[frame.route.eq(5) | frame.hour.between(1, 4)].eq(0).all():
        raise ValueError('Forced zeros violated')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f'{attempt}.csv'
    # Exclusive creation preserves the exact original bytes and rejects reused IDs.
    with target.open('xb') as output:
        output.write(source.read_bytes())
    pd.testing.assert_frame_equal(frame, pd.read_csv(target, sep=';'))
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('attempt')
    args = parser.parse_args()
    print(archive(args.source, args.attempt))
