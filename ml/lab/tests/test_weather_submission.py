"""Small regression check for climatology and immutable submissions."""
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from archive_submission import archive
from experiments.chronos_weather_experiment import climatology, covariates, tasks


def check():
    source = Path('artifacts/weather_climatology/era5_moscow_2020_2024.json')
    table = climatology(source)
    assert table.shape == (366, 2) and np.isfinite(table).all().all()
    assert table.climate_wet_probability.between(0, 1).all()
    dates = pd.date_range('2025-01-01', '2025-06-30')
    future = pd.date_range('2025-07-01', '2025-08-31')
    batch = tasks(np.ones((9, len(dates))), dates, future, table)
    assert len(batch) == 9 and len(batch[0]['target']) == 181
    assert all(len(v) == 62 for v in batch[0]['future_covariates'].values())
    for c, v in covariates(table, dates).items():
        np.testing.assert_array_equal(v, covariates(table, dates + pd.DateOffset(years=1))[c])
    with tempfile.TemporaryDirectory() as directory:
        bad = json.loads(source.read_text())
        bad['daily']['time'][-1] = '2025-01-01'
        path = Path(directory) / 'bad.json'
        path.write_text(json.dumps(bad))
        try:
            climatology(path)
        except ValueError:
            pass
        else:
            raise AssertionError('Future weather accepted')
        target = archive('submission.csv', 'test_attempt', directory)
        original = target.read_bytes()
        try:
            archive('submission.csv', 'test_attempt', directory)
        except FileExistsError:
            pass
        else:
            raise AssertionError('Reused attempt ID accepted')
        assert target.read_bytes() == original
        invalid = pd.read_csv(target, sep=';')
        invalid.loc[invalid.route.eq(5), 'prediction'] = 1
        invalid_path = Path(directory) / 'invalid.csv'
        invalid.to_csv(invalid_path, sep=';', index=False)
        try:
            archive(invalid_path, 'invalid_attempt', directory)
        except ValueError:
            pass
        else:
            raise AssertionError('Nonzero route 5 accepted')
        assert not (Path(directory) / 'invalid_attempt.csv').exists()
    print('Climatology coverage, aligned features, future-weather rejection, submission contract and immutable IDs passed.')


if __name__ == '__main__':
    check()
