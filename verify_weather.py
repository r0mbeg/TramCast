"""Recompute downloaded metrics and archive the weather attempt exactly once."""
import hashlib
import json
from pathlib import Path
import pandas as pd
from archive_submission import archive
from pipeline import KEYS, metrics


def check():
    out = Path('artifacts/chronos_weather_zhores')
    run = json.loads((out / 'run.json').read_text())
    for remote, digest in run['sha256'].items():
        if remote.endswith('artifacts/hourly_clean.csv'):
            local = Path('artifacts/hourly_clean.csv')
        elif remote == 'artifacts/results/submission.csv':
            local = out / 'submission.csv'
        elif remote.startswith('/'):
            local = Path(remote).name
        else:
            local = remote
        assert hashlib.sha256(Path(local).read_bytes()).hexdigest() == digest, local
    history = pd.read_csv('artifacts/hourly_clean.csv', sep=';', parse_dates=['date'])
    predictions = pd.read_csv(out / 'predictions.csv', sep=';', parse_dates=['date'])
    reported = pd.read_csv(out / 'metrics.csv', sep=';')
    control = pd.read_csv('artifacts/chronos_lora_zhores/predictions.csv', sep=';', parse_dates=['date'])
    control = control[control.method.eq('chronos_daily_zero_shot')]
    for (method, cutoff), group in predictions.groupby(['method', 'cutoff']):
        truth = history[history.date.between(group.date.min(), group.date.max())]
        joined = truth[KEYS+['boardings']].merge(group[KEYS+['prediction']], on=KEYS, validate='one_to_one')
        assert len(joined) == len(truth) == len(group)
        actual = metrics(joined.boardings, joined.prediction)
        row = reported[reported.method.eq(method) & reported.cutoff.eq(cutoff)].iloc[0]
        for key, value in actual.items():
            assert abs(value-row[key]) < 1e-10, (method, cutoff, key)
        if method == 'chronos_daily_total':
            old = control[control.cutoff.eq(cutoff)][KEYS+['prediction']].sort_values(KEYS).reset_index(drop=True)
            pd.testing.assert_frame_equal(old, group[KEYS+['prediction']].sort_values(KEYS).reset_index(drop=True))
    assert hashlib.sha256(Path('submission.csv').read_bytes()).hexdigest() == '4b37870e1e8e92f0d30fbfb6d0708793fa7951238827f67c1bec2d9ac5eeab7c'
    target = Path('submissions/003_chronos_daily_climatology.csv')
    if not target.exists():
        archive(out/'submission.csv', target.stem)
    assert target.read_bytes() == (out/'submission.csv').read_bytes()
    (out/'verification.json').write_text(json.dumps(dict(scores_recomputed=4,control_exact_match=True,
        source_and_code_hashes_verified=True,baseline_submission_unchanged=True,
        submission_sha256=hashlib.sha256(target.read_bytes()).hexdigest()),indent=2))
    print(reported[['method','cutoff','wape_score']].to_string(index=False))
    print('All metric, control, hash and submission checks passed.')


if __name__ == '__main__':
    check()
