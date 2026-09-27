"""Verify saved scenario evidence without refitting: python verify_bayes.py RUN."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from audit import ROOT, sha


def verify(folder):
    manifest = json.loads((folder/'manifest.json').read_text())
    for name, digest in manifest['inputs'].items():
        path = ROOT/name
        # Older runs preserve their executed source when later numerical fixes exist.
        if name.endswith('/bayes_experiment.py') and (folder/'source.py').exists():
            path = folder/'source.py'
        if name.endswith('/BAYES_PROTOCOL.md') and (folder/'BAYES_PROTOCOL.md').exists():
            path = folder/'BAYES_PROTOCOL.md'
        assert sha(path) == digest, name
    for name, digest in manifest['outputs'].items():
        assert sha(folder/name) == digest, name
    results = json.loads((folder/'results.json').read_text())
    assert results['stop_wape'] is None and results['stop_mae'] is None
    scenario = pd.read_csv(folder/'route12_scenario.csv.gz', sep=';')
    keys = ['route', 'direction_id', 'trip_id', 'stop_sequence', 'date', 'hour']
    assert not scenario.duplicated(keys).any()
    assert len(scenario) == 97*61*24 and scenario.route.eq(12).all()
    assert scenario.date.min() == '2025-11-01' and scenario.date.max() == '2025-12-31'
    assert scenario.model_version.eq(manifest['version']).all()
    assert scenario.status.eq('unvalidated_late_geography_scenario').all()
    day = scenario[~scenario.hour.between(1, 4)]
    night = scenario[scenario.hour.between(1, 4)]
    for col in ['direct_boardings', 'reconciled030_boardings', 'equal030_boardings']:
        assert np.isfinite(scenario[col]).all() and scenario[col].ge(0).all()
        assert night[col].eq(0).all()
    np.testing.assert_allclose(day.groupby(['date', 'hour']).share.sum(), 1, atol=1e-12)
    assert night.share.isna().all()
    totals = scenario.groupby(['date', 'hour']).agg(
        route030=('prediction', 'first'), reconciled=('reconciled030_boardings', 'sum'),
        equal=('equal030_boardings', 'sum'))
    np.testing.assert_allclose(totals.route030, totals.reconciled, atol=1e-9)
    np.testing.assert_allclose(totals.route030, totals.equal, atol=1e-9)
    direct = day.groupby(['weekend', 'hour', 'date']).direct_boardings.sum().reset_index()
    profile = pd.read_csv(folder/'profile_2025-10-31.csv', sep=';').query('route == 12')
    merged = direct.merge(profile, on=['weekend', 'hour'], validate='many_to_one')
    np.testing.assert_allclose(merged.direct_boardings, merged.prediction, atol=1e-9)
    for cutoff in ['2025-04-30', '2025-06-30', '2025-08-31']:
        frame = pd.read_csv(folder/f'route_predictions_{cutoff}.csv', sep=';')
        assert frame.date.gt(cutoff).all()
        assert not frame.duplicated(['route', 'date', 'hour']).any()
        for model, column in [(f"{results['selected']['likelihood']}_sd{results['selected']['prior_sd']}", 'direct'),
                              ('030_saved', 'control030')]:
            stored = next(r for r in results['metrics'] if r['cutoff'] == cutoff and r['model'] == model)
            wape = abs(frame.boardings-frame[column]).sum()/frame.boardings.sum()
            assert abs(stored['wape']-wape) < 1e-12
    assert results['diagnostics']['invariance']['max_log_route_difference'] < 1e-12
    print('Verified hashes, 142008 scenario rows, unique occurrence keys, nonnegativity,')
    print('night zeros, both route balances, saved WAPE and unavailable stop accuracy.')


if __name__ == '__main__':
    verify(Path(sys.argv[1]))
