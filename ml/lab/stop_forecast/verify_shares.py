"""Verify prior-only share artifacts, not stop accuracy."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from audit import ROOT, sha


def verify(folder):
    manifest = json.loads((folder/'manifest.json').read_text())
    for name, digest in manifest['inputs'].items():
        source = folder/Path(name).name
        if not source.is_file() or Path(name).suffix != '.py':
            source = ROOT/name
        assert sha(source) == digest, name
    for name, digest in manifest['outputs'].items():
        assert sha(folder/name) == digest, name
    results = json.loads((folder/'results.json').read_text())
    assert results['stop_wape'] is None and results['stop_mae'] is None
    assert results['supervised_blfs_status'] == 'implemented_not_fitted_missing_real_stop_labels'
    profiles = pd.read_csv(folder/'scenario_profiles.csv.gz', sep=';')
    groups = ['route', 'weekend', 'hour', 'prior', 'strength']
    assert np.isfinite(profiles.share).all() and profiles.share.gt(0).all()
    np.testing.assert_allclose(profiles.groupby(groups).share.sum(), 1, atol=1e-12)
    assert not profiles.duplicated(groups+['direction_id', 'trip_id', 'stop_sequence']).any()
    diagnostics = pd.DataFrame(results['entropy_diagnostics'])
    for _, sub in diagnostics.groupby(['cutoff', 'prior']):
        sub = sub.sort_values('strength')
        assert (np.diff(sub.mean_kl_to_prior) <= 1e-12).all()
        if sub.prior.eq('uniform').all():
            assert (np.diff(sub.normalized_entropy) >= -1e-12).all()
    scenario = pd.read_csv(folder/'route12_scenario.csv.gz', sep=';')
    assert len(scenario) == 97*61*24
    assert not scenario.duplicated(['route', 'trip_id', 'direction_id', 'stop_sequence', 'date', 'hour']).any()
    assert scenario.date.min() == '2025-11-01' and scenario.date.max() == '2025-12-31'
    assert scenario.model_version.eq(manifest['version']).all()
    assert scenario.status.eq('prior_only_entropy_scenario_not_fitted_blfs').all()
    assert scenario.nonzero_probability.isna().all()
    assert np.isfinite(scenario.boardings).all() and scenario.boardings.ge(0).all()
    assert scenario.loc[scenario.hour.between(1, 4), 'boardings'].eq(0).all()
    mass = scenario.groupby(['date', 'hour']).agg(a=('boardings', 'sum'), b=('prediction', 'first'))
    np.testing.assert_allclose(mass.a, mass.b, atol=1e-9)
    print('Verified hashes, entropy/KL sweep, 142008 unique scenario rows, route balance,')
    print('night zeros, absent learned gates and unavailable stop accuracy.')


if __name__ == '__main__':
    verify(Path(sys.argv[1]))
