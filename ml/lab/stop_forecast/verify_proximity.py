"""Verify saved proximity ablation, without refitting."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from audit import ROOT, sha
from bayes_experiment import WINDOWS


def verify(folder):
    manifest = json.loads((folder/'manifest.json').read_text())
    for name, digest in manifest['inputs'].items():
        assert sha(ROOT/name) == digest, name
    for name, digest in manifest['outputs'].items():
        assert sha(folder/name) == digest, name
    result = json.loads((folder/'results.json').read_text())
    assert result['stop_wape'] is None and result['stop_mae'] is None
    first = [r for r in result['metrics'] if r['cutoff']==WINDOWS[0][0] and r['mode']!='030' and r['wape'] is not None]
    assert result['selected'] == min(first, key=lambda r:r['wape'])['mode']
    for cutoff, end in WINDOWS[:3]:
        f = pd.read_csv(folder/f'predictions_{cutoff}.csv', sep=';')
        assert not f.duplicated(['route', 'date', 'hour']).any()
        assert f.date.gt(cutoff).all() and f.date.le(end).all()
        assert (pd.to_datetime(f.date)-pd.Timestamp(cutoff)).dt.days.between(1, 61).all()
        for mode, column in [('base', 'direct'), ('030', 'control030'), ('proximity', 'proximity')]:
            row = next(r for r in result['metrics'] if r['cutoff']==cutoff and r['mode']==mode)
            if column not in f:
                assert mode=='proximity' and row['wape'] is None
                continue
            assert np.isfinite(f[column]).all() and f[column].ge(0).all()
            errors = abs(f.boardings-f[column])
            assert abs(row['wape']-errors.sum()/f.boardings.sum()) < 1e-12
            assert abs(row['mae']-errors.mean()) < 1e-10 and row['rows']==len(f)
    for cutoff, model in result['models'].items():
        if model.get('status')=='failed_numerical_convergence':
            assert model['reason']
            continue
        assert model['gradient_max'] < .05
        p = pd.read_csv(folder/f'profiles_{cutoff}.csv.gz', sep=';')
        assert not p.duplicated(['route', 'trip_id', 'stop_sequence', 'hour', 'weekend']).any()
        assert p.status.eq('unvalidated_historical_proximity_scenario').all()
        assert np.isfinite(p[['share', 'base_share', 'direct_expectation']]).all().all()
        assert p[['share', 'base_share', 'direct_expectation']].ge(0).all().all()
        sums = p.groupby(['route', 'weekend', 'hour'])[['share', 'base_share', 'direct_expectation']].sum()
        np.testing.assert_allclose(sums[['share', 'base_share']], 1, atol=1e-12)
        tv = p.assign(tv=abs(p.share-p.base_share)/2).groupby(['route','weekend','hour']).tv.sum()
        assert abs(tv.mean()-model['mean_share_tv_vs_base']) < 1e-12
        if cutoff != WINDOWS[3][0]:
            f = pd.read_csv(folder/f'predictions_{cutoff}.csv', sep=';')
            joined = f.merge(sums.direct_expectation, on=['route', 'weekend', 'hour'], validate='many_to_one')
            np.testing.assert_allclose(joined.proximity, joined.direct_expectation, rtol=1e-10)
    print('Verified hashes, selection, 61-day masks, metrics, convergence and profile sums.')


if __name__ == '__main__':
    verify(Path(sys.argv[1]))
