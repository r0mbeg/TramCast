"""Verify the frozen-quarter ablation without refitting."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from audit import ROOT, sha


def verify(folder):
    manifest = json.loads((folder/'manifest.json').read_text())
    for name, digest in manifest['inputs'].items():
        assert sha(ROOT/name) == digest, name
    for name, digest in manifest['outputs'].items():
        assert sha(folder/name) == digest, name
    result = json.loads((folder/'results.json').read_text())
    assert result['stop_wape'] is None and result['stop_mae'] is None
    first = [r for r in result['metrics'] if r['cutoff']=='2025-04-30' and r['mode']!='030' and r['wape'] is not None]
    assert result['selected'] == min(first, key=lambda r:r['wape'])['mode']
    for cutoff in ['2025-04-30', '2025-06-30', '2025-08-31']:
        frame = pd.read_csv(folder/f'predictions_{cutoff}.csv', sep=';')
        assert not frame.duplicated(['route', 'date', 'hour']).any()
        assert frame.date.gt(cutoff).all()
        assert (pd.to_datetime(frame.date)-pd.Timestamp(cutoff)).dt.days.between(1, 61).all()
        for mode, column in [('base', 'direct'), ('030', 'control030'), ('name', 'name'), ('flow', 'flow')]:
            row = next(r for r in result['metrics'] if r['cutoff']==cutoff and r['mode']==mode)
            if row.get('status') == 'failed_numerical_convergence':
                assert row['wape'] is None and row['mae'] is None and column not in frame
                continue
            assert np.isfinite(frame[column]).all() and frame[column].ge(0).all()
            assert abs(row['wape']-abs(frame.boardings-frame[column]).sum()/frame.boardings.sum()) < 1e-12
            assert row['rows'] == len(frame)
    for cutoff, info in result['coverage'].items():
        assert info['quarter_end'] <= cutoff
        features = pd.read_csv(folder/f'features_{cutoff}.csv', sep=';')
        assert len(features) == 315 and np.isfinite(features).all().all()
        assert features.usable_flow.sum() == info['usable']
        assert features.loc[features.usable_flow.eq(0), ['log_incoming_standardized', 'log_outgoing_standardized']].eq(0).all().all()
    for model in result['models'].values():
        if model.get('status') == 'failed_numerical_convergence':
            assert model['reason']
        else:
            assert model['gradient_max'] < .05
    print('Verified hashes, frozen development selection, 61-day masks, route metrics,')
    print('quarter cutoffs, explicit missing covariates and numerical convergence.')


if __name__ == '__main__':
    verify(Path(sys.argv[1]))
