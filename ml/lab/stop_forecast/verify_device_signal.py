"""Recompute signal metrics from saved anonymous bins, without reading event rows."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from audit import ROOT, sha
from device_signal import score, summarize


def verify(folder):
    manifest = json.loads((folder/'manifest.json').read_text())
    for name,digest in manifest['inputs'].items():
        assert sha(ROOT/name)==digest,name
    for name,digest in manifest['outputs'].items():
        assert sha(folder/name)==digest,name
    f=pd.read_csv(folder/'device_hour_scores.csv',sep=';')
    s=json.loads((folder/'summary.json').read_text())
    with np.load(folder/'binned_counts.npz',allow_pickle=False) as bins:
        h,r=bins['held'],bins['reference']
    assert h.shape==r.shape==(len(f),360)
    assert not f.duplicated(['vehicle_day','hour']).any()
    assert f.hour.isin([0]+list(range(6,24))).all()
    assert f.date.between('2025-09-08','2025-09-14').all()
    assert (h.sum(1)>=10).all() and (r.sum(1)>=20).all()
    rebuilt=pd.DataFrame([score(a,b) for a,b in zip(h,r)])
    np.testing.assert_allclose(f[rebuilt.columns],rebuilt,rtol=1e-12,atol=1e-10)
    assert int(h.sum()+r.sum())==s['used_events']
    assert s['used_events']+sum(s['exclusions'].values())==s['pilot_clean_successes']
    assert s['compared_vehicle_days']==f.vehicle_day.nunique()
    assert s['stop_accuracy'] is None
    for name,value in summarize(f).items():
        np.testing.assert_allclose(value['nats_per_held_event'],s['contrasts'][name]['nats_per_held_event'],atol=1e-12)
        np.testing.assert_allclose(value['cluster_bootstrap_95'],s['contrasts'][name]['cluster_bootstrap_95'],atol=1e-12)
    print('Verified hashes, anonymous bins, device scores, exclusions, masks and bootstrap.')


if __name__=='__main__':
    verify(Path(sys.argv[1]))
