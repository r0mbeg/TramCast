"""Run from ml with PYTHONPATH=.: independently reconstruct fixed P59 compositions."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
import numpy as np
import pandas as pd
from experiments.portfolio_verify import read

OUT=Path('artifacts/portfolio_20260926/continuation/fraction_dynamics')
info=json.loads((OUT/'run_started.json').read_text())
for name, sha in info['sha256'].items():
    path=Path(name.removeprefix('/beegfs/home/m.persiyanov/codex_runs/tram-portfolio-20260926/'))
    assert hashlib.sha256(path.read_bytes()).hexdigest()==sha, path
count=0; study=OUT/'fraction_dynamics'
for folder in sorted(study.glob('trial_*'))+[study/'selected']:
    params=json.loads((folder/'parameters.json' if folder.name!='selected' else study/'selection.json').read_text())
    recipe=params.get('params',params)['recipe']
    for path in folder.glob('raw_*.csv'):
        actual=read(path)
        network, fractions, baseline=[read(Path(info['components'][key])/path.name)
            for key in ['network','fractions','baseline']]
        for frame in [network,fractions,baseline]:
            pd.testing.assert_frame_equal(actual[['route','date','hour']],frame[['route','date','hour']])
        if recipe.startswith('parent'):
            np.testing.assert_array_equal(actual.prediction,
                network.prediction if recipe=='parent025' else fractions.prediction)
        else:
            expected=fractions.prediction.copy()
            if recipe=='relative':
                old=baseline.groupby(['route','date']).prediction.transform('sum')
                new=fractions.groupby(['route','date']).prediction.transform('sum')
                assert new[old.eq(0)].eq(0).all()
                ratio=np.divide(new,old,out=np.ones(len(old)),where=old.gt(0))
                expected=network.prediction*ratio
            total=expected.groupby(network.date).transform('sum')
            expected*=network.groupby('date').prediction.transform('sum')/total
            np.testing.assert_allclose(actual.prediction,expected,rtol=1e-12,atol=1e-8)
            np.testing.assert_allclose(actual.groupby('date').prediction.sum(),
                network.groupby('date').prediction.sum(),rtol=1e-12,atol=1e-8)
        route_total=actual.groupby(['route','date']).prediction.transform('sum')
        parent_total=network.groupby(['route','date']).prediction.transform('sum')
        np.testing.assert_allclose(actual.prediction.div(route_total.where(route_total.gt(0))).fillna(0),
            network.prediction.div(parent_total.where(parent_total.gt(0))).fillna(0),rtol=1e-12,atol=1e-12)
        assert actual.prediction.ge(0).all() and np.isfinite(actual.prediction).all()
        assert actual.loc[network.prediction.eq(0),'prediction'].eq(0).all()
        count+=1
result=dict(checked_at=datetime.now(timezone.utc).isoformat(), input_code_sha256=len(info['sha256']),
    raw_forecasts_checked=count, exact_parent_controls=True, relative_composition_reconstructed=True,
    network_raw_daily_totals_preserved=True, fixed_parent_hourshares=True, structural_zeros=True,
    audit_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
