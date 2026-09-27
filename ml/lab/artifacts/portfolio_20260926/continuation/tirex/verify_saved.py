"""Run from ml with PYTHONPATH=. : independent P56 inputs, caches and raw forecast audit."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json
import numpy as np
import pandas as pd
from experiments.portfolio_experiment import load_history
from experiments.portfolio_timesfm import series_inputs,teacher_daily
from experiments.portfolio_tirex import REVISION,WEIGHTS_SHA256,WHEEL_SHA256
from experiments.portfolio_timesfm_errors import SHAPE
from experiments.portfolio_verify import read

OUT=Path('artifacts/portfolio_20260926/continuation/tirex')
history=load_history('artifacts/hourly_clean.csv')
hashes={}
for phase in ['pilot_started','run_started']:
    info=json.loads((OUT/f'{phase}.json').read_text())
    assert info['revision']==REVISION and info['weights_sha256']==WEIGHTS_SHA256 and info['wheel_sha256']==WHEEL_SHA256
    for name,sha in info['sha256'].items():
        path=Path(name.removeprefix('/beegfs/home/m.persiyanov/codex_runs/tram-portfolio-20260926/'))
        assert hashlib.sha256(path.read_bytes()).hexdigest()==sha,path
    hashes[phase]=len(info['sha256'])
wheel=Path('artifacts/portfolio_20260926/continuation/external/tirex/tirex_ts-1.4.2-py3-none-any.whl')
assert hashlib.sha256(wheel.read_bytes()).hexdigest()==WHEEL_SHA256
assert json.loads((OUT/'pilot_completed.json').read_text())['deterministic_repeat_exact']
cached={}
for path in sorted((OUT/'quantiles').glob('*.npz')):
    info=json.loads(path.with_suffix('.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==info['sha256']
    representation=path.stem[:-11];cutoff=info['origin'];end=info['end']
    matrix,restoration=series_inputs(history,cutoff,end,representation,july_verified=True)
    assert info['input_sha256']==hashlib.sha256(matrix.tobytes()).hexdigest()
    with np.load(path,allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['inputs'],matrix)
        np.testing.assert_array_equal(saved['restoration'],restoration)
        values=saved['quantiles']
    assert values.shape==(9,61,9) and np.isfinite(values).all()
    # Future values cannot alter the model inputs or restoration, including retrospective operation masks.
    poisoned=history.copy();poisoned.loc[poisoned.date.gt(cutoff),'boardings']=9999999
    after,after_rest=series_inputs(poisoned,cutoff,end,representation,july_verified=True)
    np.testing.assert_array_equal(matrix,after);np.testing.assert_array_equal(restoration,after_rest)
    cached[(representation,cutoff)]=(values,restoration)
count=0
folders=sorted((OUT/'tirex').glob('trial_*'))+[OUT/'tirex/selected',OUT/'architecture_selected']
for folder in folders:
    info=json.loads((folder/'parameters.json' if folder.name!='selected' else folder.parent/'selection.json').read_text())
    recipe=info.get('params',info)['recipe']
    for path in sorted(folder.glob('raw_*.csv')):
        cutoff=path.stem[4:];end=str((pd.Timestamp(cutoff)+pd.Timedelta(days=61)).date())
        raw=read(path);shape=read(SHAPE/path.name)
        pd.testing.assert_frame_equal(raw[['route','date','hour']],shape[['route','date','hour']])
        if recipe=='control':
            np.testing.assert_array_equal(raw.prediction,shape.prediction)
        else:
            view,index=recipe.rsplit('_',1);values,restoration=cached[(view,cutoff)]
            daily=teacher_daily(history,cutoff,end,np.repeat(values[:,:,int(index),None],10,axis=2),restoration)
            expected=raw[['route','date']].merge(daily,on=['route','date'],validate='many_to_one').prediction
            total=shape.groupby(['route','date']).prediction.transform('sum')
            shares=shape.prediction.div(total.where(total.gt(0))).fillna(0)
            np.testing.assert_allclose(raw.prediction,expected*shares,rtol=1e-12,atol=1e-8)
        assert raw.loc[raw.route.eq(5)|raw.hour.between(1,4),'prediction'].eq(0).all()
        count+=1
result=dict(checked_at=datetime.now(timezone.utc).isoformat(),input_code_sha256=hashes,
    quantile_caches_checked=len(cached),raw_forecasts_checked=count,causal_input_future_poison_exact=True,
    fixed_024_hourshares=True,exact_024_control=True,operations_transform='fixed P49 teacher_daily',
    deterministic_pilot_repeat_exact=True,model_weights_sha256=WEIGHTS_SHA256,
    model_weight_checksum='checked by download and both GPU jobs on Zhores; weights not copied locally',
    audit_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
