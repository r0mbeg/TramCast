"""P71 independent rank/parent/mode/profile audit; prior P65 causal audit reused."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from experiments.portfolio_bayes_shape import HOURS
from experiments.portfolio_tabular_fraction import MODEL_SHA,CONTROL as VOLUME
from experiments.portfolio_tabular_volume import CONTROL
from experiments.portfolio_experiment import write_json

ROOT=Path('artifacts/portfolio_20260926/continuation');OUT=ROOT/'adaptive_shape';PARENT=ROOT/'tabular_shape/study'
DAY=['route','date']


def read(path):return pd.read_csv(path,sep=';',parse_dates=['date'],float_precision='round_trip')


def run():
    # All P71 features/targets reuse these P65 caches; verify their original causal construction again.
    prior=ROOT/'tabular_shape/verify_saved.py'
    checked=subprocess.run([sys.executable,str(prior)],capture_output=True,text=True,env=os.environ)
    if checked.returncode:raise AssertionError(checked.stderr+checked.stdout)
    previous=json.loads((ROOT/'tabular_shape/independent_verification.json').read_text())
    assert previous['causal_hourly_targets_reconstructed'] and previous['paired_daily_features_reconstructed']
    parents=json.loads((OUT/'parent_manifest.json').read_text())
    for name,digest in parents.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    hashes={};fits=[]
    for phase in ['pilot','study']:
        info=json.loads((OUT/phase/'run_started.json').read_text());assert info['parent_sha256']==parents
        assert info['versions']['tabpfn']=='2.0.9' and info['versions']['scikit-learn']=='1.6.1'
        for name,digest in info['sha256'].items():
            if name.endswith('tabpfn-v2-regressor.ckpt'):
                assert digest==MODEL_SHA and MODEL_SHA in (ROOT/'tabular_fraction/checkpoint_sha256.txt').read_text()
            else:
                name=name.removeprefix('/beegfs/home/m.persiyanov/codex_runs/tram-portfolio-20260926/')
                assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
        hashes[phase]=len(info['sha256'])
    for path in sorted(OUT.glob('*/fits_*/*/matrices.npz')):
        folder=path.parent;meta=json.loads((folder/'fit.json').read_text());cutoff=meta['cutoff'];seed=meta['seed'];parent=PARENT/f'fits_{seed}'/cutoff
        matrix=np.load(path);old=np.load(parent/'matrices.npz');oldmeta=json.loads((parent/'fit.json').read_text())
        for name in ['X','test','past_share','future_share','errors','weights','active','mean']:
            np.testing.assert_array_equal(matrix[name],old[name])
        for name in ['daily_training.csv','future.csv']:assert (folder/name).read_bytes()==(parent/name).read_bytes()
        errors=matrix['errors'];weights=matrix['weights'];mean=np.average(errors,axis=0,weights=weights)
        covariance=((errors-mean)*weights[:,None]).T@(errors-mean)/weights.sum()
        values=np.maximum(np.linalg.eigvalsh(covariance)[::-1],0);rank=int(np.searchsorted(np.cumsum(values)/values.sum(),.9))+1
        assert meta['rank']==rank and rank>oldmeta['rank'] and meta['newly_fitted_components']==rank-oldmeta['rank']
        assert values[:rank].sum()/values.sum()>=.9 and values[:rank-1].sum()/values.sum()<.9
        np.testing.assert_allclose(matrix['eigenvalues'],values,rtol=1e-8,atol=1e-14)
        components=matrix['components'];oldrank=oldmeta['rank'];assert components.shape==(rank,20)
        np.testing.assert_array_equal(components[:oldrank],old['components'])
        np.testing.assert_array_equal(matrix['coordinates'][:,:oldrank],old['coordinates'])
        np.testing.assert_array_equal(matrix['predictions'][:,:oldrank],old['predictions'])
        np.testing.assert_allclose(components@components.T,np.eye(rank),rtol=0,atol=1e-12)
        np.testing.assert_allclose(covariance@components.T,components.T*values[:rank],rtol=1e-7,atol=1e-12)
        assert (components[np.arange(rank),np.argmax(abs(components),axis=1)]>=0).all()
        np.testing.assert_allclose(matrix['coordinates'],(errors-mean)@components.T,rtol=1e-10,atol=1e-14)
        future=read(folder/'future.csv');assert 'boardings' not in future and future.date.min()>pd.Timestamp(cutoff)
        assert meta['latest_target_date']<=cutoff and meta['model_sha256']==MODEL_SHA and meta['coordinate_sample_weights'] is False
        if 'pilot' in path.parts:assert meta['repeated_prediction_exact'] is True
        active=matrix['active'];delta=np.zeros_like(matrix['future_share']);delta[active]=mean+matrix['predictions']@components
        np.testing.assert_allclose(delta,matrix['delta'],rtol=1e-12,atol=1e-14)
        profile=np.maximum(0,matrix['future_share']+delta);expanded=future[DAY].copy()
        for i,hour in enumerate(HOURS):expanded[str(hour)]=profile[:,i]
        expanded=expanded.melt(id_vars=DAY,var_name='hour',value_name='value');expanded.hour=expanded.hour.astype(int)
        volume=read(VOLUME/f'raw_{cutoff}.csv');shape=volume[DAY+['hour']].merge(expanded,on=DAY+['hour'],how='left',validate='one_to_one').fillna({'value':0.})
        target=volume.groupby(DAY).prediction.transform('sum');denominator=shape.groupby(DAY).value.transform('sum')
        backup=volume.prediction.div(target.where(target.gt(0))).fillna(0)
        learned=target*shape.value.div(denominator.where(denominator.gt(0))).fillna(backup)
        raw=read(folder/f'raw_{cutoff}.csv');np.testing.assert_allclose(raw.prediction,.5*learned+.5*volume.prediction,rtol=1e-12,atol=1e-8)
        np.testing.assert_allclose(raw.groupby(DAY).prediction.sum(),volume.groupby(DAY).prediction.sum(),rtol=1e-12,atol=1e-8)
        assert raw.loc[raw.route.eq(5)|raw.hour.between(1,4),'prediction'].eq(0).all()
        fits.append(dict(cutoff=cutoff,seed=seed,rank=rank,variance_retained=meta['variance_retained'],reused_components=oldrank))
    for key in ['components','coordinates','predictions','delta']:
        np.testing.assert_array_equal(np.load(OUT/'pilot/fits_42/2025-10-31/matrices.npz')[key],np.load(OUT/'study/fits_42/2025-10-31/matrices.npz')[key])
    count=0;study=OUT/'study/adaptive_shape'
    for folder in sorted(study.glob('trial_*'))+[study/'selected',OUT/'study/alternative',OUT/'study/seed73']:
        if not folder.exists():continue
        if folder.name=='selected':recipe=json.loads((study/'selection.json').read_text())['params']['recipe']
        elif folder.name=='seed73':recipe=json.loads((OUT/'study/seed_recipe.json').read_text())['recipe']
        elif folder.name=='alternative':recipe=json.loads((OUT/'study/best_new.json').read_text())['recipe']
        else:recipe=json.loads((folder/'parameters.json').read_text())['recipe']
        seed=73 if folder.name=='seed73' else 42
        for path in folder.glob('raw_*.csv'):
            raw=read(path);control=read(CONTROL/path.name)
            if recipe=='control':np.testing.assert_array_equal(raw.prediction,control.prediction)
            else:
                full=read(OUT/f'study/fits_{seed}'/path.stem[4:]/path.name);strength=.5 if recipe=='half' else 1.
                np.testing.assert_allclose(raw.prediction,strength*full.prediction+(1-strength)*control.prediction,rtol=1e-12,atol=1e-8)
            np.testing.assert_allclose(raw.groupby(DAY).prediction.sum(),control.groupby(DAY).prediction.sum(),rtol=1e-12,atol=1e-8)
            count+=1
    result=dict(checked_at=datetime.now(timezone.utc).isoformat(),input_code_sha256=hashes,parents=len(parents),fits=fits,
        raw_forecasts_checked=count,prior_causal_audit_replayed=True,prior_audit_code_sha256=hashlib.sha256(prior.read_bytes()).hexdigest(),
        past_only_minimum_rank90=True,exact_first_modes_reused=True,pilot_and_fresh_study_exact=True,
        fixed029_daily_volumes=True,zeros=True,audit_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(OUT/'independent_verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':run()
