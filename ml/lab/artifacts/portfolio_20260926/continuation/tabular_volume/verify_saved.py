"""P66: audit paired data against audited P65, reconstruct hour-L1 targets and raw volumes."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import pandas as pd
from experiments.portfolio_verify import read
from experiments.portfolio_tabular_volume import CONTROL,HOURS,MODEL_SHA
from experiments.portfolio_school_fraction import features as daily_features

OUT=Path('artifacts/portfolio_20260926/continuation/tabular_volume');PARENT=OUT.parent/'tabular_shape'
parent_audit=json.loads((PARENT/'independent_verification.json').read_text())
assert parent_audit['causal_hourly_targets_reconstructed'] and parent_audit['paired_daily_features_reconstructed']
assert parent_audit['weighted_mean_covariance_eigenbasis_reconstructed'] and parent_audit['calendar_publication_and_matrix_checked']
assert parent_audit['audit_code_sha256']==hashlib.sha256((PARENT/'verify_saved.py').read_bytes()).hexdigest()
fits=[];hashes={}
for phase in ['pilot','study']:
    info=json.loads((OUT/phase/'run_started.json').read_text())
    for name,sha in info['sha256'].items():
        path=Path(name.removeprefix('/beegfs/home/m.persiyanov/codex_runs/tram-portfolio-20260926/'))
        if path.name=='tabpfn-v2-regressor.ckpt':assert sha==MODEL_SHA and MODEL_SHA in (OUT.parent/'tabular_fraction/checkpoint_sha256.txt').read_text()
        else:assert hashlib.sha256(path.read_bytes()).hexdigest()==sha,path
    assert info['versions']['scikit-learn']=='1.6.1' and info['versions']['tabpfn']=='2.0.9'
    hashes[phase]=len(info['sha256'])
for path in sorted(OUT.glob('*/fits_*/*/daily_training.csv')):
    folder=path.parent;cutoff=folder.name;past=read(path);past.origin=pd.to_datetime(past.origin)
    teacher=PARENT/'study/fits_42'/cutoff
    parent=read(teacher/'daily_training.csv');parent.origin=pd.to_datetime(parent.origin)
    pd.testing.assert_frame_equal(past.drop(columns=['optimal_volume','model_target']),parent)
    hourly=read(folder/'hourly_training.csv');hourly.origin=pd.to_datetime(hourly.origin)
    expected=read(teacher/'hourly_training.csv');expected.origin=pd.to_datetime(expected.origin)
    pd.testing.assert_frame_equal(hourly,expected)
    assert past.date.max()<=pd.Timestamp(cutoff) and ((past.date-past.origin).dt.days>0).all()
    assert (past.origin+pd.Timedelta(days=61)<=pd.Timestamp(cutoff)).all()
    # Independent convex optimality check, avoiding the production median algorithm.
    keys=['origin','route','date']
    matrix=hourly.pivot(index=keys,columns='hour',values='prediction').reindex(columns=HOURS)
    actual=hourly.pivot(index=keys,columns='hour',values='boardings').reindex(columns=HOURS).to_numpy()
    pd.testing.assert_frame_equal(matrix.index.to_frame(index=False).rename_axis(columns=None),past[keys])
    shares=matrix.to_numpy()/matrix.to_numpy().sum(axis=1)[:,None]
    residual=past.optimal_volume.to_numpy()[:,None]*shares-actual
    at_zero=abs(residual)<=1e-8
    derivative=(np.where(at_zero,0,np.sign(residual))*shares).sum(axis=1)
    mass=(at_zero*shares).sum(axis=1)
    assert (derivative-mass<=1e-12).all() and (derivative+mass>=-1e-12).all()
    optimum_loss=abs(residual).sum(axis=1)
    for volume in [past.actual_day.to_numpy(),past.base_day.to_numpy()]:
        assert (optimum_loss<=abs(volume[:,None]*shares-actual).sum(axis=1)+1e-8).all()
    info=json.loads((folder/'fit.json').read_text());model_target=info['target']
    expected_target=((past.actual_day if model_target=='actual' else past.optimal_volume)-past.base_day)/10000
    np.testing.assert_array_equal(past.model_target,expected_target)
    matrices=np.load(folder/'matrices.npz');parent_matrices=np.load(teacher/'matrices.npz')
    np.testing.assert_array_equal(matrices['past_share'],parent_matrices['past_share'])
    np.testing.assert_allclose(matrices['X'],np.column_stack([daily_features(past),matrices['past_share'],np.log1p(past.base_day)/10]),rtol=1e-12,atol=1e-14)
    np.testing.assert_array_equal(matrices['y'],expected_target)
    assert info['features']==51 and info['rows']==len(past) and info['model_sha256']==MODEL_SHA and info['sample_weight_used'] is False
    assert info['teacher_final_shape_family_mismatch'] is True and info['latest_target_date']<=cutoff
    if folder.relative_to(OUT).parts[0]=='pilot':assert info['repeated_prediction_exact'] is True
    future=read(folder/'future.csv');future.origin=pd.to_datetime(future.origin)
    parent_future=read(teacher/'future.csv');parent_future.origin=pd.to_datetime(parent_future.origin)
    pd.testing.assert_frame_equal(future.drop(columns=['source_volume','predicted_error','corrected_volume','factor']),parent_future)
    assert 'boardings' not in future and future.date.min()>pd.Timestamp(cutoff)
    base=read(CONTROL/f'raw_{cutoff}.csv');profile=base.pivot(index=['route','date'],columns='hour',values='prediction').reindex(columns=HOURS)
    volume=profile.sum(axis=1).to_numpy();share=np.divide(profile.to_numpy(),volume[:,None],out=np.zeros(profile.shape),where=volume[:,None]>0)
    active=future.route.ne(5).to_numpy()&(volume>0)
    np.testing.assert_array_equal(matrices['active'],active);np.testing.assert_allclose(future.source_volume,volume,rtol=1e-12,atol=1e-8)
    np.testing.assert_allclose(matrices['future_share'],share,rtol=1e-12,atol=1e-14)
    np.testing.assert_allclose(matrices['test'],np.column_stack([daily_features(future.loc[active]),share[active],np.log1p(volume[active])/10]),rtol=1e-12,atol=1e-14)
    error=np.zeros(len(future));error[active]=matrices['prediction'];np.testing.assert_array_equal(future.predicted_error,error)
    corrected=np.clip(volume+10000*error,.5*volume,2*volume)
    np.testing.assert_allclose(future.corrected_volume,corrected,rtol=1e-12,atol=1e-8)
    factor=np.divide(corrected,volume,out=np.ones(len(volume)),where=volume>0);np.testing.assert_allclose(future.factor,factor,rtol=1e-12,atol=1e-14)
    expected=base.merge(future[['route','date']].assign(factor=factor),on=['route','date'],validate='many_to_one')
    raw=read(folder/f'raw_{cutoff}.csv');np.testing.assert_allclose(raw.prediction,expected.prediction*expected.factor,rtol=1e-12,atol=1e-8)
    assert raw.loc[base.prediction.eq(0),'prediction'].eq(0).all()
    np.testing.assert_array_equal(daily_features(past),daily_features(past.assign(boardings=1e100)))
    fits.append(dict(phase=folder.relative_to(OUT).parts[0],seed=info['seed'],target=model_target,cutoff=cutoff,rows=len(past)))
count=0;study=OUT/'study/tabular_volume'
for folder in sorted(study.glob('trial_*'))+[study/'selected',OUT/'study/alternative',OUT/'study/seed73']:
    if not folder.exists():continue
    info=json.loads((folder/'parameters.json' if folder.name!='selected' else study/'selection.json').read_text());recipe=info.get('params',info)['recipe'];seed=73 if folder.name=='seed73' else 42
    for path in folder.glob('raw_*.csv'):
        actual=read(path);control=read(CONTROL/path.name)
        if recipe=='control':np.testing.assert_array_equal(actual.prediction,control.prediction)
        else:
            target,strength=recipe.split('_');learned=read(OUT/f'study/fits_{target}_{seed}'/path.stem[4:]/path.name);weight=.5 if strength=='half' else 1.
            np.testing.assert_allclose(actual.prediction,weight*learned.prediction+(1-weight)*control.prediction,rtol=1e-12,atol=1e-8)
        count+=1
chosen=study/'selected'
if json.loads((study/'selection.json').read_text())['params']['recipe']=='control':chosen=OUT/'study/alternative'
recipe=json.loads((chosen/'parameters.json' if chosen.name=='alternative' else study/'selection.json').read_text());recipe=recipe.get('params',recipe)['recipe'];target=recipe.split('_')[0]
np.testing.assert_array_equal(np.load(OUT/f'pilot/fits_{target}_42/2025-10-31/matrices.npz')['prediction'],np.load(OUT/f'study/fits_{target}_42/2025-10-31/matrices.npz')['prediction'])
seed_difference=[]
for path in chosen.glob('raw_*.csv'):
    first=read(path);second=read(OUT/'study/seed73'/path.name)
    seed_difference.append(dict(cutoff=path.stem[4:],relative_l1=float(abs(first.prediction-second.prediction).sum()/first.prediction.sum())))
result=dict(checked_at=datetime.now(timezone.utc).isoformat(),input_code_sha256=hashes,fits=fits,raw_forecasts_checked=count,
    paired_data_matches_independently_audited_P65=True,optimal_hourly_L1_targets_verified_by_convex_subgradient=True,absolute_passenger_targets_reconstructed=True,
    future_truth_free=True,native_model_unweighted=True,teacher_final_model_mismatch_documented=True,source_profiles_features_and_zero_days_checked=True,
    predicted_errors_clip_factor_and_hourshares_checked=True,exact_030_control=True,pilot_repeat_and_selected_freshfit_exact=True,
    seed73_raw_differences=seed_difference,audit_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
