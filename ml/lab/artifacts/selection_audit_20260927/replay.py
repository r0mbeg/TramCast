"""Frozen finalist replay, no study/search; run from ml inside one Slurm allocation."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from experiments import portfolio_bayes_volume as bv, portfolio_bayes_direct as bd
from experiments import portfolio_direct_daily as dd, portfolio_bayes_shape as bs
from experiments import portfolio_timesfm as tf, portfolio_timesfm_errors as te
from experiments import portfolio_route_allocation as ra, portfolio_route_fraction as rf
from experiments import portfolio_school_fraction as sf, portfolio_tabular_shape as ts
from experiments import portfolio_adaptive_shape as ads
from experiments.portfolio_combine import mix, raw_frame, transplant
from experiments.portfolio_conditional_shape import shape_forecast
from experiments.portfolio_operations import operations_forecast, normal_history, apply_operations
from experiments.portfolio_chronos import gpu_forecast
from experiments.portfolio_experiment import load_history, cpu_forecast, write_json
from experiments.portfolio_direct_blend import volume_blend

OUT = Path('artifacts/selection_audit_20260927')
OLD = Path('artifacts/portfolio_20260926/continuation')
ORIGINS = ['2025-05-31', '2025-08-14', '2025-05-29', '2025-06-02', '2025-08-12', '2025-08-16']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(folder, cutoff, frame):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'raw_{cutoff}.csv'
    if path.exists():
        raise ValueError(f'Refusing overwrite: {path}')
    frame.to_csv(path, sep=';', index=False, date_format='%Y-%m-%d')
    return frame


def run():
    import torch
    import timesfm
    from chronos import Chronos2Pipeline

    if os.environ.get('SLURM_JOB_PARTITION') != 'gpu_devel' or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected one gpu_devel GPU')
    if int(os.environ['SLURM_CPUS_PER_TASK']) != 2 or len(os.sched_getaffinity(0)) > 2:
        raise RuntimeError('Expected two bound CPU cores')
    started = time.monotonic()
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.manual_seed(42)
    history = load_history('artifacts/hourly_clean.csv')
    root = OUT / 'replay_cache'
    # Copy only reusable completed inputs; all new/overwritten outputs stay in this audit.
    if not root.exists():
        root.mkdir(parents=True)
        for p in OLD.rglob('selection.json'):
            dest = root / p.relative_to(OLD); dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dest)
        for name in ['bayes_volume/inner', 'bayes_volume_verified_july/inner', 'verified_july/inner',
                     'timesfm_teachers/daily']:
            shutil.copytree(OLD / name, root / name)
        shutil.copytree(OLD / 'conditional_shape/conditional_shape/selected',
                        root / 'conditional_shape/conditional_shape/selected')
    for module in [bv, bd, tf]:
        module.ROOT = root
    dd.ROOT = root / 'direct_daily_inputs'
    te.SOURCE = root / 'verified_july/inner'
    te.TEACHERS = root / 'timesfm_teachers/daily'
    bs.SOURCE = ts.SOURCE = root / '021'
    te.SHAPE = ra.SHAPE = rf.SHAPE = sf.SHAPE = root / '024'
    ra.VOLUME = root / 'bayes_volume_verified_july/inner'
    sf.CONTROL = root / '028'
    ts.CONTROL = ads.VOLUME = root / '029'
    ads.SOURCE = root / '021'
    ads.PARENT = root / 'tabular'
    ads.CONTROL = root / '030'
    weather = pd.read_csv(tf.WEATHER_PATH, sep=';', parse_dates=['date']) if hasattr(tf, 'WEATHER_PATH') else pd.read_csv(
        'artifacts/portfolio_20260926/continuation/external/weather_2025.csv', sep=';', parse_dates=['date'])
    ops = json.loads((OLD / 'operations/operations/selection.json').read_text())['params']
    august = json.loads((OLD / 'august_operations/august_operations/selection.json').read_text())['params']['august7']
    shape_params = json.loads((OLD / 'conditional_shape/conditional_shape/selection.json').read_text())['params']
    annual = json.loads((OLD / 'verified_july/verified_july/selection.json').read_text())['params']
    tm_path = Path('model-timesfm2p5/model.safetensors')
    tab_path = Path('model-tabpfn/tabpfn-v2-regressor.ckpt')
    if sha(tm_path) != tf.WEIGHTS_SHA256 or sha(tab_path) != ads.MODEL_SHA:
        raise ValueError('Checkpoint checksum mismatch')
    versions = {n: importlib.metadata.version(n) for n in ['numpy', 'pandas', 'scikit-learn', 'torch', 'timesfm', 'tabpfn', 'chronos-forecasting']}
    write_json(OUT / 'replay_started.json', dict(job_id=os.environ['SLURM_JOB_ID'], versions=versions,
        origins=ORIGINS, trials=0, recipes='frozen existing selections',
        protocol_sha256=sha(OUT/'PROTOCOL.md'), source_sha256=sha(__file__),
        history_sha256=sha('artifacts/hourly_clean.csv'), timesfm_sha256=sha(tm_path), tabpfn_sha256=sha(tab_path)))

    # Reconstruct the actual P28 blend; its hourly proportions depend on both volumes.
    def parent(cutoff, end):
        p = json.loads((OLD/'bayes_volume/bayes_volume/selection.json').read_text())['params']
        v = bv.bayes_volume_forecast(history, cutoff, end, p)
        p = json.loads((OLD/'direct_daily/direct_daily/selection.json').read_text())['params']
        d = dd.direct_daily_forecast(history, cutoff, end, p)
        return mix(v, d, .75)
    c, e = '2025-06-30', '2025-08-30'
    original = raw_frame(OLD/'conditional_shape/conditional_shape/selected'/f'raw_{c}.csv', c, e)
    base = raw_frame(OLD/'direct_blend/direct_blend/selected'/f'raw_{c}.csv', c, e)
    reconstructed = shape_forecast(history, c, shape_params, weather, base)
    def shares(frame):
        total = frame.groupby(['route','date']).prediction.transform('sum')
        return frame.prediction.div(total.where(total.gt(0))).fillna(0).to_numpy()
    np.testing.assert_allclose(shares(reconstructed), shares(original), rtol=1e-10, atol=1e-12)
    write_json(OUT/'parent_shape_check.json', dict(cutoff=c, matched=True,
        max_share_difference=float(abs(shares(reconstructed)-shares(original)).max())))

    for cutoff in ORIGINS:
        end = str((pd.Timestamp(cutoff)+pd.Timedelta(days=61)).date())
        if (OUT/'new/031'/f'raw_{cutoff}.csv').exists():
            continue
        if time.monotonic()-started > 1450:
            write_json(OUT/'budget_stop.json', dict(seconds=time.monotonic()-started, next_cutoff=cutoff))
            break
        stamp = time.monotonic()
        print('start', cutoff, end, flush=True)
        base = parent(cutoff, end)
        conditional = shape_forecast(history, cutoff, shape_params, weather, base)
        save(root/'conditional_shape/conditional_shape/selected', cutoff, conditional)
        raw21 = bd.forecast(history, cutoff, end, annual, root/'verified_july')
        save(root/'021', cutoff, raw21)
        raw24 = bs.forecast(history, cutoff, end, 'half', root/'bayes_shape')
        save(root/'024', cutoff, raw24)

        config = timesfm.ForecastConfig(max_context=512, max_horizon=128, per_core_batch_size=8,
            normalize_inputs=True, use_continuous_quantile_head=True, force_flip_invariance=True,
            infer_is_positive=True, fix_quantile_crossing=True)
        model = timesfm.TimesFM_2p5_200M_torch(torch_compile=False)
        model.load_checkpoint(tm_path, torch_compile=False); model.compile(config)
        for verified in [False, True]:
            matrix, restoration = tf.series_inputs(history, cutoff, end, 'normal_ratio', july_verified=verified)
            with torch.inference_mode():
                _, q = model.forecast(horizon=61, inputs=[r.copy() for r in matrix])
            np.savez_compressed(root/f'timesfm_{cutoff}_{verified}.npz', inputs=matrix, quantiles=q, restoration=restoration)
            if verified:
                daily = tf.teacher_daily(history, cutoff, end, q, restoration)
                folder = te.TEACHERS/cutoff; folder.mkdir(parents=True, exist_ok=True)
                path = folder/'daily.csv'; daily.to_csv(path, sep=';', index=False, date_format='%Y-%m-%d')
                write_json(folder/'source.json', dict(origin=cutoff, end=end, fit_latest_date=cutoff,
                    quantile_index=3, july_verified=True, model_sha256=tf.WEIGHTS_SHA256, sha256=sha(path)))
            else:
                def predictor(tasks, horizon, cross):
                    assert horizon == 61
                    return q[:,:,3]*restoration
                rawtm = gpu_forecast(history, cutoff, end, 'daily', predictor)
                rawtm = transplant(rawtm, conditional, rawtm)
                rawtm = apply_operations(rawtm, history.loc[history.date.le(cutoff)],
                    normal_history(history, cutoff, extended=True, august7=True), {**ops, 'august7':august})
                save(OUT/'new/025', cutoff, volume_blend(raw24, rawtm, .75))
        del model; torch.cuda.empty_cache()

        # Relative-error models use only fully completed monthly 61-day teachers.
        # P58 selected .5 learned + .5 base; P61 control is P58.
        raw28 = rf.forecast(history, cutoff, end, 'half', root/'route_fraction')
        save(root/'028', cutoff, raw28)
        raw29 = sf.forecast(history, cutoff, end, 'new', root/'school')
        save(root/'029', cutoff, raw29); save(OUT/'new/029', cutoff, raw29)
        raw30 = ts.forecast(history, cutoff, end, 'new', ads.PARENT, tab_path)
        save(root/'030', cutoff, raw30); save(OUT/'new/030', cutoff, raw30)
        raw31 = ads.forecast(cutoff, end, 'full', root/'adaptive', tab_path)
        save(OUT/'new/031', cutoff, raw31)
        save(OUT/'new/C0', cutoff, cpu_forecast(history, cutoff, end, 'seasonal', dict(weeks=0,statistic='mean')))

        chronos = Chronos2Pipeline.from_pretrained('/beegfs/home/m.persiyanov/codex_runs/tram-chronos-20260926/model',
                                                  device_map='cuda', torch_dtype=torch.float32)
        chronos.model.eval()
        def cpredict(tasks, horizon, cross):
            with torch.inference_mode():
                q, _ = chronos.predict_quantiles(tasks, prediction_length=horizon, quantile_levels=[.5],
                    batch_size=32, context_length=512, cross_learning=False)
            return np.stack([v.cpu().numpy().reshape(horizon) for v in q])
        save(OUT/'new/C1', cutoff, gpu_forecast(history, cutoff, end, 'daily', cpredict))
        del chronos; torch.cuda.empty_cache()
        print('done', cutoff, 'seconds', time.monotonic()-stamp, flush=True)
        write_json(OUT/f'replay_{cutoff}.json', dict(cutoff=cutoff,end=end,seconds=time.monotonic()-stamp,
            completed_inner_windows=te.inner_windows(cutoff), target_cutoff=cutoff))
    write_json(OUT/'replay_completed.json', dict(job_id=os.environ['SLURM_JOB_ID'],
        seconds=time.monotonic()-started, peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        peak_rss_kib=__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss))


if __name__ == '__main__':
    run()
