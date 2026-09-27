"""Daily Chronos-2 with pre-2025 climatology; fixed parameters, no fine-tuning."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd
from chronos_experiment import REVISION, WINDOWS, forecast, evaluate

FEATURES = ['climate_temperature', 'climate_wet_probability']


def climatology(path):
    raw = json.loads(Path(path).read_text())
    weather = pd.DataFrame(raw['daily'])
    dates = pd.DatetimeIndex(pd.to_datetime(weather.time))
    if not dates.equals(pd.date_range('2020-01-01', '2024-12-31')):
        raise ValueError('Expected complete 2020–2024 weather, no future observations')
    values = weather[['temperature_2m_mean', 'precipitation_sum']].to_numpy(float)
    if not np.isfinite(values).all() or (values[:, 1] < 0).any():
        raise ValueError('Invalid weather')
    # A leap reference year preserves month/day alignment after February.
    day = pd.to_datetime('2000-' + dates.strftime('%m-%d')).dayofyear.to_numpy()
    rows = []
    for center in range(1, 367):
        distance = np.abs(day - center)
        mask = np.minimum(distance, 366 - distance) <= 15
        rows.append([values[mask, 0].mean(), (values[mask, 1] >= 1).mean()])
    return pd.DataFrame(rows, index=np.arange(1, 367), columns=FEATURES)


def covariates(table, dates):
    days = pd.to_datetime('2000-' + dates.strftime('%m-%d')).dayofyear
    return {c: table.loc[days, c].to_numpy(dtype=np.float32) for c in FEATURES}


def tasks(matrix, dates, future, table):
    past_cov, future_cov = covariates(table, dates), covariates(table, future)
    return [dict(target=row.copy(), past_covariates=past_cov, future_covariates=future_cov) for row in matrix]


def run(args):
    import torch
    from chronos import Chronos2Pipeline

    start = time.monotonic()
    cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', '0'))
    assert 1 <= cpus <= 4 and len(os.sched_getaffinity(0)) <= cpus
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.set_num_threads(cpus)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    table = climatology(args.weather)
    table.to_csv(out / 'climatology.csv', sep=';', index_label='leap_day_of_year')
    history = pd.read_csv(args.history, sep=';', parse_dates=['date'])
    model = Chronos2Pipeline.from_pretrained(args.model_path, device_map='cuda', torch_dtype=torch.float32)
    model.model.eval()
    rows, details, predictions = [], [], []
    for cutoff, end in WINDOWS + [('2025-10-31', '2025-12-31')]:
        for weather in [False, True]:
            method = 'chronos_daily_climatology' if weather else 'chronos_daily_total'
            if cutoff == '2025-10-31' and not weather:
                continue
            dates = pd.date_range(history.date.min(), cutoff)
            future = pd.date_range(pd.Timestamp(cutoff) + pd.Timedelta(days=1), end)

            def predictor(matrix, horizon):
                batch = tasks(matrix, dates, future, table) if weather else [torch.from_numpy(r.copy()) for r in matrix]
                with torch.inference_mode():
                    quantiles, _ = model.predict_quantiles(batch, prediction_length=horizon,
                        quantile_levels=[0.5], batch_size=32, context_length=512, cross_learning=False)
                return np.stack([q.cpu().numpy().reshape(horizon) for q in quantiles])

            torch.cuda.reset_peak_memory_stats()
            t0 = time.monotonic()
            result = forecast(history, cutoff, end, 'daily_total', predictor)
            torch.cuda.synchronize()
            if cutoff == '2025-10-31':
                result.to_csv(out / 'submission.csv', sep=';', index=False, date_format='%Y-%m-%d')
            else:
                row, detail, prediction = evaluate(history, result, cutoff, end, method, time.monotonic() - t0)
                row['peak_gpu_allocated_bytes'] = torch.cuda.max_memory_allocated()
                rows.append(row)
                details.extend(detail)
                predictions.append(prediction)
                print(json.dumps(row), flush=True)
    pd.DataFrame(rows).to_csv(out / 'metrics.csv', sep=';', index=False)
    pd.DataFrame(details).to_csv(out / 'breakdown.csv', sep=';', index=False)
    pd.concat(predictions).to_csv(out / 'predictions.csv', sep=';', index=False)
    info = dict(model='amazon/chronos-2', revision=REVISION, seed=42, quantile=0.5,
        context_length=512, batch_size=32, cross_learning=False, dtype='float32', mode='zero-shot',
        features=FEATURES, weather_period=['2020-01-01', '2024-12-31'], smoothing_days=31,
        wet_threshold_mm=1, windows=WINDOWS, submission_cutoff='2025-10-31',
        job_id=os.environ.get('SLURM_JOB_ID'), gpu=torch.cuda.get_device_name(0),
        cpu_affinity=sorted(os.sched_getaffinity(0)), seconds=time.monotonic()-start,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        versions={p: importlib.metadata.version(p) for p in ['numpy','pandas','torch','chronos-forecasting','transformers']},
        sha256={str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
            [args.history, args.weather, __file__, 'chronos_experiment.py', 'pipeline.py', out/'submission.csv']},
        missing_policy='absent counts = 0; observed validation unchanged',
        postprocessing='floor(max(0,p)+0.5); route5 and hours1-4 forced0',
        availability='Weather 2020–2024 retrieved in 2026, historical release vintage not verified; modern pretrained weights')
    (out/'run.json').write_text(json.dumps(info, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--history', required=True)
    p.add_argument('--model-path', required=True)
    p.add_argument('--weather', default='artifacts/weather_climatology/era5_moscow_2020_2024.json')
    p.add_argument('--output', default='artifacts/results')
    run(p.parse_args())
