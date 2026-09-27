"""One frozen historical-metro proximity ablation; no stop truth."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from audit import ROOT, LAB, sha, write_json
from bayes_experiment import design, moments, metric, predict, WINDOWS, ROUTES, HOURS
from metro_experiment import metro_fit, augment, PARENT

HERE = Path(__file__).resolve().parent
ACCESS = HERE/'runs/access-20260927-v1'


def features(stops, access):
    keys = ['route', 'trip_id', 'stop_sequence']
    cols = ['nearest_2021_entrance_distance_m', 'metro_2021_station_line_candidates_500m']
    joined = stops[keys].merge(access[keys+cols], on=keys, how='left', validate='one_to_one', sort=False)
    raw = joined[cols].to_numpy(float)
    if not np.isfinite(raw).all() or (raw < 0).any() or (raw[:, 1] != np.floor(raw[:, 1])).any():
        raise ValueError('Missing/invalid proximity covariates')
    logged = np.log1p(raw)
    mean, sd = logged.mean(axis=0), logged.std(axis=0)
    if (sd == 0).any():
        raise ValueError('Constant proximity covariate')
    return (logged-mean)/sd, dict(columns=cols, log_mean=mean.tolist(), log_sd=sd.tolist())


def run(out):
    input_paths = []
    for folder, names in [(PARENT, ['results.json', 'occurrences.csv', 'spatial_features.npy'] +
                          [f'route_predictions_{c}.csv' for c, _ in WINDOWS[:3]]),
                          (ACCESS, ['stop_features.csv'])]:
        manifest_path = folder/'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        input_paths.append(manifest_path)
        for name in names:
            path = folder/name
            if sha(path) != manifest['outputs'][name]:
                raise ValueError(f'Changed input: {path}')
            input_paths.append(path)
    stops = pd.read_csv(PARENT/'occurrences.csv', sep=';')
    access = pd.read_csv(ACCESS/'stop_features.csv', sep=';')
    values, scale = features(stops, access)
    base, edges, groups, occurrence = design(stops, np.load(PARENT/'spatial_features.npy', allow_pickle=False))
    x = augment(base, occurrence, groups, edges, values, 'flow')
    parent = json.loads((PARENT/'results.json').read_text())
    history_path = LAB/'artifacts/hourly_clean.csv'
    parent_manifest = json.loads((PARENT/'manifest.json').read_text())
    if sha(history_path) != parent_manifest['inputs'][str(history_path.relative_to(ROOT))]:
        raise ValueError('Changed route observations')
    history = pd.read_csv(history_path, sep=';', parse_dates=['date'])
    history = history.loc[history.route.isin(ROUTES) & history.hour.isin(HOURS) & history.working_events.gt(0)].copy()
    history['weekend'] = (history.date.dt.dayofweek >= 5).astype(int)
    out.mkdir(parents=True, exist_ok=False)
    results, models, slices = [], {}, []
    selected = None
    for wi, (cutoff, end) in enumerate(WINDOWS):
        if wi == 3 and selected == 'base':
            break
        old_beta = np.asarray(parent['diagnostics'][f'{cutoff}/poisson_sd1.0']['beta'])
        old_log, old_share, _ = moments(old_beta, base, edges)
        test = pd.read_csv(PARENT/f'route_predictions_{cutoff}.csv', sep=';', parse_dates=['date']) if wi < 3 else pd.DataFrame()
        if len(test):
            test['weekend'] = (test.date.dt.dayofweek >= 5).astype(int)
            np.testing.assert_allclose(predict(test, groups, np.exp(old_log)), test.direct, rtol=1e-9)
        try:
            beta, diag = metro_fit(history.loc[history.date.le(cutoff)], stops, x, edges, groups, old_beta)
        except RuntimeError as error:
            models[cutoff] = dict(status='failed_numerical_convergence', reason=str(error))
            results.append(dict(cutoff=cutoff, mode='proximity', wape=None, mae=None, rows=len(test)))
            print(cutoff, 'failed convergence', flush=True)
        else:
            lm, shares, _ = moments(beta, x, edges)
            tv = np.add.reduceat(abs(shares-old_share), edges[:-1])/2
            np.testing.assert_allclose(np.add.reduceat(shares, edges[:-1]), 1, atol=1e-12)
            np.testing.assert_allclose(np.add.reduceat(np.exp(x@beta), edges[:-1]), np.exp(lm), rtol=1e-10)
            models[cutoff] = dict(diag, beta=beta.tolist(), mean_share_tv_vs_base=float(tv.mean()),
                                 max_share_tv_vs_base=float(tv.max()))
            profile = stops.iloc[occurrence].reset_index(drop=True).copy()
            profile['hour'] = np.repeat(groups.hour, np.diff(edges)).to_numpy()
            profile['weekend'] = np.repeat(groups.weekend, np.diff(edges)).to_numpy()
            profile['share'], profile['base_share'] = shares, old_share
            profile['direct_expectation'] = np.exp(x@beta)
            profile['status'] = 'unvalidated_historical_proximity_scenario'
            profile.to_csv(out/f'profiles_{cutoff}.csv.gz', sep=';', index=False)
            if len(test):
                test['proximity'] = predict(test, groups, np.exp(lm))
            print(cutoff, 'gradient', diag['gradient_max'], 'mean share TV', tv.mean(), flush=True)
        if len(test):
            for mode, column in [('base', 'direct'), ('030', 'control030'), ('proximity', 'proximity')]:
                if column not in test:
                    continue
                results.append(dict(cutoff=cutoff, mode=mode, **metric(test, test[column])))
                for axis in ['route', 'hour']:
                    for key, frame in test.groupby(axis):
                        slices.append(dict(cutoff=cutoff, mode=mode, axis=axis, key=int(key), **metric(frame, frame[column])))
            test.to_csv(out/f'predictions_{cutoff}.csv', sep=';', index=False)
        if wi == 0:
            selected = min([r for r in results if r['mode'] != '030' and r['wape'] is not None], key=lambda r:r['wape'])['mode']
            print('development selected:', selected, flush=True)
    write_json(out/'results.json', dict(version='P20260927-v1', selected=selected, metrics=results,
        models=models, standardization=scale, stop_wape=None, stop_mae=None,
        final_base_reference='B20260927-v4' if selected=='base' else None))
    pd.DataFrame(slices).to_csv(out/'route_hour_metrics.csv', sep=';', index=False)
    input_paths += [history_path]+[HERE/n for n in ['proximity_experiment.py', 'PROXIMITY_PROTOCOL.md',
                                                   'metro_experiment.py', 'bayes_experiment.py', 'audit.py']]
    write_json(out/'manifest.json', dict(version='P20260927-v1',
        inputs={str(p.relative_to(ROOT)):sha(p) for p in input_paths},
        outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(dict(selected=selected, metrics=results)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    with threadpool_limits(limits=1):
        run(parser.parse_args().out)
