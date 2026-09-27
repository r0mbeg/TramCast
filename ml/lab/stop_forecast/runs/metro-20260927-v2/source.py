"""Frozen-quarter metro ablation for aggregate learning, not stop evaluation."""
import argparse
from datetime import date
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from scipy.optimize import minimize

from audit import ROOT, LAB, sha, write_json
from bayes_experiment import design, objective, curvature, moments, metric, predict, WINDOWS, ROUTES, HOURS
from build_preliminary import metro_history, named_metro, METRO

HERE = Path(__file__).resolve().parent
PARENT = HERE/'runs/bayes-20260927-v4'


def features(stops, metro, cutoff):
    available = metro.loc[metro.period_end.le(cutoff)]
    if available.empty:
        raise ValueError('No completed metro quarter at cutoff')
    latest = available.loc[available.period_end.eq(available.period_end.max())]
    names = stops.stop_name.map(named_metro)
    mapping = latest.groupby('metro_name_key').size()
    unique = latest.loc[latest.metro_name_key.map(mapping).eq(1)].set_index('metro_name_key')
    incoming = names.map(unique.IncomingPassengers)
    outgoing = names.map(unique.OutgoingPassengers)
    usable = incoming.gt(0) & outgoing.gt(0)
    matrix = np.column_stack([names.notna().astype(float), usable.astype(float),
                              np.zeros(len(stops)), np.zeros(len(stops))])
    scale = {}
    for i, (label, values) in enumerate([('incoming', incoming), ('outgoing', outgoing)], 2):
        observed = np.log1p(values.loc[usable].to_numpy())
        if len(observed) < 2 or observed.std() == 0:
            raise ValueError('Insufficient nonzero unique metro covariates')
        matrix[usable, i] = (observed-observed.mean())/observed.std()
        scale[label] = dict(log_mean=float(observed.mean()), log_sd=float(observed.std()))
    return matrix, dict(quarter_end=latest.period_end.max(), named=int(names.notna().sum()),
        usable=int(usable.sum()), ambiguous=int(names.map(mapping).gt(1).sum()),
        standardization=scale, availability='quarter_completed_publication_unknown')


def augment(base, occurrence, groups, edges, values, mode):
    v = values[occurrence, :1] if mode == 'name' else values[occurrence]
    hours = np.repeat(groups.hour.to_numpy(), np.diff(edges))
    weekend = np.repeat(groups.weekend.to_numpy(), np.diff(edges))
    return np.column_stack([base, v, v*((hours >= 6)&(hours <= 10))[:, None],
                            v*((hours >= 16)&(hours <= 20))[:, None], v*weekend[:, None]])


def metro_fit(train, stops, x, edges, groups, old_beta):
    counts = groups.merge(train.groupby(['route', 'weekend', 'hour']).boardings.agg(['size', 'sum']),
                          on=['route', 'weekend', 'hour'], how='left').fillna(0)
    centre = np.zeros(x.shape[1]); precision = np.ones(x.shape[1]); precision[:4] = .25
    for j, route in enumerate(ROUTES):
        centre[j] = np.log(max(train.loc[train.route.eq(route), 'boardings'].mean(), 1)/sum(stops.route.eq(route)))
    args = (x, edges, counts['size'].to_numpy(), counts['sum'].to_numpy(), centre, precision, 'poisson')
    candidates, failures = [], []
    for delta in [0., -.05, .05]:
        start = np.r_[old_beta, np.full(x.shape[1]-len(old_beta), delta)]
        result = minimize(objective, start, args=args, jac=True, hess=curvature,
                          method='trust-exact', options={'maxiter': 1000, 'gtol': 1e-5})
        # Large-count objectives can hit floating-point precision before gtol.
        # Accept only an independently recomputed small absolute gradient.
        value, gradient = objective(result.x, *args)
        if np.isfinite(value) and abs(gradient).max() < .05:
            candidates.append((value, result.x, gradient, result.nit))
        else:
            failures.append(dict(message=str(result.message), gradient_max=float(abs(gradient).max())))
    if not candidates:
        raise RuntimeError(f'No converged metro fit: {failures}')
    value, beta, gradient, iterations = min(candidates, key=lambda a:a[0])
    return beta, dict(objective=float(value), gradient_max=float(abs(gradient).max()),
        iterations=int(iterations), successful_starts=len(candidates), failures=failures,
        prior_mean=centre.tolist(), prior_sd=(1/np.sqrt(precision)).tolist())


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((PARENT/'manifest.json').read_text())
    parent = json.loads((PARENT/'results.json').read_text())
    input_paths = [PARENT/'manifest.json', PARENT/'results.json', PARENT/'occurrences.csv', PARENT/'spatial_features.npy']
    for p in input_paths[1:]:
        if sha(p) != manifest['outputs'][p.name]:
            raise ValueError('Changed parent output')
    stops = pd.read_csv(PARENT/'occurrences.csv', sep=';')
    z = np.load(PARENT/'spatial_features.npy', allow_pickle=False)
    base, edges, groups, occurrence = design(stops, z)
    metro = metro_history(METRO, date(2025, 10, 31))
    historical_manifest = HERE/'sources/historical_manifest_20260927.json'
    source = next(a for a in json.loads(historical_manifest.read_text())['artifacts'] if a['path'].endswith('metro_passenger_traffic.csv'))
    if sha(METRO) != source['sha256']:
        raise ValueError('Changed metro source')
    history_path = LAB/'artifacts/hourly_clean.csv'
    if sha(history_path) != manifest['inputs'][str(history_path.relative_to(ROOT))]:
        raise ValueError('Changed route observations')
    history = pd.read_csv(history_path, sep=';', parse_dates=['date'])
    history = history.loc[history.route.isin(ROUTES) & history.hour.isin(HOURS) & history.working_events.gt(0)].copy()
    history['weekend'] = (history.date.dt.dayofweek >= 5).astype(int)
    results, models, coverage = [], {}, {}
    selected = None
    for wi, (cutoff, end) in enumerate(WINDOWS):
        values, coverage[cutoff] = features(stops, metro, cutoff)
        pd.DataFrame(values, columns=['explicit_name', 'usable_flow', 'log_incoming_standardized',
                                     'log_outgoing_standardized']).to_csv(out/f'features_{cutoff}.csv', sep=';', index=False)
        old_beta = np.asarray(parent['diagnostics'][f'{cutoff}/poisson_sd1.0']['beta'])
        old_log, old_share, _ = moments(old_beta, base, edges)
        train = history.loc[history.date.le(cutoff)]
        modes = ['name', 'flow'] if wi < 3 else ([selected] if selected != 'base' else [])
        if wi < 3:
            path = PARENT/f'route_predictions_{cutoff}.csv'; input_paths.append(path)
            if sha(path) != manifest['outputs'][path.name]:
                raise ValueError('Changed comparison rows')
            test = pd.read_csv(path, sep=';', parse_dates=['date'])
            test['weekend'] = (test.date.dt.dayofweek >= 5).astype(int)
            np.testing.assert_allclose(predict(test, groups, np.exp(old_log)), test.direct, rtol=1e-9)
            results += [dict(cutoff=cutoff, mode=mode, **metric(test, test[column]))
                        for mode, column in [('base', 'direct'), ('030', 'control030')]]
        else:
            test = pd.DataFrame()
        for mode in modes:
            x = augment(base, occurrence, groups, edges, values, mode)
            beta, diag = metro_fit(train, stops, x, edges, groups, old_beta)
            lm, shares, _ = moments(beta, x, edges)
            tv = np.add.reduceat(abs(shares-old_share), edges[:-1])/2
            models[f'{cutoff}/{mode}'] = dict(diag, beta=beta.tolist(), mean_share_tv_vs_base=float(tv.mean()))
            if len(test):
                prediction = predict(test, groups, np.exp(lm))
                results.append(dict(cutoff=cutoff, mode=mode, **metric(test, prediction)))
                test[mode] = prediction
            else:
                profile = stops.iloc[occurrence].reset_index(drop=True).copy()
                profile['hour'] = np.repeat(groups.hour, np.diff(edges)).to_numpy()
                profile['weekend'] = np.repeat(groups.weekend, np.diff(edges)).to_numpy()
                profile['share'] = shares
                profile['direct_expectation'] = np.exp(x@beta)
                profile['status'] = 'unvalidated_late_network_metro_scenario'
                profile.to_csv(out/'final_profiles.csv.gz', sep=';', index=False)
            print(cutoff, mode, 'gradient', diag['gradient_max'], flush=True)
        if len(test):
            test.to_csv(out/f'predictions_{cutoff}.csv', sep=';', index=False)
        if wi == 0:
            selected = min([r for r in results if r['cutoff']==cutoff and r['mode'] != '030'], key=lambda r:r['wape'])['mode']
            print('development selected:', selected, flush=True)
    write_json(out/'results.json', dict(version='M20260927-v2', selected=selected, metrics=results,
        models=models, coverage=coverage, stop_wape=None, stop_mae=None,
        final_profiles='final_profiles.csv.gz' if selected!='base' else None,
        final_base_reference='B20260927-v4' if selected=='base' else None))
    input_paths += [METRO, historical_manifest, history_path]+[HERE/n for n in
        ['metro_experiment.py', 'METRO_PROTOCOL.md', 'bayes_experiment.py', 'build_preliminary.py', 'audit.py']]
    write_json(out/'manifest.json', dict(version='M20260927-v2',
        inputs={str(p.relative_to(ROOT)):sha(p) for p in input_paths},
        outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(dict(selected=selected, metrics=results)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    with threadpool_limits(limits=1):
        run(parser.parse_args().out)
