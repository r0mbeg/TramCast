"""B20260927-v1: aggregate-only research; no observed stop labels."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

from audit import LAB, ROOT, prior, sha, write_json

ROUTES = [1, 7, 11, 12]
HOURS = [0] + list(range(5, 24))
WINDOWS = [('2025-04-30', '2025-06-30'), ('2025-06-30', '2025-08-30'),
           ('2025-08-31', '2025-10-31'), ('2025-10-31', '2025-12-31')]
CONTROL = LAB / 'artifacts/portfolio_20260926/continuation/tabular_shape/study/tabular_shape/selected'


def geography():
    for book in sorted((ROOT / 'dataset/spravochniki').glob('*.xlsx')):
        for name, _, _, records in prior.sheets(book):
            if name == 'Порядок_с_координатами':
                f = pd.DataFrame(records).rename(columns={'route_short_name': 'route'})
                f['route'] = f.route.astype(int)
                f = f.loc[f.route.isin(ROUTES)].copy()
                for c in ['stop_sequence', 'stop_lat', 'stop_lon']:
                    f[c] = pd.to_numeric(f[c], errors='raise')
                f = f.sort_values(['route', 'trip_id', 'stop_sequence']).reset_index(drop=True)
                keys = ['route', 'direction_id', 'trip_id', 'stop_sequence']
                if f.duplicated(keys).any() or set(f.route) != set(ROUTES):
                    raise ValueError('Invalid scenario support')
                g = f.groupby(['route', 'trip_id']).stop_sequence
                terminal = (f.stop_sequence.eq(g.transform('min')) | f.stop_sequence.eq(g.transform('max')))
                z = np.column_stack([np.hypot((f.stop_lat-55.75)*111,
                                              (f.stop_lon-37.62)*62), f.stop_lat, terminal.astype(float)])
                z = (z-z.mean(axis=0))/z.std(axis=0)
                f['geography_status'] = 'late_snapshot_not_verified_for_2025'
                return f, z, book
    raise ValueError('No workbook occurrence geography')


def design(f, z):
    blocks, groups, occurrence = [], [], []
    for route in ROUTES:
        ix = np.flatnonzero(f.route.eq(route))
        for weekend in [0, 1]:
            for hour in HOURS:
                x = np.zeros((len(ix), 56))
                x[:, ROUTES.index(route)] = 1
                x[:, 4+weekend*20+HOURS.index(hour)] = 1
                x[:, 44:] = np.column_stack([z[ix], z[ix]*(6 <= hour <= 10),
                                             z[ix]*(16 <= hour <= 20), z[ix]*weekend])
                blocks.append(x); groups.append((route, weekend, hour)); occurrence.extend(ix)
    sizes = np.array([len(x) for x in blocks])
    return np.concatenate(blocks), np.r_[0, sizes.cumsum()], pd.DataFrame(groups,
        columns=['route', 'weekend', 'hour']), np.array(occurrence)


def moments(beta, x, edges):
    eta = x @ beta
    maxima = np.maximum.reduceat(eta, edges[:-1])
    logm = maxima + np.log(np.add.reduceat(np.exp(eta-np.repeat(maxima, np.diff(edges))), edges[:-1]))
    p = np.exp(eta-np.repeat(logm, np.diff(edges)))
    jac = np.add.reduceat(x*p[:, None], edges[:-1], axis=0)
    return logm, p, jac


def objective(beta, x, edges, n, y, centre, precision, kind):
    lm, _, jac = moments(beta, x, edges)
    m = np.exp(lm)
    # Subtract the saturated likelihood: same optimum, better stopping precision.
    logmean = np.log(np.maximum(y/np.maximum(n, 1), 1e-300))
    if kind == 'poisson':
        loss, derivative = n*m-y-y*(lm-logmean), n*m-y
    else:
        loss = (y+n*10)*(np.logaddexp(np.log(10), lm)-
                         np.logaddexp(np.log(10), logmean))-y*(lm-logmean)
        derivative = (n*m-y)*10/(10+m)
    delta = beta-centre
    return float(loss.sum()+0.5*np.dot(delta*precision, delta)), jac.T@derivative+precision*delta


def fit(train, f, x, edges, groups, kind, sd, shift=0):
    counts = train.groupby(['route', 'weekend', 'hour']).boardings.agg(['size', 'sum'])
    counts = groups.merge(counts, on=['route', 'weekend', 'hour'], how='left').fillna(0)
    centre = np.zeros(x.shape[1]); centre[44] = shift
    for j, route in enumerate(ROUTES):
        centre[j] = np.log(max(train.loc[train.route.eq(route), 'boardings'].mean(), 1)/sum(f.route.eq(route)))
    precision = np.full(len(centre), 1/sd**2); precision[:4] = 0.25
    args = (x, edges, counts['size'].to_numpy(), counts['sum'].to_numpy(), centre, precision, kind)
    fits = []
    for delta in [0, -0.2, 0.2]:
        start = centre.copy(); start[44:] += delta
        result = minimize(objective, start, args=args, jac=True, method='L-BFGS-B',
                          options={'maxiter': 2500, 'maxcor': 50, 'ftol': 1e-12, 'gtol': 1e-6})
        if result.success:
            fits.append(result)
    if not fits:
        raise RuntimeError(result.message)
    best = min(fits, key=lambda a: a.fun)
    refined = minimize(objective, best.x, args=args, jac=True, method='BFGS',
                       options={'maxiter': 500, 'gtol': 1e-5})
    if refined.fun <= best.fun+1e-7:
        best = refined
    if abs(best.jac).max() > 0.05:
        raise RuntimeError(f'Insufficient numerical convergence: {abs(best.jac).max()}')
    return best.x, args, dict(objective=float(best.fun), iterations=int(best.nit),
        gradient_max=float(abs(best.jac).max()), successful_starts=len(fits),
        prior_mean=centre.tolist(), prior_sd=(1/np.sqrt(precision)).tolist())


def metric(frame, pred):
    error = abs(frame.boardings.to_numpy()-np.asarray(pred))
    return dict(rows=len(frame), wape=float(error.sum()/frame.boardings.sum()), mae=float(error.mean()))


def predict(frame, groups, values):
    return frame.merge(groups.assign(prediction=values), on=['route', 'weekend', 'hour'],
                       how='left', validate='many_to_one').prediction.to_numpy()


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    history_path = LAB/'artifacts/hourly_clean.csv'
    history = pd.read_csv(history_path, sep=';', parse_dates=['date'])
    if history.duplicated(['route', 'date', 'hour']).any():
        raise ValueError('Duplicate route hours')
    history['weekend'] = (history.date.dt.dayofweek >= 5).astype(int)
    history = history.loc[history.route.isin(ROUTES) & history.hour.isin(HOURS)].copy()
    history['usable'] = history.working_events.gt(0)
    f, z, book = geography()
    x, edges, groups, occurrence = design(f, z)
    f.to_csv(out/'occurrences.csv', sep=';', index=False)
    np.save(out/'spatial_features.npy', z)
    inputs = [history_path, book, Path(__file__), Path(__file__).with_name('BAYES_PROTOCOL.md'),
              Path(__file__).with_name('audit.py'), LAB/'artifacts/methodology_20260927/data/audit_data.py']
    metrics, diagnostics, sensitivity = [], {}, []
    selected = None
    for wi, (cutoff, end) in enumerate(WINDOWS):
        train = history.loc[history.date.le(cutoff) & history.usable]
        test = history.loc[history.date.gt(cutoff) & history.date.le(end) & history.usable].copy()
        candidates = [(k, s) for k in ['poisson', 'nb'] for s in [0.25, 1.]] if wi == 0 else [selected]
        window_metrics, fitted = [], {}
        for kind, sd in candidates:
            beta, args, diag = fit(train, f, x, edges, groups, kind, sd)
            lm, p, jac = moments(beta, x, edges)
            name = f'{kind}_sd{sd}'
            fitted[(kind, sd)] = (beta, args, p, jac)
            diagnostics[f'{cutoff}/{name}'] = dict(diag, beta=beta.tolist())
            print(cutoff, name, diag['iterations'], 'iterations', flush=True)
            if len(test):
                row = dict(cutoff=cutoff, model=name, **metric(test, predict(test, groups, np.exp(lm))))
                window_metrics.append(row)
        if wi == 0:
            best = min(window_metrics, key=lambda a: a['wape'])['model']
            selected = next(c for c in candidates if f'{c[0]}_sd{c[1]}' == best)
        metrics.extend(window_metrics)
        beta, args, shares, jac = fitted[selected]
        lm, _, _ = moments(beta, x, edges)
        groups.assign(prediction=np.exp(lm)).to_csv(out/f'profile_{cutoff}.csv', sep=';', index=False)
        singular = np.linalg.svd(jac, compute_uv=False)
        diagnostics[cutoff] = dict(jacobian_rank=int(sum(singular > singular[0]*1e-8)),
            parameters=len(beta), singular_values=singular.tolist())
        if len(test):
            diagnostics[cutoff]['observed_hours'] = len(test)
            diagnostics[cutoff]['expected_working_hours'] = 4*61*20
            base = train.groupby(['route', 'weekend', 'hour']).boardings.mean().rename('prediction')
            basepred = test.merge(base, on=['route', 'weekend', 'hour'], validate='many_to_one').prediction
            metrics.append(dict(cutoff=cutoff, model='seasonal_mean', **metric(test, basepred)))
            control_path = CONTROL/f'raw_{cutoff}.csv'
            control = pd.read_csv(control_path, sep=';', parse_dates=['date'])
            inputs.append(control_path)
            joined = test.merge(control[['route', 'date', 'hour', 'prediction']],
                                on=['route', 'date', 'hour'], how='left', validate='one_to_one')
            if joined.prediction.isna().any():
                raise ValueError('Incomplete 030 control')
            metrics.append(dict(cutoff=cutoff, model='030_saved', **metric(test, joined.prediction)))
            test['direct'] = predict(test, groups, np.exp(lm)); test['control030'] = joined.prediction.to_numpy()
            test[['route', 'date', 'hour', 'boardings', 'direct', 'control030']].to_csv(
                out/f'route_predictions_{cutoff}.csv', sep=';', index=False)
            for col in ['route', 'hour']:
                for key, sub in test.groupby(col):
                    metrics.append(dict(cutoff=cutoff, model='selected', slice=f'{col}={key}', **metric(sub, sub.direct)))
        if wi == 0:
            for shift in [-1, 0, 1]:
                b, _, _ = fit(train, f, x, edges, groups, *selected, shift=shift)
                newlm, newp, _ = moments(b, x, edges)
                tv = np.add.reduceat(abs(newp-shares), edges[:-1])/2
                sensitivity.append(dict(prior_shift=shift, mean_tv=float(tv.mean()), max_tv=float(tv.max()),
                    **metric(test, predict(test, groups, np.exp(newlm)))))
            # Exact non-identifiability of the nested model without interactions.
            b = beta.copy(); b[47:] = 0
            before, p0, _ = moments(b, x, edges)
            altered = b.copy(); altered[44] += 1
            after, _, _ = moments(altered, x, edges)
            for j in range(4):
                altered[j] += before[j*40]-after[j*40]
            corrected, p1, _ = moments(altered, x, edges)
            diagnostics['invariance'] = dict(max_log_route_difference=float(abs(before-corrected).max()),
                mean_share_tv=float((np.add.reduceat(abs(p1-p0), edges[:-1])/2).mean()))
        if wi == 3:
            # Laplace approximation, conditional on the chosen model and snapshot.
            eps = 1e-4; eye = np.eye(len(beta))*eps
            hessian = np.column_stack([(objective(beta+e, *args)[1]-objective(beta-e, *args)[1])/(2*eps) for e in eye])
            hessian = (hessian+hessian.T)/2
            eigen = np.linalg.eigvalsh(hessian)
            if eigen.min() <= 0:
                raise ValueError('Nonpositive posterior curvature')
            draws = np.random.default_rng(20260927).multivariate_normal(beta, np.linalg.inv(hessian), 200)
            sampled = np.array([moments(b, x, edges)[1] for b in draws])
            lo, hi = np.quantile(sampled, [0.05, 0.95], axis=0)
            profile = f.iloc[occurrence].reset_index(drop=True)
            profile['hour'] = np.repeat(groups.hour, np.diff(edges)).to_numpy()
            profile['weekend'] = np.repeat(groups.weekend, np.diff(edges)).to_numpy()
            profile['share'] = shares; profile['share_p05'] = lo; profile['share_p95'] = hi
            profile['direct_boardings'] = np.exp(x@beta)
            profile.to_csv(out/'scenario_profiles.csv.gz', sep=';', index=False)
            pilot = profile.loc[profile.route.eq(12)].copy()
            dates = pd.DataFrame({'date': pd.date_range('2025-11-01', end)})
            dates['weekend'] = (dates.date.dt.dayofweek >= 5).astype(int)
            scenario = dates.merge(pilot, on='weekend')
            night = dates.merge(f.loc[f.route.eq(12)], how='cross').merge(
                pd.DataFrame({'hour': [1, 2, 3, 4]}), how='cross')
            night['direct_boardings'] = 0.
            scenario = pd.concat([scenario, night], ignore_index=True)
            control_path = CONTROL/f'raw_{cutoff}.csv'
            inputs.append(control_path)
            control = pd.read_csv(control_path, sep=';', parse_dates=['date'])
            scenario = scenario.merge(control[['route', 'date', 'hour', 'prediction']],
                on=['route', 'date', 'hour'], how='left', validate='many_to_one')
            if scenario.prediction.isna().any():
                raise ValueError('Missing final 030 control')
            scenario['reconciled030_boardings'] = scenario.prediction*scenario.share.fillna(0)
            scenario['equal030_boardings'] = scenario.prediction/len(pilot[pilot.hour.eq(0) & pilot.weekend.eq(0)])
            scenario['status'] = 'unvalidated_late_geography_scenario'
            scenario['model_version'] = 'B20260927-v3'
            scenario.to_csv(out/'route12_scenario.csv.gz', sep=';', index=False)
            diagnostics['laplace'] = dict(min_hessian_eigenvalue=float(eigen.min()), draws=200,
                intervals='conditional approximation, uncalibrated; no stop truth')
    write_json(out/'results.json', dict(selected=dict(zip(['likelihood', 'prior_sd'], selected)),
        metrics=metrics, sensitivity=sensitivity, diagnostics=diagnostics, stop_wape=None,
        stop_mae=None, verified_stop_coverage=0, scope='late-geography scenario, 4 routes',
        observation_status='route counters observed; completeness unknown'))
    import platform, scipy
    write_json(out/'manifest.json', dict(version='B20260927-v3',
        environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__), inputs={str(p.relative_to(ROOT)): sha(p) for p in inputs},
        outputs={p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}))
    print(json.dumps(dict(selected=selected, metrics=[r for r in metrics if 'slice' not in r], sensitivity=sensitivity)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    with threadpool_limits(limits=1):
        run(parser.parse_args().out)
