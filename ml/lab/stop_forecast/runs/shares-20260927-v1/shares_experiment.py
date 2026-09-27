"""Prior-only entropy diagnostics; deliberately no supervised BL–FS fitting."""
import argparse
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy
from threadpoolctl import threadpool_limits

from audit import ROOT, sha, write_json
from bayes_experiment import design, WINDOWS
from fractional_split import entropy_projection

HERE = Path(__file__).resolve().parent
PARENT = HERE/'runs/bayes-20260927-v4'
VERSION = 'F20260927-v1'
STRENGTHS = [0., .1, 1., 10., 100.]


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    parent_manifest = json.loads((PARENT/'manifest.json').read_text())
    inputs = [PARENT/name for name in ['manifest.json', 'results.json', 'occurrences.csv', 'spatial_features.npy']]
    for path in inputs[1:]:
        if sha(path) != parent_manifest['outputs'][path.name]:
            raise ValueError(f'Changed parent artifact: {path.name}')
    for name in ['bayes_experiment.py', 'audit.py']:
        relative = str((HERE/name).relative_to(ROOT))
        if sha(HERE/name) != parent_manifest['inputs'][relative]:
            raise ValueError(f'Changed parent feature implementation: {name}')
    old = json.loads((PARENT/'results.json').read_text())
    f = pd.read_csv(PARENT/'occurrences.csv', sep=';', dtype={'trip_id': str, 'direction_id': str, 'stop_id': str})
    z = np.load(PARENT/'spatial_features.npy', allow_pickle=False)
    x, edges, groups, occurrence = design(f, z)
    rows, controls, profiles = [], [], []
    for cutoff, end in WINDOWS:
        name = f"{cutoff}/{old['selected']['likelihood']}_sd{old['selected']['prior_sd']}"
        utility = x@np.asarray(old['diagnostics'][name]['beta'])
        baseline = entropy_projection(utility, edges, np.ones(len(x)), 0.)
        by_prior = {}
        for prior_name in ['uniform', 'centre_scenario']:
            q = np.ones(len(x)) if prior_name == 'uniform' else np.exp(-.5*z[occurrence, 0])
            q = q/np.repeat(np.add.reduceat(q, edges[:-1]), np.diff(edges))
            for strength in STRENGTHS:
                p = entropy_projection(utility, edges, q, strength)
                if not np.allclose(np.add.reduceat(p, edges[:-1]), 1, atol=1e-12):
                    raise ValueError('Share balance failed')
                entropy = -np.add.reduceat(p*np.log(p), edges[:-1])
                kl = np.add.reduceat(p*np.log(p/q), edges[:-1])
                tv = np.add.reduceat(abs(p-baseline), edges[:-1])/2
                by_prior[(prior_name, strength)] = p
                rows.append(dict(cutoff=cutoff, prior=prior_name, strength=strength,
                    normalized_entropy=float(np.mean(entropy/np.log(np.diff(edges)))),
                    mean_kl_to_prior=float(kl.mean()), mean_tv_from_base=float(tv.mean())))
                if cutoff == '2025-10-31':
                    profile = f.iloc[occurrence].reset_index(drop=True).copy()
                    profile['hour'] = np.repeat(groups.hour, np.diff(edges)).to_numpy()
                    profile['weekend'] = np.repeat(groups.weekend, np.diff(edges)).to_numpy()
                    profile['prior'] = prior_name; profile['strength'] = strength; profile['share'] = p
                    profiles.append(profile)
        for row in rows:
            if row['cutoff'] == cutoff:
                a = by_prior[('uniform', row['strength'])]
                b = by_prior[('centre_scenario', row['strength'])]
                row['mean_tv_between_priors'] = float((np.add.reduceat(abs(a-b), edges[:-1])/2).mean())
        if cutoff != '2025-10-31':
            source = PARENT/f'route_predictions_{cutoff}.csv'
            inputs.append(source)
            if sha(source) != parent_manifest['outputs'][source.name]:
                raise ValueError('Changed route control')
            obs = pd.read_csv(source, sep=';')
            controls.append(dict(cutoff=cutoff, rows=len(obs),
                inherited_030_wape=float(abs(obs.boardings-obs.control030).sum()/obs.boardings.sum()),
                note='Same for all normalized allocations by construction; not stop accuracy'))
    full = pd.concat(profiles, ignore_index=True)
    full.to_csv(out/'scenario_profiles.csv.gz', sep=';', index=False)
    pilot = full.loc[full.route.eq(12) & full.prior.eq('uniform') & full.strength.eq(1.)].copy()
    source = PARENT/'route12_scenario.csv.gz'; inputs.append(source)
    if sha(source) != parent_manifest['outputs'][source.name]:
        raise ValueError('Changed parent forecast')
    parent = pd.read_csv(source, sep=';', dtype={'trip_id': str, 'direction_id': str, 'stop_id': str})
    keys = ['route', 'direction_id', 'trip_id', 'stop_sequence', 'stop_id', 'date', 'hour', 'weekend']
    result = parent[keys+['prediction', 'start_date', 'actual_date', 'geography_status']].merge(
        pilot[['route', 'trip_id', 'stop_sequence', 'weekend', 'hour', 'share']],
        on=['route', 'trip_id', 'stop_sequence', 'weekend', 'hour'], how='left', validate='many_to_one')
    working = ~result.hour.between(1, 4)
    if result.loc[working, 'share'].isna().any():
        raise ValueError('Missing working share')
    result['boardings'] = result.prediction*result.share.fillna(0)
    result['nonzero_probability'] = np.nan
    result['status'] = 'prior_only_entropy_scenario_not_fitted_blfs'
    result['support_assumption'] = 'all_snapshot_positions_available'
    result['model_version'] = VERSION
    result['strength'] = 1.; result['prior'] = 'uniform'
    if len(result) != 97*61*24 or result.duplicated(keys[:-1]).any():
        raise ValueError('Invalid scenario grid')
    balance = result.groupby(['date', 'hour']).agg(expected=('prediction', 'first'), actual=('boardings', 'sum'))
    if not np.allclose(balance.expected, balance.actual, atol=1e-9):
        raise ValueError('Route mass lost')
    result.to_csv(out/'route12_scenario.csv.gz', sep=';', index=False)
    write_json(out/'results.json', dict(version=VERSION,
        supervised_blfs_status='implemented_not_fitted_missing_real_stop_labels',
        joint_panel_status='not_implemented', stop_wape=None, stop_mae=None,
        verified_stop_coverage=0, entropy_diagnostics=rows, route_controls=controls,
        export=dict(rows=len(result), strength=1., prior='uniform', selection='illustrative_not_accuracy_selected'),
        max_route_balance_error=float(abs(balance.expected-balance.actual).max())))
    inputs += [HERE/n for n in ['fractional_split.py', 'shares_experiment.py', 'SHARES_PROTOCOL.md',
                                'bayes_experiment.py', 'audit.py']]
    write_json(out/'manifest.json', dict(version=VERSION, timezone='Europe/Moscow',
        forecast_from='2025-11-01', forecast_to_inclusive='2025-12-31',
        status='unvalidated_scenario', occupancy_available=False,
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__),
        inputs={str(p.relative_to(ROOT)): sha(p) for p in inputs},
        outputs={p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}))
    print(json.dumps(dict(final_diagnostics=[r for r in rows if r['cutoff']=='2025-10-31'],
                          rows=len(result), balance_error=float(abs(balance.expected-balance.actual).max()))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    with threadpool_limits(limits=1):
        run(parser.parse_args().out)
