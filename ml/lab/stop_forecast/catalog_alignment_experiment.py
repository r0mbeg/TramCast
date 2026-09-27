"""Conditional candidate alignment, with all trusted assignments blocked."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from audit import ROOT,LAB,sha,write_json
from catalog_sequences import catalog,pilot_cycle
from episode_alignment import align,episodes

HERE=Path(__file__).resolve().parent
ACTIVITY=HERE/'runs/activity-20260927-v1'
HOLDOUT=HERE/'runs/device-holdout-20260927-v1'
SCENARIOS=[(speed,dwell) for speed in [10,15,20] for dwell in [0,15]]


def run(out):
    f,catalog_summary,inputs=catalog()
    pilot,length,turn=pilot_cycle(f)
    for folder,names in [(ACTIVITY,['high_activity_probabilities.npz','holdout_scores.csv']),
                         (HOLDOUT,['binned_counts.npz'])]:
        manifest_path=folder/'manifest.json';inputs.append(manifest_path)
        manifest=json.loads(manifest_path.read_text())
        for name in names:
            path=folder/name
            if sha(path)!=manifest['outputs'][name]:
                raise ValueError('Changed activity input')
            inputs.append(path)
    scores=pd.read_csv(ACTIVITY/'holdout_scores.csv',sep=';')
    with np.load(ACTIVITY/'high_activity_probabilities.npz',allow_pickle=False) as data:
        high=data['holdout']
    with np.load(HOLDOUT/'binned_counts.npz',allow_pickle=False) as data:
        reference,held=data['reference'],data['held']
    if high.shape!=reference.shape or high.shape!=held.shape or len(high)!=len(scores):
        raise ValueError('Unaligned hourly matrices')
    episode_rows=[]
    for row,(q,r,h) in enumerate(zip(high,reference,held)):
        for ordinal,item in enumerate(episodes(q,r,h)):
            episode_rows.append(dict(hour_row=row,episode_id=f'{row}:{ordinal}',
                vehicle_day=int(scores.iloc[row].vehicle_day),date=scores.iloc[row].date,
                hour=int(scores.iloc[row].hour),**item))
    all_episodes=pd.DataFrame(episode_rows)
    usable=all_episodes.loc[~all_episodes.censored]
    sizes=usable.groupby('hour_row').size()
    eligible=sizes.loc[sizes.ge(6)].index.to_numpy()
    if not len(eligible):
        raise ValueError('No eligible episode sequences')
    selected=np.sort(np.random.default_rng(20260927).choice(eligible,min(64,len(eligible)),replace=False))
    targets=usable.loc[usable.hour_row.isin(selected)].reset_index(drop=True)
    probabilities=[];likelihoods=[]
    for speed,dwell in SCENARIOS:
        block=[]
        for hour,g in targets.groupby('hour_row',sort=True):
            p,z=align(g.centre_seconds.to_numpy(),length,turn,speed,dwell)
            block.append(p);likelihoods.append(dict(hour_row=int(hour),speed_kmh=speed,dwell_seconds=dwell,
                                                    conditional_log_evidence=z))
        probabilities.append(np.concatenate(block))
    probabilities=np.asarray(probabilities)
    candidate_rows=[]
    for si,(speed,dwell) in enumerate(SCENARIOS):
        for ei,p in enumerate(probabilities[si]):
            best=int(p.argmax());occ=pilot.iloc[best]
            candidate_rows.append(dict(episode_id=targets.iloc[ei].episode_id,speed_kmh=speed,dwell_seconds=dwell,
                candidate_occurrence_id=occ.occurrence_id,candidate_stop_name=occ.stop_name,
                candidate_pattern_id=occ.source_pattern_id,candidate_direction=int(occ.direction_id),
                conditional_max_probability=float(p[best]),accepted_assignment=False,
                status='conditional_late_catalog_scenario_unverified'))
    tv=[np.abs(probabilities[i]-probabilities[j]).sum(1)/2 for i in range(6) for j in range(i+1,6)]
    winner=probabilities.argmax(2)
    maximum=probabilities.max(2)
    episode_counts=int((targets.reference_events+targets.held_events).sum())
    summary=dict(version='C20260927-v1',catalog=catalog_summary,pilot_positions=len(pilot),
        pilot_start_dates=sorted(pilot.start_date.unique()),assumed_turnaround_links=int(turn.sum()),
        direct_edge_distance_sum_m=float(length.sum()),episodes_total=len(all_episodes),
        uncensored_episodes=len(usable),eligible_hours=len(eligible),selected_hours=len(selected),
        selected_episodes=len(targets),selected_episode_events=episode_counts,
        source_compared_events=int(held.sum()+reference.sum()),
        mean_pairwise_scenario_tv=float(np.mean(tv)),
        same_top_occurrence_all_scenarios_fraction=float((winner==winner[0]).all(0).mean()),
        mean_max_probability_by_scenario=[float(v) for v in maximum.mean(1)],
        max_probability_ge_08_fraction_by_scenario=[float(v) for v in (maximum>=.8).mean(1)],
        accepted_stop_assignments=0,independent_stop_labels=0,stop_accuracy=None,
        historical_network_verified=False,absolute_time_anchor_available=False,
        unassigned_ledger_modified=False,scenario_parameters=[dict(speed_kmh=s,dwell_seconds=d) for s,d in SCENARIOS])
    out.mkdir(parents=True,exist_ok=False)
    f.to_csv(out/'catalog_occurrences.csv',sep=';',index=False)
    pilot.to_csv(out/'route12_cycle.csv',sep=';',index=False)
    all_episodes.to_csv(out/'registration_episodes.csv',sep=';',index=False)
    targets.to_csv(out/'selected_episodes.csv',sep=';',index=False)
    pd.DataFrame(candidate_rows).to_csv(out/'alignment_candidates.csv',sep=';',index=False)
    pd.DataFrame(likelihoods).to_csv(out/'conditional_evidence.csv',sep=';',index=False)
    np.savez_compressed(out/'conditional_probabilities.npz',probabilities=probabilities,
                        selected_hour_rows=selected,occurrence_ids=pilot.occurrence_id.to_numpy(str))
    write_json(out/'summary.json',summary)
    inputs+=[HERE/n for n in ['catalog_alignment_experiment.py','catalog_sequences.py','episode_alignment.py',
                              'CATALOG_ALIGNMENT_PROTOCOL.md','build_metro_access.py','audit.py']]
    inputs.append(LAB/'artifacts/methodology_20260927/data/audit_data.py')
    write_json(out/'manifest.json',dict(version='C20260927-v1',inputs={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args().out)
