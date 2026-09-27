"""Replay conditional alignment and verify coverage without claiming stop truth."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from audit import ROOT,sha
from catalog_alignment_experiment import ACTIVITY,HOLDOUT,SCENARIOS
from episode_alignment import align,episodes


def verify(folder):
    manifest=json.loads((folder/'manifest.json').read_text())
    for name,digest in manifest['inputs'].items():
        assert sha(ROOT/name)==digest,name
    for name,digest in manifest['outputs'].items():
        assert sha(folder/name)==digest,name
    summary=json.loads((folder/'summary.json').read_text())
    frames={name:pd.read_csv(folder/f'{name}.csv',sep=';') for name in
            ['catalog_occurrences','route12_cycle','registration_episodes','selected_episodes','alignment_candidates','conditional_evidence']}
    pilot=frames['route12_cycle'];targets=frames['selected_episodes']
    assert pilot.occurrence_id.is_unique and len(pilot)==97
    assert pilot.groupby('direction_id').size().to_dict()=={0:50,1:47}
    assert pilot.edge_is_assumed_turnaround.sum()==2
    assert not pilot.historical_validity_verified.any()
    assert set(pilot.start_date)=={'2025-12-20'}
    all_episodes=frames['registration_episodes']
    with np.load(ACTIVITY/'high_activity_probabilities.npz',allow_pickle=False) as data:
        high=data['holdout']
    with np.load(HOLDOUT/'binned_counts.npz',allow_pickle=False) as data:
        reference,held=data['reference'],data['held']
    replay=[]
    for row,(q,r,h) in enumerate(zip(high,reference,held)):
        for ordinal,item in enumerate(episodes(q,r,h)):
            replay.append(dict(hour_row=row,episode_id=f'{row}:{ordinal}',**item))
    replay=pd.DataFrame(replay)
    pd.testing.assert_frame_equal(replay,all_episodes[replay.columns],check_exact=False)
    assert summary['source_compared_events']==int(reference.sum()+held.sum())
    usable=all_episodes.loc[~all_episodes.censored]
    sizes=usable.groupby('hour_row').size();eligible=sizes.loc[sizes.ge(6)].index.to_numpy()
    selected=np.sort(np.random.default_rng(20260927).choice(eligible,min(64,len(eligible)),replace=False))
    pd.testing.assert_frame_equal(targets,usable.loc[usable.hour_row.isin(selected)].reset_index(drop=True))
    assert summary['selected_episode_events']==int((targets.reference_events+targets.held_events).sum())
    with np.load(folder/'conditional_probabilities.npz',allow_pickle=False) as data:
        probabilities=data['probabilities']
        np.testing.assert_array_equal(data['selected_hour_rows'],selected)
        np.testing.assert_array_equal(data['occurrence_ids'],pilot.occurrence_id)
    assert probabilities.shape==(6,len(targets),97) and np.isfinite(probabilities).all()
    assert (probabilities>=0).all()
    np.testing.assert_allclose(probabilities.sum(2),1,atol=1e-12)
    for si,(speed,dwell) in enumerate(SCENARIOS):
        candidates=frames['alignment_candidates'].loc[lambda x:x.speed_kmh.eq(speed)&x.dwell_seconds.eq(dwell)]
        np.testing.assert_array_equal(candidates.episode_id,targets.episode_id)
        np.testing.assert_array_equal(candidates.candidate_occurrence_id,pilot.occurrence_id.to_numpy()[probabilities[si].argmax(1)])
        np.testing.assert_allclose(candidates.conditional_max_probability,probabilities[si].max(1),atol=1e-12)
        for hour,g in targets.groupby('hour_row',sort=True):
            p,z=align(g.centre_seconds,pilot.edge_straight_distance_m,pilot.edge_is_assumed_turnaround,speed,dwell)
            np.testing.assert_allclose(p,probabilities[si,g.index],atol=1e-12)
            evidence=frames['conditional_evidence'].loc[lambda x:x.hour_row.eq(hour)&x.speed_kmh.eq(speed)&x.dwell_seconds.eq(dwell)]
            assert len(evidence)==1
            np.testing.assert_allclose(z,evidence.conditional_log_evidence.iloc[0],atol=1e-10)
    tv=[np.abs(probabilities[i]-probabilities[j]).sum(1)/2 for i in range(6) for j in range(i+1,6)]
    np.testing.assert_allclose(np.mean(tv),summary['mean_pairwise_scenario_tv'],atol=1e-12)
    winner=probabilities.argmax(2)
    np.testing.assert_allclose((winner==winner[0]).all(0).mean(),summary['same_top_occurrence_all_scenarios_fraction'])
    np.testing.assert_allclose(probabilities.max(2).mean(1),summary['mean_max_probability_by_scenario'])
    assert not frames['alignment_candidates'].accepted_assignment.any()
    assert summary['accepted_stop_assignments']==summary['independent_stop_labels']==0
    assert summary['stop_accuracy'] is None and not summary['unassigned_ledger_modified']
    print('Verified hashes, episode counts, frozen sampling, all 384 chains, candidate probabilities and uncertainty metrics.')


if __name__=='__main__':
    verify(Path(sys.argv[1]))
