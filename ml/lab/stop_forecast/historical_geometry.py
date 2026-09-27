"""Historical OSM support and catalog correspondence candidates, never stop labels."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from audit import ROOT,sha,write_json
from build_metro_access import distances

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'sources/route12_osm_history_20260927.json'
CATALOG=HERE/'runs/catalog-alignment-20260927-v1'
# Direction labels follow endpoint names, not relation ID order or an OSM direction tag.
PATTERNS={14258874:(0,'Восточное Измайлово','МЦК Дубровка'),
          14258875:(1,'МЦК Дубровка','Восточное Измайлово')}
ROLES={'stop','stop_entry_only','stop_exit_only'}


def occurrences(data,asof):
    if data.get('remark') or not data.get('osm3s',{}).get('copyright'):
        raise ValueError('Incomplete OSM response or missing attribution')
    elements=data['elements']
    if any(set(e)&{'user','uid','changeset','timestamp','version'} for e in elements):
        raise ValueError('Use out body, without editor metadata')
    if len({(e['type'],e['id']) for e in elements})!=len(elements):
        raise ValueError('Duplicate OSM elements')
    nodes={e['id']:e for e in elements if e['type']=='node'}
    relations={e['id']:e for e in elements if e['type']=='relation'}
    if relations.keys()!=PATTERNS.keys():
        raise ValueError('Unexpected route12 patterns')
    rows=[]
    for rid,(direction,origin,destination) in PATTERNS.items():
        relation=relations[rid];tags=relation['tags']
        if (tags.get('ref'),tags.get('route'),tags.get('from'),tags.get('to'))!=('12','tram',origin,destination):
            raise ValueError('Review route or direction mapping')
        sequence=0
        for index,member in enumerate(relation['members']):
            if member['role'] not in ROLES:
                continue
            if member['type']!='node' or member['ref'] not in nodes:
                raise ValueError('Missing stop node')
            node=nodes[member['ref']];sequence+=1
            rows.append(dict(route=12,direction=direction,source_pattern_id=f'osm:relation/{rid}',
                source_stop_id=f'osm:node/{node["id"]}',stop_sequence=sequence,member_index=index,
                occurrence_id=f'osm-history:{rid}:{index}',stop_name=node.get('tags',{}).get('name',''),
                stop_lat=node['lat'],stop_lon=node['lon'],role=member['role'],
                boarding_permitted_by_osm=member['role']!='stop_exit_only',osm_asof=asof,
                relation_check_date=tags.get('check_date'),actual_service_verified=False))
        if sequence<2:
            raise ValueError('Insufficient stop sequence')
    f=pd.DataFrame(rows)
    distances(f.stop_lat,f.stop_lon,f.stop_lat,f.stop_lon)  # shared coordinate validation
    return f


def run(out):
    receipt=json.loads(SOURCE.read_text());inputs=[SOURCE,Path(__file__),HERE/'audit.py',HERE/'build_metro_access.py']
    snapshots=[]
    for item in receipt['snapshots']:
        path=ROOT/item['file']
        if sha(path)!=item['sha256']:
            raise ValueError('Changed historical snapshot')
        inputs.append(path)
        if item['purpose']=='geometry':
            snapshots.append(occurrences(json.loads(path.read_text()),item['requested_osm_date']))
    if len(snapshots)!=2:
        raise ValueError('Expected two boundary snapshots')
    f,end=snapshots
    pd.testing.assert_frame_equal(f.drop(columns='osm_asof'),end.drop(columns='osm_asof'))
    path=CATALOG/'route12_cycle.csv';manifest=CATALOG/'manifest.json'
    if sha(path)!=json.loads(manifest.read_text())['outputs'][path.name]:
        raise ValueError('Changed catalog baseline')
    late=pd.read_csv(path,sep=';');inputs.extend([path,manifest])
    pairs=[];coverage=[]
    # ponytail: fewer than 100 positions per pattern; a pairwise matrix is sufficient.
    for direction,g in f.groupby('direction'):
        catalog=late.loc[late.direction_id.eq(direction)]
        d=distances(g.stop_lat,g.stop_lon,catalog.stop_lat,catalog.stop_lon)
        for i,row in enumerate(g.itertuples()):
            for j in np.flatnonzero(d[i]<=100):
                pairs.append(dict(historical_occurrence_id=row.occurrence_id,
                    catalog_occurrence_id=catalog.iloc[j].occurrence_id,distance_m=float(d[i,j]),
                    verified_identity=False))
        for j,row in enumerate(catalog.itertuples()):
            i=int(d[:,j].argmin())
            coverage.append(dict(catalog_occurrence_id=row.occurrence_id,stop_name=row.stop_name,
                direction=direction,stop_mode=row.stop_mode,
                nearest_historical_occurrence_id=g.iloc[i].occurrence_id,
                nearest_distance_m=float(d[i,j]),candidates_within_100m=int((d[:,j]<=100).sum()),
                verified_identity=False))
    pairs=pd.DataFrame(pairs);coverage=pd.DataFrame(coverage)
    summary=dict(version='H20260927-v1',historical_occurrences=len(f),catalog_occurrences=len(late),
        counts_by_direction={str(k):int(v) for k,v in f.groupby('direction').size().items()},
        osm_stop_roles=f.role.value_counts().to_dict(),unchanged_between_boundary_snapshots=True,
        stability_between_boundaries_proven=False,actual_service_verified=False,
        snapshot_dates=[s.osm_asof.iloc[0] for s in snapshots],
        pairs_within_100m=len(pairs),catalog_positions_without_candidate=int(coverage.candidates_within_100m.eq(0).sum()),
        catalog_positions_with_multiple_candidates=int(coverage.candidates_within_100m.gt(1).sum()),
        historical_positions_without_candidate=int((~f.occurrence_id.isin(pairs.historical_occurrence_id)).sum()),
        verified_stop_assignments=0,stop_accuracy=None,forecast_artifact=None)
    out.mkdir(parents=True,exist_ok=False)
    f.to_csv(out/'historical_occurrences.csv',sep=';',index=False)
    pairs.to_csv(out/'catalog_candidates.csv',sep=';',index=False)
    coverage.to_csv(out/'catalog_coverage.csv',sep=';',index=False)
    write_json(out/'summary.json',summary)
    write_json(out/'manifest.json',dict(version=summary['version'],
        inputs={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args().out)
