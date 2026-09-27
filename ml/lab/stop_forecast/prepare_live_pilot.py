"""Prepare current route positions and a bounded stop roster, not boarding labels."""
import argparse
import json
import math
from pathlib import Path
from uuid import UUID
from collect_stop_evidence import digest,write

HERE=Path(__file__).resolve().parent
TERMS=['Первомайская','Партизанская','Семёновская','Авиамоторная','Пролетарская','МЦК Дубровка']


def positions(data):
    if not isinstance(data,dict) or data.get('number')!='12' or data.get('type')!='tram':
        raise ValueError('Expected tram 12, not a similarly numbered bus')
    UUID(data['id']);rows=[];seen=set()
    for group_index,group in enumerate(data['directions']):
        for path in group['routePaths']:
            UUID(path['id'])
            if path['id'] in seen or not path['stops']:
                raise ValueError('Duplicate or empty route pattern')
            seen.add(path['id']);numbers=[]
            for stop in path['stops']:
                UUID(stop['id']);number=stop['num'];numbers.append(number)
                if type(number) is not int or number<1 or not stop['name']:
                    raise ValueError('Invalid occurrence identity')
                for key,limit in [('lat',90),('lon',180)]:
                    if isinstance(stop[key],bool) or not math.isfinite(stop[key]) or abs(stop[key])>limit:
                        raise ValueError('Invalid coordinates')
                rows.append(dict(route=12,source_route_id=data['id'],source_direction_group=group_index,
                    source_pattern_id=path['id'],source_stop_uuid=stop['id'],stop_sequence=number,
                    occurrence_id=f'live:{path["id"]}:{number}',stop_name=stop['name'],
                    stop_lat=stop['lat'],stop_lon=stop['lon'],
                    pattern_first_stop=path['stops'][0]['name'],pattern_last_stop=path['stops'][-1]['name'],
                    actual_service_verified=False,observed_boardings=None))
            if numbers!=sorted(set(numbers)):
                raise ValueError('Nonunique/unordered stop sequence')
    if not rows:
        raise ValueError('No positions')
    return rows


def prepare(snapshot,out):
    body=snapshot.read_bytes();data=json.loads(body);rows=positions(data)
    selected=[]
    for pattern in dict.fromkeys(r['source_pattern_id'] for r in rows):
        subset=[r for r in rows if r['source_pattern_id']==pattern]
        for term in TERMS:
            match=next((r for r in subset if term in r['stop_name']),None)
            if match is None:
                raise ValueError(f'Review roster: no {term} in pattern')
            selected.append(dict(**match,selection_term=term,
                role_in_probe='announced_closed_weekend_control' if term=='Первомайская' else 'service_probe'))
    stop_ids=list(dict.fromkeys(r['source_stop_uuid'] for r in selected))
    if len(stop_ids)>20:
        raise ValueError('Roster exceeds collector bound')
    out.mkdir(parents=True,exist_ok=False)
    write(out/'positions.json',rows)
    write(out/'roster.json',selected)
    write(out/'stop_ids.json',stop_ids)
    write(out/'service_alerts.json',data.get('messages',[]))
    write(out/'summary.json',dict(version='N20260927-v1',target_route=12,
        positions=len(rows),source_direction_groups=len(data['directions']),
        patterns=len(set(r['source_pattern_id'] for r in rows)),selected_positions=len(selected),
        selected_stop_uuids=len(stop_ids),successful_validation_labels=0,actual_visits=0,
        current_period_authorized=True,source_scope='current_api_snapshot_not_historical_2025'))
    write(out/'manifest.json',dict(version='N20260927-v1',source_snapshot_sha256=digest(body),
        inputs={p.name:digest(p.read_bytes()) for p in [Path(__file__),HERE/'collect_stop_evidence.py']},
        outputs={p.name:digest(p.read_bytes()) for p in out.iterdir() if p.is_file()}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();prepare(args.snapshot,args.out)
