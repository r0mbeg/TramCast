"""Read-only source audit. Run from any directory; output must not already exist."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time

import pandas as pd

LAB = Path(__file__).resolve().parents[1]
ROOT = LAB.parents[1]
sys.path.insert(0, str(LAB))
from pipeline import aggregate, full_grid  # existing cleaning, without modifying history

# Reuse the historical XLSX reader without calling its old-path run() entrypoint.
spec = importlib.util.spec_from_file_location(
    "prior_audit", LAB / "artifacts/methodology_20260927/data/audit_data.py")
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def valid_key(series):
    return ~series.str.strip().str.lower().isin(["", "0", "null", "none", "nan"])


def degree_summary(pairs):
    mapping = defaultdict(set)
    for left, right in pairs:
        mapping[left].add(right)
    return dict(keys=len(mapping), keys_with_multiple_values=sum(len(v) > 1 for v in mapping.values()),
                maximum_values=max(map(len, mapping.values()), default=0))


def run(out):
    started, cpu = time.monotonic(), time.process_time()
    out.mkdir(parents=True, exist_ok=False)
    sources, geography = [], []
    workbook_routes = set()
    for path in sorted((ROOT / "dataset/spravochniki").glob("*.xlsx")):
        sources.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
        for name, header_row, headers, records in prior.sheets(path):
            info = dict(sheet=name, header_row=header_row, headers=headers, rows=len(records),
                        file=str(path.relative_to(ROOT)))
            for column in ("date", "start_date", "end_date", "actual_date", "route_date_start"):
                if column in headers:
                    info[column] = dict(Counter(r[column] for r in records))
            if name == "Маршруты GTFS_ROUTES":
                workbook_routes = {int(r["route_short_name"]) for r in records}
            if name == "Порядок_остановок GTFS_TRIPS_ST":
                info["routes"] = sorted({int(r["route_short_name"]) for r in records})
                info["patterns"] = len({r["trip_id"] for r in records})
                info["repeated_stop_occurrences"] = len(records) - len({
                    (r["route_id"], r["trip_id"], r["stop_id"]) for r in records})
                keys = [(r["route_id"], r["trip_id"], r["stop_sequence"]) for r in records]
                if len(keys) != len(set(keys)):
                    raise ValueError("Duplicate geography occurrence key")
                info["route_temporal_coverage"] = [dict(route=int(route),
                    positions=len(group), patterns=len({r["trip_id"] for r in group}),
                    start_dates=sorted({r["start_date"] for r in group}),
                    actual_dates=sorted({r["actual_date"] for r in group}))
                    for route in sorted({r["route_short_name"] for r in records}, key=int)
                    for group in [[r for r in records if r["route_short_name"] == route]]]
            if name == "Расписание":
                info.update(routes=sorted({r["route_short_name"] for r in records}),
                            patterns=len({r["trip_id"] for r in records}),
                            service_ids=sorted({r["service_id"] for r in records}))
            geography.append(info)
    osm = ROOT / "data/osm/tram_routes.json"
    sources.append(dict(path=str(osm.relative_to(ROOT)), sha256=sha(osm),
                        use="geography snapshot only; no event positions"))
    write_json(out / "geography.json", geography)
    raw, clean, working = Counter(), Counter(), Counter()
    key_columns = ["device_no", "garage_number", "bus_exit_no", "place_id"]
    distinct = {c: set() for c in key_columns}
    pairs, sampled_pairs, sampled_exits = set(), set(), set()
    quality = Counter()
    per_file = {}
    for filename in ("train.csv", "test.csv"):
        path = ROOT / "dataset" / filename
        before = path.stat()
        file_counts, empty = Counter(), Counter()
        for frame in pd.read_csv(path, sep=";", dtype=str, keep_default_na=False, chunksize=250_000):
            headers = list(frame.columns)
            required = set(key_columns + ["tran_date_time", "begin_date_time", "validation_result", "ngpt_route"])
            if not required.issubset(headers):
                raise ValueError("Missing required raw columns")
            empty.update({c: int(frame[c].str.strip().eq("").sum()) for c in headers})
            frame["validation_result"] = pd.to_numeric(frame.validation_result, errors="raise").astype("int64")
            a, b, c, counts = aggregate(frame, "2025-01-01", "2025-11-01")
            raw.update(a.to_dict()); clean.update(b.to_dict()); working.update(c.to_dict())
            file_counts.update(counts)
            timestamp = pd.to_datetime(frame.tran_date_time, format="%Y-%m-%d %H:%M:%S", errors="raise")
            period = timestamp.ge("2025-01-01") & timestamp.lt("2025-11-01")
            f = frame.loc[period].copy()
            success = f.validation_result.eq(1)
            minutes = timestamp.loc[period].dt.hour * 60 + timestamp.loc[period].dt.minute
            eligible = success & ((minutes < 60) | (minutes >= 330))
            quality["eligible_successes"] += int(eligible.sum())
            for col in key_columns:
                valid = valid_key(f[col])
                distinct[col].update(f.loc[valid, col].unique())
                quality[col + "_missing_or_sentinel_eligible"] += int((eligible & ~valid).sum())
            valid_both = valid_key(f.device_no) & valid_key(f.garage_number)
            pairs.update(f.loc[valid_both, ["device_no", "garage_number"]].drop_duplicates().itertuples(index=False, name=None))
            day = f.tran_date_time.str[:10]
            sample = day.str[-2:].eq("15")
            sampled = f.loc[sample & valid_both].assign(date=day)
            sampled_pairs.update(sampled[["date", "device_no", "garage_number"]].drop_duplicates().itertuples(index=False, name=None))
            exit_sample = f.loc[sample & valid_key(f.bus_exit_no) & valid_key(f.garage_number)].assign(date=day)
            sampled_exits.update(exit_sample[["date", "ngpt_route", "bus_exit_no", "garage_number"]].drop_duplicates().itertuples(index=False, name=None))
            begin = pd.to_datetime(f.begin_date_time, format="%Y-%m-%d %H:%M:%S", errors="coerce")
            delta = (timestamp.loc[period] - begin).dt.total_seconds()
            quality["begin_invalid"] += int(begin.isna().sum())
            quality["begin_equals_event"] += int(delta.eq(0).sum())
            quality["begin_after_event"] += int(delta.lt(0).sum())
            quality["begin_more_than_24h_before_event"] += int(delta.gt(86400).sum())
            if file_counts["rows"] % 5_000_000 == 0:
                print(filename, file_counts["rows"], flush=True)
        digest = sha(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Source changed during audit")
        per_file[filename] = dict(file_counts, empty_fields=dict(empty), headers=headers,
                                  bytes=after.st_size, sha256=digest)
        sources.append(dict(path=str(path.relative_to(ROOT)), sha256=digest))
    grid = full_grid()
    result = pd.DataFrame(index=grid)
    for name, counter in (("raw_boardings", raw), ("boardings", clean), ("working_events", working)):
        result[name] = pd.Series(counter).reindex(grid, fill_value=0).astype("int64")
    history = LAB / "artifacts/hourly_clean.csv"
    expected = pd.read_csv(history, sep=";").set_index(["route", "date", "hour"])
    pd.testing.assert_frame_equal(result, expected[result.columns], check_dtype=False)
    sources.append(dict(path=str(history.relative_to(ROOT)), sha256=sha(history)))
    labels = pd.concat([pd.read_csv(ROOT / "dataset/labels" / name, sep=";")
                        for name in ("labels_day_train.csv", "labels_day_test.csv")])
    if labels.duplicated(["route", "date", "hour"]).any():
        raise ValueError("Duplicate supplied label keys")
    pd.testing.assert_series_equal(result.raw_boardings,
        labels.set_index(["route", "date", "hour"]).boardings.reindex(grid, fill_value=0), check_names=False)
    for name in ("labels_day_train.csv", "labels_day_test.csv"):
        path = ROOT / "dataset/labels" / name
        sources.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
    coverage = []
    for route, frame in result.reset_index().groupby("route"):
        work = frame.loc[~frame.hour.between(1, 4)]
        coverage.append(dict(route=int(route), eligible_successes=int(frame.boardings.sum()),
            working_hours=len(work), event_present_hours=int(work.working_events.gt(0).sum()),
            event_presence=float(work.working_events.gt(0).mean()), workbook_geography=route in workbook_routes,
            verified_stop_matches=0, stop_match_fraction=0.0, independent_stop_labels=0))
    candidates = [r for r in coverage if r["workbook_geography"] and r["eligible_successes"]]
    pilot = sorted(candidates, key=lambda r: (-r["event_presence"], r["route"]))[0]["route"]
    # No AVL or direct stop labels are present in these inputs. Keep unassigned counts explicit.
    balance = result.reset_index()
    balance["matched_boardings"] = 0
    balance["unassigned_boardings"] = balance.boardings
    balance["assignment_status"] = "unavailable_no_stop_evidence"
    balance.to_csv(out / "route_hour_balance.csv", sep=";", index=False)
    balance.loc[balance.route.eq(pilot)].to_csv(out / "pilot_unassigned.csv", sep=";", index=False)
    audit = dict(status="blocked_missing_stop_evidence", files=per_file, field_quality=dict(quality),
        distinct_nonempty_keys={k: len(v) for k, v in distinct.items()},
        device_to_vehicle_whole_period=degree_summary(pairs),
        device_day_to_vehicle_sample=degree_summary(((day, dev), veh) for day, dev, veh in sampled_pairs),
        route_exit_day_to_vehicle_sample=degree_summary(((day, route, ex), veh) for day, route, ex, veh in sampled_exits),
        sample_dates=[f"2025-{month:02}-15" for month in range(1, 11)],
        coverage=coverage, pilot_candidate=pilot, independently_verified_stop_accuracy=None,
        complete_counter_fraction=None, reconciled_hourly_keys=len(result),
        clean_total=int(result.boardings.sum()), assigned_total=0,
        unassigned_total=int(result.boardings.sum()),
        note="0 is verified linkage coverage for these inputs, not a measured failure rate of an AVL matcher")
    write_json(out / "audit.json", audit)
    sources.append(dict(path=str(Path(__file__).relative_to(ROOT)), sha256=sha(Path(__file__))))
    for path in (LAB / "pipeline.py", LAB / "artifacts/methodology_20260927/data/audit_data.py", Path(__file__).with_name("PROTOCOL.md")):
        sources.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
    write_json(out / "run.json", dict(created_at=datetime.now(timezone.utc).isoformat(), sources=sources,
        python=platform.python_version(), pandas=pd.__version__, wall_seconds=time.monotonic()-started,
        cpu_seconds=time.process_time()-cpu, protocol="S20260927-v1", gpu_seconds=0,
        outputs={p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}))
    print(json.dumps(dict(pilot_candidate=pilot, clean_total=audit["clean_total"], status=audit["status"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)
