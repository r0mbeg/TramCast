"""Validated immutable hourly forecast, shared by serving and the disk cache."""
import csv
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping
from zoneinfo import ZoneInfo

from constants import ROUTES
from google.protobuf.timestamp_pb2 import Timestamp

ROOT = Path(__file__).resolve().parent
DEFAULT_BUNDLE = ROOT.parent / "bundles" / "002" / "forecast_bundle.json"
MOSCOW = ZoneInfo("Europe/Moscow")
MAX_HOURS = 1464
INT64_MAX = 2**63 - 1


def hour_seconds(stamp):
    if stamp.nanos != 0:
        raise ValueError("Timestamp nanos must be zero")
    local = stamp.ToDatetime(tzinfo=MOSCOW)  # Also checks the Protobuf range.
    if local.minute or local.second:
        raise ValueError("Timestamp must start a Moscow hour")
    return stamp.seconds


def json_hour(value):
    stamp = Timestamp()
    stamp.FromJsonString(value)
    return hour_seconds(stamp)


@dataclass(frozen=True)
class Bundle:
    model_version: str
    dataset_version: str
    start: int
    end: int
    points: Mapping[tuple[int, int], int]

    @classmethod
    def load(cls, path):
        path = Path(path)
        meta = json.loads(path.read_text(encoding="utf-8"))
        source = path.parent / meta["prediction_file"]
        if source.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("Forecast file exceeds 16 MiB")
        content = source.read_bytes()
        return cls.from_csv(meta, content)

    @classmethod
    def from_csv(cls, meta, content):
        if len(content) > 16 * 1024 * 1024:
            raise ValueError("Forecast file exceeds 16 MiB")
        for key in ("model_version", "dataset_version"):
            if not isinstance(meta[key], str) or not meta[key].strip():
                raise ValueError(f"Missing {key}")
        if meta["timezone"] != "Europe/Moscow" or meta["route_numbers"] != list(ROUTES):
            raise ValueError("Unsupported timezone or routes")
        start, end = json_hour(meta["forecast_from"]), json_hour(meta["forecast_to"])
        if not 0 < end - start <= MAX_HOURS * 3600 or json_hour(meta["history_end"]) > start:
            raise ValueError("Invalid bundle horizon")
        if hashlib.sha256(content).hexdigest() != meta["prediction_sha256"]:
            raise ValueError("Forecast checksum mismatch")
        reader = csv.DictReader(io.StringIO(content.decode("utf-8")), delimiter=";")
        if reader.fieldnames != ["route", "date", "hour", "prediction"]:
            raise ValueError("Invalid forecast CSV schema")
        points = {}
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Invalid CSV row")
            route, hour, value = int(row["route"]), int(row["hour"]), int(row["prediction"])
            if route not in ROUTES or not 0 <= hour <= 23 or not 0 <= value <= INT64_MAX:
                raise ValueError("Invalid route, hour or int64 boardings")
            second = json_hour(f'{row["date"]}T{hour:02}:00:00+03:00')
            key = (route, second)
            if key in points or not start <= second < end:
                raise ValueError("Duplicate or out-of-range forecast point")
            if (route == 5 or 1 <= hour <= 4) and value != 0:
                raise ValueError("Structural zeros violated")
            points[key] = value
        expected = {(route, second) for route in ROUTES for second in range(start, end, 3600)}
        if points.keys() != expected:
            raise ValueError("Forecast does not cover the exact hourly grid")
        # Already rounded at export; serving must never round these integers again.
        return cls(meta["model_version"], meta["dataset_version"], start, end,
                   MappingProxyType(points))

