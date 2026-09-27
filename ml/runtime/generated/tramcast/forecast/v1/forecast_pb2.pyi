import datetime

from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class StopHourStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    STOP_HOUR_STATUS_UNSPECIFIED: _ClassVar[StopHourStatus]
    STOP_HOUR_STATUS_SCENARIO: _ClassVar[StopHourStatus]
    STOP_HOUR_STATUS_NONWORKING_ZERO: _ClassVar[StopHourStatus]
    STOP_HOUR_STATUS_ROUTE5_FALLBACK: _ClassVar[StopHourStatus]
STOP_HOUR_STATUS_UNSPECIFIED: StopHourStatus
STOP_HOUR_STATUS_SCENARIO: StopHourStatus
STOP_HOUR_STATUS_NONWORKING_ZERO: StopHourStatus
STOP_HOUR_STATUS_ROUTE5_FALLBACK: StopHourStatus

class PredictRequest(_message.Message):
    __slots__ = ("route_number", "forecast_from", "forecast_to")
    ROUTE_NUMBER_FIELD_NUMBER: _ClassVar[int]
    FORECAST_FROM_FIELD_NUMBER: _ClassVar[int]
    FORECAST_TO_FIELD_NUMBER: _ClassVar[int]
    route_number: int
    forecast_from: _timestamp_pb2.Timestamp
    forecast_to: _timestamp_pb2.Timestamp
    def __init__(self, route_number: _Optional[int] = ..., forecast_from: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., forecast_to: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class PredictResponse(_message.Message):
    __slots__ = ("model_version", "dataset_version", "points")
    MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
    DATASET_VERSION_FIELD_NUMBER: _ClassVar[int]
    POINTS_FIELD_NUMBER: _ClassVar[int]
    model_version: str
    dataset_version: str
    points: _containers.RepeatedCompositeFieldContainer[PredictionPoint]
    def __init__(self, model_version: _Optional[str] = ..., dataset_version: _Optional[str] = ..., points: _Optional[_Iterable[_Union[PredictionPoint, _Mapping]]] = ...) -> None: ...

class PredictionPoint(_message.Message):
    __slots__ = ("hour_start", "boardings")
    HOUR_START_FIELD_NUMBER: _ClassVar[int]
    BOARDINGS_FIELD_NUMBER: _ClassVar[int]
    hour_start: _timestamp_pb2.Timestamp
    boardings: int
    def __init__(self, hour_start: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., boardings: _Optional[int] = ...) -> None: ...

class PredictStopsResponse(_message.Message):
    __slots__ = ("model_version", "dataset_version", "route_model_version", "route_dataset_version", "allocation_model_version", "allocation_dataset_version", "network_version", "occurrences", "hours", "estimate_status", "geography_status", "allocation_package_version")
    MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
    DATASET_VERSION_FIELD_NUMBER: _ClassVar[int]
    ROUTE_MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
    ROUTE_DATASET_VERSION_FIELD_NUMBER: _ClassVar[int]
    ALLOCATION_MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
    ALLOCATION_DATASET_VERSION_FIELD_NUMBER: _ClassVar[int]
    NETWORK_VERSION_FIELD_NUMBER: _ClassVar[int]
    OCCURRENCES_FIELD_NUMBER: _ClassVar[int]
    HOURS_FIELD_NUMBER: _ClassVar[int]
    ESTIMATE_STATUS_FIELD_NUMBER: _ClassVar[int]
    GEOGRAPHY_STATUS_FIELD_NUMBER: _ClassVar[int]
    ALLOCATION_PACKAGE_VERSION_FIELD_NUMBER: _ClassVar[int]
    model_version: str
    dataset_version: str
    route_model_version: str
    route_dataset_version: str
    allocation_model_version: str
    allocation_dataset_version: str
    network_version: str
    occurrences: _containers.RepeatedCompositeFieldContainer[StopOccurrence]
    hours: _containers.RepeatedCompositeFieldContainer[StopPredictionHour]
    estimate_status: str
    geography_status: str
    allocation_package_version: str
    def __init__(self, model_version: _Optional[str] = ..., dataset_version: _Optional[str] = ..., route_model_version: _Optional[str] = ..., route_dataset_version: _Optional[str] = ..., allocation_model_version: _Optional[str] = ..., allocation_dataset_version: _Optional[str] = ..., network_version: _Optional[str] = ..., occurrences: _Optional[_Iterable[_Union[StopOccurrence, _Mapping]]] = ..., hours: _Optional[_Iterable[_Union[StopPredictionHour, _Mapping]]] = ..., estimate_status: _Optional[str] = ..., geography_status: _Optional[str] = ..., allocation_package_version: _Optional[str] = ...) -> None: ...

class StopOccurrence(_message.Message):
    __slots__ = ("occurrence_id", "source_route_id", "pattern_key", "direction_id", "stop_sequence", "source_stop_id", "stop_name", "boarding_allowed", "boarding_role")
    OCCURRENCE_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_ROUTE_ID_FIELD_NUMBER: _ClassVar[int]
    PATTERN_KEY_FIELD_NUMBER: _ClassVar[int]
    DIRECTION_ID_FIELD_NUMBER: _ClassVar[int]
    STOP_SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_STOP_ID_FIELD_NUMBER: _ClassVar[int]
    STOP_NAME_FIELD_NUMBER: _ClassVar[int]
    BOARDING_ALLOWED_FIELD_NUMBER: _ClassVar[int]
    BOARDING_ROLE_FIELD_NUMBER: _ClassVar[int]
    occurrence_id: str
    source_route_id: str
    pattern_key: str
    direction_id: int
    stop_sequence: int
    source_stop_id: str
    stop_name: str
    boarding_allowed: bool
    boarding_role: str
    def __init__(self, occurrence_id: _Optional[str] = ..., source_route_id: _Optional[str] = ..., pattern_key: _Optional[str] = ..., direction_id: _Optional[int] = ..., stop_sequence: _Optional[int] = ..., source_stop_id: _Optional[str] = ..., stop_name: _Optional[str] = ..., boarding_allowed: _Optional[bool] = ..., boarding_role: _Optional[str] = ...) -> None: ...

class StopPredictionHour(_message.Message):
    __slots__ = ("hour_start", "route_boardings", "estimated_boardings", "status")
    HOUR_START_FIELD_NUMBER: _ClassVar[int]
    ROUTE_BOARDINGS_FIELD_NUMBER: _ClassVar[int]
    ESTIMATED_BOARDINGS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    hour_start: _timestamp_pb2.Timestamp
    route_boardings: int
    estimated_boardings: _containers.RepeatedScalarFieldContainer[int]
    status: StopHourStatus
    def __init__(self, hour_start: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., route_boardings: _Optional[int] = ..., estimated_boardings: _Optional[_Iterable[int]] = ..., status: _Optional[_Union[StopHourStatus, str]] = ...) -> None: ...
