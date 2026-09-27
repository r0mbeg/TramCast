import datetime

from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

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
