"""Interface I1: a single sensor reading and a labelled failure event entering the system."""

from datetime import datetime
from enum import StrEnum

from pydantic import Field, field_validator

from .common import Contract, to_utc


class Quality(StrEnum):
    GOOD = "good"
    SUSPECT = "suspect"
    BAD = "bad"


class Measurement(Contract):
    timestamp: datetime
    machine_id: str
    sensor_id: str
    value: float = Field(allow_inf_nan=False)
    quality: Quality = Quality.GOOD

    @field_validator("timestamp")
    @classmethod
    def _require_aware_utc(cls, value: datetime) -> datetime:
        return to_utc(value)


class FailureEvent(Contract):
    """A real failure of a machine, with its mode. Labels for training and evaluation."""

    machine_id: str
    timestamp: datetime
    failure_mode: str
    description: str = ""

    @field_validator("timestamp")
    @classmethod
    def _require_aware_utc(cls, value: datetime) -> datetime:
        return to_utc(value)
