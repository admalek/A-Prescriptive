"""Base types shared by all contracts between modules."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict

SCHEMA_VERSION = "1"


def to_utc(value: datetime) -> datetime:
    """Require a timezone-aware datetime and normalise it to UTC."""
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


class Contract(BaseModel):
    """Strict (unknown fields are errors) and versioned base class for every contract."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION


class Status(StrEnum):
    OK = "ok"
    FAILED = "failed"


class ErrorInfo(Contract):
    code: str
    message: str


class Source(Contract):
    """Where a claim comes from. Claims without a source are rejected by the review step."""

    kind: Literal["document", "tool", "data"]
    ref: str
    excerpt: str | None = None
