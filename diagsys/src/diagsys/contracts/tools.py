"""Interface I4: typed inputs and outputs of the deterministic tools."""

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from .common import Contract, Source


class Anomaly(Contract):
    sensor_id: str
    start: datetime
    end: datetime
    score: float
    description: str = ""


class AnomalyReport(Contract):
    machine_id: str
    window_start: datetime
    window_end: datetime
    method: str
    anomalies: list[Anomaly] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)


class Forecast(Contract):
    """Remaining useful life (RUL) and failure probability, always with an uncertainty range."""

    machine_id: str
    rul_hours: float = Field(ge=0)
    rul_low_hours: float = Field(ge=0)
    rul_high_hours: float = Field(ge=0)
    failure_probability: float = Field(ge=0, le=1)
    horizon_hours: float = Field(gt=0)
    model_version: str
    sources: list[Source] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_interval(self) -> "Forecast":
        if not self.rul_low_hours <= self.rul_hours <= self.rul_high_hours:
            raise ValueError("expected rul_low_hours <= rul_hours <= rul_high_hours")
        return self


class Action(Contract):
    description: str
    due_by: datetime | None = None
    expected_cost: float | None = Field(default=None, ge=0)
    risk: Literal["low", "medium", "high"] = "medium"


class Recommendation(Contract):
    machine_id: str
    actions: list[Action]
    rationale: str = ""
    # A recommendation without a source is rejected by the review step (and by this contract).
    sources: list[Source] = Field(min_length=1)


class Passage(Contract):
    """A retrieved fragment of machine documentation (RAG)."""

    doc_id: str
    chunk_id: str
    text: str
    score: float | None = None
    page: int | None = None
