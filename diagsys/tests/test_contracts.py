from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from diagsys.contracts import (
    Action,
    AgentResult,
    ErrorInfo,
    Forecast,
    Measurement,
    Recommendation,
    Source,
    Status,
)


def test_measurement_converts_to_utc():
    m = Measurement(
        timestamp="2026-01-01T01:00:00+01:00", machine_id="press-01", sensor_id="temp", value=21.5
    )
    assert m.timestamp == datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    assert m.timestamp.utcoffset().total_seconds() == 0


def test_measurement_rejects_naive_timestamp():
    with pytest.raises(ValidationError):
        Measurement(
            timestamp=datetime(2026, 1, 1), machine_id="press-01", sensor_id="temp", value=1.0
        )


def test_measurement_rejects_nan_and_unknown_fields():
    base = {"timestamp": "2026-01-01T00:00:00Z", "machine_id": "m1", "sensor_id": "t"}
    with pytest.raises(ValidationError):
        Measurement(**base, value=float("nan"))
    with pytest.raises(ValidationError):
        Measurement(**base, value=1.0, unexpected="x")


def test_forecast_interval_must_be_ordered():
    good = {
        "machine_id": "m1",
        "rul_hours": 100,
        "rul_low_hours": 80,
        "rul_high_hours": 120,
        "failure_probability": 0.2,
        "horizon_hours": 168,
        "model_version": "v1",
    }
    assert Forecast(**good).rul_hours == 100
    with pytest.raises(ValidationError):
        Forecast(**{**good, "rul_low_hours": 150})
    with pytest.raises(ValidationError):
        Forecast(**{**good, "failure_probability": 1.5})


def test_recommendation_requires_a_source():
    action = Action(description="Replace bearing")
    with pytest.raises(ValidationError):
        Recommendation(machine_id="m1", actions=[action], sources=[])
    rec = Recommendation(
        machine_id="m1",
        actions=[action],
        sources=[Source(kind="tool", ref="predict_rul:run-1")],
    )
    assert rec.sources[0].kind == "tool"


def test_failed_result_needs_error_and_ok_result_must_not_have_one():
    with pytest.raises(ValidationError):
        AgentResult(task_id="t1", status=Status.FAILED, trace_id="tr1")
    with pytest.raises(ValidationError):
        AgentResult(
            task_id="t1",
            status=Status.OK,
            trace_id="tr1",
            error=ErrorInfo(code="x", message="y"),
        )
    ok = AgentResult(task_id="t1", status=Status.OK, trace_id="tr1")
    assert ok.schema_version == "1"
