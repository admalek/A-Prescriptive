from datetime import UTC, datetime, timedelta

import pytest

from diagsys.contracts import FailureEvent, Forecast, Measurement
from diagsys.storage import DuckDBStorage

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def reading(minute: int, value: float, sensor: str = "temp", machine: str = "m1") -> Measurement:
    return Measurement(
        timestamp=BASE + timedelta(minutes=minute),
        machine_id=machine,
        sensor_id=sensor,
        value=value,
    )


@pytest.fixture
def store(tmp_path):
    with DuckDBStorage(tmp_path / "d.duckdb") as storage:
        yield storage


def test_roundtrip_and_half_open_range(store):
    assert store.write_measurements([reading(i, float(i)) for i in range(10)]) == 10
    got = store.read_timeseries(
        "m1", "temp", BASE + timedelta(minutes=2), BASE + timedelta(minutes=5)
    )
    assert [m.value for m in got] == [2.0, 3.0, 4.0]
    assert got[0].timestamp == BASE + timedelta(minutes=2)
    assert got[0].timestamp.tzinfo is not None


def test_writes_are_idempotent_and_dedupe_inside_a_batch(store):
    rows = [reading(0, 1.0), reading(0, 1.0), reading(1, 2.0)]
    assert store.write_measurements(rows) == 2
    assert store.write_measurements(rows) == 0
    assert len(store.read_timeseries("m1", "temp", BASE, BASE + timedelta(hours=1))) == 2


def test_machines_and_sensors_are_isolated(store):
    store.write_measurements(
        [reading(0, 1.0), reading(0, 9.0, machine="m2"), reading(0, 5.0, "rpm")]
    )
    end = BASE + timedelta(hours=1)
    assert [m.value for m in store.read_timeseries("m1", "temp", BASE, end)] == [1.0]
    assert [m.value for m in store.read_timeseries("m2", "temp", BASE, end)] == [9.0]
    assert [m.value for m in store.read_timeseries("m1", "rpm", BASE, end)] == [5.0]


def test_data_survives_reopening(tmp_path):
    path = tmp_path / "d.duckdb"
    with DuckDBStorage(path) as store:
        store.write_measurements([reading(0, 1.0)])
    with DuckDBStorage(path) as store:
        assert len(store.read_timeseries("m1", "temp", BASE, BASE + timedelta(hours=1))) == 1


def test_summary(store):
    store.write_measurements([reading(i, float(i)) for i in range(5)] + [reading(0, 7.0, "rpm")])
    summary = {s.sensor_id: s for s in store.summary("m1")}
    assert summary["temp"].readings == 5
    assert (summary["temp"].minimum, summary["temp"].maximum, summary["temp"].mean) == (
        0.0,
        4.0,
        2.0,
    )
    assert summary["temp"].first == BASE
    assert summary["temp"].last == BASE + timedelta(minutes=4)
    assert summary["rpm"].readings == 1
    assert store.summary("nobody") == []


def test_events_roundtrip_and_dedupe(store):
    event = FailureEvent(machine_id="m1", timestamp=BASE, failure_mode="bearing", description="x")
    assert store.write_events([event, event]) == 1
    assert store.write_events([event]) == 0
    assert store.read_events("m1") == [event]
    assert store.read_events("m2") == []


def test_results_roundtrip(store):
    forecast = Forecast(
        machine_id="m1",
        rul_hours=100,
        rul_low_hours=80,
        rul_high_hours=120,
        failure_probability=0.2,
        horizon_hours=168,
        model_version="v1",
    )
    ref = store.write_result("forecast", forecast)
    assert ref.startswith("forecast:")
    assert store.read_result(ref)["rul_hours"] == 100
    with pytest.raises(KeyError):
        store.read_result("forecast:missing")
