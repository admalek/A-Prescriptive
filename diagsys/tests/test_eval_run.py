import random
from datetime import UTC, datetime

import numpy as np
import pytest

from diagsys.contracts import FailureEvent
from diagsys.eval import (
    RunError,
    config_fingerprint,
    data_fingerprint,
    load_run,
    start_run,
)
from diagsys.machines import create_machine
from diagsys.storage import DuckDBStorage

T0 = datetime(2026, 1, 1, tzinfo=UTC)
NOW = datetime(2026, 5, 1, 10, 30, 0, tzinfo=UTC)


@pytest.fixture
def store(settings):
    create_machine(settings, "m1", "Machine 1")
    with DuckDBStorage(settings.db_path) as storage:
        from diagsys.contracts import Measurement

        storage.write_measurements(
            [
                Measurement(
                    machine_id="m1",
                    sensor_id="a",
                    timestamp=datetime(2026, 1, 1, 0, i, tzinfo=UTC),
                    value=float(i),
                )
                for i in range(5)
            ]
        )
        yield storage


def test_run_writes_manifest_and_metrics(settings, store):
    run = start_run(settings, store, "m1", "baseline", seed=7, config={"window_s": 3600}, now=NOW)
    assert run.path.name.startswith("20260501T103000_baseline_")
    manifest = load_run(run.path).manifest
    assert manifest.seed == 7 and manifest.config == {"window_s": 3600}
    assert manifest.versions["python"] and "numpy" in manifest.versions
    run.finish({"mae": 1.5})
    assert load_run(run.path).metrics == {"mae": 1.5}
    with pytest.raises(RunError, match="already finished"):
        run.finish({"mae": 2.0})


def test_run_seeds_the_global_generators(settings, store):
    start_run(settings, store, "m1", "a", seed=123, now=NOW)
    first = (random.random(), float(np.random.rand()))
    start_run(settings, store, "m1", "b", seed=123, now=NOW)
    assert (random.random(), float(np.random.rand())) == first


def test_data_fingerprint_is_stable_and_changes_with_data(settings, store):
    before = data_fingerprint(store, "m1")
    assert data_fingerprint(store, "m1") == before
    store.write_events(
        [FailureEvent(machine_id="m1", timestamp=T0, failure_mode="bearing_wear", description="")]
    )
    after = data_fingerprint(store, "m1")
    assert after != before
    assert start_run(settings, store, "m1", "x", now=NOW).manifest.data_hash == after


def test_config_fingerprint_is_stable_and_validates():
    assert config_fingerprint({"a": 1, "b": 2}) == config_fingerprint({"b": 2, "a": 1})
    assert config_fingerprint({"a": 1}) != config_fingerprint({"a": 2})
    with pytest.raises(RunError, match="JSON"):
        config_fingerprint({"a": object()})


def test_run_rejects_bad_input(settings, store):
    start_run(settings, store, "m1", "dup", now=NOW)
    with pytest.raises(RunError, match="already exists"):
        start_run(settings, store, "m1", "dup", now=NOW)
    with pytest.raises(RunError, match="invalid run name"):
        start_run(settings, store, "m1", "../evil", now=NOW)
    with pytest.raises(RunError, match="not found"):
        start_run(settings, store, "nope", "x", now=NOW)
    run = start_run(settings, store, "m1", "art", now=NOW)
    with pytest.raises(RunError, match="plain file name"):
        run.write_artifact("../x.txt", "no")
    assert run.write_artifact("report.md", "ok").read_text() == "ok"
