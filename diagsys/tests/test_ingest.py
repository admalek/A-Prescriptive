from datetime import UTC, datetime

import duckdb
import numpy as np
import pytest

from diagsys.data import IngestError, ingest_events, ingest_file, slugify
from diagsys.gateway import SimConfig, simulate, write_simulation
from diagsys.storage import DuckDBStorage

BASE = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def store(tmp_path):
    with DuckDBStorage(tmp_path / "d.duckdb") as storage:
        yield storage


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_slugify_handles_polish_and_units():
    assert slugify("Temperatura łożyska [°C]") == "temperatura_lozyska_c"
    assert slugify("  Vibration (mm/s) ") == "vibration_mm_s"
    assert slugify("???") == ""


def test_wide_csv_with_missing_cells_and_text_column(store, tmp_path):
    path = write(
        tmp_path,
        "w.csv",
        "timestamp,Temp [°C],rpm,status\n"
        "2026-01-01T00:00:00Z,20.5,1000,ok\n"
        "2026-01-01T00:01:00Z,,1010,ok\n"
        "2026-01-01T00:02:00Z,21.5,,warn\n",
    )
    report = ingest_file(store, "m1", path)
    assert report.rows_in_file == 3
    assert report.readings_found == 4
    assert report.readings_rejected == 2
    assert report.readings_new == 4
    assert report.sensors == ["rpm", "temp_c"]
    assert report.ignored_columns == ["status"]
    assert report.start == BASE and report.end == datetime(2026, 1, 1, 0, 2, tzinfo=UTC)
    got = store.read_timeseries("m1", "temp_c", BASE, datetime(2026, 1, 2, tzinfo=UTC))
    assert [m.value for m in got] == [20.5, 21.5]


def test_importing_twice_stores_nothing_twice(store, tmp_path):
    path = write(tmp_path, "w.csv", "timestamp,a\n2026-01-01T00:00:00Z,1\n2026-01-01T00:01:00Z,2\n")
    first = ingest_file(store, "m1", path)
    second = ingest_file(store, "m1", path)
    assert (first.readings_new, first.readings_duplicate) == (2, 0)
    assert (second.readings_new, second.readings_duplicate) == (0, 2)


def test_naive_timestamps_use_the_given_time_zone(store, tmp_path):
    path = write(tmp_path, "w.csv", "timestamp,a\n2026-01-01 12:00:00,1\n2026-07-01 12:00:00,2\n")
    ingest_file(store, "m1", path, tz="Europe/Warsaw")
    got = store.read_timeseries("m1", "a", BASE, datetime(2027, 1, 1, tzinfo=UTC))
    # Warsaw is UTC+1 in winter and UTC+2 in summer
    assert [m.timestamp.hour for m in got] == [11, 10]


def test_explicit_offsets_beat_the_time_zone_option(store, tmp_path):
    path = write(
        tmp_path, "w.csv", "timestamp,a\n2026-01-01T12:00:00Z,1\n2026-01-01T12:00:00+02:00,2\n"
    )
    ingest_file(store, "m1", path, tz="America/New_York")
    got = store.read_timeseries("m1", "a", BASE, datetime(2027, 1, 1, tzinfo=UTC))
    assert sorted(m.timestamp.hour for m in got) == [10, 12]


def test_semicolon_and_decimal_comma_and_custom_timestamp_format(store, tmp_path):
    path = write(
        tmp_path,
        "pl.csv",
        "Czas;Temperatura\n19.11.2026 08:00:00;21,5\n19.11.2026 08:01:00;22,25\n",
    )
    report = ingest_file(
        store,
        "m1",
        path,
        timestamp_col="Czas",
        timestamp_format="%d.%m.%Y %H:%M:%S",
        decimal=",",
        tz="Europe/Warsaw",
    )
    assert report.readings_new == 2
    got = store.read_timeseries("m1", "temperatura", BASE, datetime(2030, 1, 1, tzinfo=UTC))
    assert [m.value for m in got] == [21.5, 22.25]
    assert got[0].timestamp == datetime(2026, 11, 19, 7, 0, tzinfo=UTC)


def test_long_layout(store, tmp_path):
    path = write(
        tmp_path,
        "l.csv",
        "time,tag,reading\n"
        "2026-01-01T00:00:00Z,Temp A,1.5\n"
        "2026-01-01T00:00:00Z,Press B,7\n"
        "2026-01-01T00:01:00Z,Temp A,oops\n",
    )
    report = ingest_file(
        store,
        "m1",
        path,
        layout="long",
        timestamp_col="time",
        sensor_col="tag",
        value_col="reading",
    )
    assert report.sensors == ["press_b", "temp_a"]
    assert (report.readings_found, report.readings_rejected) == (2, 1)


def test_parquet_input(store, tmp_path):
    path = tmp_path / "p.parquet"
    con = duckdb.connect()
    con.execute(
        "COPY (SELECT TIMESTAMPTZ '2026-01-01 00:00:00+00' AS timestamp, 3.5 AS vibration) "
        f"TO '{path}' (FORMAT parquet)"
    )
    report = ingest_file(store, "m1", path)
    assert report.sensors == ["vibration"] and report.readings_new == 1


@pytest.mark.parametrize(
    ("name", "text", "kwargs", "message"),
    [
        ("a.xlsx", "x", {}, "unsupported file type"),
        ("b.csv", "when,a\n2026-01-01T00:00:00Z,1\n", {}, "timestamp column 'timestamp' not found"),
        ("c.csv", "timestamp,a\nnot a time,1\n", {}, None),
        ("d.csv", "timestamp,note\n2026-01-01T00:00:00Z,hello\n", {}, "no numeric sensor columns"),
        (
            "e.csv",
            "timestamp,a\n2026-01-01T00:00:00Z,1\n",
            {"tz": "Mars/Base; DROP"},
            "invalid time zone",
        ),
        ("f.csv", "timestamp,a\n2026-01-01T00:00:00Z,1\n", {"layout": "tall"}, "unknown layout"),
    ],
)
def test_clear_errors(store, tmp_path, name, text, kwargs, message):
    path = write(tmp_path, name, text)
    if message is None:  # unparsable timestamps are rejected rows, not a crash
        report = ingest_file(store, "m1", path)
        assert report.readings_found == 0 and report.readings_rejected == 1
        return
    with pytest.raises(IngestError, match=message):
        ingest_file(store, "m1", path, **kwargs)


def test_missing_file_and_bad_machine_id(store, tmp_path):
    with pytest.raises(IngestError, match="file not found"):
        ingest_file(store, "m1", tmp_path / "nope.csv")
    path = write(tmp_path, "w.csv", "timestamp,a\n2026-01-01T00:00:00Z,1\n")
    with pytest.raises(Exception, match="invalid machine id"):
        ingest_file(store, "x'; DROP TABLE measurements; --", path)


def test_events_import(store, tmp_path):
    path = write(
        tmp_path,
        "ev.csv",
        "timestamp,failure_mode,description\n"
        "2026-01-10T08:00:00Z, bearing_wear ,Zużycie łożyska\n"
        "2026-02-10T08:00:00Z,overheating,\n"
        "2026-03-10T08:00:00Z,,no mode\n",
    )
    report = ingest_events(store, "m1", path)
    assert (report.rows_in_file, report.events_found, report.events_new) == (3, 2, 2)
    again = ingest_events(store, "m1", path)
    assert (again.events_new, again.events_duplicate) == (0, 2)
    events = store.read_events("m1")
    assert [e.failure_mode for e in events] == ["bearing_wear", "overheating"]
    assert events[0].description == "Zużycie łożyska" and events[1].description == ""


def test_simulated_files_roundtrip_through_ingest(store, tmp_path):
    result = simulate(SimConfig(seed=4, cycles=2, sampling_s=600, dirty=True), "m1")
    csv_path, events_path = tmp_path / "sim.csv", tmp_path / "sim_failures.csv"
    write_simulation(result, csv_path, events_path)

    report = ingest_file(store, "m1", csv_path)
    present = sum(int((~np.isnan(v)).sum()) for v in result.values.values())
    assert report.readings_found == present
    assert report.readings_new == present
    assert report.readings_rejected == result.injected["missing"]
    assert report.sensors == sorted(result.values)

    ev = ingest_events(store, "m1", events_path)
    assert ev.events_new == 2
    assert [e.failure_mode for e in store.read_events("m1")] == [
        f.failure_mode for f in result.failures
    ]
    assert store.read_events("m1")[0].timestamp == result.failures[0].timestamp
