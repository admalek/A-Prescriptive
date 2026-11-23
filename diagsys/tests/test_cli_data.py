import pytest
from typer.testing import CliRunner

from diagsys import cli

runner = CliRunner()


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DIAGSYS_DATA_DIR", str(tmp_path / "data"))
    return tmp_path / "data"


def invoke(*args):
    return runner.invoke(cli.app, list(args))


def test_full_flow_simulate_ingest_summary(data_dir):
    assert invoke("machine", "new", "demo-01").exit_code == 0

    sim = invoke("simulate", "demo-01", "--cycles", "2", "--sampling-s", "600", "--seed", "7")
    assert sim.exit_code == 0, sim.output
    raw = data_dir / "machines" / "demo-01" / "raw"
    csv_path, events_path = raw / "sim_seed7.csv", raw / "sim_seed7_failures.csv"
    assert csv_path.is_file() and events_path.is_file()

    first = invoke("ingest", "demo-01", str(csv_path))
    assert first.exit_code == 0, first.output
    assert "already stored:     0" in first.output
    assert "temperature_c" in first.output
    assert "added to profile" in first.output

    again = invoke("ingest", "demo-01", str(csv_path))
    assert "new:                0" in again.output
    assert "added to profile" not in again.output

    assert invoke("ingest-events", "demo-01", str(events_path)).exit_code == 0

    summary = invoke("data", "summary", "demo-01")
    assert summary.exit_code == 0
    assert "vibration_mm_s" in summary.output
    assert "Failure events: 2" in summary.output

    profile = invoke("machine", "show", "demo-01")
    assert "id: temperature_c" in profile.output


def test_errors_are_readable_not_tracebacks(data_dir, tmp_path):
    assert invoke("simulate", "ghost-01").exit_code == 1
    assert invoke("machine", "new", "demo-01").exit_code == 0
    missing = invoke("ingest", "demo-01", str(tmp_path / "nope.csv"))
    assert missing.exit_code == 1 and "file not found" in missing.output
    bad = tmp_path / "bad.csv"
    bad.write_text("when,a\n2026-01-01,1\n", encoding="utf-8")
    result = invoke("ingest", "demo-01", str(bad))
    assert result.exit_code == 1 and "timestamp column" in result.output
    assert "Traceback" not in result.output
    empty = invoke("data", "summary", "demo-01")
    assert empty.exit_code == 0 and "No readings" in empty.output
