import json

import pyarrow.parquet as pq
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


@pytest.fixture
def dirty_machine(data_dir):
    assert invoke("machine", "new", "demo-01").exit_code == 0
    sim = invoke("simulate", "demo-01", "--cycles", "3", "--seed", "5", "--dirty")
    assert sim.exit_code == 0, sim.output
    raw = data_dir / "machines" / "demo-01" / "raw"
    assert invoke("ingest", "demo-01", str(raw / "sim_seed5.csv")).exit_code == 0
    return data_dir / "machines" / "demo-01"


def test_quality_reports_without_writing_files(dirty_machine):
    result = invoke("quality", "demo-01")
    assert result.exit_code == 0, result.output
    assert "5 stuck sensors" in result.output
    assert "time gaps" in result.output and "Stored result: quality_report:" in result.output
    assert list((dirty_machine / "processed").iterdir()) == []  # quality only reports


def test_features_writes_report_clean_and_features(dirty_machine):
    result = invoke("features", "demo-01")
    assert result.exit_code == 0, result.output
    out = dirty_machine / "processed"
    report = json.loads((out / "quality_report.json").read_text())
    assert report["machine_id"] == "demo-01" and len(report["stuck_sensors"]) == 5
    assert report["sources"]

    clean = pq.read_table(out / "clean.parquet")
    assert "temperature_c__flags" in clean.column_names and "segment_id" in clean.column_names
    assert b"diagsys.flag_legend" in clean.schema.metadata

    feats = pq.read_table(out / "features.parquet")
    assert feats.num_rows > 50 and "vibration_mm_s__kurtosis" in feats.column_names
    assert b"diagsys.quality_params" in feats.schema.metadata


def test_rerun_gives_same_features(dirty_machine):
    invoke("features", "demo-01")
    first = pq.read_table(dirty_machine / "processed" / "features.parquet")
    invoke("features", "demo-01")
    second = pq.read_table(dirty_machine / "processed" / "features.parquet")
    assert first.equals(second)


def test_readable_errors(data_dir):
    assert invoke("quality", "ghost-01").exit_code == 1
    assert invoke("machine", "new", "demo-01").exit_code == 0
    empty = invoke("features", "demo-01")
    assert empty.exit_code == 1 and "no readings" in empty.output
    assert "Traceback" not in empty.output
