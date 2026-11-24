from datetime import UTC, datetime

import openpyxl
import pytest
from typer.testing import CliRunner

from diagsys.cli import app
from diagsys.data import XlsxError, import_workbook, write_template
from diagsys.machines import load_machine
from diagsys.storage import DuckDBStorage

BASE = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def store(settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    with DuckDBStorage(settings.db_path) as storage:
        yield storage


def make_book(path, sheets):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(name)
        for row in rows:
            ws.append(row)
    wb.save(path)
    return path


DANE = [
    ["czas", "Temperatura [°C]", "Drgania [mm/s]"],
    [datetime(2026, 1, 1, 12, 0), 55.0, 2.0],
    [datetime(2026, 1, 1, 12, 1), 55.5, "2,5"],  # decimal comma written as text
    [datetime(2026, 1, 1, 12, 2), None, 3.0],  # empty cell is rejected, not stored as 0
]


def test_template_imports_without_changes(settings, store, tmp_path):
    path = tmp_path / "t.xlsx"
    write_template(path)
    report = import_workbook(settings, store, "pompa-01", path)
    assert report.machine_created
    assert report.readings.readings_new == 9
    assert report.events is not None and report.events.events_new == 1
    assert report.failure_modes == 1 and report.sensors_described == 2
    assert report.context_file is not None and report.context_file.read_text().strip()
    profile = load_machine(settings, "pompa-01")
    assert profile.name == "Pompa P-01" and profile.machine_type == "pompa wirowa"
    assert {s.id for s in profile.sensors} == {"temperatura_c", "drgania_mm_s", "prad_a"}
    temp = next(s for s in profile.sensors if s.id == "temperatura_c")
    assert (temp.min_value, temp.max_value, temp.unit) == (0.0, 120.0, "°C")
    assert profile.failure_modes[0].downtime_cost_per_hour == 1500.0


def test_only_the_data_sheet_is_required(settings, store, tmp_path):
    path = make_book(tmp_path / "a.xlsx", {"Dane": DANE})
    report = import_workbook(settings, store, "m1", path, tz="Europe/Warsaw")
    assert report.readings.readings_found == 5
    assert report.readings.readings_rejected == 1
    assert report.events is None and report.context_file is None
    assert len(report.warnings) == 3
    got = store.read_timeseries("m1", "drgania_mm_s", BASE, datetime(2027, 1, 1, tzinfo=UTC))
    assert [m.value for m in got] == [2.0, 2.5, 3.0]
    assert got[0].timestamp.hour == 11  # Warsaw is UTC+1 in January


def test_importing_the_same_workbook_twice_stores_nothing_twice(settings, store, tmp_path):
    path = make_book(tmp_path / "a.xlsx", {"dane": DANE})
    import_workbook(settings, store, "m1", path)
    second = import_workbook(settings, store, "m1", path)
    assert (second.readings.readings_new, second.readings.readings_duplicate) == (0, 5)


def test_english_sheet_names_and_missing_data_sheet(settings, store, tmp_path):
    ok = make_book(tmp_path / "en.xlsx", {"data": DANE, "Machine": [["name", "Press"]]})
    import_workbook(settings, store, "m1", ok)
    assert load_machine(settings, "m1").name == "Press"
    bad = make_book(tmp_path / "bad.xlsx", {"inne": [["a"]]})
    with pytest.raises(XlsxError, match="sheet 'dane' not found"):
        import_workbook(settings, store, "m2", bad)


def test_a_bad_workbook_changes_nothing(settings, store, tmp_path):
    path = make_book(
        tmp_path / "bad.xlsx",
        {
            "dane": DANE,
            "tryby_awarii": [["id", "nazwa", "koszt_przestoju_na_godzine"], ["x", "X", -5]],
        },
    )
    with pytest.raises(XlsxError, match="negative"):
        import_workbook(settings, store, "m3", path)
    assert not (settings.machines_dir / "m3").exists()


def test_rejects_wrong_file_types_and_unknown_files(settings, store, tmp_path):
    with pytest.raises(XlsxError, match="not found"):
        import_workbook(settings, store, "m1", tmp_path / "nope.xlsx")
    csv_path = tmp_path / "a.csv"
    csv_path.write_text("a\n1\n")
    with pytest.raises(XlsxError, match="unsupported"):
        import_workbook(settings, store, "m1", csv_path)
    broken = tmp_path / "broken.xlsx"
    broken.write_text("not a zip")
    with pytest.raises(XlsxError, match="cannot open"):
        import_workbook(settings, store, "m1", broken)


def test_cli_template_then_import(tmp_path, monkeypatch):
    monkeypatch.setenv("DIAGSYS_DATA_DIR", str(tmp_path / "data"))
    runner = CliRunner()
    template = tmp_path / "szablon.xlsx"
    assert runner.invoke(app, ["xlsx-template", str(template)]).exit_code == 0
    assert runner.invoke(app, ["xlsx-template", str(template)]).exit_code == 1  # no overwrite
    result = runner.invoke(app, ["import-xlsx", "pompa-01", str(template), "--tz", "Europe/Warsaw"])
    assert result.exit_code == 0, result.output
    assert "machine created" in result.output and "9 new" in result.output
    again = runner.invoke(app, ["import-xlsx", "pompa-01", str(template), "--tz", "Europe/Warsaw"])
    assert "0 new" in again.output
