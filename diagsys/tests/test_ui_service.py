import pytest

from diagsys.gateway import SimConfig, write_simulation
from diagsys.gateway import simulate as run_simulation
from diagsys.machines import MachineExistsError, create_machine
from diagsys.ui import service


@pytest.fixture
def sim_files(tmp_path):
    sim = run_simulation(SimConfig(cycles=4, sampling_s=300, seed=7), "sim-01")
    write_simulation(sim, tmp_path / "readings.csv", tmp_path / "awarie.csv")
    return tmp_path / "readings.csv", tmp_path / "awarie.csv"


def test_parse_paths_strips_quotes_blanks_and_duplicates():
    text = '"C:\\a\\b.xlsx"\n\n  /x/y.csv  \n/x/y.csv\n'
    assert service.parse_paths(text) == ["C:\\a\\b.xlsx", "/x/y.csv"]


def test_new_machine_normalises_the_id_and_refuses_duplicates(settings):
    assert service.new_machine(settings, " Press-01 ", "Prasa") == "press-01"
    assert [m.machine_id for m in service.machines(settings)] == ["press-01"]
    with pytest.raises(MachineExistsError):
        service.new_machine(settings, "press-01")


def test_add_files_detects_roles_and_reports_bad_paths(settings, sim_files, tmp_path):
    readings, failures = sim_files
    create_machine(settings, "sim-01", "Sim")
    odd = tmp_path / "x.exe"
    odd.write_text("nope")
    out = service.add_files(
        settings, "sim-01", [str(readings), str(failures), str(odd), str(tmp_path / "missing.csv")]
    )
    kinds = [(o.kind, o.ok) for o in out]
    assert kinds == [("readings", True), ("failures", True), ("unknown", False), ("unknown", False)]
    assert "unsupported" in out[2].message and "not found" in out[3].message
    view = service.overview(settings, "sim-01")
    assert view.sensors and len(view.failures) == 4
    again = service.add_files(settings, "sim-01", [str(readings)])
    assert "0 new" in again[0].message  # repeating an import changes nothing


def test_unknown_machine_fails_the_whole_batch(settings, sim_files):
    from diagsys.machines import MachineNotFoundError

    with pytest.raises(MachineNotFoundError):
        service.add_files(settings, "nope-01", [str(sim_files[0])])


def test_documents_are_copied_and_name_clashes_keep_both(settings, tmp_path):
    create_machine(settings, "sim-01", "Sim")
    one = tmp_path / "a" / "manual.md"
    two = tmp_path / "b" / "manual.md"
    for p, text in ((one, "# Łożyska\nWymiana co 8000 godzin."), (two, "# Inne\nCoś innego.")):
        p.parent.mkdir()
        p.write_text(text, encoding="utf-8")
    out = service.add_files(settings, "sim-01", [str(one), str(two)])
    assert all(o.kind == "document" and o.ok for o in out)
    assert any("saved as manual_2.md" in w for w in out[1].warnings)
    assert service.overview(settings, "sim-01").documents == ["manual.md", "manual_2.md"]
    hits = service.search_docs(settings, "sim-01", "wymiana łożysk")
    assert hits and hits[0].doc_id == "manual.md"


def test_text_file_with_table_header_is_data_unless_role_says_otherwise(settings, tmp_path):
    create_machine(settings, "sim-01", "Sim")
    f = tmp_path / "dane.txt"
    f.write_text("timestamp;a;b\n2026-01-01 00:00:00;1;2\n2026-01-01 00:01:00;2;3\n")
    out = service.add_files(settings, "sim-01", [str(f)], tz="UTC")
    assert out[0].kind == "readings" and out[0].ok, out[0].message


def test_analysis_gives_diagnosis_and_forecast_and_partial_results(settings, sim_files):
    readings, failures = sim_files
    create_machine(settings, "sim-01", "Sim")
    service.add_files(settings, "sim-01", [str(readings)])
    only_data = service.analyze(settings, "sim-01", exclude_sensors=("load_pct",))
    assert set(only_data.diagnosis) == {"robust_z", "iforest"}
    assert only_data.forecast is None and "failure" in only_data.errors["forecast"]

    service.add_files(settings, "sim-01", [str(failures)])
    full = service.analyze(settings, "sim-01", exclude_sensors=("load_pct",))
    assert full.forecast is not None and full.forecast_method == "quantile_gbm_cal"
    f = full.forecast.forecast
    assert f.rul_low_hours <= f.rul_hours <= f.rul_high_hours
    view = full.diagnosis["robust_z"]
    assert len(view.series["score"]) == len(view.series["window_end"])
    assert service.table_from_series(view).num_rows == len(view.series["score"])


def test_analysis_of_an_empty_machine_reports_instead_of_raising(settings):
    create_machine(settings, "sim-01", "Sim")
    out = service.analyze(settings, "sim-01")
    assert "data" in out.errors and out.forecast is None


def test_best_method_follows_the_history():
    assert [service.best_rul_method(n) for n in (0, 1, 2, 3, 9)] == [
        None,
        "age_baseline",
        "quantile_gbm",
        "quantile_gbm_cal",
        "quantile_gbm_cal",
    ]
