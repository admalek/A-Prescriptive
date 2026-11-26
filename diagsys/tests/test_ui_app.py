from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")

from diagsys.gateway import SimConfig, write_simulation  # noqa: E402
from diagsys.gateway import simulate as run_simulation  # noqa: E402

APP = str(Path(__file__).parent.parent / "src" / "diagsys" / "ui" / "app.py")


def _app(monkeypatch, tmp_path):
    monkeypatch.setenv("DIAGSYS_DATA_DIR", str(tmp_path / "data"))
    return st_testing.AppTest.from_file(APP, default_timeout=120)


def test_first_run_asks_for_a_machine_then_adds_files_and_shows_results(monkeypatch, tmp_path):
    at = _app(monkeypatch, tmp_path).run()
    assert not at.exception
    assert any("Utwórz maszynę" in i.value for i in at.info)

    at.sidebar.text_input[0].set_value("sim-01")
    at.sidebar.button[0].click()
    at = at.run()
    assert not at.exception
    assert at.sidebar.selectbox[0].value == "sim-01"

    sim = run_simulation(SimConfig(cycles=4, sampling_s=300, seed=7), "sim-01")
    write_simulation(sim, tmp_path / "r.csv", tmp_path / "f.csv")
    at.text_area[0].set_value(f"{tmp_path / 'r.csv'}\n{tmp_path / 'f.csv'}\n{tmp_path / 'nie.csv'}")
    at.button[0].click()
    at = at.run()
    assert not at.exception
    table = at.dataframe[0].value
    assert list(table["wynik"]) == ["OK", "OK", "BŁĄD"]

    analysis_button = next(b for b in at.button if b.label == "Uruchom analizę")
    analysis_button.click()
    at = at.run()
    assert not at.exception, at.exception
    labels = [m.label for m in at.metric]
    assert "Mediana pozostałego czasu" in labels
