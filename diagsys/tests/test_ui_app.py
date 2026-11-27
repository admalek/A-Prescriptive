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


def test_report_tab_recommends_and_records_a_decision(monkeypatch, tmp_path):
    at = _app(monkeypatch, tmp_path).run()
    at.sidebar.text_input[0].set_value("sim-01")
    at.sidebar.button[0].click()
    at = at.run()
    sim = run_simulation(SimConfig(cycles=4, sampling_s=300, seed=7), "sim-01")
    write_simulation(sim, tmp_path / "r.csv", tmp_path / "f.csv")
    at.text_area[0].set_value(f"{tmp_path / 'r.csv'}\n{tmp_path / 'f.csv'}")
    at.button[0].click()
    at = at.run()

    next(b for b in at.button if b.label == "Przygotuj zalecenia i raport").click()
    at = at.run()
    assert not at.exception, at.exception
    assert any("Zalecenia" in m.value for m in at.markdown)
    assert any("Brak decyzji" in c.value for c in at.caption)

    # an empty name is refused, a proper decision is stored and shows up in the log
    next(b for b in at.button if b.label == "Zapisz decyzję").click()
    at = at.run()
    assert any("who decides" in e.value for e in at.error)
    next(t for t in at.text_input if t.label.startswith("Kto decyduje")).set_value("AM")
    next(b for b in at.button if b.label == "Zapisz decyzję").click()
    at = at.run()
    assert not at.exception and any("Decyzja zapisana" in s.value for s in at.success)
    assert at.dataframe[-2].value["decyzja"].tolist() == ["Zatwierdzam"]
