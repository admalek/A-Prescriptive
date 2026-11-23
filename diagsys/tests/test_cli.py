from typer.testing import CliRunner

from diagsys import __version__, cli
from diagsys.doctor import Check

runner = CliRunner()


def test_version():
    result = runner.invoke(cli.app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_machine_new_list_show(tmp_path, monkeypatch):
    monkeypatch.setenv("DIAGSYS_DATA_DIR", str(tmp_path / "data"))
    result = runner.invoke(cli.app, ["machine", "list"])
    assert "No machines yet" in result.output

    result = runner.invoke(
        cli.app, ["machine", "new", "press-01", "--name", "Prasa", "-t", "press"]
    )
    assert result.exit_code == 0, result.output
    assert "docs_press-01" in result.output

    result = runner.invoke(cli.app, ["machine", "list"])
    assert "press-01" in result.output and "Prasa" in result.output

    result = runner.invoke(cli.app, ["machine", "show", "press-01"])
    assert result.exit_code == 0
    assert "rag_collection: docs_press-01" in result.output


def test_machine_new_rejects_bad_id_and_duplicates(tmp_path, monkeypatch):
    monkeypatch.setenv("DIAGSYS_DATA_DIR", str(tmp_path / "data"))
    assert runner.invoke(cli.app, ["machine", "new", "../evil"]).exit_code == 1
    assert runner.invoke(cli.app, ["machine", "new", "fan-01"]).exit_code == 0
    assert runner.invoke(cli.app, ["machine", "new", "fan-01"]).exit_code == 1
    assert runner.invoke(cli.app, ["machine", "show", "ghost-01"]).exit_code == 1


def test_doctor_exit_codes(monkeypatch):
    monkeypatch.setattr(cli, "run_checks", lambda s: [Check("X", "ok", "fine")])
    ok = runner.invoke(cli.app, ["doctor"])
    assert ok.exit_code == 0 and "[OK]" in ok.output

    monkeypatch.setattr(cli, "run_checks", lambda s: [Check("X", "fail", "broken")])
    bad = runner.invoke(cli.app, ["doctor"])
    assert bad.exit_code == 1 and "[FAIL]" in bad.output
