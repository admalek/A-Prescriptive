from types import SimpleNamespace

from diagsys import doctor
from diagsys.config import Settings


def test_run_checks_returns_valid_results(settings):
    checks = doctor.run_checks(settings)
    names = [c.name for c in checks]
    assert {"Python", "RAM", "CPU", "Disk", "GPU", "Data directory", "Model server"} <= set(names)
    assert all(c.status in {"ok", "warn", "fail"} for c in checks)
    assert doctor.check_python().status == "ok"
    assert settings.data_dir.is_dir()


def test_ram_thresholds(monkeypatch):
    def fake(gib):
        return lambda: SimpleNamespace(total=int(gib * 2**30))

    monkeypatch.setattr(doctor.psutil, "virtual_memory", fake(16))
    assert doctor.check_ram().status == "ok"
    monkeypatch.setattr(doctor.psutil, "virtual_memory", fake(8))
    result = doctor.check_ram()
    assert result.status == "warn" and "1.5B" in result.detail
    monkeypatch.setattr(doctor.psutil, "virtual_memory", fake(4))
    assert doctor.check_ram().status == "fail"


def test_unreachable_model_server_is_only_a_warning(tmp_path):
    settings = Settings(data_dir=tmp_path, llm_base_url="http://127.0.0.1:9/v1", _env_file=None)
    assert doctor.check_llm_server(settings).status == "warn"
