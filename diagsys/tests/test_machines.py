import pytest

from diagsys.machines import (
    SUBDIRS,
    InvalidMachineIdError,
    MachineExistsError,
    MachineNotFoundError,
    create_machine,
    list_machines,
    load_machine,
)


def test_create_makes_folders_and_roundtrips(settings):
    profile = create_machine(settings, "press-01", "Prasa hydrauliczna", "press")
    root = settings.machines_dir / "press-01"
    assert (root / "profile.yaml").is_file()
    for sub in SUBDIRS:
        assert (root / sub).is_dir()
    assert profile.rag_collection == "docs_press-01"
    loaded = load_machine(settings, "press-01")
    assert loaded == profile
    assert loaded.name == "Prasa hydrauliczna"


def test_list_machines_sorted(settings):
    assert list_machines(settings) == []
    create_machine(settings, "pump-02", "Pump 2")
    create_machine(settings, "fan-01", "Fan 1")
    assert [m.machine_id for m in list_machines(settings)] == ["fan-01", "pump-02"]


def test_duplicate_machine_rejected(settings):
    create_machine(settings, "fan-01", "Fan 1")
    with pytest.raises(MachineExistsError):
        create_machine(settings, "fan-01", "Fan 1 again")


@pytest.mark.parametrize("bad", ["../evil", "A", "x", "with space", "UPPER", "a/b", ""])
def test_invalid_ids_rejected(settings, bad):
    with pytest.raises(InvalidMachineIdError):
        create_machine(settings, bad, "x")
    with pytest.raises(InvalidMachineIdError):
        load_machine(settings, bad)
    assert not settings.machines_dir.exists() or list(settings.machines_dir.iterdir()) == []


def test_load_missing_machine(settings):
    with pytest.raises(MachineNotFoundError):
        load_machine(settings, "ghost-01")
