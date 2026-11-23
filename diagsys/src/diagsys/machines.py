"""Machine profiles: what makes the universal system specific to one machine.

Every machine gets its own directory under data/machines/<machine_id>/ with a profile.yaml,
its own documentation (docs/, indexed into a per-machine RAG collection), raw data, trained
models and reports. Adding a machine never touches the code.
"""

import re
from pathlib import Path

import yaml
from pydantic import Field, ValidationError, field_validator, model_validator

from .config import Settings
from .contracts import Contract

_SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]{1,62}$")
SUBDIRS = ("docs", "raw", "processed", "models", "reports")


class MachineError(Exception):
    """Base class for machine profile errors."""


class InvalidMachineIdError(MachineError):
    pass


class MachineExistsError(MachineError):
    pass


class MachineNotFoundError(MachineError):
    pass


def validate_machine_id(machine_id: str) -> str:
    if not _SLUG.fullmatch(machine_id):
        raise InvalidMachineIdError(
            f"invalid machine id {machine_id!r}: use 2-63 characters from a-z, 0-9, '_' and '-', "
            "starting with a letter or digit"
        )
    return machine_id


class SensorSpec(Contract):
    id: str
    name: str | None = None
    unit: str | None = None
    sampling_hz: float | None = Field(default=None, gt=0)
    min_value: float | None = None
    max_value: float | None = None


class FailureMode(Contract):
    id: str
    name: str
    description: str = ""
    downtime_cost_per_hour: float | None = Field(default=None, ge=0)


class MachineProfile(Contract):
    machine_id: str
    name: str
    machine_type: str = "generic"
    language: str = "pl"
    sensors: list[SensorSpec] = Field(default_factory=list)
    failure_modes: list[FailureMode] = Field(default_factory=list)
    rag_collection: str | None = None
    notes: str = ""

    @field_validator("machine_id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        try:
            return validate_machine_id(value)
        except InvalidMachineIdError as exc:
            raise ValueError(str(exc)) from exc

    @model_validator(mode="after")
    def _default_collection(self) -> "MachineProfile":
        if self.rag_collection is None:
            self.rag_collection = f"docs_{self.machine_id}"
        return self


def machine_dir(settings: Settings, machine_id: str) -> Path:
    return settings.machines_dir / validate_machine_id(machine_id)


def create_machine(
    settings: Settings, machine_id: str, name: str, machine_type: str = "generic"
) -> MachineProfile:
    path = machine_dir(settings, machine_id)
    if path.exists():
        raise MachineExistsError(f"machine {machine_id!r} already exists at {path}")
    try:
        profile = MachineProfile(machine_id=machine_id, name=name, machine_type=machine_type)
    except ValidationError as exc:
        raise MachineError(str(exc)) from exc
    for sub in SUBDIRS:
        (path / sub).mkdir(parents=True)
    _write_profile(path / "profile.yaml", profile)
    return profile


def load_machine(settings: Settings, machine_id: str) -> MachineProfile:
    profile_path = machine_dir(settings, machine_id) / "profile.yaml"
    if not profile_path.is_file():
        raise MachineNotFoundError(f"machine {machine_id!r} not found (looked in {profile_path})")
    data = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    try:
        return MachineProfile.model_validate(data)
    except ValidationError as exc:
        raise MachineError(f"invalid profile {profile_path}: {exc}") from exc


def list_machines(settings: Settings) -> list[MachineProfile]:
    root = settings.machines_dir
    if not root.is_dir():
        return []
    profiles = []
    for child in sorted(root.iterdir()):
        if (child / "profile.yaml").is_file():
            profiles.append(load_machine(settings, child.name))
    return profiles


def save_machine(settings: Settings, profile: MachineProfile) -> None:
    path = machine_dir(settings, profile.machine_id) / "profile.yaml"
    if not path.parent.is_dir():
        raise MachineNotFoundError(f"machine {profile.machine_id!r} not found")
    _write_profile(path, profile)


def add_sensors(settings: Settings, machine_id: str, sensor_ids: list[str]) -> list[str]:
    """Register sensors that appeared in imported data. Returns the ids that were new."""
    profile = load_machine(settings, machine_id)
    known = {sensor.id for sensor in profile.sensors}
    added = [sid for sid in sensor_ids if sid not in known]
    if added:
        profile.sensors.extend(SensorSpec(id=sid) for sid in added)
        save_machine(settings, profile)
    return added


def _write_profile(path: Path, profile: MachineProfile) -> None:
    text = yaml.safe_dump(profile.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
    path.write_text(text, encoding="utf-8")
