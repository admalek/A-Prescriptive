"""Reproducible experiment runs.

Every experiment (a model, a baseline, an agent variant) is started with `start_run`. The run gets
its own folder `data/machines/<id>/experiments/<run_id>/` holding a manifest with everything needed
to repeat it (seed, configuration, a fingerprint of the exact data, library versions, git commit)
and, once finished, the metrics. Two runs with the same manifest must give the same metrics; the
fingerprint shows when they did not because the data changed.
"""

import hashlib
import json
import platform
import random
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Settings
from ..machines import machine_dir, validate_machine_id
from ..storage import DuckDBStorage

_PACKAGES = ("diagsys", "numpy", "duckdb", "pyarrow", "scikit-learn", "lightgbm", "lifelines")


class RunError(Exception):
    """The run cannot be started or finished as requested."""


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    name: str
    machine_id: str
    created_at: str
    seed: int
    config: dict[str, Any]
    config_hash: str
    data_hash: str
    versions: dict[str, str]
    git_commit: str | None = None


@dataclass
class Run:
    path: Path
    manifest: RunManifest
    metrics: dict[str, Any] = field(default_factory=dict)

    def finish(self, metrics: dict[str, Any]) -> Path:
        """Store the metrics next to the manifest. A run can be finished once."""
        target = self.path / "metrics.json"
        if target.exists():
            raise RunError(f"run {self.manifest.run_id} is already finished")
        self.metrics = metrics
        _write_json(target, metrics)
        return target

    def write_artifact(self, name: str, text: str) -> Path:
        """Store any other output of the run (a report, a table) in the run folder."""
        if Path(name).name != name or not name:
            raise RunError(f"artifact name must be a plain file name: {name!r}")
        target = self.path / name
        target.write_text(text, encoding="utf-8")
        return target


def config_fingerprint(config: dict[str, Any]) -> str:
    try:
        text = json.dumps(config, sort_keys=True, ensure_ascii=False)
    except TypeError as exc:
        raise RunError(f"configuration must be JSON-serialisable: {exc}") from exc
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def data_fingerprint(storage: DuckDBStorage, machine_id: str) -> str:
    """Hash of every reading and failure label of a machine, independent of insertion order."""
    digest = hashlib.sha256()
    table = storage.read_table(machine_id)
    digest.update(str(table.num_rows).encode())
    if table.num_rows:
        digest.update("\x1f".join(table.column("sensor_id").to_pylist()).encode())
        digest.update(table.column("ts").cast("int64").to_numpy().tobytes())
        digest.update(table.column("value").cast("float64").to_numpy().tobytes())
    for event in sorted(
        storage.read_events(machine_id), key=lambda e: (e.timestamp, e.failure_mode)
    ):
        digest.update(f"{event.timestamp.isoformat()}|{event.failure_mode}".encode())
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    """Seed the generators used in the project. Libraries with their own generators (scikit-learn,
    LightGBM) must be given the same seed explicitly, e.g. `random_state=run.manifest.seed`."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - legacy global state, for code that still reads it


def start_run(
    settings: Settings,
    storage: DuckDBStorage,
    machine_id: str,
    name: str,
    *,
    seed: int = 42,
    config: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> Run:
    validate_machine_id(machine_id)
    if not name or any(c in name for c in "/\\ "):
        raise RunError(f"invalid run name {name!r}: use letters, digits, '-' and '_'")
    if not machine_dir(settings, machine_id).is_dir():
        raise RunError(f"machine {machine_id!r} not found")
    config = config or {}
    created = (now or datetime.now(UTC)).astimezone(UTC)
    config_hash = config_fingerprint(config)
    run_id = f"{created:%Y%m%dT%H%M%S}_{name}_{config_hash[:8]}"
    path = machine_dir(settings, machine_id) / "experiments" / run_id
    if path.exists():
        raise RunError(f"run {run_id} already exists")
    manifest = RunManifest(
        run_id=run_id,
        name=name,
        machine_id=machine_id,
        created_at=created.isoformat(),
        seed=seed,
        config=config,
        config_hash=config_hash,
        data_hash=data_fingerprint(storage, machine_id),
        versions=_versions(),
        git_commit=_git_commit(),
    )
    path.mkdir(parents=True)
    _write_json(path / "manifest.json", asdict(manifest))
    seed_everything(seed)
    return Run(path, manifest)


def load_run(path: Path) -> Run:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise RunError(f"no manifest in {path}")
    manifest = RunManifest(**json.loads(manifest_path.read_text(encoding="utf-8")))
    metrics_path = path / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    return Run(path, manifest, metrics)


def _versions() -> dict[str, str]:
    found = {"python": platform.python_version()}
    for package in _PACKAGES:
        try:
            found[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    return found


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
