"""Environment checks: is this laptop ready, and which model size fits it."""

import importlib.util
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import psutil

from .config import Settings

Level = Literal["ok", "warn", "fail"]

# Optional packages and the step that needs them.
_OPTIONAL = {
    "sklearn": "K4",
    "lightgbm": "K5",
    "lancedb": "K6",
    "ortools": "K7",
    "langgraph": "K8",
    "streamlit": "K8",
}


@dataclass(frozen=True)
class Check:
    name: str
    status: Level
    detail: str


def check_python() -> Check:
    version = sys.version_info
    text = f"{version.major}.{version.minor}.{version.micro}"
    if version >= (3, 11):
        return Check("Python", "ok", text)
    return Check("Python", "fail", f"{text}, version 3.11 or newer is required")


def check_ram() -> Check:
    gib = psutil.virtual_memory().total / 2**30
    if gib >= 15:
        return Check("RAM", "ok", f"{gib:.0f} GB: Bielik 4.5B (Q4) fits, one model at a time")
    if gib >= 7.5:
        return Check("RAM", "warn", f"{gib:.0f} GB: use Bielik 1.5B (Q4), not 4.5B")
    return Check("RAM", "fail", f"{gib:.1f} GB: too little for a local language model")


def check_cpu() -> Check:
    cores = os.cpu_count() or 0
    if cores >= 4:
        return Check("CPU", "ok", f"{cores} logical cores")
    return Check("CPU", "warn", f"{cores} logical cores: model answers will be slow")


def check_disk(settings: Settings) -> Check:
    probe = settings.data_dir.resolve()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free_gib = shutil.disk_usage(probe).free / 2**30
    if free_gib >= 30:
        return Check("Disk", "ok", f"{free_gib:.0f} GB free")
    return Check("Disk", "warn", f"{free_gib:.0f} GB free, 30 GB recommended for models and data")


def check_gpu() -> Check:
    if shutil.which("nvidia-smi"):
        return Check("GPU", "ok", "NVIDIA GPU tooling found (optional, speeds up the model)")
    return Check("GPU", "ok", "none detected (optional; the system runs on CPU)")


def check_data_dir(settings: Settings) -> Check:
    try:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=settings.data_dir):
            pass
    except OSError as exc:
        return Check("Data directory", "fail", f"{Path(settings.data_dir)} is not writable: {exc}")
    return Check("Data directory", "ok", f"{settings.data_dir} is writable")


def check_llm_server(settings: Settings) -> Check:
    url = settings.llm_base_url.rstrip("/") + "/models"
    try:
        urllib.request.urlopen(url, timeout=1.5)  # noqa: S310 - local URL from settings
    except urllib.error.HTTPError:
        return Check("Model server", "ok", f"{settings.llm_base_url} answers")
    except (OSError, ValueError):
        return Check(
            "Model server",
            "warn",
            f"{settings.llm_base_url} is not reachable (needed only for answers and report text; "
            "document search works without it)",
        )
    return Check("Model server", "ok", f"{settings.llm_base_url} answers")


def check_optional_packages() -> Check:
    missing = {step: [] for step in sorted(set(_OPTIONAL.values()))}  # step -> packages
    for package, step in _OPTIONAL.items():
        if importlib.util.find_spec(package) is None:
            missing[step].append(package)
    pending = [f"{step}: {', '.join(pkgs)}" for step, pkgs in missing.items() if pkgs]
    if not pending:
        return Check("Optional packages", "ok", "all installed")
    return Check("Optional packages", "ok", "installed in later steps: " + "; ".join(pending))


def run_checks(settings: Settings) -> list[Check]:
    return [
        check_python(),
        check_ram(),
        check_cpu(),
        check_disk(settings),
        check_gpu(),
        check_data_dir(settings),
        check_llm_server(settings),
        check_optional_packages(),
    ]
