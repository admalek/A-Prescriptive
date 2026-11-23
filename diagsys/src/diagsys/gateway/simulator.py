"""Synthetic run-to-failure data, a stand-in for a real machine until real data is connected.

A machine runs for a random lifetime, degrades gradually, fails, is repaired (a gap without data)
and starts again. Each cycle ends in one of several failure modes; every mode leaves a different
signature on different sensors. Readings also depend on a varying load and a daily cycle, so
a model has to separate "the machine works harder" from "the machine is wearing out".

The output is a wide CSV (timestamp + one column per sensor) and a failures CSV, the same shape
as files exported from a real system, and goes through the same `diagsys ingest` path.
"""

import csv
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import duckdb
import numpy as np
import pyarrow as pa

from ..contracts import FailureEvent


@dataclass(frozen=True)
class SimSensor:
    id: str
    unit: str
    base: float
    noise: float
    load_coef: float = 0.0  # change of the reading per 1.0 of load above/below the 0.7 norm
    daily_amp: float = 0.0  # amplitude of the daily (ambient) cycle
    min_value: float | None = None
    kind: Literal["process", "load"] = "process"


@dataclass(frozen=True)
class FailureSignature:
    id: str
    name: str
    # sensor id -> (shift at the moment of failure, extra noise factor at the moment of failure)
    effects: dict[str, tuple[float, float]]


DEFAULT_SENSORS = (
    SimSensor("temperature_c", "°C", base=55.0, noise=0.4, load_coef=15.0, daily_amp=2.0),
    SimSensor("vibration_mm_s", "mm/s", base=2.0, noise=0.08, load_coef=0.8, min_value=0.0),
    SimSensor("current_a", "A", base=12.0, noise=0.2, load_coef=6.0, min_value=0.0),
    SimSensor("pressure_bar", "bar", base=6.0, noise=0.05, load_coef=0.3, min_value=0.0),
    SimSensor("load_pct", "%", base=70.0, noise=0.5, kind="load"),
)

DEFAULT_FAILURES = (
    FailureSignature(
        "bearing_wear",
        "Zużycie łożyska",
        {"vibration_mm_s": (3.5, 1.5), "temperature_c": (6.0, 0.0)},
    ),
    FailureSignature(
        "overheating",
        "Przegrzewanie",
        {"temperature_c": (25.0, 0.5), "current_a": (2.5, 0.0)},
    ),
    FailureSignature(
        "pressure_leak",
        "Nieszczelność układu ciśnienia",
        {"pressure_bar": (-2.5, 1.0), "current_a": (1.0, 0.0)},
    ),
)


@dataclass(frozen=True)
class SimConfig:
    sensors: tuple[SimSensor, ...] = DEFAULT_SENSORS
    failure_modes: tuple[FailureSignature, ...] = DEFAULT_FAILURES
    cycles: int = 5
    cycle_hours: tuple[float, float] = (150.0, 300.0)  # lifetime range of one cycle
    sampling_s: int = 60
    gap_hours: float = 4.0  # repair time between cycles, no data
    start: datetime = datetime(2026, 1, 1, tzinfo=UTC)
    seed: int = 42
    dirty: bool = False  # inject missing values, spikes and a stuck sensor


@dataclass(frozen=True)
class InjectedDirt:
    """Ground truth of what `dirty` did to one sensor (row indexes into SimResult.timestamps)."""

    flat_start: int | None  # first row of the stuck run (None: sensor too short)
    flat_len: int
    spikes: np.ndarray  # bool per row: value was multiplied by 3
    missing: np.ndarray  # bool per row: value was removed


@dataclass
class SimResult:
    timestamps: np.ndarray  # datetime64[s], UTC
    values: dict[str, np.ndarray]  # sensor id -> float64, NaN = missing
    failures: list[FailureEvent]
    injected: dict[str, int]  # what `dirty` added: missing, spikes, flatlines
    dirt: dict[str, InjectedDirt] = field(default_factory=dict)  # per sensor, empty if not dirty


def simulate(config: SimConfig, machine_id: str = "sim") -> SimResult:
    if config.cycles < 1 or config.sampling_s < 1:
        raise ValueError("cycles and sampling_s must be at least 1")
    if not config.failure_modes:
        raise ValueError("at least one failure mode is required")

    rng = np.random.default_rng(config.seed)
    start = config.start.astimezone(UTC)
    t0 = np.datetime64(start.replace(tzinfo=None), "s")
    start_hour = start.hour + start.minute / 60.0
    step = config.sampling_s

    stamps: list[np.ndarray] = []
    series: dict[str, list[np.ndarray]] = {s.id: [] for s in config.sensors}
    failures: list[FailureEvent] = []
    elapsed_s = 0

    for _ in range(config.cycles):
        lifetime_h = rng.uniform(*config.cycle_hours)
        n = max(int(lifetime_h * 3600 // step), 10)
        mode = config.failure_modes[int(rng.integers(len(config.failure_modes)))]
        offsets = elapsed_s + np.arange(n) * step
        hour_of_day = (start_hour + offsets / 3600.0) % 24

        load = (
            0.7 + 0.2 * np.sin(2 * math.pi * (hour_of_day - 6) / 24) + _smooth_noise(rng, n, 0.08)
        )
        load = np.clip(load, 0.2, 1.0)
        health = (np.arange(n) / n) ** rng.uniform(2.0, 3.5)  # 0 = new, 1 = failure

        for sensor in config.sensors:
            if sensor.kind == "load":
                values = 100.0 * load + rng.normal(0.0, sensor.noise, n)
            else:
                shift, noise_gain = mode.effects.get(sensor.id, (0.0, 0.0))
                values = (
                    sensor.base
                    + sensor.load_coef * (load - 0.7)
                    + sensor.daily_amp * np.sin(2 * math.pi * (hour_of_day - 9) / 24)
                    + shift * health
                    + sensor.noise * (1.0 + noise_gain * health) * rng.standard_normal(n)
                )
            if sensor.min_value is not None:
                values = np.maximum(values, sensor.min_value)
            series[sensor.id].append(values)

        stamps.append(t0 + offsets.astype("timedelta64[s]"))
        failure_s = int(offsets[-1] + step)
        failures.append(
            FailureEvent(
                machine_id=machine_id,
                timestamp=start + timedelta(seconds=failure_s),
                failure_mode=mode.id,
                description=mode.name,
            )
        )
        elapsed_s = failure_s + int(config.gap_hours * 3600)

    values_all = {sid: np.concatenate(parts) for sid, parts in series.items()}
    injected = {"missing": 0, "spikes": 0, "flatlines": 0}
    dirt: dict[str, InjectedDirt] = {}
    if config.dirty:
        injected, dirt = _inject_dirt(rng, values_all)
    return SimResult(np.concatenate(stamps), values_all, failures, injected, dirt)


def write_simulation(result: SimResult, csv_path: Path, events_path: Path | None = None) -> None:
    """Write a wide CSV (ISO timestamps in UTC, empty cell = missing) and optionally failures."""
    columns: dict[str, pa.Array] = {
        "timestamp": pa.array(
            result.timestamps.astype("datetime64[us]"), type=pa.timestamp("us", tz="UTC")
        )
    }
    for sensor_id, values in result.values.items():
        columns[sensor_id] = pa.array(values, type=pa.float64(), from_pandas=True)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(":memory:")
    con.execute("SET TimeZone='UTC'")
    con.register("sim", pa.table(columns))
    selected = ", ".join(f'round("{sid}", 4) AS "{sid}"' for sid in result.values)
    target = str(csv_path).replace("'", "''")
    con.execute(
        f"""
        COPY (
            SELECT strftime("timestamp" AT TIME ZONE 'UTC', '%Y-%m-%dT%H:%M:%SZ') AS timestamp,
                   {selected}
            FROM sim ORDER BY "timestamp"
        ) TO '{target}' (HEADER, DELIMITER ',')
        """
    )
    con.close()

    if events_path is not None:
        with events_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp", "failure_mode", "description"])
            for event in result.failures:
                stamp = event.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
                writer.writerow([stamp, event.failure_mode, event.description])


def _smooth_noise(rng: np.random.Generator, n: int, scale: float) -> np.ndarray:
    """Slowly varying noise with standard deviation `scale` (white noise, exponential smoothing)."""
    kernel = np.exp(-np.arange(120) / 30.0)
    kernel /= kernel.sum()
    noise = np.convolve(rng.standard_normal(n + kernel.size - 1), kernel, mode="valid")
    deviation = noise.std()
    return scale * noise / deviation if deviation > 0 else noise


def _inject_dirt(
    rng: np.random.Generator, values: dict[str, np.ndarray]
) -> tuple[dict[str, int], dict[str, InjectedDirt]]:
    counts = {"missing": 0, "spikes": 0, "flatlines": 0}
    truth: dict[str, InjectedDirt] = {}
    flat_len = 60
    for sensor_id, series in values.items():
        n = series.size
        flat_start: int | None = None
        if n > 200:  # a stuck sensor: 60 identical readings in a row
            flat_start = int(rng.integers(0, n - flat_len))
            series[flat_start : flat_start + flat_len] = series[flat_start]
            counts["flatlines"] += 1
        spikes = rng.random(n) < 0.001
        series[spikes] *= 3.0
        counts["spikes"] += int(spikes.sum())
        missing = rng.random(n) < 0.005
        series[missing] = np.nan
        counts["missing"] += int(missing.sum())
        truth[sensor_id] = InjectedDirt(flat_start, flat_len, spikes, missing)
    return counts, truth
