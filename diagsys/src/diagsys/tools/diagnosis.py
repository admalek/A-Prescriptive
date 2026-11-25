"""K4 diagnosis: anomalous windows, alarm episodes and which sensors caused them.

Input is the window feature table from K3 (`diagsys features`). No labels are needed.

Assumption (state it wherever results are reported): the machine is healthy at the start of every
continuous data segment, i.e. after a gap such as a repair. The first `reference_frac` of the valid
windows of each segment form its reference. Every window is compared with the reference of its own
segment, so a machine that behaves differently after a repair is judged against its new normal.

Two detectors share everything else (reference, calibration, alarm rule, attribution), so the only
difference between them is the score:
  robust_z  baseline: largest robust z-score over sensors and statistics (a threshold rule)
  iforest   Isolation Forest on the same robust z-scores (finds unusual combinations of sensors)

The alarm threshold is calibrated on the reference scores (`quantile` x `margin`) and an alarm
episode needs `min_run` consecutive alarm windows. The time of detection is the end of the window
in which the episode became an alarm, so nothing in the result uses data from after that moment.
"""

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pyarrow as pa

from ..contracts import Anomaly, AnomalyReport
from ..contracts.common import Source

METHODS = ("robust_z", "iforest")
DEFAULT_STATS = ("mean", "std", "slope_per_h")
_META = {"window_start", "window_end", "segment_id", "n_points", "valid_frac"}
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_Z_CLIP = 50.0


class DiagnosisError(Exception):
    """The diagnosis cannot be computed as requested; the message says why."""


@dataclass(frozen=True)
class Episode:
    segment_id: int
    start: datetime  # first alarm window starts
    detected_at: datetime  # end of the window completing `min_run` consecutive alarms
    end: datetime  # last alarm window ends
    n_windows: int
    peak_score: float
    mean_score: float
    # (sensor, statistic, mean signed z over the episode, mean strength of the sensor) by strength
    drivers: list[tuple[str, str, float, float]]


@dataclass
class DiagnosisResult:
    method: str
    params: dict[str, Any]
    sensors: list[str]
    threshold: float
    window_end: np.ndarray  # int64, microseconds since epoch
    segment: np.ndarray
    score: np.ndarray  # NaN where the window has missing features
    reference: np.ndarray  # bool, windows used as the healthy reference
    alarm: np.ndarray  # bool, above the threshold (never inside the reference)
    episodes: list[Episode] = field(default_factory=list)
    skipped_segments: list[int] = field(default_factory=list)
    monitored_hours: float = 0.0  # time covered by windows outside the reference


def diagnose(
    features: pa.Table,
    *,
    method: str = "iforest",
    sensors: list[str] | None = None,
    exclude_sensors: tuple[str, ...] = (),
    stats: tuple[str, ...] = DEFAULT_STATS,
    reference_frac: float = 0.25,
    min_reference_windows: int = 12,
    reference_windows: int | None = None,
    quantile: float = 0.995,
    margin: float = 1.0,
    min_run: int = 3,
    merge_gap: int = 2,
    seed: int = 42,
    n_estimators: int = 200,
) -> DiagnosisResult:
    if method not in METHODS:
        raise DiagnosisError(f"unknown method {method!r}: use one of {', '.join(METHODS)}")
    if not 0.05 <= reference_frac <= 0.9:
        raise DiagnosisError("reference_frac must be between 0.05 and 0.9")
    if reference_windows is not None and reference_windows < 5:
        raise DiagnosisError("reference_windows must be at least 5")
    if not 0.5 <= quantile < 1.0 or margin <= 0 or min_run < 1 or merge_gap < 0:
        raise DiagnosisError("invalid alarm settings (quantile, margin, min_run, merge_gap)")

    names = sensors if sensors is not None else sensors_in(features)
    names = [s for s in names if s not in exclude_sensors]
    if not names:
        raise DiagnosisError("no sensors to diagnose")
    cube = _cube(features, names, stats)  # (windows, sensors, stats)
    seg = features.column("segment_id").to_numpy().astype(np.int64)
    w_end = _int64(features, "window_end")
    w_start = _int64(features, "window_start")
    order = np.lexsort((w_end, seg))
    if not np.array_equal(order, np.arange(order.size)):
        raise DiagnosisError("features must be ordered by segment and time")

    valid = ~np.isnan(cube).any(axis=(1, 2))
    z, reference, skipped = reference_zscores(
        cube,
        seg,
        valid,
        reference_frac=reference_frac,
        min_reference_windows=min_reference_windows,
        reference_windows=reference_windows,
        min_tail=min_run,
    )
    usable = reference | (valid & ~np.isin(seg, skipped))
    if not reference.any():
        raise DiagnosisError(
            "no segment is long enough for a reference "
            f"({min_reference_windows} windows) plus an alarm run; skipped segments: {skipped}"
        )

    z = np.clip(z, -_Z_CLIP, _Z_CLIP)
    strength = np.abs(z).max(axis=2)  # (windows, sensors): how unusual each sensor is
    if method == "robust_z":
        score = strength.max(axis=1)
    else:
        score = _iforest_scores(z, usable, reference, seed, n_estimators)
    score = np.where(usable, score, np.nan)

    threshold = float(np.quantile(score[reference], quantile)) * margin
    alarm = usable & ~reference & (score > threshold)
    episodes = _episodes(
        alarm, score, z, strength, seg, w_start, w_end, names, stats, min_run, merge_gap
    )

    monitored = 0.0
    for s in np.unique(seg):
        mask = (seg == s) & usable & ~reference
        if mask.sum() > 1:
            monitored += (w_end[mask].max() - w_end[mask].min()) / 3.6e9
    params = {
        "method": method,
        "stats": list(stats),
        "exclude_sensors": list(exclude_sensors),
        "reference_frac": reference_frac,
        "min_reference_windows": min_reference_windows,
        "reference_windows": reference_windows,
        "quantile": quantile,
        "margin": margin,
        "min_run": min_run,
        "merge_gap": merge_gap,
        "seed": seed,
        "n_estimators": n_estimators if method == "iforest" else None,
    }
    return DiagnosisResult(
        method=method,
        params=params,
        sensors=names,
        threshold=threshold,
        window_end=w_end,
        segment=seg,
        score=score,
        reference=reference,
        alarm=alarm,
        episodes=episodes,
        skipped_segments=skipped,
        monitored_hours=float(monitored),
    )


def reference_zscores(
    cube: np.ndarray,
    seg: np.ndarray,
    valid: np.ndarray,
    *,
    reference_frac: float = 0.25,
    min_reference_windows: int = 12,
    reference_windows: int | None = None,
    min_tail: int = 1,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Robust z-scores of every window against the healthy start of its own segment.

    The reference is the first `reference_windows` valid windows of the segment. When that is not
    given, it is the first `reference_frac` of the segment's valid windows, which depends on how
    long the segment turns out to be: fine for analysing recorded data, but it looks into the
    future, so use a fixed `reference_windows` wherever a result must be possible in real time.
    Returns (z, reference mask, ids of segments too short for a reference plus `min_tail` windows).
    """
    z = np.full_like(cube, np.nan)
    reference = np.zeros(seg.size, dtype=bool)
    skipped: list[int] = []
    for s in np.unique(seg):
        idx = np.flatnonzero((seg == s) & valid)
        if reference_windows is not None:
            n_ref = reference_windows
        else:
            n_ref = max(math.ceil(reference_frac * idx.size), min_reference_windows)
        if idx.size < n_ref + min_tail:
            skipped.append(int(s))
            continue
        ref_idx = idx[:n_ref]
        reference[ref_idx] = True
        z[seg == s] = _robust_z(cube[seg == s], cube[ref_idx])
    return z, reference, skipped


def sensors_in(features: pa.Table) -> list[str]:
    seen: list[str] = []
    for name in features.column_names:
        if name in _META or "__" not in name:
            continue
        sensor = name.split("__", 1)[0]
        if sensor not in seen:
            seen.append(sensor)
    return seen


def to_anomaly_report(
    result: DiagnosisResult, machine_id: str, *, drivers_per_episode: int = 3
) -> AnomalyReport:
    """The episodes as the typed contract of interface I4 (one anomaly per driving sensor)."""
    anomalies = [
        Anomaly(
            sensor_id=sensor,
            start=episode.start,
            end=episode.end,
            score=round(strength, 3),
            description=_describe(sensor, stat, signed_z),
        )
        for episode in result.episodes
        for sensor, stat, signed_z, strength in episode.drivers[:drivers_per_episode]
    ]
    first = _EPOCH + timedelta(microseconds=int(result.window_end.min()))
    last = _EPOCH + timedelta(microseconds=int(result.window_end.max()))
    return AnomalyReport(
        machine_id=machine_id,
        window_start=first,
        window_end=last,
        method=result.method,
        anomalies=anomalies,
        sources=[Source(kind="tool", ref=f"diagnose:{result.method}")],
    )


def _describe(sensor: str, stat: str, signed_z: float) -> str:
    direction = "above" if signed_z > 0 else "below"
    return f"{sensor}: {stat} {abs(signed_z):.1f} robust deviations {direction} its reference"


# --- internals ----------------------------------------------------------------------------------


def _int64(table: pa.Table, name: str) -> np.ndarray:
    return table.column(name).cast(pa.int64()).to_numpy().astype(np.int64)


def _cube(features: pa.Table, sensors: list[str], stats: tuple[str, ...]) -> np.ndarray:
    columns = []
    for sensor in sensors:
        for stat in stats:
            name = f"{sensor}__{stat}"
            if name not in features.column_names:
                raise DiagnosisError(f"feature column {name!r} not found in the feature table")
            columns.append(features.column(name).to_numpy(zero_copy_only=False).astype(float))
    return np.stack(columns, axis=1).reshape(features.num_rows, len(sensors), len(stats))


def _robust_z(window: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Signed deviation from the reference in robust standard deviations (per sensor/statistic)."""
    median = np.median(ref, axis=0)
    mad = np.median(np.abs(ref - median), axis=0) * 1.4826
    # A floor keeps a nearly constant reference from turning tiny changes into huge scores.
    scale = np.maximum(np.maximum(mad, 0.25 * ref.std(axis=0)), 1e-9)
    return (window - median) / scale


def _iforest_scores(
    z: np.ndarray, usable: np.ndarray, reference: np.ndarray, seed: int, n_estimators: int
) -> np.ndarray:
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError as exc:  # pragma: no cover - depends on the optional 'ml' extra
        raise DiagnosisError("method 'iforest' needs scikit-learn: uv sync --extra ml") from exc
    flat = z.reshape(z.shape[0], -1)
    model = IsolationForest(n_estimators=n_estimators, random_state=seed, n_jobs=1)
    model.fit(flat[reference])
    score = np.full(z.shape[0], np.nan)
    score[usable] = -model.score_samples(flat[usable])  # higher = more anomalous
    return score


def _episodes(
    alarm: np.ndarray,
    score: np.ndarray,
    z: np.ndarray,
    strength: np.ndarray,
    seg: np.ndarray,
    w_start: np.ndarray,
    w_end: np.ndarray,
    sensors: list[str],
    stats: tuple[str, ...],
    min_run: int,
    merge_gap: int,
) -> list[Episode]:
    out: list[Episode] = []
    for s in np.unique(seg):
        idx = np.flatnonzero(seg == s)
        hits = idx[alarm[idx]]
        if hits.size == 0:
            continue
        groups: list[list[int]] = [[int(hits[0])]]
        for i in hits[1:]:
            previous = groups[-1][-1]
            # windows between two alarms belong to the same episode when the gap is short
            if int(np.searchsorted(idx, i) - np.searchsorted(idx, previous)) - 1 <= merge_gap:
                groups[-1].append(int(i))
            else:
                groups.append([int(i)])
        for group in groups:
            detected = _run_completion(group, idx, min_run)
            if detected is None:
                continue
            member = np.array(group)
            out.append(
                _episode(
                    member, detected, int(s), score, z, strength, w_start, w_end, sensors, stats
                )
            )
    return out


def _run_completion(group: list[int], idx: np.ndarray, min_run: int) -> int | None:
    """Window that completes the first run of `min_run` consecutive alarms (None: no such run)."""
    positions = np.searchsorted(idx, group)
    run = 1
    if min_run <= 1:
        return group[0]
    for k in range(1, len(group)):
        run = run + 1 if positions[k] == positions[k - 1] + 1 else 1
        if run >= min_run:
            return group[k]
    return None


def _episode(
    member: np.ndarray,
    detected: int,
    segment: int,
    score: np.ndarray,
    z: np.ndarray,
    strength: np.ndarray,
    w_start: np.ndarray,
    w_end: np.ndarray,
    sensors: list[str],
    stats: tuple[str, ...],
) -> Episode:
    mean_strength = strength[member].mean(axis=0)  # (sensors,)
    mean_z = z[member].mean(axis=0)  # (sensors, stats)
    drivers = []
    for j in np.argsort(-mean_strength):
        k = int(np.argmax(np.abs(mean_z[j])))
        drivers.append((sensors[j], stats[k], float(mean_z[j, k]), float(mean_strength[j])))
    return Episode(
        segment_id=segment,
        start=_EPOCH + timedelta(microseconds=int(w_start[member[0]])),
        detected_at=_EPOCH + timedelta(microseconds=int(w_end[detected])),
        end=_EPOCH + timedelta(microseconds=int(w_end[member[-1]])),
        n_windows=int(member.size),
        peak_score=float(np.nanmax(score[member])),
        mean_score=float(np.nanmean(score[member])),
        drivers=drivers,
    )
