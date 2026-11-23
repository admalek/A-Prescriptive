"""Data quality check: missing readings, spikes, stuck sensors, range violations, time gaps.

Deterministic and read-only: it never changes the values, it only reports and returns per-reading
flags. Cleaning (`clean.py`) decides what to do with the flagged readings.

Definitions
  missing   a grid cell without a reading from this sensor
  stuck     `flatline_min_points` or more identical consecutive readings of a sensor that normally
            varies (sensors that mostly repeat values are treated as discrete and skipped)
  spike     a reading far from the median of its neighbours (Hampel filter, robust sigma); only
            isolated outliers are found, a lasting level change is not a spike
  gap       a pause longer than `max_gap_s` with no data from any sensor
"""

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ..contracts import (
    QualityParams,
    QualityReport,
    SensorQuality,
    Source,
    SpikeExample,
    StuckRun,
    TimeGap,
)
from .frame import FLAG_MISSING, FLAG_RANGE, FLAG_SPIKE, FLAG_STUCK, SensorFrame

MAX_GAPS_IN_REPORT = 50
MAX_SPIKE_EXAMPLES = 5
_CHUNK = 200_000


@dataclass
class QualityResult:
    report: QualityReport
    flags: np.ndarray  # uint8 (rows, sensors), bits from frame.FLAG_*


def default_params(frame: SensorFrame) -> QualityParams:
    return QualityParams(
        step_s=frame.step_s,
        max_gap_s=frame.max_gap_us / 1e6,
        flatline_min_points=20,
        spike_window=21,
        spike_threshold=6.0,
        max_interp_points=5,
    )


def _dt(us: int) -> datetime:
    return datetime.fromtimestamp(us / 1e6, tz=UTC)


def runs_of(mask: np.ndarray, segment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Starts and lengths of runs of True in `mask`; a run never crosses a segment boundary."""
    if mask.size == 0 or not mask.any():
        return np.empty(0, np.int64), np.empty(0, np.int64)
    prev_mask = np.r_[False, mask[:-1]]
    prev_seg = np.r_[segment[0] - 1, segment[:-1]]
    starts = np.flatnonzero(mask & ~(prev_mask & (prev_seg == segment)))
    next_mask = np.r_[mask[1:], False]
    next_seg = np.r_[segment[1:], segment[-1] + 1]
    ends = np.flatnonzero(mask & ~(next_mask & (next_seg == segment)))
    return starts, ends - starts + 1


def _robust_sigma(values: np.ndarray, same_segment: np.ndarray) -> float:
    """Noise level from first differences (insensitive to trends and to rare spikes)."""
    d = np.diff(values)[same_segment]
    if d.size < 3:
        return 0.0
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0))


def _hampel(x: np.ndarray, window: int, sigma_floor: float) -> tuple[np.ndarray, np.ndarray]:
    """Per point: deviation from the rolling median in local robust sigmas, and that median."""
    m = x.size
    w = min(window, m)
    if w % 2 == 0:
        w -= 1
    if w < 5:
        return np.zeros(m), x.copy()
    half = w // 2
    padded = np.pad(x, half, mode="edge")
    score = np.empty(m)
    median = np.empty(m)
    for a in range(0, m, _CHUNK):
        b = min(a + _CHUNK, m)
        win = sliding_window_view(padded[a : b + 2 * half], w)
        med = np.median(win, axis=1)
        mad = np.median(np.abs(win - med[:, None]), axis=1)
        scale = np.maximum(1.4826 * mad, sigma_floor)
        score[a:b] = np.abs(x[a:b] - med) / scale
        median[a:b] = med
    return score, median


def check_quality(
    frame: SensorFrame,
    *,
    machine_id: str,
    params: QualityParams | None = None,
    limits: dict[str, tuple[float | None, float | None]] | None = None,
) -> QualityResult:
    params = params or default_params(frame)
    limits = limits or {}
    n, s = frame.values.shape
    flags = np.zeros((n, s), dtype=np.uint8)
    sensors: list[SensorQuality] = []
    warnings: list[str] = []

    for j, sensor_id in enumerate(frame.sensors):
        col = frame.values[:, j]
        observed = ~np.isnan(col)
        flags[~observed, j] |= FLAG_MISSING
        _, miss_len = runs_of(~observed, frame.segment)

        obs_idx = np.flatnonzero(observed)
        vals = col[obs_idx]
        segs = frame.segment[obs_idx]
        same = (vals[1:] == vals[:-1]) & (segs[1:] == segs[:-1])
        constant = bool(vals.size > 0 and np.all(vals == vals[0]))
        equal_fraction = float(same.mean()) if same.size else 0.0
        discrete = (not constant) and equal_fraction > 0.5

        stuck_runs: list[StuckRun] = []
        if vals.size and not constant and not discrete:
            new_run = np.r_[True, ~same]
            run_id = np.cumsum(new_run) - 1
            starts = np.flatnonzero(new_run)
            lengths = np.diff(np.r_[starts, vals.size])
            long = lengths >= params.flatline_min_points
            if long.any():
                flags[obs_idx[long[run_id]], j] |= FLAG_STUCK
                for st, ln in zip(starts[long], lengths[long], strict=True):
                    stuck_runs.append(
                        StuckRun(
                            start=_dt(int(frame.ts[obs_idx[st]])),
                            end=_dt(int(frame.ts[obs_idx[st + ln - 1]])),
                            points=int(ln),
                            value=float(vals[st]),
                        )
                    )

        out_of_range = 0
        lo, hi = limits.get(sensor_id, (None, None))
        if lo is not None or hi is not None:
            bad = np.zeros(n, dtype=bool)
            if lo is not None:
                bad |= observed & (col < lo)
            if hi is not None:
                bad |= observed & (col > hi)
            flags[bad, j] |= FLAG_RANGE
            out_of_range = int(bad.sum())

        spike_count = 0
        examples: list[SpikeExample] = []
        candidates = np.flatnonzero(observed & ((flags[:, j] & (FLAG_STUCK | FLAG_RANGE)) == 0))
        if candidates.size >= 5 and not constant and not discrete:
            cvals = col[candidates]
            cseg = frame.segment[candidates]
            sigma = _robust_sigma(cvals, cseg[1:] == cseg[:-1])
            if sigma > 0:
                cuts = np.r_[0, np.flatnonzero(np.diff(cseg) != 0) + 1, candidates.size]
                found: list[tuple[float, int, float]] = []  # score, row, expected
                for a, b in zip(cuts[:-1], cuts[1:], strict=True):
                    score, median = _hampel(cvals[a:b], params.spike_window, 0.5 * sigma)
                    hits = np.flatnonzero(score > params.spike_threshold)
                    rows = candidates[a + hits]
                    flags[rows, j] |= FLAG_SPIKE
                    found.extend(zip(score[hits], rows, median[hits], strict=True))
                spike_count = len(found)
                found.sort(key=lambda item: -item[0])
                examples = [
                    SpikeExample(
                        timestamp=_dt(int(frame.ts[row])),
                        value=float(col[row]),
                        expected=float(exp),
                    )
                    for _, row, exp in found[:MAX_SPIKE_EXAMPLES]
                ]

        interval = frame.intervals_us.get(sensor_id)
        sensors.append(
            SensorQuality(
                sensor_id=sensor_id,
                expected_points=n,
                observed_points=int(observed.sum()),
                missing_points=int((~observed).sum()),
                missing_runs=int(miss_len.size),
                longest_missing_run=int(miss_len.max()) if miss_len.size else 0,
                spike_count=spike_count,
                spike_examples=examples,
                stuck_runs=stuck_runs,
                out_of_range_count=out_of_range,
                median_interval_s=interval / 1e6 if interval else None,
                constant=constant,
                discrete=discrete,
            )
        )

    gaps_all = sorted(frame.gaps, key=lambda g: g[0] - g[1])  # longest first
    gaps = [
        TimeGap(
            start=_dt(a),
            end=_dt(b),
            duration_s=(b - a) / 1e6,
            missing_points=max(int(round((b - a) / frame.step_us)) - 1, 0),
        )
        for a, b in gaps_all[:MAX_GAPS_IN_REPORT]
    ]
    warnings = _warnings(sensors, frame, len(frame.gaps), sum(g[1] - g[0] for g in frame.gaps))
    report = QualityReport(
        machine_id=machine_id,
        start=_dt(int(frame.ts[0])),
        end=_dt(int(frame.ts[-1])),
        step_s=frame.step_s,
        rows=n,
        segments=int(frame.segment[-1]) + 1,
        gap_count=len(frame.gaps),
        gap_total_s=sum(g[1] - g[0] for g in frame.gaps) / 1e6,
        gaps=gaps,
        sensors=sensors,
        missing_points=sum(q.missing_points for q in sensors),
        spike_count=sum(q.spike_count for q in sensors),
        stuck_sensors=[q.sensor_id for q in sensors if q.stuck_runs],
        constant_sensors=[q.sensor_id for q in sensors if q.constant],
        warnings=warnings,
        params=params,
        sources=[
            Source(kind="data", ref=f"duckdb:measurements/{machine_id}"),
            Source(kind="tool", ref="diagsys.data.quality.check_quality"),
        ],
    )
    return QualityResult(report, flags)


def _warnings(
    sensors: list[SensorQuality], frame: SensorFrame, gap_count: int, gap_total_us: int
) -> list[str]:
    out: list[str] = []
    for q in sensors:
        frac = q.missing_points / q.expected_points if q.expected_points else 0.0
        if frac > 0.01:
            out.append(f"{q.sensor_id}: {frac:.1%} of readings missing")
        if q.constant:
            out.append(f"{q.sensor_id}: constant over the whole period (useless for diagnosis)")
        for run in q.stuck_runs:
            out.append(
                f"{q.sensor_id}: stuck at {run.value:g} for {run.points} readings "
                f"({run.start:%Y-%m-%d %H:%M} to {run.end:%Y-%m-%d %H:%M} UTC)"
            )
        if q.out_of_range_count:
            out.append(f"{q.sensor_id}: {q.out_of_range_count} readings outside profile limits")
        if q.median_interval_s and q.median_interval_s > 1.5 * frame.step_s:
            out.append(
                f"{q.sensor_id}: sampled every {q.median_interval_s:g} s, slower than the "
                f"{frame.step_s:g} s grid; consider --step-s {q.median_interval_s:g}"
            )
    if gap_count:
        out.append(
            f"{gap_count} time gaps with no data from any sensor, {gap_total_us / 3.6e9:.1f} h total"
        )
    return out
