"""A regular time grid built from stored readings: the common input of quality, cleaning, features.

Readings of all sensors are snapped to one grid (step = typical sampling interval). A pause longer
than `max_gap_s` splits the data into segments; the grid has no rows inside a pause. A reading
missing from a grid cell is NaN, so "missing" becomes visible and comparable between sensors.
"""

from dataclasses import dataclass

import numpy as np
import pyarrow as pa

# Bit flags kept per reading (uint8): what the checks found and what cleaning did.
FLAG_MISSING = 1
FLAG_SPIKE = 2
FLAG_STUCK = 4
FLAG_RANGE = 8
FLAG_IMPUTED = 16
FLAG_LEGEND = {
    "missing": FLAG_MISSING,
    "spike": FLAG_SPIKE,
    "stuck": FLAG_STUCK,
    "out_of_range": FLAG_RANGE,
    "imputed": FLAG_IMPUTED,
}

MAX_GRID_ROWS = 50_000_000


class FrameError(Exception):
    """The readings cannot be put on a grid; the message says why."""


@dataclass
class SensorFrame:
    sensors: list[str]
    ts: np.ndarray  # int64 microseconds since epoch (UTC), one per grid row
    segment: np.ndarray  # int32 segment id per row
    values: np.ndarray  # float64 (rows, sensors); NaN = no (valid) reading
    step_us: int
    max_gap_us: int
    gaps: list[tuple[int, int]]  # (last ts before pause, first ts after pause), microseconds
    intervals_us: dict[str, int | None]  # typical interval between each sensor's own readings

    @property
    def step_s(self) -> float:
        return self.step_us / 1e6

    @property
    def rows(self) -> int:
        return int(self.ts.size)

    def with_values(self, values: np.ndarray) -> "SensorFrame":
        return SensorFrame(
            self.sensors, self.ts, self.segment, values, self.step_us, self.max_gap_us,
            self.gaps, self.intervals_us,
        )


def build_frame(
    table: pa.Table, *, step_s: float | None = None, max_gap_s: float | None = None
) -> SensorFrame:
    """`table` has columns sensor_id (string), ts (timestamp, UTC) and value (double)."""
    if table.num_rows == 0:
        raise FrameError("no readings to analyse")
    ts = np.asarray(table["ts"].cast(pa.timestamp("us", tz="UTC")).cast(pa.int64()))
    encoded = table["sensor_id"].combine_chunks().dictionary_encode()
    names = encoded.dictionary.to_pylist()
    codes = np.asarray(encoded.indices)
    values = np.asarray(table["value"].cast(pa.float64()))

    finite = np.isfinite(values)
    ts, codes, values = ts[finite], codes[finite], values[finite]
    if ts.size == 0:
        raise FrameError("no finite readings to analyse")

    order = sorted(range(len(names)), key=names.__getitem__)  # deterministic sensor order
    remap = np.empty(len(names), dtype=np.int64)
    for new, old in enumerate(order):
        remap[old] = new
    sensors = [names[i] for i in order]
    codes = remap[codes]

    master = np.unique(ts)
    if master.size < 2:
        raise FrameError("need readings at two or more different times")
    diffs = np.diff(master)
    if step_s is None:
        step_us = max(int(round(float(np.median(diffs)) / 1000.0)) * 1000, 1000)
    else:
        step_us = max(int(round(step_s * 1e6)), 1)
    max_gap_us = int(round(max_gap_s * 1e6)) if max_gap_s is not None else 5 * step_us
    if max_gap_us < step_us:
        raise FrameError("max gap must be at least one grid step")

    breaks = np.flatnonzero(diffs > max_gap_us)
    seg_starts = master[np.r_[0, breaks + 1]]
    seg_ends = master[np.r_[breaks, master.size - 1]]
    seg_len = np.rint((seg_ends - seg_starts) / step_us).astype(np.int64) + 1
    rows = int(seg_len.sum())
    if rows > MAX_GRID_ROWS:
        raise FrameError(f"the grid would have {rows} rows; use a larger --step-s")
    offsets = np.r_[0, np.cumsum(seg_len)[:-1]]

    seg_of_reading = np.searchsorted(seg_starts, ts, side="right") - 1
    cell = np.rint((ts - seg_starts[seg_of_reading]) / step_us).astype(np.int64)
    flat = offsets[seg_of_reading] + cell

    matrix = np.full((rows, len(sensors)), np.nan)
    intervals: dict[str, int | None] = {}
    for j, name in enumerate(sensors):
        mine = codes == j
        sums = np.bincount(flat[mine], weights=values[mine], minlength=rows)
        counts = np.bincount(flat[mine], minlength=rows)
        np.divide(sums, counts, out=matrix[:, j], where=counts > 0)
        own = np.unique(ts[mine])
        intervals[name] = int(np.median(np.diff(own))) if own.size > 1 else None

    row_seg = np.repeat(np.arange(seg_len.size, dtype=np.int32), seg_len)
    row_idx = np.arange(rows, dtype=np.int64) - offsets[row_seg]
    grid_ts = seg_starts[row_seg] + row_idx * step_us
    gaps = [(int(master[i]), int(master[i + 1])) for i in breaks]
    return SensorFrame(sensors, grid_ts, row_seg, matrix, step_us, max_gap_us, gaps, intervals)
