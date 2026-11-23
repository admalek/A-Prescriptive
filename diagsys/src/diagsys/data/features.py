"""Window features: summary statistics of each sensor over sliding windows of a segment.

A window never crosses a pause in the data. Its row is labelled with the time of its LAST reading
(`window_end`) and uses only readings up to that time, so features can be used for prediction
without looking into the future. Statistics ignore NaN; a sensor with too few valid readings in
a window gets NaN features for that window.
"""

import numpy as np
import pyarrow as pa
from numpy.lib.stride_tricks import sliding_window_view

from .frame import SensorFrame

STATS = ("mean", "std", "min", "max", "rms", "slope_per_h", "kurtosis")
_CHUNK_WINDOWS = 20_000


class FeatureError(Exception):
    """Features cannot be computed with these settings; the message says why."""


def window_points(frame: SensorFrame, window_s: float, stride_s: float) -> tuple[int, int]:
    length = int(round(window_s / frame.step_s))
    stride = max(int(round(stride_s / frame.step_s)), 1)
    if length < 5:
        raise FeatureError(
            f"window of {window_s:g} s is only {length} grid points ({frame.step_s:g} s each); "
            "use at least 5 points"
        )
    return length, stride


def compute_features(
    frame: SensorFrame, *, window_s: float = 3600.0, stride_s: float = 600.0,
    min_valid_frac: float = 0.8,
) -> pa.Table:
    length, stride = window_points(frame, window_s, stride_s)
    t_hours = np.arange(length) * frame.step_s / 3600.0
    seg_ids, seg_first = np.unique(frame.segment, return_index=True)
    seg_bounds = np.r_[seg_first, frame.rows]

    meta_start: list[np.ndarray] = []
    meta_end: list[np.ndarray] = []
    meta_seg: list[np.ndarray] = []
    blocks: list[np.ndarray] = []  # each (windows, sensors, stats + n_valid)
    for k, seg in enumerate(seg_ids):
        a, b = int(seg_bounds[k]), int(seg_bounds[k + 1])
        if b - a < length:
            continue
        starts = np.arange(0, b - a - length + 1, stride)
        meta_start.append(frame.ts[a + starts])
        meta_end.append(frame.ts[a + starts + length - 1])
        meta_seg.append(np.full(starts.size, seg, dtype=np.int32))
        per_sensor = [
            _window_stats(frame.values[a:b, j], length, stride, t_hours, min_valid_frac)
            for j in range(len(frame.sensors))
        ]
        blocks.append(np.stack(per_sensor, axis=1))
    if not blocks:
        raise FeatureError(
            f"no segment is long enough for a window of {window_s:g} s; shorten --window-s"
        )

    stats = np.concatenate(blocks)  # (windows, sensors, len(STATS) + 1)
    n_valid = stats[:, :, -1]
    columns: dict[str, pa.Array] = {
        "window_start": pa.array(np.concatenate(meta_start), pa.timestamp("us", tz="UTC")),
        "window_end": pa.array(np.concatenate(meta_end), pa.timestamp("us", tz="UTC")),
        "segment_id": pa.array(np.concatenate(meta_seg), pa.int32()),
        "n_points": pa.array(np.full(stats.shape[0], length, dtype=np.int32)),
        "valid_frac": pa.array(n_valid.min(axis=1) / length, pa.float64()),
    }
    for j, sensor in enumerate(frame.sensors):
        for i, stat in enumerate(STATS):
            columns[f"{sensor}__{stat}"] = pa.array(stats[:, j, i], pa.float64(), from_pandas=True)
    return pa.table(columns)


def _window_stats(
    x: np.ndarray, length: int, stride: int, t: np.ndarray, min_valid_frac: float
) -> np.ndarray:
    windows = sliding_window_view(x, length)[::stride]
    out = np.full((windows.shape[0], len(STATS) + 1), np.nan)
    for a in range(0, windows.shape[0], _CHUNK_WINDOWS):
        w = windows[a : a + _CHUNK_WINDOWS]
        ok = ~np.isnan(w)
        n = ok.sum(axis=1)
        w0 = np.where(ok, w, 0.0)
        safe_n = np.maximum(n, 1)
        mean = w0.sum(axis=1) / safe_n
        dev = np.where(ok, w - mean[:, None], 0.0)
        var_pop = (dev**2).sum(axis=1) / safe_n
        std = np.sqrt((dev**2).sum(axis=1) / np.maximum(n - 1, 1))
        m4 = (dev**4).sum(axis=1) / safe_n
        with np.errstate(divide="ignore", invalid="ignore"):
            kurt = np.where(var_pop > 1e-12, m4 / var_pop**2 - 3.0, np.nan)
        tt = np.where(ok, t[None, :], 0.0)
        st, sx, stx, stt = tt.sum(1), w0.sum(1), (tt * w0).sum(1), (tt * tt).sum(1)
        denom = n * stt - st**2
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = np.where(denom > 1e-12, (n * stx - st * sx) / denom, np.nan)
        res = np.stack(
            [
                mean,
                std,
                np.where(ok, w, np.inf).min(axis=1),
                np.where(ok, w, -np.inf).max(axis=1),
                np.sqrt((w0**2).sum(axis=1) / safe_n),
                slope,
                kurt,
            ],
            axis=1,
        )
        res[n < max(min_valid_frac * length, 2)] = np.nan
        out[a : a + w.shape[0], :-1] = res
        out[a : a + w.shape[0], -1] = n
    return out
