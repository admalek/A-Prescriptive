"""Cleaning: remove readings the quality check found untrustworthy, fill only short holes.

Spikes, stuck readings and out-of-range readings become NaN. Runs of NaN that are at most
`max_interp_points` long and have a valid reading on both sides (inside one segment) are filled
by linear interpolation and flagged IMPUTED. Longer holes stay NaN: inventing a long stretch
of data would hide the very problem a diagnosis should see.
"""

from dataclasses import dataclass

import numpy as np

from .frame import FLAG_IMPUTED, FLAG_RANGE, FLAG_SPIKE, FLAG_STUCK, SensorFrame
from .quality import runs_of

_DROP = FLAG_SPIKE | FLAG_STUCK | FLAG_RANGE


@dataclass
class CleanFrame:
    frame: SensorFrame  # cleaned values
    flags: np.ndarray  # detection flags plus FLAG_IMPUTED


def clean_frame(frame: SensorFrame, flags: np.ndarray, max_interp_points: int = 5) -> CleanFrame:
    values = frame.values.copy()
    out_flags = flags.copy()
    values[(flags & _DROP) != 0] = np.nan
    for j in range(values.shape[1]):
        col = values[:, j]
        starts, lengths = runs_of(np.isnan(col), frame.segment)
        for st, ln in zip(starts, lengths, strict=True):
            end = st + ln  # first row after the hole
            if ln > max_interp_points or st == 0 or end >= col.size:
                continue
            if frame.segment[st - 1] != frame.segment[st] or frame.segment[end] != frame.segment[st]:
                continue
            col[st:end] = np.linspace(col[st - 1], col[end], ln + 2)[1:-1]
            out_flags[st:end, j] |= FLAG_IMPUTED
    return CleanFrame(frame.with_values(values), out_flags)
