"""Interface I4 (tool output): the data quality report produced by `diagsys.data.quality`.

Every number in the report comes from the deterministic check; nothing in it is estimated by
a language model. The report is a summary: the per-reading detail lives in the flags column of
the cleaned Parquet file.
"""

from datetime import datetime

from pydantic import Field

from .common import Contract, Source


class QualityParams(Contract):
    """Thresholds of the check, stored in the report so that a result can be reproduced."""

    step_s: float = Field(gt=0)  # grid step used for the analysis
    max_gap_s: float = Field(gt=0)  # a pause longer than this splits the data into segments
    flatline_min_points: int = Field(ge=3)  # identical consecutive readings that mean "stuck"
    spike_window: int = Field(ge=5)  # points in the rolling window of the spike filter
    spike_threshold: float = Field(gt=0)  # deviation from the local median, in robust sigmas
    max_interp_points: int = Field(ge=0)  # longest run of bad points that cleaning will fill


class TimeGap(Contract):
    """A pause with no data from any sensor (for example a repair)."""

    start: datetime  # last reading before the pause
    end: datetime  # first reading after the pause
    duration_s: float
    missing_points: int  # grid points that fall inside the pause


class StuckRun(Contract):
    """Identical consecutive readings of a sensor that normally varies."""

    start: datetime
    end: datetime
    points: int
    value: float


class SpikeExample(Contract):
    timestamp: datetime
    value: float
    expected: float  # median of the neighbouring readings


class SensorQuality(Contract):
    sensor_id: str
    expected_points: int  # rows of the regular grid
    observed_points: int
    missing_points: int
    missing_runs: int
    longest_missing_run: int  # in grid points
    spike_count: int
    spike_examples: list[SpikeExample] = Field(default_factory=list)  # the strongest ones
    stuck_runs: list[StuckRun] = Field(default_factory=list)
    out_of_range_count: int = 0  # against min_value/max_value from the machine profile
    median_interval_s: float | None = None  # typical time between this sensor's readings
    constant: bool = False  # the same value over the whole period
    discrete: bool = False  # mostly repeated values (a state or counter): stuck/spike checks off


class QualityReport(Contract):
    machine_id: str
    start: datetime
    end: datetime
    step_s: float
    rows: int  # rows of the regular grid (pauses excluded)
    segments: int  # continuous stretches of data
    gap_count: int
    gap_total_s: float
    gaps: list[TimeGap] = Field(default_factory=list)  # the longest ones, at most 50
    sensors: list[SensorQuality]
    missing_points: int
    spike_count: int
    stuck_sensors: list[str] = Field(default_factory=list)
    constant_sensors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    params: QualityParams
    sources: list[Source] = Field(default_factory=list)
