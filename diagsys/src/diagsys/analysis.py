"""The analysis of one machine: data summary, diagnosis (both methods) and the RUL forecast.

Used by the supervisor window and by the pipeline. Nothing is estimated here: the numbers come from
the deterministic tools. Each part fails on its own, with a message, so a machine with readings but
no failure history still gets its diagnosis.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
import pyarrow as pa

from .config import Settings
from .data import FeatureError, FrameError, prepare
from .machines import load_machine
from .storage import DuckDBStorage
from .tools import (
    METHODS,
    DiagnosisError,
    PredictionError,
    build_dataset,
    diagnose,
    fit_model,
    forecast_latest,
)
from .tools.diagnosis import DiagnosisResult
from .tools.prediction import ForecastResult, Model


@dataclass
class DiagnosisView:
    method: str
    threshold: float
    episodes: list[dict]
    monitored_hours: float
    series: dict  # window_end (list of datetime), score, alarm, reference
    result: DiagnosisResult


@dataclass
class Analysis:
    machine_id: str
    created: datetime
    diagnosis: dict[str, DiagnosisView] = field(default_factory=dict)
    forecast: ForecastResult | None = None
    forecast_method: str | None = None
    model: Model | None = None
    failures: list[datetime] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)  # part -> why it is missing
    notes: list[str] = field(default_factory=list)


def best_rul_method(n_cycles: int) -> str | None:
    """The most capable RUL method the history allows (see the minimums in `fit_model`)."""
    if n_cycles >= 3:
        return "quantile_gbm_cal"
    if n_cycles == 2:
        return "quantile_gbm"
    if n_cycles == 1:
        return "age_baseline"
    return None


def analyze(
    settings: Settings,
    machine_id: str,
    *,
    exclude_sensors: tuple[str, ...] = (),
    horizon_h: float = 72.0,
    window_s: float = 3600.0,
    stride_s: float = 600.0,
    seed: int = 42,
) -> Analysis:
    profile = load_machine(settings, machine_id)
    out = Analysis(machine_id, datetime.now(UTC))
    with DuckDBStorage(settings.db_path) as storage:
        try:
            features = prepare(
                storage, profile, with_features=True, window_s=window_s, stride_s=stride_s
            ).features
        except (FrameError, FeatureError) as exc:
            out.errors["data"] = str(exc)
            return out
        out.failures = [e.timestamp for e in storage.read_events(machine_id)]
    for method in METHODS:
        try:
            result = diagnose(features, method=method, exclude_sensors=exclude_sensors, seed=seed)
            out.diagnosis[method] = _view(result)
        except DiagnosisError as exc:
            out.errors[f"diagnosis:{method}"] = str(exc)
    try:
        ds = build_dataset(features, out.failures, exclude_sensors=exclude_sensors)
        cycles = len(np.unique(ds.cycle[ds.usable & (ds.cycle >= 0)]))
        method = best_rul_method(cycles)
        if method is None:
            raise PredictionError(
                f"no completed failure cycle with usable data ({len(out.failures)} failures "
                "recorded): import the failure history to get a forecast"
            )
        out.model = fit_model(method, ds, seed=seed)
        out.forecast = forecast_latest(out.model, ds, machine_id, horizon_h=horizon_h)
        out.forecast_method = method
    except PredictionError as exc:
        out.errors["forecast"] = str(exc)
    if "forecast" not in out.errors and out.forecast_method == "age_baseline":
        out.notes.append("one failure cycle only: the forecast uses age alone, not the sensors")
    return out


def _view(result: DiagnosisResult) -> DiagnosisView:
    episodes = [
        {
            "start": e.start,
            "detected_at": e.detected_at,
            "end": e.end,
            "hours": round((e.end - e.start).total_seconds() / 3600, 1),
            "peak_score": round(e.peak_score, 2),
            "drivers": ", ".join(f"{s} ({st})" for s, st, _, _ in e.drivers[:3]),
        }
        for e in result.episodes
    ]
    ends = [datetime.fromtimestamp(us / 1e6, tz=UTC) for us in result.window_end]
    series = {
        "window_end": ends,
        "score": [None if np.isnan(x) else float(x) for x in result.score],
        "alarm": [bool(x) for x in result.alarm],
        "reference": [bool(x) for x in result.reference],
        "segment": [int(x) for x in result.segment],
    }
    return DiagnosisView(
        result.method, float(result.threshold), episodes, result.monitored_hours, series, result
    )


def table_from_series(view: DiagnosisView) -> pa.Table:
    """The score series as a small table (used by the page to plot)."""
    return pa.table(view.series)
