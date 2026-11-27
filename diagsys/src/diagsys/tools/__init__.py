"""Deterministic tools (interface I4): anomalies, RUL, optimization, document search: K4-K7."""

from .diagnosis import (
    METHODS,
    DiagnosisError,
    DiagnosisResult,
    Episode,
    diagnose,
    reference_zscores,
    to_anomaly_report,
)

__all__ = [
    "METHODS",
    "DiagnosisError",
    "DiagnosisResult",
    "Episode",
    "diagnose",
    "reference_zscores",
    "to_anomaly_report",
]

from .prediction import (  # noqa: E402
    RUL_METHODS,
    ForecastResult,
    PredictionError,
    build_dataset,
    failure_probability,
    fit_model,
    forecast_latest,
    predict_windows,
    reliability_warnings,
)

__all__ += [
    "RUL_METHODS",
    "ForecastResult",
    "PredictionError",
    "build_dataset",
    "failure_probability",
    "fit_model",
    "forecast_latest",
    "predict_windows",
    "reliability_warnings",
]

from .prescription import (  # noqa: E402
    CostModel,
    Prescription,
    PrescriptionError,
    ServiceOption,
    prescribe,
    profile_downtime_cost,
)

__all__ += [
    "CostModel",
    "Prescription",
    "PrescriptionError",
    "ServiceOption",
    "prescribe",
    "profile_downtime_cost",
]
