"""Deterministic tools (interface I4): anomalies, RUL, optimization, document search: K4-K7."""

from .diagnosis import (
    METHODS,
    DiagnosisError,
    DiagnosisResult,
    Episode,
    diagnose,
    to_anomaly_report,
)

__all__ = [
    "METHODS",
    "DiagnosisError",
    "DiagnosisResult",
    "Episode",
    "diagnose",
    "to_anomaly_report",
]
