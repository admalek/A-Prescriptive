"""Scoring of the diagnosis against known failures, and the reproducible experiment around it.

Rules (fixed before looking at results, so they can be stated in the thesis):
  * An alarm episode is matched to the NEXT failure after the moment it was detected.
  * It is a true alarm when it was detected at most `horizon_h` hours before that failure,
    otherwise it is a false alarm (too early, or no failure follows).
  * A failure is detected when at least one true alarm is matched to it; the lead time is the time
    between the earliest such detection and the failure.
  * False alarms are reported per 100 hours of monitored operation (windows outside the reference).
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pyarrow as pa

from ..config import Settings
from ..machines import MachineProfile
from ..storage import DuckDBStorage
from ..tools.diagnosis import METHODS, DiagnosisResult, diagnose
from .run import Run, start_run


@dataclass(frozen=True)
class DiagnosisScore:
    method: str
    horizon_h: float
    failures: int
    detected: int
    recall: float | None  # None when there are no failures to detect
    median_lead_h: float | None
    mean_lead_h: float | None
    episodes: int
    true_alarms: int
    false_alarms: int
    monitored_hours: float
    false_alarms_per_100h: float | None
    lead_times_h: list[float]


def score_diagnosis(
    result: DiagnosisResult, failures: list[datetime], horizon_h: float
) -> DiagnosisScore:
    if horizon_h <= 0:
        raise ValueError("horizon_h must be positive")
    ordered = sorted(failures)
    first_detection: dict[int, datetime] = {}  # failure index -> earliest true detection
    true_alarms = 0
    for episode in result.episodes:
        nxt = next((i for i, f in enumerate(ordered) if f >= episode.detected_at), None)
        if nxt is None:
            continue
        lead = (ordered[nxt] - episode.detected_at).total_seconds() / 3600.0
        if lead <= horizon_h:
            true_alarms += 1
            if nxt not in first_detection or episode.detected_at < first_detection[nxt]:
                first_detection[nxt] = episode.detected_at
    leads = [
        (ordered[i] - at).total_seconds() / 3600.0 for i, at in sorted(first_detection.items())
    ]
    false_alarms = len(result.episodes) - true_alarms
    hours = result.monitored_hours
    return DiagnosisScore(
        method=result.method,
        horizon_h=horizon_h,
        failures=len(ordered),
        detected=len(first_detection),
        recall=len(first_detection) / len(ordered) if ordered else None,
        median_lead_h=float(np.median(leads)) if leads else None,
        mean_lead_h=float(np.mean(leads)) if leads else None,
        episodes=len(result.episodes),
        true_alarms=true_alarms,
        false_alarms=false_alarms,
        monitored_hours=hours,
        false_alarms_per_100h=false_alarms / hours * 100.0 if hours > 0 else None,
        lead_times_h=leads,
    )


def run_diagnosis_experiment(
    settings: Settings,
    storage: DuckDBStorage,
    profile: MachineProfile,
    features: pa.Table,
    *,
    methods: tuple[str, ...] = METHODS,
    horizon_h: float = 72.0,
    seed: int = 42,
    diagnose_options: dict[str, Any] | None = None,
    name: str = "diagnosis",
) -> tuple[Run, list[DiagnosisScore]]:
    """Run every method on the same features and failures and store manifest, metrics, episodes."""
    options = dict(diagnose_options or {})
    failures = [e.timestamp for e in storage.read_events(profile.machine_id)]
    config = {"methods": list(methods), "horizon_h": horizon_h, "diagnose": options}
    run = start_run(settings, storage, profile.machine_id, name, seed=seed, config=config)
    scores, episodes = [], {}
    for method in methods:
        result = diagnose(features, method=method, seed=seed, **options)
        scores.append(score_diagnosis(result, failures, horizon_h))
        episodes[method] = [_episode_dict(e) for e in result.episodes]
    run.finish({"failures": len(failures), "scores": [asdict(s) for s in scores]})
    run.write_artifact("episodes.json", json.dumps(episodes, indent=2, ensure_ascii=False))
    return run, scores


def _episode_dict(episode) -> dict[str, Any]:
    return {
        "segment_id": episode.segment_id,
        "start": episode.start.isoformat(),
        "detected_at": episode.detected_at.isoformat(),
        "end": episode.end.isoformat(),
        "n_windows": episode.n_windows,
        "peak_score": episode.peak_score,
        "drivers": [
            {"sensor": s, "statistic": stat, "signed_z": z, "strength": strength}
            for s, stat, z, strength in episode.drivers
        ],
    }
