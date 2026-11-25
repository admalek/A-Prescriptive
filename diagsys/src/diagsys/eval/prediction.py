"""Scoring of the RUL prediction by leave-one-cycle-out cross-validation.

Every failure cycle is held out in turn: the model is fitted on the other cycles and forecasts every
usable window of the held-out one. Scores are pooled over all out-of-fold forecasts.

Protocol (fixed before looking at results):
  * Point accuracy: MAE of the median forecast in hours, overall and for windows whose true RUL is
    at most 72 h and 24 h (the part that matters for maintenance decisions).
  * Calibration: coverage of the 80% interval (target 0.80) and its mean width.
  * Failure within `horizon_h`: Brier score of the derived probability, and the skill against always
    predicting the base rate of the scored windows (1 = perfect, 0 = no better, negative = worse).
Limitation to state with the results: the cycles used for training include ones recorded after the
held-out cycle, so this measures generalisation between cycles, not a strict forward-in-time
backtest (`diagsys predict --at` does the strict version for one point in time).
"""

import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pyarrow as pa

from ..config import Settings
from ..machines import MachineProfile
from ..storage import DuckDBStorage
from ..tools.prediction import (
    RUL_METHODS,
    Dataset,
    PredictionError,
    build_dataset,
    failure_probability,
    fit_model,
    predict_windows,
)
from .run import Run, start_run

_BINS = ((0.0, 24.0), (24.0, 72.0), (72.0, 168.0), (168.0, float("inf")))


@dataclass(frozen=True)
class PredictionScore:
    method: str
    horizon_h: float
    windows: int
    cycles: int
    mae_h: float
    mae_last72_h: float | None
    mae_last24_h: float | None
    coverage80: float
    coverage80_last72: float | None
    mean_width_h: float
    brier: float
    brier_base_rate: float
    brier_skill: float | None
    by_true_rul: dict[str, dict[str, float]]


def cross_validate(ds: Dataset, method: str, *, seed: int = 42) -> np.ndarray:
    """Out-of-fold quantile forecasts (hours); rows that were not scored are NaN."""
    labelled = ds.usable & ~np.isnan(ds.rul_h)
    cycles = np.unique(ds.cycle[labelled])
    if cycles.size < 2:
        raise PredictionError(
            f"cross-validation needs at least 2 failure cycles with usable data, found "
            f"{cycles.size}"
        )
    out = np.full((ds.X.shape[0], 5), np.nan)
    for c in cycles:
        held = labelled & (ds.cycle == c)
        model = fit_model(method, ds, train=~(ds.cycle == c), seed=seed)
        out[held] = predict_windows(model, ds, held)[held]
    return out


def score_predictions(
    pred: np.ndarray, ds: Dataset, method: str, horizon_h: float = 72.0
) -> PredictionScore:
    if horizon_h <= 0:
        raise PredictionError("horizon_h must be positive")
    rows = ~np.isnan(pred).any(axis=1) & ~np.isnan(ds.rul_h)
    if not rows.any():
        raise PredictionError("no forecasts to score")
    truth = ds.rul_h[rows]
    q = pred[rows]
    err = np.abs(q[:, 2] - truth)
    inside = (truth >= q[:, 0]) & (truth <= q[:, -1])
    late72, late24 = truth <= 72.0, truth <= 24.0
    prob = np.clip(failure_probability(q, horizon_h), 0.0, 1.0)
    event = (truth <= horizon_h).astype(float)
    brier = float(np.mean((prob - event) ** 2))
    base = float(np.mean((event.mean() - event) ** 2))
    bins = {}
    for lo, hi in _BINS:
        sel = (truth >= lo) & (truth < hi)
        if sel.any():
            label = f"{lo:g}-{hi:g}h" if np.isfinite(hi) else f">{lo:g}h"
            bins[label] = {
                "windows": int(sel.sum()),
                "mae_h": float(err[sel].mean()),
                "coverage80": float(inside[sel].mean()),
            }
    return PredictionScore(
        method=method,
        horizon_h=horizon_h,
        windows=int(rows.sum()),
        cycles=int(np.unique(ds.cycle[rows]).size),
        mae_h=float(err.mean()),
        mae_last72_h=float(err[late72].mean()) if late72.any() else None,
        mae_last24_h=float(err[late24].mean()) if late24.any() else None,
        coverage80=float(inside.mean()),
        coverage80_last72=float(inside[late72].mean()) if late72.any() else None,
        mean_width_h=float((q[:, -1] - q[:, 0]).mean()),
        brier=brier,
        brier_base_rate=base,
        brier_skill=1.0 - brier / base if base > 0 else None,
        by_true_rul=bins,
    )


def run_prediction_experiment(
    settings: Settings,
    storage: DuckDBStorage,
    profile: MachineProfile,
    features: pa.Table,
    *,
    methods: tuple[str, ...] = RUL_METHODS,
    horizon_h: float = 72.0,
    seed: int = 42,
    dataset_options: dict[str, Any] | None = None,
    name: str = "prediction",
) -> tuple[Run, list[PredictionScore]]:
    """Cross-validate every method on the same data and store manifest, scores and a summary."""
    options = dict(dataset_options or {})
    failures = [e.timestamp for e in storage.read_events(profile.machine_id)]
    ds = build_dataset(features, failures, **options)
    config = {"methods": list(methods), "horizon_h": horizon_h, "dataset": options}
    run = start_run(settings, storage, profile.machine_id, name, seed=seed, config=config)
    scores = [
        score_predictions(cross_validate(ds, m, seed=seed), ds, m, horizon_h) for m in methods
    ]
    run.finish({"failures": len(failures), "scores": [asdict(s) for s in scores]})
    run.write_artifact(
        "features.json", json.dumps({"names": ds.names, "sensors": ds.sensors}, indent=2)
    )
    return run, scores
