"""K5 prediction: remaining useful life (RUL) with an uncertainty range, from window features and
the recorded failures.

What is predicted: the time from the end of a feature window to the next recorded failure, as
quantiles (10/25/50/75/90 %). The Forecast contract carries the median, the 80% interval and an
approximate probability of failure within a horizon.

Rules that keep the result honest:
  * Only data up to the end of the window is used (features, age, the healthy reference). The
    reference is a FIXED number of windows at the start of each segment, so the same forecast would
    have been possible in real time. Windows inside the reference are not forecast (burn-in).
  * A "cycle" is the time between two failures (or from the start of a segment to a failure).
    Windows with no later failure are censored: they cannot be used for training or scoring.
  * Few failures means an unreliable model. `reliability_warnings` says so, and fitting refuses
    to run on less than the minimum.

Two methods share the dataset and the output format:
  age_baseline   no sensors: lifetimes of the training cycles minus the age since the last repair
  quantile_gbm   gradient boosting with quantile loss on health features (robust z-scores of
                 each sensor against its reference, their trailing means, and the age)
  quantile_gbm_cal  the same, with the 80% interval widened by a conformal correction: the
                 out-of-fold miss of q10/q90 on the training cycles (inner leave-one-cycle-out)
                 sets how far the outer quantiles move; the median is unchanged
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import pyarrow as pa

from ..contracts import Forecast
from ..contracts.common import Source
from .diagnosis import DiagnosisError, _cube, _int64, reference_zscores, sensors_in

RUL_METHODS = ("age_baseline", "quantile_gbm", "quantile_gbm_cal")
CAL_ALPHA = 0.2  # the calibrated interval targets 80% coverage (quantiles 0.1 and 0.9)
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
DEFAULT_STATS = ("mean", "std", "slope_per_h")
DEFAULT_TRAILING = (6, 36, 144)  # windows: 1 h, 6 h, 24 h at the default 10 min step
DEFAULT_REFERENCE_WINDOWS = 144  # 24 h at the default step: one full daily cycle
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_Z_CLIP = 50.0


class PredictionError(Exception):
    """The prediction cannot be computed as requested; the message says why."""


@dataclass
class Dataset:
    X: np.ndarray  # (windows, features); NaN where a feature is missing
    names: list[str]
    rul_h: np.ndarray  # hours to the next failure, NaN when no failure follows (censored)
    age_h: np.ndarray  # hours since the last repair (segment start or previous failure)
    cycle: np.ndarray  # index of the failure that ends the window's cycle, -1 if censored
    window_end: np.ndarray  # int64, microseconds since epoch
    segment: np.ndarray
    usable: np.ndarray  # window has a complete feature row (not burn-in, no missing values)
    sensors: list[str]
    n_failures: int


@dataclass
class Model:
    method: str
    n_train_cycles: int
    n_train_windows: int
    max_train_life_h: float
    seed: int
    _predict: object  # callable (X, age_h) -> (n, len(QUANTILES)) hours

    def predict(self, X: np.ndarray, age_h: np.ndarray) -> np.ndarray:
        raw = np.asarray(self._predict(X, age_h), dtype=float)  # type: ignore[operator]
        return np.maximum.accumulate(np.clip(raw, 0.0, None), axis=1)


@dataclass
class ForecastResult:
    forecast: Forecast
    as_of: datetime
    age_h: float
    warnings: list[str]


def build_dataset(
    features: pa.Table,
    failures: list[datetime],
    *,
    exclude_sensors: tuple[str, ...] = (),
    stats: tuple[str, ...] = DEFAULT_STATS,
    trailing: tuple[int, ...] = DEFAULT_TRAILING,
    reference_windows: int = DEFAULT_REFERENCE_WINDOWS,
) -> Dataset:
    names = [s for s in sensors_in(features) if s not in exclude_sensors]
    if not names:
        raise PredictionError("no sensors to predict from")
    try:
        cube = _cube(features, names, stats)
    except DiagnosisError as exc:
        raise PredictionError(str(exc)) from exc
    seg = features.column("segment_id").to_numpy().astype(np.int64)
    w_end = _int64(features, "window_end")
    w_start = _int64(features, "window_start")
    if not np.array_equal(np.lexsort((w_end, seg)), np.arange(seg.size)):
        raise PredictionError("features must be ordered by segment and time")

    valid = ~np.isnan(cube).any(axis=(1, 2))
    z, reference, skipped = reference_zscores(
        cube, seg, valid, reference_windows=reference_windows, min_tail=1
    )
    z = np.clip(z, -_Z_CLIP, _Z_CLIP).reshape(seg.size, -1)
    z_names = [f"{s}__{st}__z" for s in names for st in stats]

    blocks, columns = [z], list(z_names)
    for width in trailing:
        smooth = np.full_like(z, np.nan)
        for s in np.unique(seg):
            rows = seg == s
            smooth[rows] = _trailing_mean(z[rows], width)
        blocks.append(smooth)
        columns += [f"{n}_tr{width}" for n in z_names]

    fail_us = np.sort(
        np.array([int((f - _EPOCH).total_seconds() * 1e6) for f in failures], dtype=np.int64)
    )
    seg_start = {int(s): int(w_start[seg == s].min()) for s in np.unique(seg)}
    start_us = np.array([seg_start[int(s)] for s in seg], dtype=np.int64)
    never = np.iinfo(np.int64).min
    if fail_us.size:
        prev = np.searchsorted(fail_us, w_end, side="left") - 1  # last failure before the window
        last_fail = np.where(prev >= 0, fail_us[np.maximum(prev, 0)], never)
        nxt = np.searchsorted(fail_us, w_end, side="left")
        censored = nxt >= fail_us.size
        following = fail_us[np.minimum(nxt, fail_us.size - 1)]
        rul_h = np.where(censored, np.nan, (following - w_end) / 3.6e9)
        cycle = np.where(censored, -1, nxt)
    else:  # no recorded failure: every window is censored
        last_fail = np.full(seg.size, never, dtype=np.int64)
        rul_h = np.full(seg.size, np.nan)
        cycle = np.full(seg.size, -1)
    age_h = (w_end - np.maximum(start_us, last_fail)) / 3.6e9

    X = np.hstack([*blocks, age_h[:, None]])
    columns.append("age_h")
    usable = valid & ~reference & ~np.isin(seg, skipped) & ~np.isnan(z).any(axis=1)
    return Dataset(
        X=X,
        names=columns,
        rul_h=rul_h,
        age_h=age_h,
        cycle=cycle.astype(np.int64),
        window_end=w_end,
        segment=seg,
        usable=usable,
        sensors=names,
        n_failures=int(fail_us.size),
    )


def fit_model(
    method: str, ds: Dataset, train: np.ndarray | None = None, *, seed: int = 42
) -> Model:
    """Fit on the labelled (non-censored) usable windows selected by `train` (default: all)."""
    if method not in RUL_METHODS:
        raise PredictionError(f"unknown method {method!r}: use one of {', '.join(RUL_METHODS)}")
    mask = ds.usable & ~np.isnan(ds.rul_h)
    if train is not None:
        mask &= train
    cycles = np.unique(ds.cycle[mask])
    minimum = {"age_baseline": 1, "quantile_gbm": 2, "quantile_gbm_cal": 3}[method]
    if cycles.size < minimum:
        raise PredictionError(
            f"{method} needs at least {minimum} completed failure cycle(s) with usable data, "
            f"found {cycles.size}; import more failure history (sheet 'historia')"
        )
    life = np.array(
        [
            np.median(ds.age_h[mask & (ds.cycle == c)] + ds.rul_h[mask & (ds.cycle == c)])
            for c in cycles
        ]
    )
    if method == "age_baseline":
        quantiles = np.quantile(life, QUANTILES)

        def predict(X, age_h):
            return np.clip(quantiles[None, :] - age_h[:, None], 0.0, None)

    else:
        target = np.log1p(ds.rul_h[mask])
        models = _fit_gbm(ds.X[mask], target, QUANTILES, seed)
        correction = 0.0
        if method == "quantile_gbm_cal":
            correction = _conformal_correction(ds, mask, cycles, seed)

        def predict(X, age_h):
            log_q = np.column_stack([m.predict(X) for m in models])
            return np.expm1(_widen(log_q, correction))

    return Model(
        method=method,
        n_train_cycles=int(cycles.size),
        n_train_windows=int(mask.sum()),
        max_train_life_h=float(life.max()),
        seed=seed,
        _predict=predict,
    )


def _fit_gbm(X: np.ndarray, y: np.ndarray, levels, seed: int) -> list:
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError as exc:  # pragma: no cover - depends on the optional 'ml' extra
        raise PredictionError("quantile methods need scikit-learn: uv sync --extra ml") from exc
    return [
        HistGradientBoostingRegressor(
            loss="quantile",
            quantile=q,
            max_iter=150,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=40,
            l2_regularization=1.0,
            random_state=seed,
        ).fit(X, y)
        for q in levels
    ]


def _conformal_correction(ds: Dataset, mask: np.ndarray, cycles: np.ndarray, seed: int) -> float:
    """Conformalised quantile regression in log time (Romano et al. 2019), cycles as folds.

    For each training cycle the q10/q90 models are fitted on the other cycles and scored on it:
    E = max(q10 - y, y - q90), positive when the true value fell outside the interval. The
    correction is the (1 - alpha)(1 + 1/n) quantile of E over all out-of-fold windows. Windows of
    one cycle are strongly correlated, so the guarantee is approximate; the held-out evaluation
    says whether coverage really improved.
    """
    low, high = QUANTILES[0], QUANTILES[-1]
    y_all = np.log1p(ds.rul_h)
    scores = []
    for c in cycles:
        fit = mask & (ds.cycle != c)
        held = mask & (ds.cycle == c)
        lo, hi = _fit_gbm(ds.X[fit], y_all[fit], (low, high), seed)
        y = y_all[held]
        scores.append(np.maximum(lo.predict(ds.X[held]) - y, y - hi.predict(ds.X[held])))
    e = np.concatenate(scores)
    level = min(1.0, (1 - CAL_ALPHA) * (1 + 1 / e.size))
    return float(np.quantile(e, level))


def _widen(log_q: np.ndarray, correction: float) -> np.ndarray:
    """Move the lower quantiles down and the upper ones up by `correction` (log hours)."""
    if correction == 0.0:
        return log_q
    out = log_q.copy()
    mid = len(QUANTILES) // 2
    out[:, :mid] = np.minimum(out[:, :mid] - correction, out[:, [mid]])
    out[:, mid + 1 :] = np.maximum(out[:, mid + 1 :] + correction, out[:, [mid]])
    return np.clip(out, 0.0, None)


def predict_windows(model: Model, ds: Dataset, rows: np.ndarray | None = None) -> np.ndarray:
    """Quantile forecasts (hours) for the usable windows in `rows`; other rows are NaN."""
    chosen = ds.usable if rows is None else ds.usable & rows
    out = np.full((ds.X.shape[0], len(QUANTILES)), np.nan)
    if chosen.any():
        out[chosen] = model.predict(ds.X[chosen], ds.age_h[chosen])
    return out


def failure_probability(quantile_hours: np.ndarray, horizon_h: float) -> np.ndarray:
    """P(RUL <= horizon) from the predicted quantiles: a piecewise-linear CDF in log time.

    Approximate by construction (five quantiles, tails extended linearly); evaluate it with the
    Brier score before relying on it.
    """
    q = np.atleast_2d(quantile_hours)
    x = np.log1p(q)
    levels = np.array(QUANTILES)
    left = np.clip(x[:, 0] - (x[:, 1] - x[:, 0]) * (levels[0] / (levels[1] - levels[0])), 0, None)
    right = x[:, -1] + (x[:, -1] - x[:, -2]) * ((1 - levels[-1]) / (levels[-1] - levels[-2]))
    xs = np.column_stack([left, x, right]) + np.arange(x.shape[1] + 2) * 1e-9
    ps = np.concatenate([[0.0], levels, [1.0]])
    target = np.log1p(horizon_h)
    return np.array([np.interp(target, row, ps) for row in xs])


def reliability_warnings(model: Model, age_h: float | None = None) -> list[str]:
    out = []
    n = model.n_train_cycles
    if n < 3:
        out.append(
            f"only {n} completed failure cycle(s) in the history: the forecast is not reliable"
        )
    elif n < 10:
        out.append(f"{n} failure cycles in the history: treat the interval as a rough guide")
    if age_h is not None and age_h > model.max_train_life_h:
        out.append(
            f"the machine has run {age_h:.0f} h since the last repair, longer than any training "
            f"cycle ({model.max_train_life_h:.0f} h): extrapolation"
        )
    return out


def forecast_latest(
    model: Model, ds: Dataset, machine_id: str, *, horizon_h: float = 72.0
) -> ForecastResult:
    """Forecast for the most recent usable window (what the machine looks like now)."""
    if horizon_h <= 0:
        raise PredictionError("horizon_h must be positive")
    rows = np.flatnonzero(ds.usable)
    if rows.size == 0:
        raise PredictionError(
            "no window can be forecast yet: each segment needs its reference period "
            "(first 24 h of data after a repair) plus complete features"
        )
    i = int(rows[-1])
    q = model.predict(ds.X[i : i + 1], ds.age_h[i : i + 1])[0]
    prob = float(np.clip(failure_probability(q, horizon_h)[0], 0.0, 1.0))
    forecast = Forecast(
        machine_id=machine_id,
        rul_hours=float(q[2]),
        rul_low_hours=float(q[0]),
        rul_high_hours=float(q[-1]),
        failure_probability=prob,
        horizon_hours=horizon_h,
        model_version=f"{model.method}-{model.n_train_cycles}cycles",
        sources=[Source(kind="tool", ref=f"predict:{model.method}")],
    )
    return ForecastResult(
        forecast=forecast,
        as_of=_EPOCH + timedelta(microseconds=int(ds.window_end[i])),
        age_h=float(ds.age_h[i]),
        warnings=reliability_warnings(model, float(ds.age_h[i])),
    )


def _trailing_mean(x: np.ndarray, width: int) -> np.ndarray:
    """Mean of the last `width` rows (the row itself included), ignoring NaN; needs half valid."""
    ok = ~np.isnan(x)
    zeros = np.zeros((1, x.shape[1]))
    sums = np.vstack([zeros, np.cumsum(np.where(ok, x, 0.0), axis=0)])
    counts = np.vstack([zeros, np.cumsum(ok, axis=0)])
    hi = np.arange(1, x.shape[0] + 1)
    lo = np.maximum(hi - width, 0)
    total = sums[hi] - sums[lo]
    count = counts[hi] - counts[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count >= max(1, width // 2), total / np.maximum(count, 1), np.nan)
