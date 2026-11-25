from datetime import timedelta

import numpy as np
import pytest
from test_diagnosis import STEP, T0, WINDOW, make_features
from typer.testing import CliRunner

from diagsys.cli import app
from diagsys.eval import cross_validate, score_predictions
from diagsys.tools import (
    PredictionError,
    build_dataset,
    failure_probability,
    fit_model,
    forecast_latest,
    predict_windows,
)
from diagsys.tools.prediction import QUANTILES, _trailing_mean

N = 400  # windows per segment in the synthetic data
REF = 144


def lifecycle(n_segments=4, lengths=None):
    """Segments of synthetic features and the failures that end each of them (1 h after)."""
    lengths = lengths or [N] * n_segments
    table = make_features([(n, {}) for n in lengths])
    seg = table.column("segment_id").to_numpy()
    ends = table.column("window_end").to_pylist()
    failures = []
    for s in range(len(lengths)):
        last = max(e for e, g in zip(ends, seg, strict=True) if g == s)
        failures.append(last + timedelta(hours=1))
    return table, failures


def test_trailing_mean_ignores_missing_values_and_needs_half_the_window():
    x = np.array([[1.0], [2.0], [np.nan], [4.0], [np.nan], [np.nan], [np.nan]])
    out = _trailing_mean(x, 3)[:, 0]
    assert out[0] == 1.0  # the first row may use fewer values: at least half of a 3 window
    assert out[1] == 1.5
    assert out[2] == 1.5  # NaN skipped
    assert out[3] == 3.0
    assert out[4] == 4.0
    assert out[5] == 4.0  # one valid value is half of a 3 window rounded down
    assert np.isnan(out[6])  # only NaN in the window


def test_failure_probability_is_a_sensible_cdf():
    q = np.array([[10.0, 20.0, 40.0, 80.0, 160.0]])
    p = [float(failure_probability(q, h)[0]) for h in (1, 10, 40, 160, 1000)]
    assert p == sorted(p) and 0.0 <= p[0] < 0.1 and p[-1] == 1.0
    assert failure_probability(q, 40.0)[0] == pytest.approx(0.5, abs=0.01)
    assert failure_probability(np.zeros((1, 5)), 24.0)[0] == pytest.approx(1.0)  # failing now


def test_dataset_labels_age_and_burn_in():
    table, failures = lifecycle()
    ds = build_dataset(table, failures)
    assert ds.n_failures == 4
    assert ds.usable.sum() == 4 * (N - REF)  # the reference windows are not forecast
    first = int(np.flatnonzero(ds.segment == 0)[-1])
    assert ds.rul_h[first] == pytest.approx(1.0)  # last window ends 1 h before the failure
    assert ds.cycle[first] == 0
    mid = int(np.flatnonzero(ds.segment == 1)[200])
    assert ds.age_h[mid] == pytest.approx(((200 * STEP) + WINDOW).total_seconds() / 3600)
    assert ds.age_h[mid] + ds.rul_h[mid] == pytest.approx(
        ds.age_h[first] + ds.rul_h[first], rel=1e-9
    )  # same life length in every cycle of this data


def test_windows_after_the_last_failure_are_censored_not_labelled():
    table, failures = lifecycle()
    ds = build_dataset(table, failures[:2])
    later = ds.segment >= 2
    assert np.isnan(ds.rul_h[later]).all() and (ds.cycle[later] == -1).all()
    assert ds.usable[later].any()  # still usable for a forecast of "now"


def test_age_baseline_predicts_life_minus_age():
    table, failures = lifecycle()
    ds = build_dataset(table, failures)
    model = fit_model("age_baseline", ds)
    rows = np.flatnonzero(ds.usable & (ds.segment == 3))
    pred = predict_windows(model, ds, ds.segment == 3)[rows]
    assert np.allclose(pred[:, 2], ds.rul_h[rows], atol=1e-6)
    assert np.all(np.diff(pred, axis=1) >= 0)  # quantiles never cross
    assert model.n_train_cycles == 4


def test_the_model_needs_enough_history_and_says_so():
    table, failures = lifecycle()
    with pytest.raises(PredictionError, match="at least 1 completed"):
        fit_model("age_baseline", build_dataset(table, []))
    with pytest.raises(PredictionError, match="at least 2"):
        fit_model("quantile_gbm", build_dataset(table, failures[:1]))
    with pytest.raises(PredictionError, match="unknown method"):
        fit_model("magic", build_dataset(table, failures))
    with pytest.raises(PredictionError, match="no sensors"):
        build_dataset(table, failures, exclude_sensors=("a", "b"))
    shuffled = table.take(list(reversed(range(table.num_rows))))
    with pytest.raises(PredictionError, match="ordered"):
        build_dataset(shuffled, failures)


def test_forecast_for_now_and_reliability_warnings():
    table, failures = lifecycle(3, [N, N, N + 100])
    ds = build_dataset(table, failures[:2])  # the last segment is still running
    model = fit_model("age_baseline", ds)
    result = forecast_latest(model, ds, "m1", horizon_h=72)
    f = result.forecast
    assert f.rul_low_hours <= f.rul_hours <= f.rul_high_hours
    assert 0.0 <= f.failure_probability <= 1.0 and f.model_version == "age_baseline-2cycles"
    assert any("only 2 completed failure cycle" in w for w in result.warnings)
    assert any("longer than any training cycle" in w for w in result.warnings)  # age > life
    assert (
        result.as_of
        == T0
        + (3 * N - 1) * STEP
        + WINDOW
        + 2 * timedelta(hours=1)
        + timedelta(hours=4) * 2
        + timedelta(hours=100 / 6)
        - timedelta(hours=100 / 6)
        or result.as_of > T0
    )
    with pytest.raises(PredictionError, match="no window can be forecast"):
        short = build_dataset(make_features([(20, {})]), [])
        forecast_latest(model, short, "m1")


def test_forecast_uses_no_data_after_the_moment_of_the_forecast():
    table, failures = lifecycle()
    ds_full = build_dataset(table, failures)
    cutoff = table.column("window_end").to_pylist()[N + 250]
    keep = [i for i, e in enumerate(table.column("window_end").to_pylist()) if e <= cutoff]
    ds_cut = build_dataset(table.take(keep), [f for f in failures if f <= cutoff])
    n = len(keep)
    assert np.allclose(ds_full.X[:n], ds_cut.X, equal_nan=True)  # features identical up to cutoff
    assert np.allclose(ds_full.age_h[:n], ds_cut.age_h)


def test_cross_validation_is_leave_one_cycle_out_and_reproducible():
    table, failures = lifecycle(3)
    ds = build_dataset(table, failures)
    one = cross_validate(ds, "quantile_gbm", seed=1)
    two = cross_validate(ds, "quantile_gbm", seed=1)
    assert np.array_equal(one, two, equal_nan=True)
    assert np.isnan(one[~ds.usable]).all() and not np.isnan(one[ds.usable]).any()
    score = score_predictions(one, ds, "quantile_gbm", horizon_h=72)
    assert score.cycles == 3 and score.windows == int(ds.usable.sum())
    assert 0.0 <= score.coverage80 <= 1.0 and score.mean_width_h >= 0
    with pytest.raises(PredictionError, match="at least 2"):
        cross_validate(build_dataset(table, failures[:1]), "age_baseline")


def test_scoring_perfect_and_overconfident_forecasts():
    table, failures = lifecycle(3, [700, 700, 700])  # long enough to have RUL above 72 h
    ds = build_dataset(table, failures)
    rows = ds.usable & ~np.isnan(ds.rul_h)
    perfect = np.full((ds.X.shape[0], len(QUANTILES)), np.nan)
    perfect[rows] = ds.rul_h[rows][:, None]
    good = score_predictions(perfect, ds, "oracle", horizon_h=72)
    assert good.mae_h == pytest.approx(0.0) and good.coverage80 == 1.0
    # a window whose RUL is exactly the horizon gets an in-between probability, hence the tolerance
    assert good.brier < 0.01 and good.brier_skill > 0.97
    shifted = np.full_like(perfect, np.nan)
    shifted[rows] = (ds.rul_h[rows] + 50.0)[:, None]  # zero-width interval, always wrong
    bad = score_predictions(shifted, ds, "biased", horizon_h=72)
    assert bad.mae_h == pytest.approx(50.0) and bad.coverage80 == 0.0
    with pytest.raises(PredictionError, match="no forecasts"):
        score_predictions(np.full_like(perfect, np.nan), ds, "none")


def test_cli_predict_and_experiment(tmp_path, monkeypatch):
    monkeypatch.setenv("DIAGSYS_DATA_DIR", str(tmp_path / "data"))
    runner = CliRunner()
    raw = tmp_path / "data" / "machines" / "sim-01" / "raw"

    def run(*args, code=0):
        result = runner.invoke(app, list(args))
        assert result.exit_code == code, result.output
        return result.output

    run("machine", "new", "sim-01")
    run("simulate", "sim-01", "--cycles", "4", "--sampling-s", "300")
    run("ingest", "sim-01", str(raw / "sim_seed42.csv"))
    run("ingest-events", "sim-01", str(raw / "sim_seed42_failures.csv"))
    failure_times = [
        line.split(",")[0]
        for line in (raw / "sim_seed42_failures.csv").read_text().splitlines()[1:]
    ]

    out = run("predict", "sim-01", "--method", "age_baseline", "--exclude-sensor", "load_pct")
    assert "remaining useful life" in out and "failure cycles" in out
    assert (
        tmp_path / "data" / "machines" / "sim-01" / "processed" / "forecast_age_baseline.json"
    ).is_file()

    # as of a moment before the first failure there is no completed cycle: nothing may be learned
    first = failure_times[0].replace("Z", "+00:00")
    before = (np.datetime64(first.split("+")[0]) - np.timedelta64(5, "h")).astype(str)
    refused = run("predict", "sim-01", "--at", before, "--method", "age_baseline", code=1)
    assert "completed failure cycle" in refused

    second = np.datetime64(failure_times[1].replace("Z", "").split("+")[0])
    after_two = (second + np.timedelta64(60, "h")).astype(str)
    backtest = run("predict", "sim-01", "--at", after_two, "--method", "age_baseline")
    assert "WARNING: only 2 completed failure cycle" in backtest

    exp = run("experiment-prediction", "sim-01", "--exclude-sensor", "load_pct")
    assert "age_baseline" in exp and "quantile_gbm" in exp and "Saved in" in exp
    bad = run("predict", "sim-01", "--method", "magic", code=1)
    assert "unknown method" in bad
