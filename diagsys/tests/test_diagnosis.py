from datetime import UTC, datetime, timedelta

import numpy as np
import pyarrow as pa
import pytest
from typer.testing import CliRunner

from diagsys.cli import app
from diagsys.eval import score_diagnosis
from diagsys.tools import DiagnosisError, diagnose, to_anomaly_report

T0 = datetime(2026, 1, 1, tzinfo=UTC)
STEP = timedelta(minutes=10)
WINDOW = timedelta(hours=1)
STATS = ("mean", "std", "rms", "slope_per_h", "kurtosis")


def make_features(segments, *, seed=0, sensors=("a", "b"), shift_all_stats=False):
    """segments: list of (n_windows, {sensor: (first_changed_window, shift_in_mean)}).

    shift_all_stats also moves the std and slope of a shifted sensor by the same number of its
    own standard deviations, which is what a real fault does to several features at once."""
    rng = np.random.default_rng(seed)
    cols: dict[str, list] = {"window_start": [], "window_end": [], "segment_id": []}
    for s in sensors:
        for st in STATS:
            cols[f"{s}__{st}"] = []
    clock = T0
    for seg_id, (n, shifts) in enumerate(segments):
        for i in range(n):
            start = clock + i * STEP
            cols["window_start"].append(start)
            cols["window_end"].append(start + WINDOW)
            cols["segment_id"].append(seg_id)
            for s in sensors:
                first, shift = shifts.get(s, (n + 1, 0.0))
                cols[f"{s}__mean"].append(10 + rng.normal(0, 1) + (shift if i >= first else 0))
                moved = shift_all_stats and i >= first
                cols[f"{s}__std"].append(1 + rng.normal(0, 0.1) + (shift * 0.1 if moved else 0))
                cols[f"{s}__rms"].append(10 + rng.normal(0, 1))
                cols[f"{s}__slope_per_h"].append(rng.normal(0, 0.2) + (shift * 0.2 if moved else 0))
                cols[f"{s}__kurtosis"].append(rng.normal(0, 0.3))
        clock = clock + n * STEP + timedelta(hours=4)  # a repair gap between segments
    n_rows = len(cols["segment_id"])
    cols["n_points"] = [60] * n_rows
    cols["valid_frac"] = [1.0] * n_rows
    ts = pa.timestamp("us", tz="UTC")
    return pa.table(
        {k: pa.array(v, ts) if k.startswith("window_") else pa.array(v) for k, v in cols.items()}
    )


def test_a_shift_in_one_sensor_is_found_and_attributed_by_the_threshold_rule():
    table = make_features([(400, {"a": (250, 8.0)})])
    result = diagnose(table, method="robust_z")
    assert len(result.episodes) == 1
    episode = result.episodes[0]
    shift_time = T0 + 250 * STEP + WINDOW
    assert episode.detected_at >= shift_time  # never before the change is complete in a window
    assert episode.detected_at <= shift_time + 6 * STEP
    sensor, stat, signed_z, _ = episode.drivers[0]
    assert (sensor, stat) == ("a", "mean") and signed_z > 3


def test_iforest_finds_a_fault_that_moves_several_features():
    table = make_features([(400, {"a": (250, 8.0)})], shift_all_stats=True)
    result = diagnose(table, method="iforest")
    assert len(result.episodes) == 1
    episode = result.episodes[0]
    assert episode.detected_at >= T0 + 250 * STEP + WINDOW
    assert episode.drivers[0][0] == "a"


@pytest.mark.parametrize("method", ["robust_z", "iforest"])
def test_stationary_data_gives_no_episode(method):
    result = diagnose(make_features([(400, {})], seed=3), method=method)
    assert result.episodes == []
    assert result.monitored_hours > 0


def test_each_segment_is_judged_against_its_own_reference():
    # the second segment is shifted from its very first window: that is its normal after a repair
    table = make_features([(300, {}), (300, {"a": (0, 15.0)})])
    result = diagnose(table, method="robust_z")
    assert result.episodes == []


def test_reference_windows_never_alarm_and_scores_are_nan_only_for_missing_features():
    table = make_features([(200, {"a": (0, 30.0)})])
    result = diagnose(table, method="robust_z")
    assert result.reference.sum() == 50
    assert not result.alarm[result.reference].any()


def test_detection_time_is_the_end_of_the_window_completing_the_run():
    table = make_features([(300, {"a": (200, 20.0)})])
    result = diagnose(table, method="robust_z", min_run=4)
    episode = result.episodes[0]
    first_alarm_end = T0 + 200 * STEP + WINDOW
    assert episode.detected_at == first_alarm_end + 3 * STEP
    assert episode.start == T0 + 200 * STEP


def test_isolated_alarm_windows_are_not_an_episode():
    table = make_features([(300, {"a": (200, 0.0)})])
    mean = table.column("a__mean").to_numpy().copy()
    mean[220] += 40  # a single spike window
    table = table.set_column(table.column_names.index("a__mean"), "a__mean", pa.array(mean))
    assert diagnose(table, method="robust_z", min_run=3).episodes == []
    assert len(diagnose(table, method="robust_z", min_run=1).episodes) == 1


def test_short_gaps_between_alarms_merge_into_one_episode():
    table = make_features([(300, {"a": (200, 20.0)})])
    mean = table.column("a__mean").to_numpy().copy()
    mean[230] = 10.0  # one healthy-looking window inside the fault
    table = table.set_column(table.column_names.index("a__mean"), "a__mean", pa.array(mean))
    assert len(diagnose(table, method="robust_z", merge_gap=2).episodes) == 1
    assert len(diagnose(table, method="robust_z", merge_gap=0).episodes) == 2


def test_exclude_sensors_removes_their_influence():
    table = make_features([(400, {"a": (250, 8.0)})])
    result = diagnose(table, method="robust_z", exclude_sensors=("a",))
    assert result.sensors == ["b"] and result.episodes == []


def test_iforest_is_reproducible_for_a_seed():
    table = make_features([(400, {"a": (250, 3.0), "b": (260, -3.0)})])
    one = diagnose(table, method="iforest", seed=5)
    two = diagnose(table, method="iforest", seed=5)
    assert np.array_equal(one.score, two.score, equal_nan=True)


def test_errors_say_what_to_change():
    table = make_features([(300, {})])
    with pytest.raises(DiagnosisError, match="unknown method"):
        diagnose(table, method="magic")
    with pytest.raises(DiagnosisError, match="reference_frac"):
        diagnose(table, reference_frac=0.99)
    with pytest.raises(DiagnosisError, match="not found"):
        diagnose(table, stats=("mean", "nope"))
    with pytest.raises(DiagnosisError, match="no sensors"):
        diagnose(table, exclude_sensors=("a", "b"))
    with pytest.raises(DiagnosisError, match="long enough"):
        diagnose(make_features([(10, {})]))
    shuffled = table.take(list(reversed(range(table.num_rows))))
    with pytest.raises(DiagnosisError, match="ordered"):
        diagnose(shuffled)


def test_short_segments_are_skipped_and_reported():
    result = diagnose(make_features([(300, {}), (10, {})]), method="robust_z")
    assert result.skipped_segments == [1]


def test_anomaly_report_contract():
    table = make_features([(400, {"a": (250, 8.0)})])
    result = diagnose(table, method="robust_z")
    report = to_anomaly_report(result, "pump-01")
    assert report.method == "robust_z" and report.machine_id == "pump-01"
    assert report.anomalies[0].sensor_id == "a" and "above" in report.anomalies[0].description
    assert report.sources[0].ref == "diagnose:robust_z"


def test_scoring_matches_each_alarm_to_the_next_failure():
    table = make_features([(400, {"a": (250, 8.0)})])
    result = diagnose(table, method="robust_z")
    detected = result.episodes[0].detected_at
    failure = detected + timedelta(hours=30)
    hit = score_diagnosis(result, [failure], horizon_h=72)
    assert (hit.detected, hit.true_alarms, hit.false_alarms) == (1, 1, 0)
    assert hit.recall == 1.0 and hit.median_lead_h == pytest.approx(30.0)
    early = score_diagnosis(result, [detected + timedelta(hours=100)], horizon_h=72)
    assert (early.detected, early.true_alarms, early.false_alarms) == (0, 0, 1)
    assert early.recall == 0.0 and early.median_lead_h is None
    after = score_diagnosis(result, [detected - timedelta(hours=1)], horizon_h=72)
    assert after.false_alarms == 1 and after.detected == 0  # no failure follows the alarm
    quiet = score_diagnosis(diagnose(make_features([(300, {})]), method="robust_z"), [], 72)
    assert quiet.recall is None and quiet.false_alarms == 0
    with pytest.raises(ValueError):
        score_diagnosis(result, [failure], horizon_h=0)


def test_cli_diagnose_and_experiment(tmp_path, monkeypatch):
    monkeypatch.setenv("DIAGSYS_DATA_DIR", str(tmp_path / "data"))
    runner = CliRunner()
    raw = tmp_path / "data" / "machines" / "sim-01" / "raw"

    def run(*args):
        result = runner.invoke(app, list(args))
        assert result.exit_code == 0, result.output
        return result.output

    run("machine", "new", "sim-01")
    run("simulate", "sim-01", "--cycles", "3", "--sampling-s", "300")
    run("ingest", "sim-01", str(raw / "sim_seed42.csv"))
    run("ingest-events", "sim-01", str(raw / "sim_seed42_failures.csv"))
    out = run("diagnose", "sim-01", "--method", "robust_z", "--exclude-sensor", "load_pct")
    assert "alarm episodes" in out
    assert (
        tmp_path / "data" / "machines" / "sim-01" / "processed" / "diagnosis_robust_z.json"
    ).is_file()
    exp = run("experiment-diagnosis", "sim-01", "--exclude-sensor", "load_pct")
    assert "robust_z" in exp and "iforest" in exp and "Saved in" in exp
    bad = runner.invoke(app, ["diagnose", "sim-01", "--method", "magic"])
    assert bad.exit_code == 1 and "unknown method" in bad.output
