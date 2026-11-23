"""K3 acceptance: on --dirty data the check finds what the simulator injected."""

import numpy as np
import pyarrow as pa
import pytest

from diagsys.data import (
    FeatureError,
    FrameError,
    build_frame,
    check_quality,
    clean_frame,
    compute_features,
)
from diagsys.data.frame import FLAG_IMPUTED, FLAG_MISSING, FLAG_RANGE, FLAG_SPIKE, FLAG_STUCK
from diagsys.gateway import SimConfig, simulate


def to_table(result) -> pa.Table:
    """Long table of the readings that exist (NaN = never stored), as the database returns it."""
    ts = result.timestamps.astype("datetime64[us]")
    names, stamps, vals = [], [], []
    for name, v in result.values.items():
        ok = ~np.isnan(v)
        names += [name] * int(ok.sum())
        stamps.append(ts[ok])
        vals.append(v[ok])
    return pa.table(
        {
            "sensor_id": pa.array(names),
            "ts": pa.array(np.concatenate(stamps), pa.timestamp("us", tz="UTC")),
            "value": pa.array(np.concatenate(vals)),
        }
    )


@pytest.fixture(scope="module")
def dirty():
    result = simulate(SimConfig(cycles=4, cycle_hours=(40, 60), seed=42, dirty=True))
    frame = build_frame(to_table(result))
    quality = check_quality(frame, machine_id="m")
    rows = np.searchsorted(frame.ts, result.timestamps.astype("datetime64[us]").astype("int64"))
    return result, frame, quality, rows


def test_missing_readings_found_exactly(dirty):
    result, frame, quality, rows = dirty
    for j, name in enumerate(frame.sensors):
        detected = (quality.flags[rows, j] & FLAG_MISSING) != 0
        assert np.array_equal(detected, result.dirt[name].missing)
    assert quality.report.missing_points == result.injected["missing"]


def test_spikes_all_found_and_few_false_alarms(dirty):
    result, frame, quality, rows = dirty
    true_hits = false_alarms = injected = 0
    for j, name in enumerate(frame.sensors):
        detected = (quality.flags[rows, j] & FLAG_SPIKE) != 0
        truth = result.dirt[name].spikes & ~result.dirt[name].missing
        true_hits += int((detected & truth).sum())
        false_alarms += int((detected & ~truth).sum())
        injected += int(truth.sum())
    assert injected > 20
    assert true_hits == injected  # recall 100% (x3 spikes are far outside the noise)
    # measured on seeds 1-4, 42: 0.6e-4 to 2.1e-4 false alarms per point (rare Gaussian tails)
    assert false_alarms <= 5e-4 * frame.rows * len(frame.sensors)
    assert false_alarms < injected / 2


def test_five_stuck_sensors_found(dirty):
    result, frame, quality, rows = dirty
    assert sorted(quality.report.stuck_sensors) == sorted(result.values)
    assert len(quality.report.stuck_sensors) == 5
    for j, name in enumerate(frame.sensors):
        d = result.dirt[name]
        flagged = (quality.flags[rows, j] & FLAG_STUCK) != 0
        truth = np.zeros(flagged.size, dtype=bool)
        truth[d.flat_start : d.flat_start + d.flat_len] = True
        assert flagged[~truth].sum() == 0  # nothing stuck outside the injected run
        assert flagged[truth].sum() >= 20  # the injected run is (mostly) covered


def test_time_gaps_are_the_repairs(dirty):
    result, frame, quality, _ = dirty
    report = quality.report
    assert report.gap_count == 3 and report.segments == 4
    assert all(abs(g.duration_s - (4 * 3600 + 60)) < 1 for g in report.gaps)
    assert all(g.missing_points == 240 for g in report.gaps)


def test_clean_data_has_no_findings():
    result = simulate(SimConfig(cycles=3, cycle_hours=(40, 60), seed=3, dirty=False))
    quality = check_quality(build_frame(to_table(result)), machine_id="m")
    report = quality.report
    assert report.missing_points == 0 and report.stuck_sensors == []
    # same rare noise tails as above (measured 0.7e-4 to 2.0e-4 per point), no real spikes
    assert report.spike_count <= 5e-4 * report.rows * len(report.sensors)
    assert report.gap_count == 2


def test_cleaning_removes_flagged_and_fills_only_short_holes(dirty):
    _, frame, quality, _ = dirty
    cleaned = clean_frame(frame, quality.flags, max_interp_points=5)
    flags, values = cleaned.flags, cleaned.frame.values
    assert not np.isnan(values[(flags & FLAG_IMPUTED) != 0]).any()
    assert np.isnan(values[(flags & FLAG_STUCK) != 0]).all()  # stuck runs are too long to fill
    spikes_left = (flags & FLAG_SPIKE != 0) & ~np.isnan(values) & (flags & FLAG_IMPUTED == 0)
    assert not spikes_left.any()
    originals = ~np.isnan(frame.values) & ((flags & (FLAG_SPIKE | FLAG_STUCK)) == 0)
    assert np.array_equal(values[originals], frame.values[originals])  # good values untouched


def test_range_limits_from_profile_flag_values():
    result = simulate(SimConfig(cycles=2, cycle_hours=(30, 40), seed=1))
    frame = build_frame(to_table(result))
    quality = check_quality(frame, machine_id="m", limits={"temperature_c": (None, 60.0)})
    j = frame.sensors.index("temperature_c")
    assert (quality.flags[:, j] & FLAG_RANGE).any()
    sensor = quality.report.sensors[j]
    assert sensor.out_of_range_count == int(((quality.flags[:, j] & FLAG_RANGE) != 0).sum())


def test_constant_and_discrete_sensors_are_not_called_stuck_or_spiky():
    n = 600
    ts = np.arange(n) * 60 * 10**6
    rng = np.random.default_rng(0)
    state = np.repeat(rng.integers(0, 3, n // 100).astype(float), 100)  # a state, long plateaus
    table = pa.table(
        {
            "sensor_id": pa.array(["const"] * n + ["state"] * n + ["temp"] * n),
            "ts": pa.array(np.tile(ts, 3), pa.timestamp("us", tz="UTC")),
            "value": pa.array(np.r_[np.full(n, 5.0), state, 50 + rng.normal(0, 0.3, n)]),
        }
    )
    report = check_quality(build_frame(table), machine_id="m").report
    by = {s.sensor_id: s for s in report.sensors}
    assert by["const"].constant and not by["const"].stuck_runs
    assert by["state"].discrete and not by["state"].stuck_runs and by["state"].spike_count == 0
    assert report.constant_sensors == ["const"]
    assert report.stuck_sensors == []


def test_slow_sensor_triggers_hint():
    ts = np.arange(300) * 10 * 10**6  # 10 s grid
    slow = ts[::6]  # one sensor every 60 s
    table = pa.table(
        {
            "sensor_id": pa.array(["fast"] * ts.size + ["slow"] * slow.size),
            "ts": pa.array(np.r_[ts, slow], pa.timestamp("us", tz="UTC")),
            "value": pa.array(np.r_[np.sin(ts / 1e8), np.cos(slow / 1e8)]),
        }
    )
    report = check_quality(build_frame(table), machine_id="m").report
    assert any("slow" in w and "--step-s" in w for w in report.warnings)


def test_empty_or_degenerate_input_is_a_readable_error():
    empty = pa.table(
        {
            "sensor_id": pa.array([], pa.string()),
            "ts": pa.array([], pa.timestamp("us", tz="UTC")),
            "value": pa.array([], pa.float64()),
        }
    )
    with pytest.raises(FrameError):
        build_frame(empty)


# --- features -----------------------------------------------------------------------------


def test_features_exist_and_have_expected_shape(dirty):
    _, frame, quality, _ = dirty
    cleaned = clean_frame(frame, quality.flags)
    feats = compute_features(cleaned.frame, window_s=3600, stride_s=600)
    assert feats.num_rows > 100
    assert {"window_start", "window_end", "segment_id", "valid_frac"} <= set(feats.column_names)
    assert (
        "vibration_mm_s__rms" in feats.column_names
        and "temperature_c__slope_per_h" in feats.column_names
    )
    assert feats.num_columns == 5 + 5 * 7
    ends = feats["window_end"].to_pylist()
    assert ends == sorted(ends)


def test_windows_never_cross_a_pause(dirty):
    _, frame, quality, _ = dirty
    feats = compute_features(clean_frame(frame, quality.flags).frame, window_s=3600, stride_s=600)
    gap_starts = [a for a, _ in frame.gaps]
    for start, end in zip(
        feats["window_start"].cast(pa.int64()).to_pylist(),
        feats["window_end"].cast(pa.int64()).to_pylist(),
        strict=True,
    ):
        assert not any(start <= g < end for g in gap_starts)
        assert end - start == 59 * 60 * 10**6


def test_feature_values_match_numpy_on_a_known_window():
    n = 120
    ts = np.arange(n) * 60 * 10**6
    x = np.linspace(10, 20, n) + np.sin(np.arange(n))
    table = pa.table(
        {
            "sensor_id": pa.array(["s"] * n),
            "ts": pa.array(ts, pa.timestamp("us", tz="UTC")),
            "value": pa.array(x),
        }
    )
    feats = compute_features(build_frame(table), window_s=3600, stride_s=3600)
    w = x[:60]
    row = {c: feats[c][0].as_py() for c in feats.column_names}
    assert row["s__mean"] == pytest.approx(w.mean())
    assert row["s__std"] == pytest.approx(w.std(ddof=1))
    assert row["s__min"] == pytest.approx(w.min()) and row["s__max"] == pytest.approx(w.max())
    assert row["s__rms"] == pytest.approx(np.sqrt((w**2).mean()))
    assert row["s__slope_per_h"] == pytest.approx(np.polyfit(np.arange(60) / 60.0, w, 1)[0])
    kurt = ((w - w.mean()) ** 4).mean() / w.var() ** 2 - 3
    assert row["s__kurtosis"] == pytest.approx(kurt)


def test_no_lookahead_features_of_a_window_ignore_later_data():
    n = 240
    ts = np.arange(n) * 60 * 10**6
    x = np.random.default_rng(1).normal(0, 1, n)

    def feats_of(values):
        table = pa.table(
            {
                "sensor_id": pa.array(["s"] * n),
                "ts": pa.array(ts, pa.timestamp("us", tz="UTC")),
                "value": pa.array(values),
            }
        )
        return compute_features(build_frame(table), window_s=3600, stride_s=600)

    a = feats_of(x)
    y = x.copy()
    y[100:] += 50.0  # change the future of the first window
    b = feats_of(y)
    assert a["s__mean"][0].as_py() == b["s__mean"][0].as_py()


def test_window_too_short_or_too_long_is_a_readable_error(dirty):
    _, frame, _, _ = dirty
    with pytest.raises(FeatureError, match="at least 5"):
        compute_features(frame, window_s=120, stride_s=60)
    with pytest.raises(FeatureError, match="long enough"):
        compute_features(frame, window_s=10**7, stride_s=600)
