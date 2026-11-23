import numpy as np
import pytest

from diagsys.gateway import SimConfig, simulate
from diagsys.gateway.simulator import DEFAULT_FAILURES

SMALL = {"cycles": 3, "sampling_s": 300}


def test_same_seed_same_data_different_seed_different_data():
    a = simulate(SimConfig(seed=1, **SMALL))
    b = simulate(SimConfig(seed=1, **SMALL))
    c = simulate(SimConfig(seed=2, **SMALL))
    assert np.array_equal(a.timestamps, b.timestamps)
    assert all(np.array_equal(a.values[k], b.values[k]) for k in a.values)
    assert not np.array_equal(a.values["temperature_c"][:100], c.values["temperature_c"][:100])


def test_cycles_failures_and_repair_gaps():
    result = simulate(SimConfig(seed=3, **SMALL), machine_id="press-01")
    assert len(result.failures) == 3
    assert all(f.machine_id == "press-01" for f in result.failures)
    assert {f.failure_mode for f in result.failures} <= {m.id for m in DEFAULT_FAILURES}
    steps = np.diff(result.timestamps).astype("timedelta64[s]").astype(int)
    assert (steps > 0).all()
    assert (steps == 300).sum() > 0.99 * steps.size
    assert (steps > 3 * 3600).sum() == 2  # one repair gap between each pair of cycles
    # each failure comes right after the last reading of its cycle
    assert (
        result.failures[0].timestamp.replace(tzinfo=None)
        > result.timestamps[0].astype("datetime64[us]").item()
    )


def test_degradation_is_visible_on_the_affected_sensor():
    result = simulate(SimConfig(seed=5, cycles=1, sampling_s=300))
    mode = next(m for m in DEFAULT_FAILURES if m.id == result.failures[0].failure_mode)
    sensor, (shift, _) = max(mode.effects.items(), key=lambda kv: abs(kv[1][0]))
    values = result.values[sensor]
    tenth = values.size // 10
    change = values[-tenth:].mean() - values[:tenth].mean()
    assert change * shift > 0  # moves in the direction of the failure signature
    assert abs(change) > 0.3 * abs(shift)


def test_readings_follow_the_load():
    result = simulate(SimConfig(seed=7, cycles=1, sampling_s=300))
    corr = np.corrcoef(result.values["load_pct"], result.values["current_a"])[0, 1]
    assert corr > 0.5


def test_clean_data_has_no_gaps_and_dirty_data_does():
    clean = simulate(SimConfig(seed=1, **SMALL))
    assert not any(np.isnan(v).any() for v in clean.values.values())
    assert clean.injected == {"missing": 0, "spikes": 0, "flatlines": 0}
    dirty = simulate(SimConfig(seed=1, dirty=True, **SMALL))
    nans = sum(int(np.isnan(v).sum()) for v in dirty.values.values())
    assert nans == dirty.injected["missing"] > 0
    assert dirty.injected["flatlines"] == len(dirty.values)


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        simulate(SimConfig(cycles=0))
    with pytest.raises(ValueError):
        simulate(SimConfig(failure_modes=()))
