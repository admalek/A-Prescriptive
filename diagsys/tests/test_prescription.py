from datetime import UTC, datetime

import pytest

from diagsys.contracts import Forecast
from diagsys.contracts.common import Source
from diagsys.tools import CostModel, PrescriptionError, prescribe
from diagsys.tools.prediction import ForecastResult

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def forecast(q=(40.0, 60.0, 100.0, 150.0, 220.0), warnings=None):
    f = Forecast(
        machine_id="m-01",
        rul_hours=q[2],
        rul_low_hours=q[0],
        rul_high_hours=q[-1],
        failure_probability=0.1,
        horizon_hours=72,
        model_version="test",
        sources=[Source(kind="tool", ref="predict:test")],
    )
    return ForecastResult(f, T0, 100.0, warnings or [], list(q))


COSTS = CostModel(
    downtime_cost_per_hour=1000.0,
    planned_downtime_h=4,
    unplanned_downtime_h=24,
    planned_service_cost=2000.0,
    unplanned_extra_cost=10000.0,
    lead_time_h=24,
)


def test_risk_rule_recommends_the_quantile_of_the_tolerance():
    p = prescribe(forecast(), CostModel(lead_time_h=12), risk_tolerance=0.10)
    assert p.basis == "risk" and p.cost_start_h is None
    assert p.risk_start_h == pytest.approx(40.0, abs=0.5)  # q10 of the forecast
    assert p.recommended.risk == pytest.approx(0.10, abs=0.02)
    assert any("no costs" in w for w in p.warnings)
    assert p.recommended.expected_cost is None


def test_costs_never_push_the_service_later_than_the_risk_tolerance():
    p = prescribe(forecast(), COSTS, risk_tolerance=0.10)
    assert p.recommended.start_h <= max(COSTS.lead_time_h, p.risk_start_h) + 1.0
    assert p.cost_start_h is not None
    assert p.run_to_failure_cost == COSTS.failure_cost
    assert p.recommended.expected_cost < COSTS.failure_cost  # servicing beats waiting for it


def test_cost_rule_can_ask_for_an_earlier_service_than_the_risk_rule():
    # failure is very expensive compared with a service: a low rate favours servicing early
    costs = CostModel(downtime_cost_per_hour=100.0, planned_service_cost=100.0,
                      unplanned_extra_cost=1_000_000.0, lead_time_h=1.0)  # fmt: skip
    p = prescribe(forecast(), costs, risk_tolerance=0.5)
    assert p.basis == "risk+cost"
    assert p.recommended.start_h < p.risk_start_h


def test_lead_time_longer_than_the_risk_time_warns_and_recommends_the_earliest_start():
    p = prescribe(forecast(), CostModel(lead_time_h=100.0), risk_tolerance=0.10)
    assert p.recommended.start_h == pytest.approx(100.0, abs=1.0)
    assert any("act at once" in w for w in p.warnings)
    assert p.recommendation.actions[0].risk in {"medium", "high"}


def test_risk_grows_with_the_start_time_and_options_are_not_before_the_lead_time():
    p = prescribe(forecast(), COSTS)
    risks = [o.risk for o in p.options]
    assert risks == sorted(risks)
    assert min(o.start_h for o in p.options) >= COSTS.lead_time_h - 1e-9


def test_recommendation_has_sources_actions_and_passes_forecast_warnings():
    guidance_src = [Source(kind="document", ref="manual.md", excerpt="x")]
    p = prescribe(
        forecast(warnings=["only 2 failure cycles"]),
        COSTS,
        guidance="Sprawdzić łożyska [1].",
        guidance_sources=guidance_src,
    )
    rec = p.recommendation
    assert rec.actions[0].due_by is not None and rec.actions[-1].risk == "high"
    assert {s.kind for s in rec.sources} == {"tool", "document"}
    assert "Sprawdzić łożyska" in rec.rationale
    assert "only 2 failure cycles" in p.warnings


def test_bad_inputs_are_refused_and_odd_costs_are_flagged():
    with pytest.raises(PrescriptionError, match="negative"):
        prescribe(forecast(), CostModel(downtime_cost_per_hour=-1.0))
    with pytest.raises(PrescriptionError, match="risk_tolerance"):
        prescribe(forecast(), COSTS, risk_tolerance=0.9)
    cheap_failure = CostModel(downtime_cost_per_hour=10.0, planned_downtime_h=50,
                              unplanned_downtime_h=1)  # fmt: skip
    assert any("not more expensive" in w for w in prescribe(forecast(), cheap_failure).warnings)
    no_quantiles = forecast()
    no_quantiles.quantiles_h = []
    with pytest.raises(PrescriptionError, match="no quantiles"):
        prescribe(no_quantiles, COSTS)


# --- policy replay -------------------------------------------------------------------------------

import numpy as np  # noqa: E402
from test_prediction import lifecycle  # noqa: E402

from diagsys.eval import replay_cycles, score_policies  # noqa: E402
from diagsys.tools import build_dataset  # noqa: E402


def perfect_forecast(ds):
    q = np.full((ds.X.shape[0], 5), np.nan)
    rows = ds.usable & ~np.isnan(ds.rul_h)
    # the truth with a +-20% spread: a step-shaped CDF would trigger only once failure is certain,
    # i.e. after it is too late to start a service that takes `lead_h` to begin
    q[rows] = ds.rul_h[rows][:, None] * (1 + 0.2 * np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    return q


def test_replay_with_a_perfect_forecast_services_just_in_time():
    table, failures = lifecycle(4)
    ds = build_dataset(table, failures)
    out = replay_cycles(ds, {"perfect": perfect_forecast(ds)}, lead_h=24.0, risks=(0.1,))
    by = {}
    for o in out:
        by.setdefault(o.policy, []).append(o)
    assert all(o.failed for o in by["run_to_failure"])
    assert not any(o.failed for o in by["oracle"])
    assert not any(o.failed for o in by["perfect(r=0.1)"])
    for o in by["perfect(r=0.1)"]:  # served before the end of life, with some life left unused
        assert 0 < o.service_age_h < o.life_h


def test_a_forecast_that_never_warns_ends_every_cycle_in_failure():
    table, failures = lifecycle(3)
    ds = build_dataset(table, failures)
    far = np.where(np.isnan(perfect_forecast(ds)), np.nan, 1e6)
    far = far + np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    out = replay_cycles(ds, {"blind": far}, risks=(0.1,))
    assert all(o.failed for o in out if o.policy == "blind(r=0.1)")


def test_fixed_age_policy_uses_only_the_other_cycles():
    table, failures = lifecycle(3, lengths=[300, 300, 600])
    ds = build_dataset(table, failures)
    out = [o for o in replay_cycles(ds, {}, fixed_fractions=(0.8,)) if o.policy == "fixed_age(0.8)"]
    assert len(out) == 3
    long_cycle = max(out, key=lambda o: o.life_h)
    assert not long_cycle.failed  # judged by the two short cycles, not by itself


def test_scores_follow_the_cost_ratio_and_punish_early_service():
    table, failures = lifecycle(4)
    ds = build_dataset(table, failures)
    out = replay_cycles(ds, {"perfect": perfect_forecast(ds)}, risks=(0.1,))
    cheap = {s.policy: s for s in score_policies(out, 3.0)}
    dear = {s.policy: s for s in score_policies(out, 10.0)}
    assert cheap["run_to_failure"].mean_cost == 3.0 and dear["run_to_failure"].mean_cost == 10.0
    assert dear["perfect(r=0.1)"].mean_cost == 1.0
    assert cheap["perfect(r=0.1)"].cost_per_1000h < cheap["run_to_failure"].cost_per_1000h
    assert cheap["oracle"].cost_per_1000h <= cheap["perfect(r=0.1)"].cost_per_1000h
    with pytest.raises(ValueError, match="ratio"):
        score_policies(out, 1.0)


def test_cli_prescribe(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from diagsys.cli import app

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
    out = run("prescribe", "sim-01", "--exclude-sensor", "load_pct", "--downtime-cost", "1000",
              "--service-cost", "2000")  # fmt: skip
    assert "Prescription for sim-01" in out and "basis:" in out and "Saved:" in out
    saved = tmp_path / "data" / "machines" / "sim-01" / "processed" / "prescription.json"
    assert saved.is_file() and "sources" in saved.read_text()
    bad = run("prescribe", "sim-01", "--risk", "0.9", code=1)
    assert "risk_tolerance" in bad
