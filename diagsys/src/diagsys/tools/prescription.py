"""K7 prescription: when to service, from the RUL forecast and the costs. Deterministic.

The forecast gives a distribution of the time to failure (five quantiles, a piecewise-linear CDF
in log time). From it, for every possible start of a planned service at time t (hours from now,
not earlier than the lead time needed to organise it):

  risk(t)   probability that the machine fails before t         F(t)
  cost(t)   expected cost  C_plan * (1 - F(t)) + C_fail * F(t)
  rate(t)   cost(t) / E[min(RUL, t)]: expected cost per operating hour gained, assuming the
            service renews the machine (the classic age-replacement criterion)

Two rules are combined, and both are shown:
  * risk rule: do not wait beyond the time at which the failure probability reaches the tolerance
    (default 10%), i.e. the matching quantile of the forecast;
  * cost rule: the t that minimises rate(t), available only when costs are known.
The recommended start is the earlier of the two (never earlier than the lead time).

Nothing is invented: without a downtime cost and service costs, only the risk rule is used and
the output says so. The numbers are only as good as the forecast: its warnings are passed on.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from ..contracts import Action, Recommendation
from ..contracts.common import Source
from .prediction import ForecastResult, cdf_points

DEFAULT_RISK = 0.10


class PrescriptionError(Exception):
    """The prescription cannot be computed as requested; the message says why."""


@dataclass(frozen=True)
class CostModel:
    downtime_cost_per_hour: float | None = None  # money per hour the machine is stopped
    planned_downtime_h: float = 4.0  # stop for a planned service
    unplanned_downtime_h: float = 24.0  # stop after a failure (diagnosis, parts, repair)
    planned_service_cost: float = 0.0  # parts and labour of a planned service
    unplanned_extra_cost: float = 0.0  # additional damage and parts after a failure
    lead_time_h: float = 24.0  # earliest start of a planned service (spare parts, crew)

    def validate(self) -> None:
        values = (
            self.planned_downtime_h,
            self.unplanned_downtime_h,
            self.planned_service_cost,
            self.unplanned_extra_cost,
            self.lead_time_h,
        )
        if any(v < 0 for v in values) or (self.downtime_cost_per_hour or 0) < 0:
            raise PrescriptionError("costs and times must not be negative")

    @property
    def known(self) -> bool:
        """Costs are known when the downtime or the service costs were given."""
        return bool(self.downtime_cost_per_hour) or self.planned_service_cost > 0

    @property
    def planned_cost(self) -> float:
        return (
            self.planned_service_cost + (self.downtime_cost_per_hour or 0) * self.planned_downtime_h
        )

    @property
    def failure_cost(self) -> float:
        return (
            self.planned_service_cost
            + self.unplanned_extra_cost
            + (self.downtime_cost_per_hour or 0) * self.unplanned_downtime_h
        )


@dataclass
class ServiceOption:
    start_h: float  # hours from now
    risk: float  # P(failure before the service starts)
    expected_cost: float | None
    cost_rate: float | None  # expected cost per operating hour gained


@dataclass
class Prescription:
    machine_id: str
    as_of: datetime
    options: list[ServiceOption]
    recommended: ServiceOption
    basis: str  # "risk" or "risk+cost": which rules produced the recommended time
    risk_tolerance: float
    risk_start_h: float  # latest start keeping the risk within the tolerance (may be < lead time)
    cost_start_h: float | None  # start with the lowest cost rate, None without costs
    run_to_failure_cost: float | None
    recommendation: Recommendation
    warnings: list[str] = field(default_factory=list)


def prescribe(
    result: ForecastResult,
    costs: CostModel | None = None,
    *,
    risk_tolerance: float = DEFAULT_RISK,
    guidance: str = "",
    guidance_sources: list[Source] | None = None,
) -> Prescription:
    """Service timing from a forecast. `guidance` is optional, already source-checked text."""
    costs = costs or CostModel()
    costs.validate()
    if not 0.01 <= risk_tolerance <= 0.5:
        raise PrescriptionError("risk_tolerance must be between 0.01 and 0.5")
    if len(result.quantiles_h) != 5:
        raise PrescriptionError("the forecast carries no quantiles")
    q = np.array(result.quantiles_h, dtype=float)
    machine = result.forecast.machine_id
    warnings = list(result.warnings)
    if costs.known and costs.failure_cost <= costs.planned_cost:
        warnings.append(
            "a failure is not more expensive than a planned service with these costs: check them"
        )

    xs, ps = cdf_points(q)

    def cdf(hours):
        return np.interp(np.log1p(hours), xs[0], ps)

    horizon = max(costs.lead_time_h, float(np.expm1(xs[0, -1])))
    grid = np.unique(np.concatenate([np.linspace(0.0, horizon, 2001), [costs.lead_time_h]]))
    survival = 1.0 - cdf(grid)
    operated = np.concatenate(
        [[0.0], np.cumsum((survival[1:] + survival[:-1]) / 2 * np.diff(grid))]
    )  # E[min(RUL, t)]

    risk_start = float(np.interp(risk_tolerance, ps, np.expm1(xs[0])))
    if risk_start < costs.lead_time_h:
        warnings.append(
            f"the earliest possible service ({costs.lead_time_h:g} h) already carries a failure "
            f"probability of {float(cdf(costs.lead_time_h)):.0%}, above the tolerance "
            f"{risk_tolerance:.0%}: act at once, e.g. shorten the lead time or reduce the load"
        )

    candidates = grid[(grid >= costs.lead_time_h) & (operated > 0)]
    options = []
    for t in candidates:
        risk = float(cdf(t))
        cost = rate = None
        if costs.known:
            cost = costs.planned_cost * (1 - risk) + costs.failure_cost * risk
            rate = cost / float(np.interp(t, grid, operated))
        options.append(ServiceOption(float(t), risk, cost, rate))
    if not options:
        raise PrescriptionError("no service time can be evaluated (the forecast is zero)")

    cost_start = None
    if costs.known:
        cost_start = min(options, key=lambda o: o.cost_rate).start_h  # type: ignore[arg-type]
    chosen = max(costs.lead_time_h, risk_start)
    basis = "risk"
    if cost_start is not None and cost_start < chosen:
        chosen, basis = max(costs.lead_time_h, cost_start), "risk+cost"
    recommended = min(options, key=lambda o: abs(o.start_h - chosen))
    if not costs.known:
        warnings.append(
            "no costs given (downtime cost per hour, service costs): the timing follows from the "
            "failure risk only; add costs in the 'maszyna' / 'tryby_awarii' sheet for a cost rule"
        )

    run_to_failure = costs.failure_cost if costs.known else None
    recommendation = _recommendation(
        result, costs, options, recommended, basis, risk_tolerance, risk_start,
        guidance, guidance_sources or [], machine,
    )  # fmt: skip
    return Prescription(
        machine_id=machine,
        as_of=result.as_of,
        options=options,
        recommended=recommended,
        basis=basis,
        risk_tolerance=risk_tolerance,
        risk_start_h=risk_start,
        cost_start_h=cost_start,
        run_to_failure_cost=run_to_failure,
        recommendation=recommendation,
        warnings=warnings,
    )


def _level(risk: float) -> str:
    return "low" if risk < 0.1 else "medium" if risk < 0.3 else "high"


def _recommendation(
    result, costs, options, recommended, basis, tolerance, risk_start, guidance, guidance_sources,
    machine,
) -> Recommendation:  # fmt: skip
    as_of = result.as_of
    earliest = options[0]
    rule = (
        f"ryzyko awarii nie większe niż {tolerance:.0%}"
        if basis == "risk"
        else "najniższy oczekiwany koszt na godzinę pracy (z ograniczeniem ryzyka)"
    )
    actions = [
        Action(
            description=(
                f"Zaplanować serwis z rozpoczęciem za {recommended.start_h:.0f} h "
                f"(kryterium: {rule}). Prawdopodobieństwo awarii przed serwisem: "
                f"{recommended.risk:.0%}."
            ),
            due_by=as_of + timedelta(hours=recommended.start_h),
            expected_cost=recommended.expected_cost,
            risk=_level(recommended.risk),
        )
    ]
    if abs(earliest.start_h - recommended.start_h) > 1.0:
        actions.append(
            Action(
                description=(
                    f"Alternatywa: serwis najszybciej możliwy, za {earliest.start_h:.0f} h "
                    f"(ryzyko przed serwisem {earliest.risk:.0%}), jeśli koszt zbędnego "
                    "przestoju jest akceptowalny."
                ),
                due_by=as_of + timedelta(hours=earliest.start_h),
                expected_cost=earliest.expected_cost,
                risk=_level(earliest.risk),
            )
        )
    actions.append(
        Action(
            description="Nie planować serwisu i pracować do awarii (nie zalecane).",
            expected_cost=costs.failure_cost if costs.known else None,
            risk="high",
        )
    )
    f = result.forecast
    rationale = (
        f"Prognoza pozostałego czasu: mediana {f.rul_hours:.0f} h, przedział 80% "
        f"{f.rul_low_hours:.0f} do {f.rul_high_hours:.0f} h. Serwis nie może zacząć się wcześniej "
        f"niż za {costs.lead_time_h:g} h. Ryzyko {tolerance:.0%} przypada na {risk_start:.0f} h."
    )
    if guidance:
        rationale += "\n\nZ dokumentacji maszyny: " + guidance
    sources = [
        Source(kind="tool", ref="prescribe:risk" + ("+cost" if costs.known else "")),
        *f.sources,
        *guidance_sources,
    ]
    return Recommendation(machine_id=machine, actions=actions, rationale=rationale, sources=sources)
