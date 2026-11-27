"""Scoring of maintenance policies on recorded failure cycles (K7).

The question: does acting on the forecast beat the simple alternatives? Every failure cycle is
replayed window by window (forecasts are the out-of-fold ones from `cross_validate`, so the policy
never sees the cycle's own failure):

  run_to_failure      no service: every cycle ends in a failure
  fixed_age(f)        service at f x the median life of the machine's other cycles
  predictive(r)       the risk rule: at the first window where the forecast puts the probability of
                      failing within the lead time at r or more, order a service, which starts
                      `lead_h` later; a failure before it starts counts as a failure
  hybrid(f, r)        whichever comes first: the fixed-age service or the predictive one
  oracle              service exactly at the end of life (unreachable, shows the headroom)

A cycle ends either in a failure (cost `ratio`) or in a planned service (cost 1), in units of one
planned service. Reported per policy: share of cycles ending in failure, mean cost per cycle, mean
share of life left unused at the service, and cost per operating hour (this is what punishes
servicing too early). The cost ratio is an assumption, so several values are reported.
"""

from dataclasses import dataclass

import numpy as np

from ..tools.prediction import Dataset, failure_probability

POLICIES_NOTE = "cost unit = one planned service; failure = `ratio` units"


@dataclass(frozen=True)
class CycleOutcome:
    policy: str
    failed: bool
    service_age_h: float  # age at which the service started (nan when the cycle ended in failure)
    life_h: float  # time from the last repair to the failure


@dataclass(frozen=True)
class PolicyScore:
    policy: str
    ratio: float
    cycles: int
    failure_share: float
    mean_cost: float
    unused_life_share: float  # mean over serviced cycles of (life - service age) / life
    cost_per_1000h: float


def replay_cycles(
    ds: Dataset,
    forecasts: dict[str, np.ndarray],
    *,
    lead_h: float = 24.0,
    risks: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.5),
    fixed_fractions: tuple[float, ...] = (0.6, 0.8),
) -> list[CycleOutcome]:
    """Outcome of every policy for every labelled cycle of one machine.

    `forecasts` maps a name to out-of-fold quantile forecasts (rows x 5, NaN when not forecast).
    """
    cycles = [int(c) for c in np.unique(ds.cycle[ds.usable & (ds.cycle >= 0)])]
    life = {c: float(np.median((ds.age_h + ds.rul_h)[ds.usable & (ds.cycle == c)])) for c in cycles}
    out: list[CycleOutcome] = []
    for c in cycles:
        rows = np.flatnonzero(ds.usable & (ds.cycle == c))
        rows = rows[np.argsort(ds.age_h[rows])]
        age = ds.age_h[rows]
        L = life[c]
        others = [life[o] for o in cycles if o != c]
        out.append(CycleOutcome("run_to_failure", True, float("nan"), L))
        out.append(CycleOutcome("oracle", False, L, L))
        fixed = {f: f * float(np.median(others)) for f in fixed_fractions if others}
        for f, t in fixed.items():
            out.append(_serviced_at(f"fixed_age({f:g})", t, L))
        for name, preds in forecasts.items():
            p = failure_probability(preds[rows], lead_h)
            for r in risks:
                hit = np.flatnonzero(p >= r)
                start = float(age[hit[0]]) + lead_h if hit.size else float("inf")
                out.append(_serviced_at(f"{name}(r={r:g})", start, L))
                for f, t in fixed.items():
                    if f >= 0.8:  # a fixed-age fallback that is not itself very early
                        out.append(_serviced_at(f"hybrid({f:g},{name},r={r:g})", min(start, t), L))
    return out


def _serviced_at(policy: str, start_age_h: float, life_h: float) -> CycleOutcome:
    if start_age_h >= life_h:  # no service was ordered, or it started after the failure
        return CycleOutcome(policy, True, float("nan"), life_h)
    return CycleOutcome(policy, False, start_age_h, life_h)


def score_policies(outcomes: list[CycleOutcome], ratio: float) -> list[PolicyScore]:
    if ratio <= 1:
        raise ValueError("a failure must cost more than a planned service (ratio > 1)")
    scores = []
    for policy in dict.fromkeys(o.policy for o in outcomes):
        rows = [o for o in outcomes if o.policy == policy]
        failed = np.array([o.failed for o in rows])
        cost = np.where(failed, ratio, 1.0)
        operated = np.array([o.life_h if o.failed else o.service_age_h for o in rows])
        unused = [(o.life_h - o.service_age_h) / o.life_h for o in rows if not o.failed]
        scores.append(
            PolicyScore(
                policy=policy,
                ratio=ratio,
                cycles=len(rows),
                failure_share=float(failed.mean()),
                mean_cost=float(cost.mean()),
                unused_life_share=float(np.mean(unused)) if unused else float("nan"),
                cost_per_1000h=float(cost.sum() / operated.sum() * 1000.0),
            )
        )
    return scores
