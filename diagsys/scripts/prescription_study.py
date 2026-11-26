"""Multi-seed study of K7 maintenance policies on simulator data.

    uv run python scripts/prescription_study.py --seeds 1-10 [--lead-h 24] [--json out.json]

Forecasts are the out-of-fold ones of leave-one-cycle-out cross-validation (see eval/prediction.py).
Costs are relative (one planned service = 1, a failure = ratio), reported for several ratios.
Use development seeds while choosing, the fresh test seeds only for the reported numbers.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from prediction_study import parse_seeds, run_seed

from diagsys.eval import replay_cycles, score_policies

RATIOS = (3.0, 5.0, 10.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1-10")
    ap.add_argument("--cycles", type=int, default=6)
    ap.add_argument("--sampling-s", type=int, default=60)
    ap.add_argument("--lead-h", type=float, default=24.0)
    ap.add_argument("--methods", default="age_baseline,quantile_gbm_cal")
    ap.add_argument("--exclude", default="load_pct")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    methods = args.methods.split(",")
    exclude = tuple(x for x in args.exclude.split(",") if x)

    outcomes = []
    for seed in parse_seeds(args.seeds):
        ds, preds = run_seed(seed, args.cycles, args.sampling_s, methods, exclude)
        outcomes += replay_cycles(ds, preds, lead_h=args.lead_h)
        print(f"seed {seed} done", flush=True)

    result = {}
    for ratio in RATIOS:
        scores = score_policies(outcomes, ratio)
        result[str(ratio)] = [asdict(s) for s in scores]
        print(f"\nfailure costs {ratio:g}x a planned service, lead time {args.lead_h:g} h")
        print(f"{'policy':<28}{'cycles':>7}{'failed':>8}{'cost':>7}{'unused':>8}{'per 1000h':>11}")
        for s in scores:
            unused = (
                f"{s.unused_life_share:.0%}" if s.unused_life_share == s.unused_life_share else "-"
            )
            print(
                f"{s.policy:<28}{s.cycles:>7}{s.failure_share:>8.0%}{s.mean_cost:>7.2f}"
                f"{unused:>8}{s.cost_per_1000h:>11.2f}"
            )
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
