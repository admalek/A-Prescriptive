"""Multi-seed study of the K4 diagnosis methods on simulator data.

    uv run python scripts/diagnosis_study.py --seeds 1-10 [--horizon-h 72] [--json out.json]

Every seed is a separate machine (own random lifetimes, failure modes and dirt). Failures and
false alarms are pooled over seeds. Use development seeds while designing, and the test seeds
(101-110) only for the numbers that are reported.
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from diagsys.config import Settings
from diagsys.data import ingest_events, ingest_file, prepare
from diagsys.eval import score_diagnosis
from diagsys.gateway import SimConfig, simulate, write_simulation
from diagsys.machines import create_machine, load_machine
from diagsys.storage import DuckDBStorage
from diagsys.tools import METHODS, diagnose


def parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        seeds.extend(range(int(lo), int(hi or lo) + 1))
    return seeds


def run_seed(seed: int, cycles: int, sampling_s: int, horizons: list[float], methods, exclude):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        settings = Settings(data_dir=tmp / "data", _env_file=None)
        create_machine(settings, "sim-01", "sim")
        sim = simulate(
            SimConfig(cycles=cycles, sampling_s=sampling_s, seed=seed, dirty=True), "sim-01"
        )
        write_simulation(sim, tmp / "d.csv", tmp / "f.csv")
        settings.data_dir.mkdir(exist_ok=True)
        with DuckDBStorage(settings.db_path) as storage:
            ingest_file(storage, "sim-01", tmp / "d.csv")
            ingest_events(storage, "sim-01", tmp / "f.csv")
            features = prepare(
                storage, load_machine(settings, "sim-01"), with_features=True
            ).features
            failures = [e.timestamp for e in storage.read_events("sim-01")]
    out = {}
    for method in methods:
        result = diagnose(features, method=method, exclude_sensors=exclude, seed=seed)
        out[method] = {h: score_diagnosis(result, failures, h) for h in horizons}
    return out


def pool(scores):
    failures = sum(s.failures for s in scores)
    detected = sum(s.detected for s in scores)
    leads = [x for s in scores for x in s.lead_times_h]
    false = sum(s.false_alarms for s in scores)
    hours = sum(s.monitored_hours for s in scores)
    return {
        "failures": failures,
        "detected": detected,
        "recall": detected / failures if failures else None,
        "median_lead_h": float(np.median(leads)) if leads else None,
        "false_alarms": false,
        "monitored_hours": hours,
        "false_per_100h": false / hours * 100 if hours else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1-10")
    ap.add_argument("--cycles", type=int, default=6)
    ap.add_argument("--sampling-s", type=int, default=60)
    ap.add_argument("--horizon-h", type=float, default=72.0)
    ap.add_argument("--sensitivity", default="24,48", help="extra horizons, comma separated")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--exclude", default="load_pct")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)
    horizons = [args.horizon_h] + [float(x) for x in args.sensitivity.split(",") if x]
    methods = args.methods.split(",")
    exclude = tuple(x for x in args.exclude.split(",") if x)
    per_seed = []
    for seed in seeds:
        per_seed.append(run_seed(seed, args.cycles, args.sampling_s, horizons, methods, exclude))
        print(f"seed {seed} done", flush=True)
    summary = {m: {h: pool([r[m][h] for r in per_seed]) for h in horizons} for m in methods}
    print(f"\nseeds {args.seeds}, {args.cycles} cycles each, sampling {args.sampling_s} s")
    print(
        f"{'method':<14}{'horizon':>8}{'detected':>10}{'recall':>8}{'med.lead':>10}"
        f"{'false':>7}{'per100h':>9}"
    )
    for m in methods:
        for h in horizons:
            p = summary[m][h]
            lead = f"{p['median_lead_h']:.1f}h" if p["median_lead_h"] is not None else "-"
            print(
                f"{m:<14}{h:>7g}h{p['detected']:>6}/{p['failures']:<3}{p['recall']:>8.2f}"
                f"{lead:>10}{p['false_alarms']:>7}{p['false_per_100h']:>9.2f}"
            )
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {m: {str(h): v for h, v in d.items()} for m, d in summary.items()}, indent=2
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
