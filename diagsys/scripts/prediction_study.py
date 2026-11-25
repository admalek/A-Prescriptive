"""Multi-seed study of the K5 RUL prediction on simulator data (leave-one-cycle-out per machine).

    uv run python scripts/prediction_study.py --seeds 1-10 [--horizon-h 72] [--json out.json]

Every seed is a separate machine; its cycles are cross-validated among themselves and the
out-of-fold forecasts of all machines are pooled. Use development seeds while designing, and the
test seeds (101-110) only for the numbers that are reported.
"""

import argparse
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from diagsys.config import Settings
from diagsys.data import ingest_events, ingest_file, prepare
from diagsys.eval import cross_validate, score_predictions
from diagsys.gateway import SimConfig, simulate, write_simulation
from diagsys.machines import create_machine, load_machine
from diagsys.storage import DuckDBStorage
from diagsys.tools import RUL_METHODS, build_dataset


def parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        seeds.extend(range(int(lo), int(hi or lo) + 1))
    return seeds


def run_seed(seed, cycles, sampling_s, methods, exclude):
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
    ds = build_dataset(features, failures, exclude_sensors=exclude)
    return ds, {m: cross_validate(ds, m, seed=seed) for m in methods}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1-10")
    ap.add_argument("--cycles", type=int, default=6)
    ap.add_argument("--sampling-s", type=int, default=60)
    ap.add_argument("--horizon-h", type=float, default=72.0)
    ap.add_argument("--methods", default=",".join(RUL_METHODS))
    ap.add_argument("--exclude", default="load_pct")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    methods = args.methods.split(",")
    exclude = tuple(x for x in args.exclude.split(",") if x)
    seeds = parse_seeds(args.seeds)

    truth, cycle, preds = [], [], {m: [] for m in methods}
    for seed in seeds:
        ds, out = run_seed(seed, args.cycles, args.sampling_s, methods, exclude)
        truth.append(ds.rul_h)
        cycle.append(np.where(ds.cycle >= 0, ds.cycle + seed * 100, -1))
        for m in methods:
            preds[m].append(out[m])
        print(f"seed {seed} done", flush=True)
    pooled = SimpleNamespace(rul_h=np.concatenate(truth), cycle=np.concatenate(cycle))
    scores = {m: score_predictions(np.vstack(preds[m]), pooled, m, args.horizon_h) for m in methods}

    print(
        f"\nseeds {args.seeds}, {args.cycles} cycles each, sampling {args.sampling_s} s, "
        f"horizon {args.horizon_h:g} h"
    )
    print(
        f"{'method':<14}{'cycles':>7}{'MAE':>8}{'<=72h':>8}{'<=24h':>8}{'cov80':>7}"
        f"{'cov<=72':>9}{'width':>7}{'Brier skill':>12}"
    )
    for m, s in scores.items():
        print(
            f"{m:<14}{s.cycles:>7}{s.mae_h:>7.1f}h{s.mae_last72_h:>7.1f}h{s.mae_last24_h:>7.1f}h"
            f"{s.coverage80:>7.2f}{s.coverage80_last72:>9.2f}{s.mean_width_h:>6.0f}h"
            f"{s.brier_skill:>12.2f}"
        )
    if args.json:
        Path(args.json).write_text(
            json.dumps({m: asdict(s) for m, s in scores.items()}, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
