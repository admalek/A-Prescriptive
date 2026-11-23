"""End-to-end preparation for one machine: stored readings -> quality report -> clean -> features.

Everything is recomputed from the database on every call (deterministic), so the report, the clean
file and the feature file written together always describe the same snapshot of the data.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..config import Settings
from ..contracts import QualityParams, QualityReport
from ..machines import MachineProfile, machine_dir
from ..ports import Storage
from .clean import CleanFrame, clean_frame
from .features import compute_features
from .frame import FLAG_LEGEND, FrameError, SensorFrame, build_frame
from .quality import QualityResult, check_quality, default_params


@dataclass
class Prepared:
    frame: SensorFrame
    quality: QualityResult
    clean: CleanFrame
    features: pa.Table | None


def processed_dir(settings: Settings, machine_id: str) -> Path:
    return machine_dir(settings, machine_id) / "processed"


def prepare(
    storage: Storage,
    profile: MachineProfile,
    *,
    step_s: float | None = None,
    max_gap_s: float | None = None,
    flatline_min_points: int = 20,
    spike_window: int = 21,
    spike_threshold: float = 6.0,
    max_interp_points: int = 5,
    with_features: bool = False,
    window_s: float = 3600.0,
    stride_s: float = 600.0,
    min_valid_frac: float = 0.8,
) -> Prepared:
    table = storage.read_table(profile.machine_id)
    if table.num_rows == 0:
        raise FrameError(f"no readings for {profile.machine_id}; import some with diagsys ingest")
    frame = build_frame(table, step_s=step_s, max_gap_s=max_gap_s)
    params = QualityParams(
        **{
            **default_params(frame).model_dump(exclude={"schema_version"}),
            "flatline_min_points": flatline_min_points,
            "spike_window": spike_window,
            "spike_threshold": spike_threshold,
            "max_interp_points": max_interp_points,
        }
    )
    limits = {
        s.id: (s.min_value, s.max_value)
        for s in profile.sensors
        if s.min_value is not None or s.max_value is not None
    }
    result = check_quality(frame, machine_id=profile.machine_id, params=params, limits=limits)
    cleaned = clean_frame(frame, result.flags, max_interp_points)
    features = None
    if with_features:
        features = compute_features(
            cleaned.frame, window_s=window_s, stride_s=stride_s, min_valid_frac=min_valid_frac
        )
    return Prepared(frame, result, cleaned, features)


def _metadata(kind: str, machine_id: str, report: QualityReport, extra: dict) -> dict[bytes, bytes]:
    meta = {
        "diagsys.kind": kind,
        "diagsys.machine_id": machine_id,
        "diagsys.created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "diagsys.quality_params": report.params.model_dump_json(),
        "diagsys.flag_legend": json.dumps(FLAG_LEGEND),
        **{f"diagsys.{k}": json.dumps(v) for k, v in extra.items()},
    }
    return {k.encode(): v.encode() for k, v in meta.items()}


def clean_table(cleaned: CleanFrame) -> pa.Table:
    frame = cleaned.frame
    columns: dict[str, pa.Array] = {
        "timestamp": pa.array(frame.ts, pa.int64()).cast(pa.timestamp("us", tz="UTC")),
        "segment_id": pa.array(frame.segment, pa.int32()),
    }
    for j, sensor in enumerate(frame.sensors):
        columns[sensor] = pa.array(frame.values[:, j], pa.float64(), from_pandas=True)
        columns[f"{sensor}__flags"] = pa.array(cleaned.flags[:, j].astype(np.uint8), pa.uint8())
    return pa.table(columns)


def write_outputs(
    settings: Settings, prepared: Prepared, *, write_features: bool
) -> dict[str, Path]:
    report = prepared.quality.report
    out = processed_dir(settings, report.machine_id)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"report": out / "quality_report.json", "clean": out / "clean.parquet"}
    paths["report"].write_text(report.model_dump_json(indent=2), encoding="utf-8")

    table = clean_table(prepared.clean)
    table = table.replace_schema_metadata(_metadata("clean", report.machine_id, report, {}))
    pq.write_table(table, paths["clean"], compression="zstd")

    if write_features and prepared.features is not None:
        paths["features"] = out / "features.parquet"
        feats = prepared.features.replace_schema_metadata(
            _metadata("features", report.machine_id, report, {"sensors": prepared.frame.sensors})
        )
        pq.write_table(feats, paths["features"], compression="zstd")
    return paths
