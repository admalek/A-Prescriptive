"""Logic behind the supervisor window, free of any UI code so it can be tested directly.

What the supervisor can do: see and create machines, add files by path (Excel workbook, readings,
failure history, documentation), and run the analysis for one machine: data summary, diagnosis
(both methods) and the RUL forecast. Nothing is estimated here: the numbers come from the same tools
as the command line. Each part of an analysis fails on its own, with a message, so a machine with
readings but no failure history still gets its diagnosis.
"""

import csv
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa

from ..config import Settings
from ..contracts import Passage
from ..data import (
    FeatureError,
    FrameError,
    IngestError,
    XlsxError,
    import_workbook,
    ingest_events,
    ingest_file,
    prepare,
)
from ..llm import LLMError, OpenAICompatClient
from ..machines import (
    MachineError,
    MachineProfile,
    add_sensors,
    create_machine,
    list_machines,
    load_machine,
    machine_dir,
)
from ..rag import RagError, answer_question, index_machine, make_embedder, search_machine
from ..rag.answer import Answer
from ..rag.loaders import MAX_FILE_BYTES, SUPPORTED
from ..storage import DuckDBStorage
from ..tools import (
    METHODS,
    DiagnosisError,
    PredictionError,
    build_dataset,
    diagnose,
    fit_model,
    forecast_latest,
)
from ..tools.diagnosis import DiagnosisResult
from ..tools.prediction import ForecastResult, Model

MAX_DATA_BYTES = 1024 * 1024 * 1024
ROLES = ("auto", "readings", "failures")  # how a CSV/Parquet file is understood
_TABULAR = {".csv", ".tsv", ".txt", ".parquet"}
_EVENT_HEADERS = {"failure_mode", "awaria", "tryb_awarii", "failure"}


@dataclass
class FileOutcome:
    path: str
    kind: str  # workbook | readings | failures | document | unknown
    ok: bool
    message: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class Overview:
    profile: MachineProfile
    sensors: list[dict]  # sensor, readings, first, last, minimum, maximum, mean
    failures: list[dict]  # timestamp, mode, description
    documents: list[str]


@dataclass
class DiagnosisView:
    method: str
    threshold: float
    episodes: list[dict]
    monitored_hours: float
    series: dict  # window_end (list of datetime), score, alarm, reference
    result: DiagnosisResult


@dataclass
class Analysis:
    machine_id: str
    created: datetime
    diagnosis: dict[str, DiagnosisView] = field(default_factory=dict)
    forecast: ForecastResult | None = None
    forecast_method: str | None = None
    model: Model | None = None
    failures: list[datetime] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)  # part -> why it is missing
    notes: list[str] = field(default_factory=list)


# --- machines -----------------------------------------------------------------------------------


def machines(settings: Settings) -> list[MachineProfile]:
    return list_machines(settings)


def new_machine(settings: Settings, machine_id: str, name: str = "", machine_type: str = "") -> str:
    machine_id = machine_id.strip().lower()
    create_machine(
        settings, machine_id, name.strip() or machine_id, machine_type.strip() or "generic"
    )
    return machine_id


def overview(settings: Settings, machine_id: str) -> Overview:
    profile = load_machine(settings, machine_id)
    with DuckDBStorage(settings.db_path) as storage:
        rows = storage.summary(machine_id)
        events = storage.read_events(machine_id)
    docs_dir = machine_dir(settings, machine_id) / "docs"
    documents = (
        sorted(p.name for p in docs_dir.iterdir() if p.is_file()) if docs_dir.is_dir() else []
    )
    return Overview(
        profile=profile,
        sensors=[
            {
                "sensor": r.sensor_id,
                "readings": r.readings,
                "first": r.first,
                "last": r.last,
                "minimum": r.minimum,
                "maximum": r.maximum,
                "mean": r.mean,
            }
            for r in rows
        ],
        failures=[
            {"timestamp": e.timestamp, "mode": e.failure_mode, "description": e.description}
            for e in events
        ],
        documents=documents,
    )


# --- adding files by path -----------------------------------------------------------------------


def parse_paths(text: str) -> list[str]:
    """One path per line; surrounding quotes (as pasted from Explorer) and blanks are dropped."""
    out = []
    for line in text.splitlines():
        line = line.strip().strip("'\"").strip()
        if line and line not in out:
            out.append(line)
    return out


def add_files(
    settings: Settings,
    machine_id: str,
    paths: list[str],
    *,
    tz: str = "UTC",
    role: str = "auto",
    index_documents: bool = True,
) -> list[FileOutcome]:
    """Import every path; a bad file is reported and does not stop the others."""
    load_machine(settings, machine_id)  # unknown machine: fail loudly, for the whole batch
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    outcomes = []
    copied_docs = False
    for raw in paths:
        outcome = _add_one(settings, machine_id, raw, tz, role)
        copied_docs |= outcome.kind == "document" and outcome.ok
        outcomes.append(outcome)
    if copied_docs and index_documents:
        note = _reindex(settings, machine_id)
        for outcome in outcomes:
            if outcome.kind == "document" and outcome.ok:
                outcome.warnings.append(note)
    return outcomes


def _add_one(settings: Settings, machine_id: str, raw: str, tz: str, role: str) -> FileOutcome:
    path = Path(raw).expanduser()
    if not path.is_file():
        return FileOutcome(raw, "unknown", False, "file not found (is the path complete?)")
    suffix = path.suffix.lower()
    try:
        size = path.stat().st_size
        if suffix == ".xlsx":
            _limit(size, MAX_DATA_BYTES)
            return _import_xlsx(settings, machine_id, path, tz)
        if suffix in _TABULAR and not (
            suffix == ".txt" and role == "auto" and _looks_like_text(path)
        ):
            _limit(size, MAX_DATA_BYTES)
            kind = role if role != "auto" else _guess_role(path)
            return _import_table(settings, machine_id, path, tz, kind)
        if suffix in SUPPORTED:
            _limit(size, MAX_FILE_BYTES)
            return _copy_document(settings, machine_id, path)
    except (MachineError, IngestError, XlsxError, RagError, OSError, ValueError) as exc:
        return FileOutcome(raw, "unknown", False, str(exc))
    return FileOutcome(
        raw,
        "unknown",
        False,
        f"unsupported file type {suffix!r}: use .xlsx, .csv, .tsv, .parquet (data) or "
        ".txt, .md, .pdf, .docx (documentation)",
    )


def _limit(size: int, maximum: int) -> None:
    if size > maximum:
        raise ValueError(f"file is larger than {maximum // (1024 * 1024)} MB")


def _looks_like_text(path: Path) -> bool:
    """A .txt is documentation unless its first line looks like a delimited table header."""
    line = _first_line(path)
    return line.count(";") + line.count(",") + line.count("\t") < 2


def _first_line(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        return handle.readline()


def _guess_role(path: Path) -> str:
    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq

            names = pq.ParquetFile(path).schema_arrow.names
        except Exception as exc:  # noqa: BLE001 - reported as an unreadable file
            raise ValueError(f"cannot read the Parquet file: {exc}") from exc
    else:
        line = _first_line(path)
        delimiter = max(";,\t", key=line.count)
        names = next(csv.reader([line], delimiter=delimiter), [])
    return "failures" if {n.strip().lower() for n in names} & _EVENT_HEADERS else "readings"


def _import_xlsx(settings: Settings, machine_id: str, path: Path, tz: str) -> FileOutcome:
    with DuckDBStorage(settings.db_path) as storage:
        report = import_workbook(settings, storage, machine_id, path, tz=tz)
    r = report.readings
    parts = [f"{r.readings_new} new readings ({r.readings_duplicate} already stored)"]
    if report.events is not None:
        parts.append(f"{report.events.events_new} new failures")
    if report.context_file:
        parts.append("context saved")
    return FileOutcome(str(path), "workbook", True, ", ".join(parts), list(report.warnings))


def _import_table(
    settings: Settings, machine_id: str, path: Path, tz: str, kind: str
) -> FileOutcome:
    with DuckDBStorage(settings.db_path) as storage:
        if kind == "failures":
            report = ingest_events(storage, machine_id, path, tz=tz)
            message = f"{report.events_new} new failures ({report.events_duplicate} already stored)"
        else:
            report = ingest_file(storage, machine_id, path, tz=tz)
            add_sensors(settings, machine_id, report.sensors)
            message = (
                f"{report.readings_new} new readings ({report.readings_duplicate} already stored)"
            )
    warnings = []
    if kind == "readings" and report.readings_rejected:
        warnings.append(f"{report.readings_rejected} empty or invalid cells were skipped")
    if kind == "readings" and report.ignored_columns:
        warnings.append("ignored non-numeric columns: " + ", ".join(report.ignored_columns))
    return FileOutcome(str(path), kind, True, message, warnings)


def _copy_document(settings: Settings, machine_id: str, path: Path) -> FileOutcome:
    docs_dir = machine_dir(settings, machine_id) / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    target = docs_dir / path.name
    warnings = []
    if target.exists() and target.read_bytes() != path.read_bytes():
        stem, n = target.stem, 2
        while target.exists():
            target = docs_dir / f"{stem}_{n}{path.suffix}"
            n += 1
        warnings.append(f"a different file named {path.name} exists; saved as {target.name}")
    if not target.exists():
        shutil.copyfile(path, target)
    return FileOutcome(str(path), "document", True, f"copied to docs as {target.name}", warnings)


def _reindex(settings: Settings, machine_id: str) -> str:
    try:
        report = index_machine(settings, machine_id, make_embedder(settings))
    except (RagError, LLMError) as exc:
        return f"the documentation index was not updated: {exc}"
    skipped = "; ".join(f"{n}: {why}" for n, why in report.skipped)
    return f"{report.passages_total} passages indexed" + (
        f" (skipped: {skipped})" if skipped else ""
    )


# --- analysis -----------------------------------------------------------------------------------


def best_rul_method(n_cycles: int) -> str | None:
    """The most capable RUL method the history allows (see the minimums in `fit_model`)."""
    if n_cycles >= 3:
        return "quantile_gbm_cal"
    if n_cycles == 2:
        return "quantile_gbm"
    if n_cycles == 1:
        return "age_baseline"
    return None


def analyze(
    settings: Settings,
    machine_id: str,
    *,
    exclude_sensors: tuple[str, ...] = (),
    horizon_h: float = 72.0,
    window_s: float = 3600.0,
    stride_s: float = 600.0,
    seed: int = 42,
) -> Analysis:
    profile = load_machine(settings, machine_id)
    out = Analysis(machine_id, datetime.now(UTC))
    with DuckDBStorage(settings.db_path) as storage:
        try:
            features = prepare(
                storage, profile, with_features=True, window_s=window_s, stride_s=stride_s
            ).features
        except (FrameError, FeatureError) as exc:
            out.errors["data"] = str(exc)
            return out
        out.failures = [e.timestamp for e in storage.read_events(machine_id)]
    for method in METHODS:
        try:
            result = diagnose(features, method=method, exclude_sensors=exclude_sensors, seed=seed)
            out.diagnosis[method] = _view(result)
        except DiagnosisError as exc:
            out.errors[f"diagnosis:{method}"] = str(exc)
    try:
        ds = build_dataset(features, out.failures, exclude_sensors=exclude_sensors)
        cycles = len(np.unique(ds.cycle[ds.usable & (ds.cycle >= 0)]))
        method = best_rul_method(cycles)
        if method is None:
            raise PredictionError(
                f"no completed failure cycle with usable data ({len(out.failures)} failures "
                "recorded): import the failure history to get a forecast"
            )
        out.model = fit_model(method, ds, seed=seed)
        out.forecast = forecast_latest(out.model, ds, machine_id, horizon_h=horizon_h)
        out.forecast_method = method
    except PredictionError as exc:
        out.errors["forecast"] = str(exc)
    if "forecast" not in out.errors and out.forecast_method == "age_baseline":
        out.notes.append("one failure cycle only: the forecast uses age alone, not the sensors")
    return out


def _view(result: DiagnosisResult) -> DiagnosisView:
    episodes = [
        {
            "start": e.start,
            "detected_at": e.detected_at,
            "end": e.end,
            "hours": round((e.end - e.start).total_seconds() / 3600, 1),
            "peak_score": round(e.peak_score, 2),
            "drivers": ", ".join(f"{s} ({st})" for s, st, _, _ in e.drivers[:3]),
        }
        for e in result.episodes
    ]
    ends = [datetime.fromtimestamp(us / 1e6, tz=UTC) for us in result.window_end]
    series = {
        "window_end": ends,
        "score": [None if np.isnan(x) else float(x) for x in result.score],
        "alarm": [bool(x) for x in result.alarm],
        "reference": [bool(x) for x in result.reference],
        "segment": [int(x) for x in result.segment],
    }
    return DiagnosisView(
        result.method, float(result.threshold), episodes, result.monitored_hours, series, result
    )


# --- documentation ------------------------------------------------------------------------------


def search_docs(settings: Settings, machine_id: str, query: str, k: int = 5) -> list[Passage]:
    return search_machine(settings, machine_id, make_embedder(settings), query, k)


def ask_docs(settings: Settings, machine_id: str, question: str, k: int = 5) -> Answer:
    passages = search_docs(settings, machine_id, question, k)
    return answer_question(OpenAICompatClient.from_settings(settings), question, passages)


def table_from_series(view: DiagnosisView) -> pa.Table:
    """The score series as a small table (used by the page to plot)."""
    return pa.table(view.series)
