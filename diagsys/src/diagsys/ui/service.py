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
from pathlib import Path

from ..analysis import (  # noqa: F401  (re-exported for the window and its tests)
    Analysis,
    DiagnosisView,
    analyze,
    best_rul_method,
    table_from_series,
)
from ..config import Settings
from ..contracts import Passage
from ..data import (
    IngestError,
    XlsxError,
    import_workbook,
    ingest_events,
    ingest_file,
)
from ..decisions import Decision, Verdict, list_decisions, record_decision
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
from ..orchestrator import PipelineOptions, PipelineRun, refresh_report, run_pipeline
from ..rag import RagError, answer_question, index_machine, make_embedder, search_machine
from ..rag.answer import Answer
from ..rag.loaders import MAX_FILE_BYTES, SUPPORTED
from ..storage import DuckDBStorage
from ..tools import CostModel

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


# --- documentation ------------------------------------------------------------------------------


def search_docs(settings: Settings, machine_id: str, query: str, k: int = 5) -> list[Passage]:
    return search_machine(settings, machine_id, make_embedder(settings), query, k)


def ask_docs(settings: Settings, machine_id: str, question: str, k: int = 5) -> Answer:
    passages = search_docs(settings, machine_id, question, k)
    return answer_question(OpenAICompatClient.from_settings(settings), question, passages)


# --- recommendations, decisions, report ---------------------------------------------------------


def make_report(
    settings: Settings,
    machine_id: str,
    *,
    exclude_sensors: tuple[str, ...] = (),
    horizon_h: float = 72.0,
    risk_tolerance: float = 0.10,
    costs: CostModel | None = None,
    use_llm: bool = False,
) -> PipelineRun:
    """The whole chain for one machine: analysis, prescription, report (saved to reports/)."""
    options = PipelineOptions(
        exclude_sensors=exclude_sensors,
        horizon_h=horizon_h,
        risk_tolerance=risk_tolerance,
        costs=costs,
        use_llm=use_llm,
    )
    return run_pipeline(settings, machine_id, options)


def decide(
    settings: Settings,
    run: PipelineRun,
    action_index: int,
    verdict: Verdict,
    *,
    decided_by: str,
    comment: str = "",
    modified_description: str | None = None,
) -> Decision:
    """Record the supervisor's decision on one action and refresh the report with it."""
    if run.prescription is None:
        raise ValueError("there is no recommendation to decide on")
    decision = record_decision(
        settings,
        run.prescription.recommendation,
        action_index,
        verdict,
        decided_by=decided_by,
        comment=comment,
        modified_description=modified_description,
    )
    refresh_report(settings, run)
    return decision


def decision_log(settings: Settings, machine_id: str) -> list[Decision]:
    return list_decisions(settings, machine_id)
