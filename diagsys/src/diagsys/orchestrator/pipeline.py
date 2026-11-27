"""One analysis run from data to report, with a trace of every step.

Steps (the roles of the architecture: data, diagnosis, prediction, prescription, report):
  analysis      data preparation, diagnosis (both methods) and the RUL forecast: `analysis.analyze`
  guidance      what the machine's documentation says about the symptoms (needs the model + index)
  prescription  service timing from the forecast and costs (deterministic)
  summary       a short model-written summary, accepted only if its numbers come from the tools
  report        Markdown report written to the machine's reports/ folder

Each step records its status, duration and sources. A failed or skipped step never stops the
ones that do not depend on it, and the model parts (guidance, summary) are optional switches, so
the same run can be repeated without the model or without the documentation (ablations).
The pipeline is the deterministic baseline: a model that plans the steps itself can be compared
against it, because both call the same tools.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ..analysis import Analysis, analyze
from ..config import Settings
from ..contracts import Recommendation
from ..decisions import Decision, list_decisions, recommendation_hash
from ..llm import LLMError, OpenAICompatClient
from ..machines import load_machine, machine_dir
from ..ports import LLMClient
from ..rag import Embedder, RagError, make_embedder
from ..rag.guidance import Guidance, documented_guidance
from ..report import ReportData, build_report, grounded_summary
from ..storage import DuckDBStorage
from ..tools import CostModel, Prescription, PrescriptionError, prescribe, profile_downtime_cost


@dataclass
class PipelineOptions:
    exclude_sensors: tuple[str, ...] = ()
    horizon_h: float = 72.0
    risk_tolerance: float = 0.10
    costs: CostModel | None = None  # None: the machine's downtime cost, default times
    use_llm: bool = False  # model parts: guidance from the documentation and the summary
    use_rag: bool = True  # with use_llm: look up the documentation
    seed: int = 42


@dataclass
class StepRecord:
    step: str
    status: str  # ok | skipped | failed
    seconds: float
    detail: str = ""
    sources: list[str] = field(default_factory=list)


@dataclass
class PipelineRun:
    run_id: str
    machine_id: str
    steps: list[StepRecord]
    analysis: Analysis
    prescription: Prescription | None
    guidance: Guidance | None
    report: str
    report_dir: Path | None
    decisions: list[Decision] = field(default_factory=list)

    def step(self, name: str) -> StepRecord:
        return next(s for s in self.steps if s.step == name)


def default_costs(profile, options: PipelineOptions) -> CostModel:
    if options.costs is not None:
        return options.costs
    return CostModel(downtime_cost_per_hour=profile_downtime_cost(profile))


def decisions_for(settings: Settings, machine_id: str, rec: Recommendation) -> list[Decision]:
    """The decisions made on exactly this recommendation."""
    wanted = recommendation_hash(rec)
    return [d for d in list_decisions(settings, machine_id) if d.recommendation_hash == wanted]


def run_pipeline(
    settings: Settings,
    machine_id: str,
    options: PipelineOptions | None = None,
    *,
    llm: LLMClient | None = None,
    embedder: Embedder | None = None,
    now: datetime | None = None,
    write: bool = True,
) -> PipelineRun:
    options = options or PipelineOptions()
    profile = load_machine(settings, machine_id)
    now = now or datetime.now(UTC)
    steps: list[StepRecord] = []

    def timed(name, fn):
        start = time.perf_counter()
        try:
            detail, sources, status = fn()
        except (LLMError, RagError, PrescriptionError) as exc:
            detail, sources, status = str(exc), [], "failed"
        steps.append(StepRecord(name, status, time.perf_counter() - start, detail, sources))

    holder: dict = {}

    def do_analysis():
        a = analyze(
            settings,
            machine_id,
            exclude_sensors=options.exclude_sensors,
            horizon_h=options.horizon_h,
            seed=options.seed,
        )
        holder["analysis"] = a
        problems = "; ".join(f"{k}: {v}" for k, v in a.errors.items())
        done = [f"diagnose:{m}" for m in a.diagnosis]
        if a.forecast_method:
            done.append(f"predict:{a.forecast_method}")
        return problems, done, "ok" if a.diagnosis or a.forecast else "failed"

    timed("analysis", do_analysis)
    analysis: Analysis = holder["analysis"]

    use_model = options.use_llm
    guidance: Guidance | None = None

    def do_guidance():
        nonlocal guidance
        sensors = _symptom_sensors(analysis)
        modes = [m.name for m in profile.failure_modes]
        client = llm or OpenAICompatClient.from_settings(settings)
        index = embedder or make_embedder(settings)
        guidance = documented_guidance(
            settings, machine_id, client, index, sensors=sensors, failure_modes=modes
        )
        if guidance.text:
            return "", [s.ref for s in guidance.sources], "ok"
        return "; ".join(guidance.warnings), [], "ok"

    if use_model and options.use_rag:
        timed("guidance", do_guidance)
    else:
        steps.append(StepRecord("guidance", "skipped", 0.0, "model or documentation switched off"))

    prescription: Prescription | None = None

    def do_prescription():
        nonlocal prescription
        if analysis.forecast is None:
            return "no forecast: " + analysis.errors.get("forecast", ""), [], "skipped"
        prescription = prescribe(
            analysis.forecast,
            default_costs(profile, options),
            risk_tolerance=options.risk_tolerance,
            guidance=guidance.text if guidance else "",
            guidance_sources=guidance.sources if guidance else [],
        )
        return "", [s.ref for s in prescription.recommendation.sources], "ok"

    timed("prescription", do_prescription)

    with DuckDBStorage(settings.db_path) as storage:
        n_sensors = len(storage.summary(machine_id))
        n_failures = len(storage.read_events(machine_id))
    decisions = (
        decisions_for(settings, machine_id, prescription.recommendation) if prescription else []
    )
    data = ReportData(
        profile=profile,
        analysis=analysis,
        prescription=prescription,
        guidance=guidance,
        decisions=decisions,
        generated_at=now,
        n_sensors=n_sensors,
        n_failures=n_failures,
    )

    def do_summary():
        client = llm or OpenAICompatClient.from_settings(settings)
        text, notes = grounded_summary(client, data)
        data.summary, data.notes = text, notes
        return "; ".join(notes), [], "ok" if text else "failed"

    if use_model and (analysis.forecast is not None or analysis.diagnosis):
        timed("summary", do_summary)
    else:
        steps.append(StepRecord("summary", "skipped", 0.0, "model switched off or no results"))

    report = build_report(data)
    steps.append(StepRecord("report", "ok", 0.0, "", []))
    run_id = now.strftime("%Y%m%dT%H%M%SZ")
    directory = (
        _write(settings, machine_id, run_id, options, steps, report, prescription)
        if write
        else None
    )
    return PipelineRun(
        run_id, machine_id, steps, analysis, prescription, guidance, report, directory, decisions
    )


def _symptom_sensors(analysis: Analysis, limit: int = 4) -> list[str]:
    """Sensors that drive the anomalies, most frequent first."""
    counts: dict[str, int] = {}
    for view in analysis.diagnosis.values():
        for episode in view.result.episodes:
            for sensor, _stat, _z, _strength in episode.drivers[:3]:
                counts[sensor] = counts.get(sensor, 0) + 1
    return [s for s, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:limit]]


def _write(settings, machine_id, run_id, options, steps, report, prescription) -> Path:
    directory = machine_dir(settings, machine_id) / "reports" / run_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.md").write_text(report, encoding="utf-8")
    trace = {
        "run_id": run_id,
        "machine_id": machine_id,
        "options": {
            **{k: v for k, v in asdict(options).items() if k != "costs"},
            "costs": asdict(options.costs) if options.costs else None,
        },
        "steps": [asdict(s) for s in steps],
        "recommendation_hash": recommendation_hash(prescription.recommendation)
        if prescription
        else None,
    }
    (directory / "trace.json").write_text(
        json.dumps(trace, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    if prescription:
        (directory / "recommendation.json").write_text(
            prescription.recommendation.model_dump_json(indent=2), encoding="utf-8"
        )
    return directory
