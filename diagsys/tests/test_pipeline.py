import json
from datetime import UTC, datetime

import pytest
from stub_llm import StubLLM

from diagsys.config import Settings
from diagsys.decisions import list_decisions, record_decision
from diagsys.gateway import SimConfig, write_simulation
from diagsys.gateway import simulate as run_simulation
from diagsys.llm import OpenAICompatClient
from diagsys.machines import create_machine, machine_dir
from diagsys.orchestrator import PipelineOptions, decisions_for, run_pipeline
from diagsys.rag import HashingEmbedder, index_machine
from diagsys.report import (
    ReportData,
    build_report,
    check_numbers,
    default_summary,
    grounded_summary,
    numbers_in,
)
from diagsys.ui import service

NOW = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pipeline")
    settings = Settings(data_dir=tmp / "data", _env_file=None)
    create_machine(settings, "sim-01", "Pompa testowa")
    sim = run_simulation(SimConfig(cycles=4, sampling_s=300, seed=7), "sim-01")
    write_simulation(sim, tmp / "r.csv", tmp / "f.csv")
    service.add_files(settings, "sim-01", [str(tmp / "r.csv"), str(tmp / "f.csv")])
    doc = tmp / "lozyska.md"
    doc.write_text(
        "# Łożyska\nPrzy wzroście drgań sprawdzić łożyska i smarowanie.", encoding="utf-8"
    )
    service.add_files(settings, "sim-01", [str(doc)], index_documents=False)
    index_machine(settings, "sim-01", HashingEmbedder())
    options = PipelineOptions(exclude_sensors=("load_pct",))
    plain = run_pipeline(settings, "sim-01", options, now=NOW)
    return settings, options, plain


def test_pipeline_without_the_model_gives_a_report_and_a_trace(world):
    settings, _, run = world
    assert [s.step for s in run.steps] == [
        "analysis",
        "guidance",
        "prescription",
        "summary",
        "report",
    ]
    assert run.step("analysis").status == "ok" and run.step("prescription").status == "ok"
    assert run.step("guidance").status == "skipped" and run.step("summary").status == "skipped"
    text = run.report
    for heading in ("## Podsumowanie", "## Diagnoza", "## Prognoza", "## Zalecenia", "## Źródła"):
        assert heading in text
    assert "zalecenia czekają na zatwierdzenie" in text
    assert "prescribe:" in text and "diagnose:robust_z" in text
    trace = json.loads((run.report_dir / "trace.json").read_text())
    assert trace["run_id"] == "20260301T120000Z" and trace["recommendation_hash"]
    assert (run.report_dir / "report.md").read_text(encoding="utf-8") == text
    assert (run.report_dir / "recommendation.json").is_file()
    assert run.report_dir.parent == machine_dir(settings, "sim-01") / "reports"


def test_decisions_attach_to_the_exact_recommendation(world):
    settings, options, run = world
    assert run.decisions == []
    rec = run.prescription.recommendation
    record_decision(settings, rec, 0, "approved", decided_by="AM", now=NOW)
    assert len(decisions_for(settings, "sim-01", rec)) == 1
    again = run_pipeline(settings, "sim-01", options, now=NOW, write=False)
    assert again.report_dir is None
    assert len(again.decisions) == 1 and "zatwierdzone" in again.report
    assert "Brak decyzji" not in again.report
    other = rec.model_copy(update={"rationale": "inne"})
    assert (
        decisions_for(settings, "sim-01", other) == []
    )  # a changed recommendation is not approved
    assert len(list_decisions(settings, "sim-01")) == 1


def test_model_parts_use_the_documentation_and_a_summary_with_foreign_numbers_is_rejected(world):
    settings, options, _ = world
    with StubLLM() as stub:
        stub.chat_reply = "Sprawdzić łożyska i smarowanie [1]. Awaria za 4242 h."
        client = OpenAICompatClient(stub.url, "m")
        run = run_pipeline(
            settings,
            "sim-01",
            PipelineOptions(exclude_sensors=("load_pct",), use_llm=True),
            llm=client,
            embedder=HashingEmbedder(),
            now=NOW,
            write=False,
        )
    assert run.step("guidance").status == "ok" and run.guidance.sources
    assert "Z dokumentacji maszyny" in run.report and "lozyska.md" in run.report
    assert any(s.kind == "document" for s in run.prescription.recommendation.sources)
    assert run.step("summary").status == "failed"  # 4242 is in no tool result
    assert "summary was rejected" in run.report and "4242" in run.report
    assert "4242 h" not in run.report.split("## Dane")[0]  # the standard summary is shown instead


def test_a_model_that_is_down_does_not_stop_the_run(world):
    settings, options, _ = world
    run = run_pipeline(
        settings,
        "sim-01",
        PipelineOptions(exclude_sensors=("load_pct",), use_llm=True),
        llm=OpenAICompatClient("http://127.0.0.1:9/v1", "m", timeout_s=2),
        embedder=HashingEmbedder(),
        now=NOW,
        write=False,
    )
    assert run.step("guidance").status == "failed" and run.step("summary").status == "failed"
    assert run.step("prescription").status == "ok" and "## Zalecenia" in run.report


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply

    def chat(self, messages, **kwargs):
        self.prompt = messages[-1].content
        return self.reply

    def embed(self, texts):  # pragma: no cover - not used
        raise NotImplementedError


def test_summary_is_accepted_only_when_every_number_is_in_the_facts(world):
    settings, options, run = world
    data = ReportData(
        run.analysis.forecast and service.load_machine(settings, "sim-01"),
        run.analysis,
        run.prescription,
    )
    median = f"{run.analysis.forecast.forecast.rul_hours:.0f}"
    good, notes = grounded_summary(FakeLLM(f"Mediana to {median} h. Sprawa wymaga uwagi."), data)
    assert good and not notes
    bad, notes = grounded_summary(FakeLLM("Mediana to 99999 h."), data)
    assert bad is None and "99999" in notes[0]
    assert median in default_summary(data) and "Prognozowany" in build_report(data)


def test_number_helpers_normalise_decimals_and_ignore_words():
    assert numbers_in("od 12,50 do 3 h; 07.0") == {"12.5", "3", "7"}
    assert check_numbers("mediana 40 h, ryzyko 10%", "mediana: 40 h\nryzyko: 10%") == []
    assert check_numbers("mediana 41 h", "mediana: 40 h") == ["41"]
