"""What the machine's own documentation says to do about the symptoms found (K7).

The question is built from facts the tools produced (the sensors behind the anomaly, the known
failure modes), the passages are retrieved from the machine's index, and the model answers only
from them (`answer_question` checks the citations). If the documentation does not answer, there
is no guidance and the reason is returned; nothing is filled in from the model's own knowledge.
"""

from dataclasses import dataclass, field

from ..config import Settings
from ..contracts.common import Source
from ..ports import LLMClient
from .answer import answer_question
from .embedding import Embedder
from .service import search_machine


@dataclass
class Guidance:
    text: str = ""
    sources: list[Source] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def build_question(sensors: list[str], failure_modes: list[str]) -> str:
    parts = ["Jakie czynności serwisowe, kontrole i środki ostrożności zalecają dokumenty"]
    if sensors:
        parts.append("przy nietypowych odczytach czujników: " + ", ".join(sensors))
    if failure_modes:
        parts.append("oraz przy awariach: " + ", ".join(failure_modes))
    return " ".join(parts) + "?"


def documented_guidance(
    settings: Settings,
    machine_id: str,
    llm: LLMClient,
    embedder: Embedder,
    *,
    sensors: list[str],
    failure_modes: list[str],
    k: int = 5,
) -> Guidance:
    question = build_question(sensors, failure_modes)
    passages = search_machine(settings, machine_id, embedder, question, k)
    answer = answer_question(llm, question, passages)
    if not answer.grounded:
        return Guidance(
            warnings=["the documentation gives no guidance: " + "; ".join(answer.warnings)]
        )
    refs = "; ".join(
        f"[{n}] {s.ref}" for n, s in zip(answer.citations, answer.sources, strict=True)
    )
    return Guidance(f"{answer.text} (źródła: {refs})", answer.sources, answer.warnings)
