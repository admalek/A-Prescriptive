"""The report: every number comes from a tool, every claim lists its sources.

`build_report` assembles the Markdown from structured results only. The one piece of free text a
model may write is the short summary, and it is accepted only if every number in it also appears
in the facts the tools produced (`check_numbers`); otherwise the deterministic summary is used and
the rejection is noted in the report. The report states the limits of the analysis every time.
"""

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime

from .analysis import Analysis
from .contracts import ChatMessage
from .contracts.common import Source
from .decisions import Decision
from .machines import MachineProfile
from .ports import LLMClient
from .rag.guidance import Guidance
from .tools.prescription import Prescription

_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
METHOD_LABELS = {"robust_z": "reguła progowa (robust z-score)", "iforest": "Isolation Forest"}
LIMITS = (
    "System tylko doradza: nie steruje maszyną, a decyzję podejmuje nadzorca.",
    "Liczby pochodzą z narzędzi obliczeniowych; model językowy nie liczy ani nie ocenia ryzyka.",
    "Prognoza i diagnoza są tak dobre, jak historia awarii i dane z czujników; przy małej liczbie "
    "awarii przedział prognozy jest orientacyjny (patrz ostrzeżenia).",
    "Jakość metod sprawdzono dotąd na danych symulowanych; na danych rzeczywistych wyniki mogą "
    "być gorsze.",
)

SUMMARY_PROMPT = (
    "Jesteś asystentem diagnostyki maszyn. Napisz po polsku podsumowanie w 3-4 zdaniach, "
    "WYŁĄCZNIE na podstawie podanych faktów. Nie podawaj żadnych liczb ani dat, których nie ma "
    "w faktach, nie dodawaj zaleceń spoza faktów i nie obiecuj, że maszyna nie ulegnie awarii."
)


@dataclass
class ReportData:
    profile: MachineProfile
    analysis: Analysis
    prescription: Prescription | None = None
    guidance: Guidance | None = None
    decisions: list[Decision] = field(default_factory=list)
    summary: str | None = None  # model summary that already passed `check_numbers`
    notes: list[str] = field(default_factory=list)  # e.g. why the model summary was rejected
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    n_sensors: int = 0
    n_failures: int = 0


def numbers_in(text: str) -> set[str]:
    return {f"{float(m.replace(',', '.')):g}" for m in _NUMBER.findall(text)}


def check_numbers(text: str, facts: str) -> list[str]:
    """Numbers in `text` that are not in `facts` (empty list: the text is grounded)."""
    return sorted(numbers_in(text) - numbers_in(facts), key=float)


def facts_lines(data: ReportData) -> list[str]:
    a = data.analysis
    lines = [f"maszyna: {data.profile.name}", f"czujniki z danymi: {data.n_sensors}"]
    lines.append(f"zapisane awarie: {data.n_failures}")
    for method, view in a.diagnosis.items():
        lines.append(
            f"diagnoza, {METHOD_LABELS.get(method, method)}: {len(view.episodes)} okresów "
            f"nietypowej pracy w {view.monitored_hours:.0f} h monitorowania"
        )
    if a.forecast is not None:
        f = a.forecast.forecast
        lines += [
            f"mediana pozostałego czasu: {f.rul_hours:.0f} h",
            f"przedział 80%: od {f.rul_low_hours:.0f} do {f.rul_high_hours:.0f} h",
            f"prawdopodobieństwo awarii w {f.horizon_hours:g} h: {f.failure_probability:.0%}",
            f"czas od ostatniej naprawy: {a.forecast.age_h:.0f} h",
        ]
    p = data.prescription
    if p is not None:
        r = p.recommended
        lines.append(
            f"zalecany start serwisu za {r.start_h:.0f} h, "
            f"ryzyko awarii przed serwisem {r.risk:.0%}"
        )
        if r.expected_cost is not None:
            lines.append(f"oczekiwany koszt przy zalecanym terminie: {r.expected_cost:.0f}")
        if p.run_to_failure_cost is not None:
            lines.append(f"koszt pracy do awarii: {p.run_to_failure_cost:.0f}")
    warnings = _all_warnings(data)
    lines += [f"ostrzeżenie: {w}" for w in warnings]
    return lines


def default_summary(data: ReportData) -> str:
    a = data.analysis
    parts = []
    if a.forecast is not None:
        f = a.forecast.forecast
        parts.append(
            f"Prognozowany pozostały czas pracy {data.profile.name}: mediana {f.rul_hours:.0f} h "
            f"(przedział 80%: {f.rul_low_hours:.0f} do {f.rul_high_hours:.0f} h)."
        )
    else:
        parts.append("Prognozy czasu do awarii nie ma: " + a.errors.get("forecast", "brak danych."))
    if data.prescription is not None:
        r = data.prescription.recommended
        parts.append(
            f"Zalecany start serwisu za {r.start_h:.0f} h, przy ryzyku awarii przed serwisem "
            f"{r.risk:.0%}."
        )
    if a.diagnosis:
        counts = ", ".join(
            f"{METHOD_LABELS.get(m, m)}: {len(v.episodes)}" for m, v in a.diagnosis.items()
        )
        parts.append(f"Okresy nietypowej pracy w danych: {counts}.")
    if _all_warnings(data):
        parts.append("Wyniki mają ostrzeżenia, patrz niżej.")
    return " ".join(parts)


def grounded_summary(llm: LLMClient, data: ReportData) -> tuple[str | None, list[str]]:
    """A model-written summary, or None and the reason if its numbers are not all in the facts."""
    facts = "\n".join(facts_lines(data))
    text = llm.chat(
        [
            ChatMessage(role="system", content=SUMMARY_PROMPT),
            ChatMessage(role="user", content="Fakty:\n" + facts),
        ],
        temperature=0.1,
        max_tokens=400,
    )
    extra = check_numbers(text, facts)
    if extra:
        return None, [
            "the model summary was rejected: it contains numbers that are not in the tool "
            f"results ({', '.join(extra)}); the standard summary is used"
        ]
    return text.strip(), []


def _all_warnings(data: ReportData) -> list[str]:
    out: list[str] = []
    if data.prescription is not None:
        out += data.prescription.warnings
    elif data.analysis.forecast is not None:
        out += data.analysis.forecast.warnings
    out += data.analysis.notes
    out += data.guidance.warnings if data.guidance else []
    return list(dict.fromkeys(out))


def _sources(data: ReportData) -> list[Source]:
    seen: dict[tuple[str, str], Source] = {}
    if data.prescription is not None:
        for s in data.prescription.recommendation.sources:
            seen.setdefault((s.kind, s.ref), s)
    elif data.analysis.forecast is not None:
        for s in data.analysis.forecast.forecast.sources:
            seen.setdefault((s.kind, s.ref), s)
    for method in data.analysis.diagnosis:
        seen.setdefault(
            ("tool", f"diagnose:{method}"), Source(kind="tool", ref=f"diagnose:{method}")
        )
    return list(seen.values())


def _cell(text: object) -> str:
    return str(text).replace("|", "/").replace("\n", " ")


def build_report(data: ReportData) -> str:
    a = data.analysis
    p = data.prescription
    out = [f"# Raport: {data.profile.name} ({data.profile.machine_id})", ""]
    out.append(f"Wygenerowano: {data.generated_at:%Y-%m-%d %H:%M} UTC")
    out += [
        "",
        f"> {LIMITS[0]}",
        "",
        "## Podsumowanie",
        "",
        data.summary or default_summary(data),
        "",
    ]
    for note in data.notes:
        out += [f"*Uwaga: {note}*", ""]

    out += ["## Dane", ""]
    out.append(
        f"Czujniki z danymi: {data.n_sensors}. Zapisane awarie: {data.n_failures}."
        + (f" Błąd danych: {a.errors['data']}" if "data" in a.errors else "")
    )

    out += ["", "## Diagnoza", ""]
    if not a.diagnosis:
        out.append("Brak wyniku: " + ("; ".join(a.errors.values()) or "brak danych."))
    for method, view in a.diagnosis.items():
        out.append(
            f"**{METHOD_LABELS.get(method, method)}**: {len(view.episodes)} okresów nietypowej "
            f"pracy w {view.monitored_hours:.0f} h monitorowania "
            f"(próg alarmu {view.threshold:.2f}). "
            f"Źródło: diagnose:{method}."
        )
        if view.episodes:
            out += [
                "",
                "| początek | koniec | czas [h] | szczyt wyniku | główne czujniki |",
                "| --- | --- | --- | --- | --- |",
            ]
            for e in view.episodes[-5:]:
                out.append(
                    f"| {e['start']:%Y-%m-%d %H:%M} | {e['end']:%Y-%m-%d %H:%M} | {e['hours']} | "
                    f"{e['peak_score']} | {_cell(e['drivers'])} |"
                )
            if len(view.episodes) > 5:
                out.append(f"\nPokazano 5 ostatnich z {len(view.episodes)}.")
        out.append("")

    out += ["## Prognoza czasu do awarii", ""]
    if a.forecast is None:
        out.append("Brak prognozy: " + a.errors.get("forecast", "brak danych."))
    else:
        f = a.forecast.forecast
        out += [
            f"- mediana pozostałego czasu: **{f.rul_hours:.0f} h**",
            f"- przedział 80%: {f.rul_low_hours:.0f} do {f.rul_high_hours:.0f} h",
            f"- prawdopodobieństwo awarii w {f.horizon_hours:g} h: {f.failure_probability:.0%} "
            "(przybliżone)",
            f"- stan na {a.forecast.as_of:%Y-%m-%d %H:%M} UTC, {a.forecast.age_h:.0f} h od "
            f"ostatniej naprawy; model {f.model_version}",
        ]

    out += ["", "## Zalecenia", ""]
    if p is None:
        out.append("Brak zaleceń, bo nie ma prognozy.")
    else:
        out.append(f"Podstawa terminu: {p.basis}; akceptowane ryzyko {p.risk_tolerance:.0%}.")
        out.append("")
        for i, action in enumerate(p.recommendation.actions, start=1):
            cost = f", oczekiwany koszt {action.expected_cost:.0f}" if action.expected_cost else ""
            out.append(f"{i}. [{action.risk}] {action.description}{cost}")
        if data.guidance and data.guidance.text:
            out += ["", "Z dokumentacji maszyny: " + data.guidance.text]

    warnings = _all_warnings(data)
    if warnings:
        out += ["", "## Ostrzeżenia", ""] + [f"- {w}" for w in warnings]

    out += ["", "## Decyzje nadzorcy", ""]
    if not data.decisions:
        out.append("Brak decyzji: zalecenia czekają na zatwierdzenie przez nadzorcę.")
    else:
        out += [
            "| kiedy | kto | decyzja | działanie | uzasadnienie |",
            "| --- | --- | --- | --- | --- |",
        ]
        verdicts = {"approved": "zatwierdzone", "rejected": "odrzucone", "modified": "zmienione"}
        for d in data.decisions:
            what = d.modified_description or d.action.description
            out.append(
                f"| {d.decided_at:%Y-%m-%d %H:%M} | {_cell(d.decided_by)} | "
                f"{verdicts[d.verdict]} | {_cell(what)} | {_cell(d.comment)} |"
            )

    out += ["", "## Ograniczenia", ""] + [f"- {x}" for x in LIMITS[1:]]
    out += ["", "## Źródła", ""]
    for s in _sources(data):
        kind = {"tool": "narzędzie", "document": "dokument", "data": "dane"}[s.kind]
        out.append(f"- {kind}: {s.ref}" + (f" ({_cell(s.excerpt)[:120]})" if s.excerpt else ""))
    return "\n".join(out) + "\n"
