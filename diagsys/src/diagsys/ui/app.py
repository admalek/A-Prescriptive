"""Supervisor window (Streamlit): machines, adding files by path, results per machine.

Run with `diagsys ui`. The page only displays what `diagsys.ui.service` returns; it computes
nothing itself. The server listens on this computer only.
"""

import altair as alt
import pandas as pd
import streamlit as st

from diagsys.config import load_settings
from diagsys.decisions import DecisionError
from diagsys.llm import LLMError
from diagsys.machines import MachineError
from diagsys.rag import RagError
from diagsys.tools import CostModel, PrescriptionError, profile_downtime_cost
from diagsys.ui import service

LABELS = {"robust_z": "Reguła progowa (robust z-score)", "iforest": "Isolation Forest"}
KINDS = {
    "workbook": "skoroszyt Excel",
    "readings": "odczyty czujników",
    "failures": "historia awarii",
    "document": "dokumentacja",
    "unknown": "?",
}
ROLE_LABELS = {
    "auto": "wykryj automatycznie",
    "readings": "odczyty czujników",
    "failures": "historia awarii",
}


def main() -> None:
    st.set_page_config(page_title="diagsys: okno nadzorcy", layout="wide")
    settings = load_settings()
    st.title("diagsys: okno nadzorcy")
    st.caption(
        "System tylko doradza. Liczby pochodzą z narzędzi obliczeniowych, nie z modelu językowego."
    )

    machine_id = _sidebar(settings)
    if machine_id is None:
        st.info("Utwórz maszynę w panelu bocznym, aby zacząć.")
        return
    files_tab, results_tab, report_tab = st.tabs(["Pliki", "Wyniki", "Zalecenia i raport"])
    with files_tab:
        _files_tab(settings, machine_id)
    with results_tab:
        _results_tab(settings, machine_id)
    with report_tab:
        _report_tab(settings, machine_id)


def _sidebar(settings) -> str | None:
    with st.sidebar:
        st.header("Maszyna")
        try:
            profiles = service.machines(settings)
        except MachineError as exc:
            st.error(str(exc))
            profiles = []
        ids = [p.machine_id for p in profiles]
        names = {p.machine_id: f"{p.machine_id}: {p.name}" for p in profiles}
        chosen = None
        if ids:
            created = st.session_state.pop("select_machine", None)
            if created in ids:  # a machine was just created: show it
                st.session_state["machine"] = created
            chosen = st.selectbox("Wybierz maszynę", ids, format_func=names.get, key="machine")
        with st.expander("Nowa maszyna", expanded=not ids):
            with st.form("new_machine", clear_on_submit=True):
                new_id = st.text_input("Identyfikator", placeholder="np. pompa-01")
                new_name = st.text_input("Nazwa")
                new_type = st.text_input("Typ maszyny", placeholder="np. pompa")
                if st.form_submit_button("Utwórz"):
                    try:
                        created = service.new_machine(settings, new_id, new_name, new_type)
                        st.session_state["select_machine"] = created
                        st.rerun()
                    except MachineError as exc:
                        st.error(str(exc))
        return chosen


def _files_tab(settings, machine_id: str) -> None:
    st.subheader("Dodaj pliki po ścieżce")
    st.write(
        "Wpisz pełne ścieżki do plików na tym komputerze, po jednej w wierszu: skoroszyt "
        "`.xlsx` (dane, opis maszyny, historia, kontekst), `.csv`/`.tsv`/`.parquet` (odczyty lub "
        "historia awarii) albo dokumentacja `.txt`, `.md`, `.pdf`, `.docx`."
    )
    with st.form("add_files"):
        text = st.text_area("Ścieżki plików", height=140, placeholder=r"C:\dane\pompa.xlsx")
        left, right = st.columns(2)
        tz = left.text_input("Strefa czasowa znaczników bez przesunięcia", value="Europe/Warsaw")
        role = right.selectbox(
            "Jak traktować pliki CSV/Parquet",
            list(ROLE_LABELS),
            format_func=ROLE_LABELS.get,
        )
        submitted = st.form_submit_button("Dodaj do maszyny", type="primary")
    if submitted:
        paths = service.parse_paths(text)
        if not paths:
            st.warning("Nie podano żadnej ścieżki.")
        else:
            try:
                with st.spinner("Importuję pliki…"):
                    outcomes = service.add_files(settings, machine_id, paths, tz=tz, role=role)
            except (MachineError, ValueError) as exc:
                st.error(str(exc))
            else:
                st.session_state.pop(f"analysis:{machine_id}", None)
                _show_outcomes(outcomes)
    st.divider()
    _show_overview(settings, machine_id)


def _show_outcomes(outcomes) -> None:
    rows = [
        {
            "plik": o.path,
            "rodzaj": KINDS.get(o.kind, o.kind),
            "wynik": "OK" if o.ok else "BŁĄD",
            "szczegóły": o.message,
            "uwagi": "; ".join(o.warnings),
        }
        for o in outcomes
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _show_overview(settings, machine_id: str) -> None:
    view = service.overview(settings, machine_id)
    st.subheader(f"Co jest zapisane dla {machine_id}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Czujniki z danymi", len(view.sensors))
    c2.metric("Zapisane awarie", len(view.failures))
    c3.metric("Pliki dokumentacji", len(view.documents))
    if view.sensors:
        st.dataframe(pd.DataFrame(view.sensors), hide_index=True, use_container_width=True)
    if view.failures:
        st.write("Historia awarii")
        st.dataframe(pd.DataFrame(view.failures), hide_index=True, use_container_width=True)
    if view.documents:
        st.write("Dokumentacja: " + ", ".join(view.documents))


def _results_tab(settings, machine_id: str) -> None:
    view = service.overview(settings, machine_id)
    st.subheader(f"Wyniki dla {view.profile.name} ({machine_id})")
    if not view.sensors:
        st.info("Ta maszyna nie ma jeszcze odczytów. Dodaj plik w zakładce „Pliki”.")
    else:
        sensor_ids = [s["sensor"] for s in view.sensors]
        with st.form("analysis"):
            left, right = st.columns(2)
            excluded = left.multiselect(
                "Pomiń czujniki w analizie (np. obciążenie)",
                sensor_ids,
                default=[s for s in sensor_ids if s == "load_pct"],
            )
            horizon = right.number_input("Horyzont prawdopodobieństwa awarii [h]", 1.0, 720.0, 72.0)
            run = st.form_submit_button("Uruchom analizę", type="primary")
        key = f"analysis:{machine_id}"
        if run:
            with st.spinner("Liczę diagnozę i prognozę…"):
                st.session_state[key] = service.analyze(
                    settings, machine_id, exclude_sensors=tuple(excluded), horizon_h=horizon
                )
        analysis = st.session_state.get(key)
        if analysis is not None:
            _show_forecast(analysis)
            _show_diagnosis(analysis)
    st.divider()
    _docs_section(settings, machine_id)


def _show_forecast(analysis) -> None:
    st.markdown("### Prognoza czasu do awarii")
    if analysis.errors.get("data"):
        st.error(analysis.errors["data"])
        return
    if analysis.forecast is None:
        st.info(analysis.errors.get("forecast", "Brak prognozy."))
        return
    f = analysis.forecast.forecast
    c1, c2, c3 = st.columns(3)
    c1.metric("Mediana pozostałego czasu", f"{f.rul_hours:.0f} h")
    c2.metric("Przedział 80%", f"{f.rul_low_hours:.0f} – {f.rul_high_hours:.0f} h")
    c3.metric(
        f"Prawdopodobieństwo awarii w {f.horizon_hours:g} h",
        f"{f.failure_probability:.0%}",
        help="Wartość przybliżona, wyznaczona z kwantyli prognozy.",
    )
    st.caption(
        f"Stan na {analysis.forecast.as_of:%Y-%m-%d %H:%M} UTC, {analysis.forecast.age_h:.0f} h od "
        f"ostatniej naprawy. Model: {f.model_version}."
    )
    for warning in analysis.forecast.warnings + analysis.notes:
        st.warning(warning)
    st.caption("Źródło: " + ", ".join(s.ref for s in f.sources))


def _show_diagnosis(analysis) -> None:
    st.markdown("### Diagnoza: okresy nietypowej pracy")
    if not analysis.diagnosis:
        st.info("; ".join(analysis.errors.values()) or "Brak diagnozy.")
        return
    tabs = st.tabs([LABELS.get(m, m) for m in analysis.diagnosis])
    for tab, view in zip(tabs, analysis.diagnosis.values(), strict=True):
        with tab:
            st.caption(
                f"Próg alarmu {view.threshold:.2f}. Monitorowane: {view.monitored_hours:.0f} h. "
                f"Okresów nietypowych: {len(view.episodes)}."
            )
            st.altair_chart(_score_chart(view, analysis.failures), use_container_width=True)
            if view.episodes:
                rows = [
                    {
                        "początek": e["start"],
                        "wykryto": e["detected_at"],
                        "koniec": e["end"],
                        "czas [h]": e["hours"],
                        "szczyt wyniku": e["peak_score"],
                        "główne czujniki": e["drivers"],
                    }
                    for e in view.episodes
                ]
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            else:
                st.success("Nie wykryto okresów nietypowej pracy.")
    for name, reason in analysis.errors.items():
        if name.startswith("diagnosis:"):
            st.warning(f"{name.split(':', 1)[1]}: {reason}")


def _score_chart(view, failures) -> alt.Chart:
    data = pd.DataFrame(view.series)
    data = data[data["score"].notna() & ~data["reference"]]
    line = (
        alt.Chart(data)
        .mark_line(strokeWidth=1)
        .encode(
            x=alt.X("window_end:T", title="czas"),
            y=alt.Y("score:Q", title="wynik nietypowości"),
            detail="segment:N",  # no line across a repair gap
        )
    )
    threshold = (
        alt.Chart(pd.DataFrame({"y": [view.threshold]}))
        .mark_rule(strokeDash=[5, 4], color="#c0392b")
        .encode(y="y:Q")
    )
    layers = [line, threshold]
    if failures:
        marks = (
            alt.Chart(pd.DataFrame({"t": failures}))
            .mark_rule(color="#2c3e50", strokeWidth=2)
            .encode(x="t:T")
        )
        layers.append(marks)
    return alt.layer(*layers).properties(height=260)


def _docs_section(settings, machine_id: str) -> None:
    st.markdown("### Dokumentacja maszyny")
    question = st.text_input("Szukaj w dokumentacji lub zadaj pytanie", key=f"q:{machine_id}")
    if not question.strip():
        return
    ask = st.button("Zapytaj model (z cytowaniem źródeł)")
    try:
        if ask:
            answer = service.ask_docs(settings, machine_id, question)
            st.write(answer.text)
            for n, source in zip(answer.citations, answer.sources, strict=True):
                st.caption(f"[{n}] {source.ref}")
            for warning in answer.warnings:
                st.warning(warning)
        else:
            passages = service.search_docs(settings, machine_id, question)
            if not passages:
                st.info("Brak pasujących fragmentów. Dodaj dokumentację w zakładce „Pliki”.")
            for p in passages:
                where = p.doc_id + (f", s. {p.page}" if p.page else "")
                with st.expander(f"{where}  (zgodność {p.score:.2f})"):
                    st.write(p.text)
    except (RagError, LLMError, MachineError) as exc:
        st.error(str(exc))


VERDICTS = {"approved": "Zatwierdzam", "rejected": "Odrzucam", "modified": "Zmieniam"}


def _report_tab(settings, machine_id: str) -> None:
    st.subheader("Zalecenia i raport")
    view = service.overview(settings, machine_id)
    if not view.sensors:
        st.info("Ta maszyna nie ma jeszcze odczytów. Dodaj plik w zakładce „Pliki”.")
        return
    sensor_ids = [s["sensor"] for s in view.sensors]
    profile_cost = profile_downtime_cost(view.profile)
    with st.form("report"):
        st.write("Koszty (zostaw 0, jeśli nieznane: wtedy termin wynika tylko z ryzyka awarii).")
        a, b, c = st.columns(3)
        downtime = a.number_input("Koszt godziny postoju", 0.0, value=float(profile_cost or 0.0))
        service_cost = b.number_input("Koszt planowego serwisu (części, robocizna)", 0.0, value=0.0)
        extra = c.number_input("Dodatkowy koszt awarii (zniszczenia, części)", 0.0, value=0.0)
        d, e, f = st.columns(3)
        planned_h = d.number_input("Postój planowego serwisu [h]", 0.0, value=4.0)
        unplanned_h = e.number_input("Postój po awarii [h]", 0.0, value=24.0)
        lead = f.number_input("Najwcześniejszy start serwisu [h]", 0.0, value=24.0)
        g, h = st.columns(2)
        risk = g.slider("Akceptowane ryzyko awarii przed serwisem", 1, 50, 10, format="%d%%")
        excluded = h.multiselect(
            "Pomiń czujniki",
            sensor_ids,
            default=[x for x in sensor_ids if x == "load_pct"],
            key="report_excluded",
        )
        use_llm = st.checkbox(
            "Użyj modelu językowego (dokumentacja i podsumowanie; wymaga serwera modelu)"
        )
        go = st.form_submit_button("Przygotuj zalecenia i raport", type="primary")
    key = f"run:{machine_id}"
    if go:
        costs = CostModel(
            downtime_cost_per_hour=downtime or None,
            planned_downtime_h=planned_h,
            unplanned_downtime_h=unplanned_h,
            planned_service_cost=service_cost,
            unplanned_extra_cost=extra,
            lead_time_h=lead,
        )
        try:
            with st.spinner("Liczę analizę, zalecenia i raport…"):
                st.session_state[key] = service.make_report(
                    settings,
                    machine_id,
                    exclude_sensors=tuple(excluded),
                    risk_tolerance=risk / 100,
                    costs=costs,
                    use_llm=use_llm,
                )
        except (PrescriptionError, ValueError) as exc:
            st.error(str(exc))
    run = st.session_state.get(key)
    if run is None:
        return
    _show_run(settings, machine_id, run)


def _show_run(settings, machine_id: str, run) -> None:
    st.markdown("### Zalecenia")
    if run.prescription is None:
        st.info(run.step("prescription").detail or "Brak zaleceń.")
    else:
        rec = run.prescription.recommendation
        for i, action in enumerate(rec.actions, start=1):
            due = f" (do {action.due_by:%Y-%m-%d %H:%M} UTC)" if action.due_by else ""
            cost = f", oczekiwany koszt {action.expected_cost:.0f}" if action.expected_cost else ""
            st.markdown(f"**{i}.** [{action.risk}] {action.description}{due}{cost}")
        for warning in run.prescription.warnings:
            st.warning(warning)
        st.caption("Źródła: " + ", ".join(dict.fromkeys(s.ref for s in rec.sources)))
        _decision_form(settings, machine_id, run)
    st.markdown("### Decyzje nadzorcy")
    decisions = service.decision_log(settings, machine_id)
    if decisions:
        rows = [
            {
                "kiedy": d.decided_at,
                "kto": d.decided_by,
                "decyzja": VERDICTS[d.verdict],
                "działanie": d.modified_description or d.action.description,
                "uzasadnienie": d.comment,
            }
            for d in decisions
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.caption("Brak decyzji.")
    st.markdown("### Raport")
    st.download_button(
        "Pobierz raport (Markdown)",
        run.report.encode("utf-8"),
        file_name=f"raport_{machine_id}_{run.run_id}.md",
        mime="text/markdown",
    )
    if run.report_dir is not None:
        st.caption(f"Zapisano w {run.report_dir}")
    with st.expander("Podgląd raportu", expanded=False):
        st.markdown(run.report)
    with st.expander("Przebieg kroków"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "krok": s.step,
                        "status": s.status,
                        "czas [s]": round(s.seconds, 1),
                        "szczegóły": s.detail,
                    }
                    for s in run.steps
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )


def _decision_form(settings, machine_id: str, run) -> None:
    actions = run.prescription.recommendation.actions
    st.markdown("#### Twoja decyzja")
    with st.form("decision", clear_on_submit=True):
        index = st.selectbox(
            "Działanie",
            range(len(actions)),
            format_func=lambda i: f"{i + 1}. {actions[i].description[:90]}",
        )
        verdict = st.radio("Decyzja", list(VERDICTS), format_func=VERDICTS.get, horizontal=True)
        who = st.text_input("Kto decyduje (imię lub inicjały)")
        comment = st.text_area("Uzasadnienie (wymagane przy odrzuceniu i zmianie)", height=70)
        changed = st.text_input("Co zamiast tego (tylko przy zmianie)")
        if st.form_submit_button("Zapisz decyzję"):
            try:
                service.decide(
                    settings,
                    run,
                    index,
                    verdict,
                    decided_by=who,
                    comment=comment,
                    modified_description=changed or None,
                )
                st.success("Decyzja zapisana, raport odświeżony.")
            except (DecisionError, ValueError) as exc:
                st.error(str(exc))


main()
