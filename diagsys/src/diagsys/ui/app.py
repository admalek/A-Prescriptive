"""Supervisor window (Streamlit): machines, adding files by path, results per machine.

Run with `diagsys ui`. The page only displays what `diagsys.ui.service` returns; it computes
nothing itself. The server listens on this computer only.
"""

import altair as alt
import pandas as pd
import streamlit as st

from diagsys.config import load_settings
from diagsys.llm import LLMError
from diagsys.machines import MachineError
from diagsys.rag import RagError
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
    files_tab, results_tab = st.tabs(["Pliki", "Wyniki"])
    with files_tab:
        _files_tab(settings, machine_id)
    with results_tab:
        _results_tab(settings, machine_id)


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


main()
