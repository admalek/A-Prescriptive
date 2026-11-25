import pytest
from stub_llm import StubLLM
from typer.testing import CliRunner

from diagsys.cli import app
from diagsys.contracts import Passage
from diagsys.machines import create_machine, machine_dir
from diagsys.rag import (
    NO_ANSWER,
    HashingEmbedder,
    RagError,
    answer_question,
    build_messages,
    index_machine,
    search_machine,
)

DOCS = {
    "lozyska.md": "# Łożyska\n\nŁożyska pompy należy wymieniać co 8000 godzin pracy. Objawem "
    "zużycia jest wzrost drgań i temperatury obudowy łożyska.\n\n## Smarowanie\nŁożyska smaruje "
    "się smarem EP2 co 500 godzin. Nadmiar smaru powoduje przegrzewanie.",
    "uszczelnienie.md": "# Uszczelnienie mechaniczne\n\nPrzeciek uszczelnienia powyżej 10 kropli "
    "na minutę wymaga wymiany uszczelnienia. Suchobieg niszczy uszczelnienie w kilka minut.\n\n"
    "# Rozruch\nPrzed rozruchem należy zalać pompę cieczą i otworzyć zawór ssawny.",
    "cisnienie.md": "# Ciśnienie i przepływ\n\nSpadek ciśnienia tłoczenia może oznaczać zużycie "
    "wirnika lub zatkany filtr ssawny. Normalne ciśnienie pracy to 4 bary.\n\n# Kawitacja\n"
    "Kawitacja objawia się hałasem podobnym do przesypywanego żwiru i drganiami. Przyczyną jest "
    "zbyt niskie ciśnienie na ssaniu.",
    "silnik.md": "# Silnik napędowy\n\nTemperatura uzwojeń silnika nie powinna przekraczać 120 "
    "stopni. Wysoki prąd przy normalnym obciążeniu wskazuje na zwarcie zwojowe lub zablokowany "
    "wirnik.\n\n# Bezpieczeństwo\nPrzed pracami serwisowymi odłącz zasilanie i zabezpiecz "
    "wyłącznik przed załączeniem.",
}
QUESTIONS = [
    ("jak często wymieniać łożyska?", "lozyska.md"),
    ("czym smarować łożysko", "lozyska.md"),
    ("co robić gdy uszczelnienie przecieka", "uszczelnienie.md"),
    ("jak uruchomić pompę po postoju, rozruch", "uszczelnienie.md"),
    ("spadek ciśnienia tłoczenia przyczyny", "cisnienie.md"),
    ("maksymalna temperatura uzwojenia silnika", "silnik.md"),
    ("bezpieczeństwo przed serwisem odłączyć zasilanie", "silnik.md"),
    ("zbyt wysoki prąd silnika", "silnik.md"),
    ("suchobieg", "uszczelnienie.md"),
    ("wibracje łożysk", "lozyska.md"),
    ("kawitacja na ssaniu", "cisnienie.md"),
    ("zwarcie w uzwojeniu", "silnik.md"),
]


@pytest.fixture
def machine(settings):
    create_machine(settings, "m1", "Machine 1")
    return machine_dir(settings, "m1") / "docs"


def write_docs(folder, docs=DOCS):
    for name, text in docs.items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def test_indexing_is_incremental(settings, machine):
    e = HashingEmbedder()
    write_docs(machine)
    first = index_machine(settings, "m1", e)
    assert sorted(first.added) == sorted(DOCS) and first.passages_total >= 4
    again = index_machine(settings, "m1", e)
    assert sorted(again.unchanged) == sorted(DOCS) and again.passages_added == 0
    (machine / "silnik.md").write_text("# Silnik\n\nNowa instrukcja silnika.", encoding="utf-8")
    (machine / "cisnienie.md").unlink()
    changed = index_machine(settings, "m1", e)
    assert changed.updated == ["silnik.md"] and changed.removed == ["cisnienie.md"]
    assert {p.doc_id for p in search_machine(settings, "m1", e, "kawitacja ssanie", 10)} == {
        "lozyska.md",
        "uszczelnienie.md",
        "silnik.md",
    }
    assert search_machine(settings, "m1", e, "nowa instrukcja silnika", 1)[0].text.endswith(
        "Nowa instrukcja silnika."
    )


def test_files_that_cannot_be_indexed_are_reported_not_fatal(settings, machine, tmp_path):
    e = HashingEmbedder()
    write_docs(
        machine, {"ok.md": "# Pompa\n\nInstrukcja pompy.", "sub/nested.txt": "Zagnieżdżony plik."}
    )
    (machine / "tabela.xlsx").write_text("x")
    (machine / "pusty.txt").write_text("  ")
    (machine / ".ukryty.md").write_text("ukryty")
    secret = tmp_path / "secret.txt"
    secret.write_text("poza folderem")
    (machine / "link.txt").symlink_to(secret)
    report = index_machine(settings, "m1", e)
    assert sorted(report.added) == ["ok.md", "sub/nested.txt"]  # relative posix paths
    reasons = dict(report.skipped)
    assert "unsupported" in reasons["tabela.xlsx"] and "no text" in reasons["pusty.txt"]
    assert "symbolic" in reasons["link.txt"] and ".ukryty.md" not in reasons
    assert search_machine(settings, "m1", e, "poza folderem", 5)[0].doc_id != "link.txt"


def test_changing_the_embedder_needs_a_rebuild(settings, machine):
    write_docs(machine)
    index_machine(settings, "m1", HashingEmbedder())
    other = HashingEmbedder(dim=256)
    with pytest.raises(RagError, match="--rebuild"):
        search_machine(settings, "m1", other, "łożysko")
    with pytest.raises(RagError, match="--rebuild"):
        index_machine(settings, "m1", other)
    index_machine(settings, "m1", other, rebuild=True)
    assert search_machine(settings, "m1", other, "łożysko")[0].doc_id == "lozyska.md"


def test_machines_do_not_see_each_others_documents(settings, machine):
    create_machine(settings, "m2", "Machine 2")
    write_docs(machine)
    write_docs(machine_dir(settings, "m2") / "docs", {"inna.md": "Prasa hydrauliczna olej."})
    e = HashingEmbedder()
    index_machine(settings, "m1", e)
    index_machine(settings, "m2", e)
    assert {p.doc_id for p in search_machine(settings, "m2", e, "łożysko pompa", 10)} == {"inna.md"}
    assert search_machine(settings, "m1", e, "prasa hydrauliczna olej", 1)[0].doc_id != "inna.md"


def test_empty_folder_and_empty_question(settings, machine):
    e = HashingEmbedder()
    report = index_machine(settings, "m1", e)
    assert report.passages_total == 0 and report.added == []
    assert search_machine(settings, "m1", e, "cokolwiek") == []
    with pytest.raises(RagError, match="empty"):
        search_machine(settings, "m1", e, "  ")


def test_retrieval_regression_on_polish_questions(settings, machine):
    """Guards against regressions with the model-free embedder; not a claim about quality in
    general (a small corpus written for this test, questions that share word stems)."""
    write_docs(machine)
    e = HashingEmbedder()
    index_machine(settings, "m1", e)
    hits = sum(
        search_machine(settings, "m1", e, q, 1)[0].doc_id == expected for q, expected in QUESTIONS
    )
    assert hits >= 11, f"only {hits}/{len(QUESTIONS)} questions found their document"


# --- answers with citations ------------------------------------------------------------------

P = [
    Passage(doc_id="lozyska.md", chunk_id="lozyska.md#0", text="Wymiana łożysk co 8000 h.", page=3),
    Passage(doc_id="silnik.md", chunk_id="silnik.md#0", text="Temperatura uzwojeń do 120 stopni."),
]


class FakeLLM:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def chat(self, messages, *, temperature=0.2, max_tokens=1024):
        self.calls.append((list(messages), temperature, max_tokens))
        return self.reply

    def embed(self, texts):
        raise AssertionError("not used")


def test_prompt_numbers_the_passages_and_forbids_outside_knowledge():
    messages = build_messages("Jak często wymieniać łożyska?", P)
    assert messages[0].role == "system" and "WYŁĄCZNIE" in messages[0].content
    user = messages[1].content
    assert "[1] (lozyska.md, s. 3)" in user and "[2] (silnik.md)" in user
    assert user.endswith("Pytanie: Jak często wymieniać łożyska?")


def test_a_cited_answer_carries_its_sources():
    llm = FakeLLM("Łożyska wymienia się co 8000 godzin [1]. Silnik do 120 stopni [2][1].")
    answer = answer_question(llm, "pytanie", P)
    assert answer.grounded and answer.citations == [1, 2] and answer.warnings == []
    assert [s.ref for s in answer.sources] == ["lozyska.md s. 3", "silnik.md"]
    assert answer.sources[0].kind == "document" and "8000" in answer.sources[0].excerpt
    assert llm.calls[0][1] == 0.1  # low temperature for factual answers


def test_answers_without_valid_sources_are_not_trusted():
    nocite = answer_question(FakeLLM("Wymienia się co 8000 godzin."), "pytanie", P)
    assert not nocite.grounded and nocite.sources == []
    assert any("cites no source" in w for w in nocite.warnings)
    invented = answer_question(FakeLLM("Co 8000 godzin [7]."), "pytanie", P)
    assert not invented.grounded and invented.citations == []
    assert any("[7]" in w for w in invented.warnings)
    mixed = answer_question(FakeLLM("Tak [1] i nie [9]."), "pytanie", P)
    assert mixed.grounded and mixed.citations == [1] and any("[9]" in w for w in mixed.warnings)


def test_no_answer_and_no_passages():
    declined = answer_question(FakeLLM(f"{NO_ANSWER}."), "pytanie", P)
    assert declined.text == NO_ANSWER and not declined.grounded and declined.citations == []
    llm = FakeLLM("should not be called")
    empty = answer_question(llm, "pytanie", [])
    assert empty.text == NO_ANSWER and llm.calls == []
    with pytest.raises(RagError, match="empty"):
        answer_question(llm, " ", P)


# --- CLI --------------------------------------------------------------------------------------


def test_cli_docs_index_search_ask_and_llm_check(tmp_path, monkeypatch):
    monkeypatch.setenv("DIAGSYS_DATA_DIR", str(tmp_path / "data"))
    runner = CliRunner()

    def run(*args, code=0):
        result = runner.invoke(app, list(args))
        assert result.exit_code == code, result.output
        return result.output

    run("machine", "new", "pump-01")
    docs = tmp_path / "data" / "machines" / "pump-01" / "docs"
    write_docs(docs)

    out = run("docs", "index", "pump-01")
    assert "4 new" in out and "embedder hash-ngram" in out
    assert "4 unchanged" in run("docs", "index", "pump-01")
    found = run("docs", "search", "pump-01", "jak często wymieniać łożyska", "-k", "2")
    assert found.startswith("[1]") and "lozyska.md" in found.splitlines()[0]
    run("docs", "search", "pump-01", "łożysko", "-k", "0", code=1)

    with StubLLM() as stub:
        monkeypatch.setenv("DIAGSYS_LLM_BASE_URL", stub.url)
        monkeypatch.setenv("DIAGSYS_LLM_MODEL", "bielik-test")
        stub.chat_reply = "Łożyska wymienia się co 8000 godzin [1]."
        asked = run("docs", "ask", "pump-01", "jak często wymieniać łożyska?")
        assert "co 8000 godzin [1]" in asked and "  [1] lozyska.md" in asked
        assert "WARNING" not in asked
        sent = [r for r in stub.requests if r[1].endswith("/chat/completions")][0][2]
        assert "Fragmenty dokumentacji" in sent["messages"][1]["content"]
        stub.chat_reply = "Wymienia się co 8000 godzin."
        assert "WARNING: the answer cites no source" in run(
            "docs", "ask", "pump-01", "jak często wymieniać łożyska?"
        )
        checked = run("llm", "check")
        assert "bielik-test" in checked and "answered" in checked


def test_cli_ask_without_a_server_says_what_to_do(tmp_path, monkeypatch):
    monkeypatch.setenv("DIAGSYS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DIAGSYS_LLM_BASE_URL", "http://127.0.0.1:9/v1")  # discard port: closed
    monkeypatch.setenv("DIAGSYS_LLM_TIMEOUT_S", "2")
    runner = CliRunner()
    runner.invoke(app, ["machine", "new", "pump-01"])
    docs = tmp_path / "data" / "machines" / "pump-01" / "docs"
    write_docs(docs)
    runner.invoke(app, ["docs", "index", "pump-01"])
    result = runner.invoke(app, ["docs", "ask", "pump-01", "jak często wymieniać łożyska?"])
    assert result.exit_code == 1 and "not reachable" in result.output
    assert runner.invoke(app, ["llm", "check"]).exit_code == 1
