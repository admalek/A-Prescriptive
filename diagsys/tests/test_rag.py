import sys
import zipfile

import numpy as np
import pytest

from diagsys.config import Settings
from diagsys.contracts import Passage
from diagsys.rag import (
    DuckDBVectorIndex,
    HashingEmbedder,
    RagError,
    ServerEmbedder,
    chunk_pages,
    load_pages,
    make_embedder,
)
from diagsys.rag import loaders as loaders_module
from diagsys.rag.embedding import fold


def cos(a, b):
    return float(np.dot(a, b))


# --- embedder ---------------------------------------------------------------------------------


def test_hashing_embedder_is_deterministic_normalised_and_ignores_diacritics():
    e = HashingEmbedder()
    one = e.embed_query("Zużycie łożyska pompy")
    assert one == e.embed_query("Zużycie łożyska pompy")
    assert one == e.embed_query("zuzycie lozyska pompy")
    assert np.linalg.norm(one) == pytest.approx(1.0)
    assert fold("Łódź, ŻÓŁĆ") == "lodz, zolc"
    assert e.embed_query("") == [0.0] * 512 and e.embed_query("a i w") == [0.0] * 512


def test_hashing_embedder_ranks_related_text_above_unrelated():
    e = HashingEmbedder()
    q = e.embed_query("wymiana łożyska")
    related = e.embed_passages(["Procedura wymiany łożysk w pompie"])[0]
    unrelated = e.embed_passages(["Kalibracja czujnika ciśnienia w instalacji"])[0]
    assert cos(q, related) > cos(q, unrelated) + 0.1
    assert e.id == "hash-ngram-v1-d512"
    with pytest.raises(ValueError):
        HashingEmbedder(dim=4)


def test_server_embedder_adds_e5_prefixes_and_normalises():
    class Fake:
        def __init__(self):
            self.seen = []

        def embed(self, texts):
            self.seen.append(list(texts))
            return [[3.0, 4.0] for _ in texts]

    fake = Fake()
    e5 = ServerEmbedder(fake, "multilingual-e5-small")
    assert e5.embed_passages(["a"])[0] == pytest.approx([0.6, 0.8])
    e5.embed_query("pytanie")
    assert (
        fake.seen == [["passage: a"], ["query: pytanie"]]
        and e5.id == "server:multilingual-e5-small"
    )
    plain = ServerEmbedder(fake, "bge-m3")
    plain.embed_query("x")
    assert fake.seen[-1] == ["x"]


def test_make_embedder_follows_the_setting():
    assert make_embedder(Settings(embedding_backend="hash", _env_file=None)).id.startswith("hash")
    server = make_embedder(
        Settings(embedding_backend="server", embedding_model="m", _env_file=None)
    )
    assert server.id == "server:m"


# --- chunking ---------------------------------------------------------------------------------


def test_chunks_respect_the_size_limit_even_for_one_huge_word():
    text = "x" * 10_000 + "\n\n" + " ".join(["słowo"] * 500)
    chunks = chunk_pages([(None, text)], max_chars=300, overlap=50)
    assert chunks and all(len(c.text) <= 300 + 50 + 1 for c in chunks)
    assert "".join(c.text for c in chunks).count("x") >= 10_000


def test_chunks_never_span_pages_and_keep_their_page_number():
    chunks = chunk_pages([(1, "Pierwsza strona."), (2, "Druga strona."), (3, "   ")], 200, 20)
    assert [(c.page, c.text) for c in chunks] == [(1, "Pierwsza strona."), (2, "Druga strona.")]


def test_overlap_repeats_the_end_of_the_previous_chunk_and_can_be_off():
    text = ". ".join(f"Zdanie numer {i} o pompie" for i in range(40))
    with_overlap = chunk_pages([(None, text)], 200, 60)
    assert len(with_overlap) > 2
    first_tail = with_overlap[0].text.split()[-3:]
    assert " ".join(first_tail) in with_overlap[1].text
    without = chunk_pages([(None, text)], 200, 0)
    assert " ".join(without[0].text.split()[-3:]) not in without[1].text[:40]


def test_headings_stay_in_front_of_their_section_and_sections_do_not_mix():
    text = "# Łożysko\n\nWymiana łożyska.\n\n## Smarowanie\nSmar EP2 co 500 h.\nSprawdź poziom."
    texts = [c.text for c in chunk_pages([(None, text)], 300, 40)]
    assert texts == [
        "[Łożysko] Wymiana łożyska.",
        "[Smarowanie] Smar EP2 co 500 h. Sprawdź poziom.",
    ]


def test_chunking_validates_its_settings_and_handles_empty_text():
    assert chunk_pages([(None, ""), (None, "\n\n  \n")]) == []
    with pytest.raises(ValueError):
        chunk_pages([(None, "x")], max_chars=50)
    with pytest.raises(ValueError):
        chunk_pages([(None, "x")], max_chars=200, overlap=200)


# --- loaders ----------------------------------------------------------------------------------


def make_docx(path, paragraphs):
    ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    body = ""
    for style, text in paragraphs:
        ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
        body += f"<w:p>{ppr}<w:r><w:t>{text}</w:t></w:r></w:p>"
    xml = f'<?xml version="1.0"?><w:document xmlns:w="{ns}"><w:body>{body}</w:body></w:document>'
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("word/document.xml", xml)


def make_pdf(path, lines):
    """A minimal valid one-page PDF with the given lines of ASCII text."""
    stream = "BT /F1 12 Tf 72 720 Td 14 TL " + " ".join(f"({t}) Tj T*" for t in lines) + " ET"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = "%PDF-1.4\n", []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{obj}\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    path.write_bytes(out.encode("latin-1"))


def test_text_files_utf8_and_windows_polish_encoding(tmp_path):
    (tmp_path / "a.txt").write_text("Łożysko pompy", encoding="utf-8")
    (tmp_path / "b.md").write_bytes("Łożysko pompy".encode("cp1250"))
    assert load_pages(tmp_path / "a.txt") == [(None, "Łożysko pompy")]
    assert load_pages(tmp_path / "b.md") == [(None, "Łożysko pompy")]


def test_docx_text_and_heading_styles(tmp_path):
    make_docx(tmp_path / "a.docx", [("Heading1", "Łożysko"), (None, "Wymiana co 8000 h.")])
    assert load_pages(tmp_path / "a.docx") == [(None, "# Łożysko\n\nWymiana co 8000 h.")]
    (tmp_path / "bad.docx").write_text("not a zip")
    with pytest.raises(RagError, match="cannot read the Word file"):
        load_pages(tmp_path / "bad.docx")


def test_pdf_pages_and_the_reasons_a_file_is_rejected(tmp_path):
    pytest.importorskip("pypdf")
    make_pdf(tmp_path / "a.pdf", ["Pump bearing replacement", "every 8000 hours"])
    pages = load_pages(tmp_path / "a.pdf")
    assert pages[0][0] == 1 and "bearing replacement" in pages[0][1]
    (tmp_path / "broken.pdf").write_bytes(b"%PDF-1.4 garbage")
    with pytest.raises(RagError, match="cannot read the PDF"):
        load_pages(tmp_path / "broken.pdf")
    make_pdf(tmp_path / "blank.pdf", [])
    with pytest.raises(RagError, match="no text found .*scanned"):
        load_pages(tmp_path / "blank.pdf")


def test_pdf_without_pypdf_says_how_to_install_it(tmp_path, monkeypatch):
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4")
    monkeypatch.setitem(sys.modules, "pypdf", None)
    with pytest.raises(RagError, match="needs pypdf"):
        load_pages(tmp_path / "a.pdf")


def test_empty_unsupported_and_oversized_files(tmp_path, monkeypatch):
    (tmp_path / "empty.txt").write_text("  \n")
    with pytest.raises(RagError, match="no text found"):
        load_pages(tmp_path / "empty.txt")
    (tmp_path / "a.xls").write_text("x")
    with pytest.raises(RagError, match="unsupported"):
        load_pages(tmp_path / "a.xls")
    (tmp_path / "big.txt").write_text("x" * 100)
    monkeypatch.setattr(loaders_module, "MAX_FILE_BYTES", 10)
    with pytest.raises(RagError, match="larger than"):
        load_pages(tmp_path / "big.txt")


# --- index ------------------------------------------------------------------------------------

DOCS = [
    Passage(doc_id="a.md", chunk_id="a.md#0", text="Wymiana łożysk w pompie co 8000 godzin pracy."),
    Passage(
        doc_id="a.md",
        chunk_id="a.md#1",
        text="Smarowanie łożyska smarem EP2 co 500 godzin.",
        page=2,
    ),
    Passage(doc_id="b.md", chunk_id="b.md#0", text="Czujnik ciśnienia kalibruje się raz w roku."),
]


@pytest.fixture
def index(tmp_path):
    with DuckDBVectorIndex(tmp_path / "rag.duckdb", HashingEmbedder()) as idx:
        yield idx


def test_add_is_idempotent_and_search_ranks_the_relevant_passage_first(index):
    assert index.add("docs_m1", DOCS) == 3
    assert index.add("docs_m1", DOCS) == 0  # same chunk ids are skipped
    assert index.add("docs_m1", [DOCS[0], DOCS[0]]) == 0
    hits = index.search("docs_m1", "kalibracja czujnika ciśnienia", k=2)
    assert [h.chunk_id for h in hits][0] == "b.md#0" and len(hits) == 2
    assert hits[0].score >= hits[1].score and hits[0].page is None
    assert index.search("docs_m1", "smarowanie łożyska", k=1)[0].page == 2
    assert index.stats("docs_m1") == {
        "passages": 3,
        "documents": 0,
        "embedder": "hash-ngram-v1-d512",
    }


def test_collections_are_isolated_and_missing_ones_are_empty(index):
    index.add("docs_m1", DOCS[:1])
    index.add("docs_m2", DOCS[2:])
    assert index.search("docs_m1", "ciśnienia czujnik")[0].doc_id == "a.md"
    assert index.search("docs_m2", "ciśnienia czujnik")[0].doc_id == "b.md"
    assert index.search("docs_none", "cokolwiek") == []
    with pytest.raises(RagError, match="between 1 and 100"):
        index.search("docs_m1", "x", k=0)


def test_remove_document_and_drop(index):
    index.add("docs_m1", DOCS)
    index.set_document("docs_m1", "a.md", "key1", 2)
    assert index.documents("docs_m1") == {"a.md": "key1"}
    assert index.remove_document("docs_m1", "a.md") == 2
    assert index.documents("docs_m1") == {}
    assert {h.doc_id for h in index.search("docs_m1", "łożysko", k=5)} == {"b.md"}
    index.drop("docs_m1")
    assert index.search("docs_m1", "łożysko") == []


def test_the_index_survives_reopening_and_refuses_another_embedder(tmp_path):
    path = tmp_path / "rag.duckdb"
    with DuckDBVectorIndex(path, HashingEmbedder()) as idx:
        idx.add("docs_m1", DOCS)
    with DuckDBVectorIndex(path, HashingEmbedder()) as idx:
        assert idx.search("docs_m1", "wymiana łożysk")[0].chunk_id == "a.md#0"
    with DuckDBVectorIndex(path, HashingEmbedder(dim=256)) as other:
        with pytest.raises(RagError, match="--rebuild"):
            other.search("docs_m1", "wymiana")
        with pytest.raises(RagError, match="--rebuild"):
            other.add("docs_m1", [Passage(doc_id="c", chunk_id="c#0", text="nowy tekst")])
