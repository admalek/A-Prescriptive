"""Reading machine documentation: .txt, .md, .pdf (pypdf) and .docx (standard library only).

Each loader returns pages as (page number or None, text). A file that cannot be read raises
`RagError` with the reason; scanned PDFs without a text layer are reported, not silently skipped
(OCR is not supported).
"""

import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

SUPPORTED = {".txt": "text", ".md": "text", ".markdown": "text", ".pdf": "pdf", ".docx": "docx"}
MAX_FILE_BYTES = 50 * 1024 * 1024
_MAX_XML_BYTES = 30 * 1024 * 1024
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


class RagError(Exception):
    """A document or index operation failed; the message says why."""


def load_pages(path: Path) -> list[tuple[int | None, str]]:
    kind = SUPPORTED.get(path.suffix.lower())
    if kind is None:
        raise RagError(f"unsupported file type {path.suffix!r} (use .txt, .md, .pdf or .docx)")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise RagError(f"file is larger than {MAX_FILE_BYTES // (1024 * 1024)} MB")
    pages = {"text": _text, "pdf": _pdf, "docx": _docx}[kind](path)
    if not any(text.strip() for _, text in pages):
        raise RagError(
            "no text found" + (" (a scanned PDF? OCR is not supported)" if kind == "pdf" else "")
        )
    return pages


def _text(path: Path) -> list[tuple[int | None, str]]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1250", errors="replace")  # common for Polish files from Windows
    return [(None, text)]


def _pdf(path: Path) -> list[tuple[int | None, str]]:
    try:
        from pypdf import PdfReader
        from pypdf.errors import PyPdfError
    except ImportError as exc:
        raise RagError("reading PDF files needs pypdf: uv pip install pypdf") from exc
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            raise RagError("the PDF is encrypted")
        return [(i, page.extract_text() or "") for i, page in enumerate(reader.pages, start=1)]
    except (PyPdfError, ValueError, OSError, KeyError) as exc:
        raise RagError(f"cannot read the PDF: {exc}") from exc


def _docx(path: Path) -> list[tuple[int | None, str]]:
    try:
        with zipfile.ZipFile(path) as archive:
            info = archive.getinfo("word/document.xml")
            if info.file_size > _MAX_XML_BYTES:
                raise RagError("the document body is unreasonably large")
            root = ET.fromstring(archive.read(info))
    except (zipfile.BadZipFile, KeyError, ET.ParseError, OSError) as exc:
        raise RagError(f"cannot read the Word file: {exc}") from exc
    lines = []
    for paragraph in root.iter(f"{_W}p"):
        parts = []
        for node in paragraph.iter():
            if node.tag == f"{_W}t" and node.text:
                parts.append(node.text)
            elif node.tag == f"{_W}tab":
                parts.append("\t")
            elif node.tag == f"{_W}br":
                parts.append(" ")
        line = "".join(parts).strip()
        style = paragraph.find(f"{_W}pPr/{_W}pStyle")
        if line and style is not None and style.get(f"{_W}val", "").lower().startswith("heading"):
            line = "# " + line
        lines.append(line)
    return [(None, "\n\n".join(lines))]
