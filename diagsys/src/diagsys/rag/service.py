"""Indexing the documentation folder of a machine and searching it.

`index_machine` is incremental: a file is read again only when its content (or the chunking
settings) changed, passages of deleted files are removed, and one unreadable file does not stop
the others (it is reported with the reason).
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Settings
from ..contracts import Passage
from ..machines import load_machine, machine_dir
from .chunking import CHUNKER_VERSION, chunk_pages
from .embedding import Embedder
from .index import DuckDBVectorIndex
from .loaders import SUPPORTED, RagError, load_pages

DEFAULT_MAX_CHARS = 900
DEFAULT_OVERLAP = 150


@dataclass
class IndexReport:
    collection: str
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (file, reason)
    passages_added: int = 0
    passages_total: int = 0


def index_path(settings: Settings, machine_id: str) -> Path:
    """The index is a derived artifact, kept with the machine's models and rebuilt on demand."""
    return machine_dir(settings, machine_id) / "models" / "rag.duckdb"


def open_index(settings: Settings, machine_id: str, embedder: Embedder) -> DuckDBVectorIndex:
    return DuckDBVectorIndex(index_path(settings, machine_id), embedder)


def index_machine(
    settings: Settings,
    machine_id: str,
    embedder: Embedder,
    *,
    rebuild: bool = False,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> IndexReport:
    profile = load_machine(settings, machine_id)
    collection = profile.rag_collection or f"docs_{machine_id}"
    docs_dir = machine_dir(settings, machine_id) / "docs"
    report = IndexReport(collection)
    files, skipped = _scan(docs_dir)
    report.skipped.extend(skipped)
    chunk_key = f"{CHUNKER_VERSION}:{max_chars}:{overlap}"

    with open_index(settings, machine_id, embedder) as index:
        if rebuild:
            index.drop(collection)
        else:  # an index built by another embedder cannot be extended, even if no file changed
            index.check_embedder(collection)
        indexed = index.documents(collection)
        for doc_id in sorted(set(indexed) - set(files)):
            index.remove_document(collection, doc_id)
            report.removed.append(doc_id)
        for doc_id, path in files.items():
            key = f"{_sha256(path)}:{chunk_key}"
            if indexed.get(doc_id) == key:
                report.unchanged.append(doc_id)
                continue
            try:
                chunks = chunk_pages(load_pages(path), max_chars, overlap)
            except RagError as exc:
                report.skipped.append((doc_id, str(exc)))
                continue
            if not chunks:
                report.skipped.append((doc_id, "no text found"))
                continue
            index.remove_document(collection, doc_id)
            passages = [
                Passage(doc_id=doc_id, chunk_id=f"{doc_id}#{n:04d}", text=c.text, page=c.page)
                for n, c in enumerate(chunks)
            ]
            report.passages_added += index.add(collection, passages)
            index.set_document(collection, doc_id, key, len(passages))
            (report.updated if doc_id in indexed else report.added).append(doc_id)
        report.passages_total = int(index.stats(collection)["passages"])
    return report


def search_machine(
    settings: Settings, machine_id: str, embedder: Embedder, query: str, k: int = 5
) -> list[Passage]:
    profile = load_machine(settings, machine_id)
    collection = profile.rag_collection or f"docs_{machine_id}"
    if not query.strip():
        raise RagError("the question is empty")
    with open_index(settings, machine_id, embedder) as index:
        return index.search(collection, query, k)


def _scan(docs_dir: Path) -> tuple[dict[str, Path], list[tuple[str, str]]]:
    """Readable files by doc_id (relative path), and the files that were left out with a reason."""
    files: dict[str, Path] = {}
    skipped: list[tuple[str, str]] = []
    if not docs_dir.is_dir():
        return files, skipped
    root = docs_dir.resolve()
    for path in sorted(docs_dir.rglob("*")):
        rel = path.relative_to(docs_dir).as_posix()
        if path.is_dir() or any(part.startswith(".") for part in path.relative_to(docs_dir).parts):
            continue
        if path.is_symlink():
            skipped.append((rel, "symbolic links are not followed"))
        elif not path.resolve().is_relative_to(root):
            skipped.append((rel, "outside the documentation folder"))
        elif path.suffix.lower() not in SUPPORTED:
            skipped.append((rel, f"unsupported file type {path.suffix!r}"))
        else:
            files[rel] = path
    return files, skipped


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
