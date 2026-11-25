"""Per-machine document index on DuckDB: implements `diagsys.ports.VectorIndex`.

Search is exact (cosine similarity against every stored passage), which is fast and simple for the
documentation of one machine (thousands of passages). For a much larger corpus swap this class for
another `VectorIndex` (LanceDB, Qdrant) without changing anything above the port.

The index remembers which embedder built each collection and refuses another one.
"""

import threading
from collections.abc import Sequence
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa

from ..contracts import Passage
from .embedding import Embedder
from .loaders import RagError

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rag_meta (
    collection VARCHAR PRIMARY KEY, embedder_id VARCHAR NOT NULL, dim INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS rag_docs (
    collection VARCHAR NOT NULL, doc_id VARCHAR NOT NULL, content_key VARCHAR NOT NULL,
    n_passages INTEGER NOT NULL, PRIMARY KEY (collection, doc_id));
CREATE TABLE IF NOT EXISTS rag_passages (
    collection VARCHAR NOT NULL, doc_id VARCHAR NOT NULL, chunk_id VARCHAR NOT NULL,
    page INTEGER, text VARCHAR NOT NULL, embedding FLOAT[] NOT NULL,
    PRIMARY KEY (collection, chunk_id));
"""


class DuckDBVectorIndex:
    def __init__(self, path: Path, embedder: Embedder) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self._conn = duckdb.connect(str(path))
        self._conn.execute(_SCHEMA).fetchall()
        self._lock = threading.Lock()
        self._cache: dict[str, tuple[list[tuple[str, str, int | None, str]], np.ndarray]] = {}

    def __enter__(self) -> "DuckDBVectorIndex":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    # --- VectorIndex port -------------------------------------------------------------------

    def add(self, collection: str, passages: Sequence[Passage]) -> int:
        """Store passages that are not stored yet (same chunk_id); returns how many were new."""
        with self._lock:
            known = {
                row[0]
                for row in self._run(
                    "SELECT chunk_id FROM rag_passages WHERE collection = ?", [collection]
                )
            }
            seen: set[str] = set()
            fresh = []
            for p in passages:
                if p.chunk_id not in known and p.chunk_id not in seen:
                    seen.add(p.chunk_id)
                    fresh.append(p)
            if not fresh:
                return 0
            vectors = self.embedder.embed_passages([p.text for p in fresh])
            if len(vectors) != len(fresh):
                raise RagError("the embedder returned a different number of vectors than texts")
            self._claim(collection, len(vectors[0]))
            table = pa.table(
                {
                    "collection": [collection] * len(fresh),
                    "doc_id": [p.doc_id for p in fresh],
                    "chunk_id": [p.chunk_id for p in fresh],
                    "page": pa.array([p.page for p in fresh], pa.int32()),
                    "text": [p.text for p in fresh],
                    "embedding": pa.array(vectors, pa.list_(pa.float32())),
                }
            )
            self._conn.register("incoming_passages", table)
            try:
                self._run("INSERT INTO rag_passages SELECT * FROM incoming_passages")
            finally:
                self._conn.unregister("incoming_passages")
            self._cache.pop(collection, None)
            return len(fresh)

    def search(self, collection: str, query: str, k: int = 5) -> list[Passage]:
        if not 1 <= k <= 100:
            raise RagError("k must be between 1 and 100")
        with self._lock:
            if self._meta(collection) is None:
                return []
            self.check_embedder(collection)
            rows, matrix = self._load(collection)
            if not rows:
                return []
            q = np.asarray(self.embedder.embed_query(query), dtype=np.float32)
            if q.shape[0] != matrix.shape[1]:
                raise RagError("the query vector has a different size than the indexed vectors")
            norm = float(np.linalg.norm(q))
            scores = matrix @ (q / norm) if norm > 0 else np.zeros(len(rows), dtype=np.float32)
            top = np.argsort(-scores, kind="stable")[:k]
            return [
                Passage(
                    doc_id=rows[i][0],
                    chunk_id=rows[i][1],
                    page=rows[i][2],
                    text=rows[i][3],
                    score=float(scores[i]),
                )
                for i in top
            ]

    # --- documents and housekeeping -----------------------------------------------------------

    def documents(self, collection: str) -> dict[str, str]:
        """doc_id -> content key of the version that is indexed."""
        with self._lock:
            return dict(
                self._run(
                    "SELECT doc_id, content_key FROM rag_docs WHERE collection = ?", [collection]
                )
            )

    def set_document(self, collection: str, doc_id: str, content_key: str, n_passages: int) -> None:
        with self._lock:
            self._run(
                "INSERT OR REPLACE INTO rag_docs VALUES (?, ?, ?, ?)",
                [collection, doc_id, content_key, n_passages],
            )

    def remove_document(self, collection: str, doc_id: str) -> int:
        """Delete a document and its passages; returns how many passages were removed."""
        with self._lock:
            removed = self._run(
                "SELECT count(*) FROM rag_passages WHERE collection = ? AND doc_id = ?",
                [collection, doc_id],
            )[0][0]
            self._run(
                "DELETE FROM rag_passages WHERE collection = ? AND doc_id = ?", [collection, doc_id]
            )
            self._run(
                "DELETE FROM rag_docs WHERE collection = ? AND doc_id = ?", [collection, doc_id]
            )
            self._cache.pop(collection, None)
            return int(removed)

    def drop(self, collection: str) -> None:
        with self._lock:
            for table in ("rag_passages", "rag_docs", "rag_meta"):
                self._run(f"DELETE FROM {table} WHERE collection = ?", [collection])
            self._cache.pop(collection, None)

    def stats(self, collection: str) -> dict[str, object]:
        with self._lock:
            meta = self._meta(collection)
            passages = self._run(
                "SELECT count(*) FROM rag_passages WHERE collection = ?", [collection]
            )[0][0]
            docs = self._run("SELECT count(*) FROM rag_docs WHERE collection = ?", [collection])[0][
                0
            ]
        return {
            "passages": int(passages),
            "documents": int(docs),
            "embedder": meta[0] if meta else None,
        }

    # --- internals ----------------------------------------------------------------------------

    def _run(self, sql: str, params: list | None = None) -> list[tuple]:
        """Execute and always consume the result, or DuckDB blocks the next statement."""
        result = self._conn.execute(sql, params) if params is not None else self._conn.execute(sql)
        return result.fetchall()

    def _meta(self, collection: str) -> tuple[str, int] | None:
        rows = self._run("SELECT embedder_id, dim FROM rag_meta WHERE collection = ?", [collection])
        return (rows[0][0], rows[0][1]) if rows else None

    def check_embedder(self, collection: str) -> None:
        meta = self._meta(collection)
        if meta is not None and meta[0] != self.embedder.id:
            raise RagError(
                f"collection {collection!r} was indexed with {meta[0]!r} but the current setting "
                f"uses {self.embedder.id!r}; run `diagsys docs index <machine> --rebuild`"
            )

    def _claim(self, collection: str, dim: int) -> None:
        meta = self._meta(collection)
        if meta is None:
            self._run("INSERT INTO rag_meta VALUES (?, ?, ?)", [collection, self.embedder.id, dim])
            return
        self.check_embedder(collection)
        if meta[1] != dim:
            raise RagError(f"vectors of size {dim} do not fit the collection (size {meta[1]})")

    def _load(self, collection: str) -> tuple[list[tuple[str, str, int | None, str]], np.ndarray]:
        cached = self._cache.get(collection)
        if cached is not None:
            return cached
        data = self._run(
            "SELECT doc_id, chunk_id, page, text, embedding FROM rag_passages "
            "WHERE collection = ? ORDER BY doc_id, chunk_id",
            [collection],
        )
        rows = [(d, c, p, t) for d, c, p, t, _ in data]
        matrix = np.asarray([e for *_, e in data], dtype=np.float32).reshape(len(data), -1)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.where(norms > 0, norms, 1.0)
        self._cache[collection] = (rows, matrix)
        return rows, matrix
