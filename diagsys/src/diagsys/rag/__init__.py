"""Per-machine documentation index: loading, chunking, embeddings, DuckDB VectorIndex: K6."""

from .answer import NO_ANSWER, Answer, answer_question, build_messages
from .chunking import Chunk, chunk_pages
from .embedding import Embedder, HashingEmbedder, ServerEmbedder, make_embedder
from .index import DuckDBVectorIndex
from .loaders import RagError, load_pages
from .service import IndexReport, index_machine, index_path, open_index, search_machine

__all__ = [
    "NO_ANSWER",
    "Answer",
    "Chunk",
    "DuckDBVectorIndex",
    "Embedder",
    "HashingEmbedder",
    "IndexReport",
    "RagError",
    "ServerEmbedder",
    "answer_question",
    "build_messages",
    "chunk_pages",
    "index_machine",
    "index_path",
    "load_pages",
    "make_embedder",
    "open_index",
    "search_machine",
]
