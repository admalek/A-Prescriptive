"""Text embedders for the document index.

  HashingEmbedder  local, no model: hashed character n-grams of accent-folded words. Works offline
                   and tolerates Polish inflection and missing diacritics, but matches wording,
                   not meaning (a question and a passage must share word stems).
  ServerEmbedder   vectors from the model server (e.g. a multilingual-e5 or bge-m3 model served by
                   Ollama or llama.cpp); understands meaning, needs the server.

An index remembers the `id` of the embedder that built it and refuses another one, because vectors
from different embedders cannot be compared.
"""

import math
import re
import unicodedata
import zlib
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from ..config import Settings
from ..ports import LLMClient


class Embedder(Protocol):
    id: str

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


_WORD = re.compile(r"[a-z0-9]+")


def fold(text: str) -> str:
    """Lower case without diacritics: 'Zużycie łożyska' -> 'zuzycie lozyska'."""
    text = unicodedata.normalize("NFKD", text.lower().replace("ł", "l"))
    return "".join(c for c in text if not unicodedata.combining(c))


class HashingEmbedder:
    def __init__(self, dim: int = 512, ngrams: tuple[int, ...] = (3, 4, 5)) -> None:
        if dim < 16 or any(n < 2 for n in ngrams):
            raise ValueError("dim must be at least 16 and n-grams at least 2 characters")
        self.dim = dim
        self.ngrams = ngrams
        self.id = f"hash-ngram-v1-d{dim}"

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float64)
        for word in _WORD.findall(fold(text)):
            if len(word) <= 2 and not word.isdigit():
                continue  # short function words only add noise
            self._add(vec, "w:" + word, 2.0)
            padded = f"<{word}>"
            for n in self.ngrams:
                for i in range(len(padded) - n + 1):
                    self._add(vec, padded[i : i + n], 1.0)
        vec = np.sign(vec) * np.log1p(np.abs(vec))
        norm = float(np.linalg.norm(vec))
        return (vec / norm).tolist() if norm > 0 else vec.tolist()

    def _add(self, vec: np.ndarray, gram: str, weight: float) -> None:
        h = zlib.crc32(gram.encode("utf-8"))
        vec[h % self.dim] += weight if (h >> 31) & 1 == 0 else -weight


class ServerEmbedder:
    def __init__(self, llm: LLMClient, model: str) -> None:
        self.llm = llm
        self.model = model
        self.id = f"server:{model}"
        # e5 models are trained with these prefixes and lose quality without them
        self._is_e5 = "e5" in model.lower()

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        prefix = "passage: " if self._is_e5 else ""
        return [_unit(v) for v in self.llm.embed([prefix + t for t in texts])]

    def embed_query(self, text: str) -> list[float]:
        prefix = "query: " if self._is_e5 else ""
        return _unit(self.llm.embed([prefix + text])[0])


def make_embedder(settings: Settings, llm: LLMClient | None = None) -> Embedder:
    if settings.embedding_backend == "hash":
        return HashingEmbedder()
    if llm is None:
        from ..llm import OpenAICompatClient

        llm = OpenAICompatClient.from_settings(settings)
    return ServerEmbedder(llm, settings.embedding_model)


def _unit(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm > 0 else list(vector)
