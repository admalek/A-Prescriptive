"""Ports: the interfaces behind which implementations can be swapped (laptop now, server later).

Laptop: DuckDB storage, LanceDB index, llama.cpp or Ollama.
Server: TimescaleDB, Qdrant, vLLM. Nothing above these ports changes.
"""

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

import pyarrow as pa

from .contracts import ChatMessage, Contract, FailureEvent, Measurement, Passage


class Storage(Protocol):
    """Interfaces I2 and I7: time series, failure labels and results.

    Writes are idempotent: a reading or event that is already stored is skipped, and the
    return value counts only what was newly written.
    """

    def write_measurements(self, rows: Sequence[Measurement]) -> int: ...

    def read_timeseries(
        self, machine_id: str, sensor_id: str, start: datetime, end: datetime
    ) -> list[Measurement]:
        """Readings with start <= timestamp < end, oldest first."""
        ...

    def read_table(
        self, machine_id: str, start: datetime | None = None, end: datetime | None = None
    ) -> pa.Table:
        """Bulk read for analysis: columns sensor_id, ts, value; start <= ts < end; by time."""
        ...

    def write_events(self, events: Sequence[FailureEvent]) -> int: ...

    def read_events(self, machine_id: str) -> list[FailureEvent]: ...

    def write_result(self, kind: str, payload: Contract) -> str:
        """Persist a result and return a reference to it."""
        ...


class VectorIndex(Protocol):
    """Interface I7: per-machine documentation index (RAG). One collection per machine."""

    def add(self, collection: str, passages: Sequence[Passage]) -> int: ...

    def search(self, collection: str, query: str, k: int = 5) -> list[Passage]: ...


class LLMClient(Protocol):
    """Interface I5: the only boundary to the language model."""

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...
