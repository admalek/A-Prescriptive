"""Storage adapters behind diagsys.ports.Storage. Laptop: DuckDB (K2); server later."""

from .duckdb_store import DuckDBStorage, SensorSummary

__all__ = ["DuckDBStorage", "SensorSummary"]
