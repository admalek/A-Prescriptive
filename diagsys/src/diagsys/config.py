"""Runtime settings, read from environment variables (prefix DIAGSYS_) and an optional .env file."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DIAGSYS_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")

    # Interface I5: the only boundary to the language model (OpenAI-compatible API).
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "bielik-4.5b-instruct"
    llm_timeout_s: float = 120.0
    embedding_model: str = "multilingual-e5-small"
    # "hash": local character n-gram vectors, no model needed (works offline, weaker semantics)
    # "server": embeddings from the model server (needs embedding_model to be served)
    embedding_backend: Literal["hash", "server"] = "hash"

    max_agent_steps: int = 12
    log_level: str = "INFO"

    @property
    def machines_dir(self) -> Path:
        return self.data_dir / "machines"

    @property
    def db_path(self) -> Path:
        """Single DuckDB file with all machines' time series, events and results."""
        return self.data_dir / "diagsys.duckdb"


def load_settings() -> Settings:
    """Read settings at call time, so tests and the CLI always see the current environment."""
    return Settings()
