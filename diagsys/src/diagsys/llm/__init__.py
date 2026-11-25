"""Language model client behind diagsys.ports.LLMClient (OpenAI-compatible API): K6."""

from .client import LLMError, OpenAICompatClient

__all__ = ["LLMError", "OpenAICompatClient"]
