"""Client for an OpenAI-compatible model server (Ollama, llama.cpp `llama-server`, vLLM).

It implements `diagsys.ports.LLMClient` with the standard library only, so nothing has to be
installed to talk to a local server. Every failure is an `LLMError` whose message says what to do.
"""

import json
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlparse

from ..config import Settings
from ..contracts import ChatMessage

_EMBED_BATCH = 32
_MAX_BODY = 32 * 1024 * 1024


class LLMError(Exception):
    """The model server cannot be reached or answered something unusable."""


class OpenAICompatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        embedding_model: str | None = None,
        timeout_s: float = 120.0,
        api_key: str = "not-needed",
    ) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise LLMError(f"model server URL must start with http:// or https://: {base_url!r}")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.embedding_model = embedding_model or model
        self.timeout_s = timeout_s
        self._headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    @classmethod
    def from_settings(cls, settings: Settings) -> "OpenAICompatClient":
        return cls(
            settings.llm_base_url,
            settings.llm_model,
            embedding_model=settings.embedding_model,
            timeout_s=settings.llm_timeout_s,
        )

    # --- LLMClient port ---------------------------------------------------------------------

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        if not messages:
            raise LLMError("chat needs at least one message")
        body = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        data = self._post("/chat/completions", body, self.model)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected chat answer from {self.base_url}: {_short(data)}") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMError(f"model {self.model!r} returned an empty answer")
        return content.strip()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_BATCH):
            batch = list(texts[start : start + _EMBED_BATCH])
            data = self._post(
                "/embeddings", {"model": self.embedding_model, "input": batch}, self.embedding_model
            )
            try:
                rows = sorted(data["data"], key=lambda r: r["index"])
                vectors = [[float(x) for x in r["embedding"]] for r in rows]
            except (KeyError, TypeError, ValueError) as exc:
                raise LLMError(
                    f"unexpected embeddings answer from {self.base_url}: {_short(data)}"
                ) from exc
            if len(vectors) != len(batch):
                raise LLMError(f"asked for {len(batch)} embeddings, got {len(vectors)}")
            out.extend(vectors)
        return out

    # --- diagnostics ------------------------------------------------------------------------

    def models(self) -> list[str]:
        """Names of the models the server offers (GET /models)."""
        data = self._request("GET", "/models", None, "")
        try:
            return [str(m["id"]) for m in data["data"]]
        except (KeyError, TypeError) as exc:
            raise LLMError(
                f"unexpected /models answer from {self.base_url}: {_short(data)}"
            ) from exc

    # --- transport --------------------------------------------------------------------------

    def _post(self, path: str, body: dict[str, Any], model: str) -> Any:
        return self._request("POST", path, body, model)

    def _request(self, method: str, path: str, body: dict[str, Any] | None, model: str) -> Any:
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(  # noqa: S310 - scheme checked in __init__
            self.base_url + path, data=payload, headers=self._headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:  # noqa: S310
                raw = response.read(_MAX_BODY + 1)
        except urllib.error.HTTPError as exc:
            detail = exc.read(2000).decode("utf-8", "replace")
            if exc.code == 404 and model:
                raise LLMError(
                    f"the server does not know model {model!r} (HTTP 404). With Ollama run "
                    f"`ollama pull {model}`; otherwise check DIAGSYS_LLM_MODEL / "
                    f"DIAGSYS_EMBEDDING_MODEL. Server said: {_short(detail)}"
                ) from exc
            raise LLMError(f"model server answered HTTP {exc.code}: {_short(detail)}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(
                f"model server {self.base_url} is not reachable ({exc.reason}). Start Ollama or "
                "llama-server, or fix DIAGSYS_LLM_BASE_URL (see docs/modele.md)"
            ) from exc
        except TimeoutError as exc:
            raise LLMError(
                f"model server did not answer within {self.timeout_s:g} s "
                "(raise DIAGSYS_LLM_TIMEOUT_S for slow CPU models)"
            ) from exc
        except OSError as exc:
            raise LLMError(f"connection to the model server failed: {exc}") from exc
        if len(raw) > _MAX_BODY:
            raise LLMError("model server answer is too large")
        try:
            return json.loads(raw)
        except ValueError as exc:
            raise LLMError(
                f"model server answered something that is not JSON: {_short(raw)}"
            ) from exc


def _short(value: Any, limit: int = 300) -> str:
    text = value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "..."
