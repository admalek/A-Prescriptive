import socket

import pytest
from stub_llm import StubLLM

from diagsys.contracts import ChatMessage
from diagsys.llm import LLMError, OpenAICompatClient


@pytest.fixture
def stub():
    with StubLLM() as server:
        yield server


def client(stub, **kw):
    return OpenAICompatClient(stub.url, "bielik-test", embedding_model="e5-test", **kw)


def test_chat_sends_messages_and_returns_the_text(stub):
    out = client(stub).chat(
        [ChatMessage(role="system", content="s"), ChatMessage(role="user", content="u")],
        temperature=0.0,
        max_tokens=50,
    )
    assert out == "Odpowiedź testowa [1]."
    method, path, body = stub.requests[-1]
    assert (method, path) == ("POST", "/v1/chat/completions")
    assert (
        body["model"] == "bielik-test" and body["temperature"] == 0.0 and body["max_tokens"] == 50
    )
    assert body["messages"] == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ]
    assert body["stream"] is False


def test_embed_keeps_order_and_batches(stub):
    texts = [f"{'x' * i}" for i in range(1, 71)]
    vectors = client(stub).embed(texts)
    assert [v[0] for v in vectors] == [float(i) for i in range(1, 71)]  # restored to input order
    posts = [r for r in stub.requests if r[1].endswith("/embeddings")]
    assert [len(r[2]["input"]) for r in posts] == [32, 32, 6]
    assert posts[0][2]["model"] == "e5-test"
    assert client(stub).embed([]) == []


def test_models_lists_what_the_server_offers(stub):
    assert client(stub).models() == ["bielik-test", "e5-test"]


def test_errors_say_what_to_do(stub):
    c = client(stub)
    stub.fail_with = (404, "model not found")
    with pytest.raises(LLMError, match="ollama pull bielik-test"):
        c.chat([ChatMessage(role="user", content="hi")])
    stub.fail_with = (500, "boom")
    with pytest.raises(LLMError, match="HTTP 500"):
        c.chat([ChatMessage(role="user", content="hi")])
    stub.fail_with = None
    stub.raw_reply = b"<html>not json</html>"
    with pytest.raises(LLMError, match="not JSON"):
        c.chat([ChatMessage(role="user", content="hi")])
    stub.raw_reply = b'{"choices": []}'
    with pytest.raises(LLMError, match="unexpected chat answer"):
        c.chat([ChatMessage(role="user", content="hi")])
    stub.raw_reply = b'{"choices": [{"message": {"content": "  "}}]}'
    with pytest.raises(LLMError, match="empty answer"):
        c.chat([ChatMessage(role="user", content="hi")])
    stub.raw_reply = b'{"data": [{"index": 0, "embedding": [1.0]}]}'
    with pytest.raises(LLMError, match="asked for 2 embeddings"):
        c.embed(["a", "b"])
    with pytest.raises(LLMError, match="at least one message"):
        c.chat([])


def test_unreachable_server_and_bad_urls():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]  # nothing listens here once the socket is closed
    dead = OpenAICompatClient(f"http://127.0.0.1:{port}/v1", "m", timeout_s=2)
    with pytest.raises(LLMError, match="not reachable"):
        dead.chat([ChatMessage(role="user", content="hi")])
    for bad in ("file:///etc/passwd", "localhost:11434", "ftp://x/y", ""):
        with pytest.raises(LLMError, match="http"):
            OpenAICompatClient(bad, "m")
