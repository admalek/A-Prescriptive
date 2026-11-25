"""A tiny OpenAI-compatible server for tests: records requests, answers from a script."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class StubLLM:
    def __init__(self):
        self.requests: list[tuple[str, str, dict | None]] = []
        self.chat_reply = "Odpowiedź testowa [1]."
        self.fail_with: tuple[int, str] | None = None
        self.raw_reply: bytes | None = None
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence
                pass

            def _send(self, code, payload):
                raw = stub.raw_reply if stub.raw_reply is not None else json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _handle(self, method):
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length)) if length else None
                stub.requests.append((method, self.path, body))
                if stub.fail_with:
                    return self._send(stub.fail_with[0], {"error": stub.fail_with[1]})
                if self.path.endswith("/models"):
                    return self._send(200, {"data": [{"id": "bielik-test"}, {"id": "e5-test"}]})
                if self.path.endswith("/chat/completions"):
                    msg = {"role": "assistant", "content": stub.chat_reply}
                    return self._send(200, {"choices": [{"message": msg}]})
                if self.path.endswith("/embeddings"):
                    texts = body["input"]
                    rows = [
                        {"index": i, "embedding": [float(len(t)), 1.0, float(i)]}
                        for i, t in enumerate(texts)
                    ]
                    rows.reverse()  # a server may answer out of order
                    return self._send(200, {"data": rows})
                return self._send(404, {"error": "not found"})

            def do_GET(self):
                self._handle("GET")

            def do_POST(self):
                self._handle("POST")

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_port}/v1"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()
