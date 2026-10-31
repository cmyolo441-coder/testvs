"""serde-alias regression: never send reasoning AND reasoning_content.

Serde-based OpenAI-compatible gateways alias ``reasoning`` to
``reasoning_content``. A message object that carries both spellings is
therefore parsed as the same field twice and the whole request fails
with::

    Bad Request: Failed to deserialize the JSON body into the target
    type: duplicate field `reasoning_content` at line 1 column N

Older FullAgent history wrote both keys (assistant_message set the
alias "for maximum provider compatibility"), so every later turn on such
a session was rejected. This file proves:

  A. assistant_message() stores the canonical key only.
  B. _sanitize_messages() heals legacy turns (both keys, alias-only,
     non-string alias) before the wire.
  C. build_payload()/chat_stream() send a body with no alias, even when
     the history came from an old session.
  D. the provider's duplicate-field error is recognised for recovery.

Run:  python3 -m pytest tests/test_reasoning_wire_alias.py -q
"""

import sys as _sys
from pathlib import Path as _Path
_ROOT_DIR = str(_Path(__file__).resolve().parent.parent)
if _ROOT_DIR not in _sys.path:
    _sys.path.insert(0, _ROOT_DIR)

import copy
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    import pytest
except ImportError:
    import importlib.util as _ilu
    _compat_spec = _ilu.spec_from_file_location(
        "pytest", str(_Path(__file__).resolve().parent / "_pytest_compat.py")
    )
    pytest = _ilu.module_from_spec(_compat_spec)
    _sys.modules["pytest"] = pytest
    _sys.modules["tests._pytest_compat"] = pytest
    _compat_spec.loader.exec_module(pytest)


import fullagent.client as _client
from fullagent.client import (
    APIError,
    _is_reasoning_content_error,
    _sanitize_messages,
    assistant_message,
    build_payload,
    chat_stream,
)
from fullagent.config import Effort, Model, Provider

DUPLICATE_ERROR = (
    "Bad Request: Failed to deserialize the JSON body into the target "
    "type: duplicate field `reasoning_content` at line 1 column 47670"
)


def _model():
    return _client.Model(id="mistralai/mistral-large-4-1t-a52b",
                         provider="nvidia", label="Mistral Large")


def _effort():
    return Effort("high", "HIGH", "#50fa7b", None, 0.6, "medium", "d")


def _legacy_history():
    return [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo",
         "reasoning_content": "thought", "reasoning": "thought"},
        {"role": "user", "content": "again"},
    ]


# ==========================================================================
# A. assistant_message() — canonical key only
# ==========================================================================

class TestAssistantMessage:
    def test_reasoning_uses_canonical_key_only(self):
        m = assistant_message("answer", [], "thought")
        assert m["reasoning_content"] == "thought"
        assert "reasoning" not in m

    def test_tool_turn_without_reasoning_has_no_alias(self):
        tc = {"id": "call_1", "type": "function",
              "function": {"name": "read", "arguments": "{}"}}
        m = assistant_message(None, [tc], "")
        assert m["reasoning_content"] == ""
        assert "reasoning" not in m


# ==========================================================================
# B. _sanitize_messages() — heal legacy sessions at the wire
# ==========================================================================

class TestSanitizeHealsAlias:
    def test_both_keys_keeps_canonical_drops_alias(self):
        msgs = _legacy_history()
        assert _sanitize_messages(msgs) is True
        assert msgs[1]["reasoning_content"] == "thought"
        assert "reasoning" not in msgs[1]

    def test_alias_only_migrates_to_canonical(self):
        msgs = [{"role": "user", "content": "hi"},
                {"role": "assistant", "content": "yo",
                 "reasoning": "only alias"}]
        assert _sanitize_messages(msgs) is True
        assert msgs[1]["reasoning_content"] == "only alias"
        assert "reasoning" not in msgs[1]

    def test_non_string_alias_is_coerced(self):
        msgs = [{"role": "user", "content": "hi"},
                {"role": "assistant", "content": "yo",
                 "reasoning": {"text": "x"}}]
        assert _sanitize_messages(msgs) is True
        assert isinstance(msgs[1]["reasoning_content"], str)
        assert "reasoning" not in msgs[1]

    def test_clean_history_untouched(self):
        msgs = [{"role": "user", "content": "hi"},
                {"role": "assistant", "content": "yo",
                 "reasoning_content": "thought"}]
        before = copy.deepcopy(msgs)
        assert _sanitize_messages(msgs) is False
        assert msgs == before


# ==========================================================================
# C. build_payload() — the real wire body
# ==========================================================================

class TestWirePayload:
    def test_legacy_history_never_sends_alias(self):
        msgs = _legacy_history()
        obj = build_payload(_model(), _effort(), msgs, None)
        asm = obj["messages"][1]
        assert "reasoning" not in asm
        assert asm["reasoning_content"] == "thought"
        json.dumps(obj)  # wire-serializable

    def test_built_turn_has_single_reasoning_field(self):
        msgs = [{"role": "user", "content": "hi"},
                assistant_message("yo", [], "thinking")]
        obj = build_payload(_model(), _effort(), msgs, None)
        wire = json.dumps(obj, ensure_ascii=False)
        asm = obj["messages"][1]
        assert "reasoning" not in asm
        # one object must never carry the field twice (serde duplicate)
        assert wire.count('"reasoning_content"') == 1

    def test_chat_stream_sends_canonical_body_for_legacy_history(self):
        seen = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                seen.append(self.rfile.read(n))
                body = b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n' \
                       b"data: [DONE]\n\n"
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            port = srv.server_address[1]
            prov = Provider(key="t", name="t",
                            base_url=f"http://127.0.0.1:{port}",
                            api_key="dummy", color="red")
            mod = Model(id="t", provider="t", label="t")
            eff = Effort("t", "T", "red", None, 0.0, None, "t")
            res = chat_stream(prov, mod, eff, _legacy_history(), None,
                              timeout=15)
        finally:
            srv.shutdown()
            srv.server_close()
        assert res.content == "ok"
        assert len(seen) == 1
        wire = json.loads(seen[0])
        asm = wire["messages"][1]
        assert "reasoning" not in asm
        assert asm["reasoning_content"] == "thought"


# ==========================================================================
# D. duplicate-field provider error is recognised for recovery
# ==========================================================================

class TestErrorDetection:
    def test_serde_duplicate_field_error(self):
        assert _is_reasoning_content_error(APIError(DUPLICATE_ERROR, status=400))

    def test_required_reasoning_error_still_detected(self):
        err = APIError("messages[2].reasoning_content is required for "
                       "thinking tool-call history", status=400)
        assert _is_reasoning_content_error(err)

    def test_unrelated_errors_not_detected(self):
        assert not _is_reasoning_content_error(
            APIError("duplicate field `model` at line 1 column 12", status=400))
        assert not _is_reasoning_content_error(
            APIError("invalid api key", status=401))


if __name__ == "__main__":
    import importlib.util as _ilu
    if "tests._pytest_compat" not in _sys.modules:
        _cs = _ilu.spec_from_file_location(
            "tests._pytest_compat", str(_Path(__file__).resolve().parent / "_pytest_compat.py")
        )
        _cm = _ilu.module_from_spec(_cs)
        _sys.modules["tests._pytest_compat"] = _cm
        _cs.loader.exec_module(_cm)
    raise SystemExit(_sys.modules["tests._pytest_compat"].run_module_tests(_sys.modules[__name__]))
