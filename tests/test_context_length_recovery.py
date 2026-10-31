"""NVIDIA NIM context_length 400 — parse, recalibrate, retry.

NVIDIA rejects an over-window request with a generic body::

    {"message": "Invalid request", "type": "Bad Request", "code": 400,
     "details": {"type": "context_length", "limit": 1048576,
                 "actual": 1048933}}

The client used to discard ``details``, so this looked like an ordinary
invalid request: errorrecovery pruned history and retried with the same
oversized max_tokens, and the turn died. This file proves:

  A. the error message keeps the machine-readable details;
  B. the context-overflow parser recovers window + real input tokens
     (actual - max_tokens the request asked for);
  C. chat_stream re-clamps max_tokens from those numbers and retries
     instead of pruning and giving up;
  D. NVIDIA stream requests ask for usage (calibration needs it) and the
     correction learner drops stale contradictory samples.

Run:  python3 -m pytest tests/test_context_length_recovery.py -q
"""

import sys as _sys
from pathlib import Path as _Path
_ROOT_DIR = str(_Path(__file__).resolve().parent.parent)
if _ROOT_DIR not in _sys.path:
    _sys.path.insert(0, _ROOT_DIR)

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
    Effort,
    Model,
    Provider,
    _extract_error_message,
    _parse_overflow,
    build_payload,
    chat_stream,
    is_context_overflow,
)

NVIDIA_BODY = json.dumps({
    "message": "Invalid request",
    "type": "Bad Request",
    "code": 400,
    "details": {"type": "context_length",
                "limit": 1_048_576, "actual": 1_048_933},
})

WINDOW = 1_000_000
ACTUAL_INPUT = 20_000


def _model(context_window=1_048_576, provider="nvidia"):
    return Model(id="ctx-test", provider=provider, label="ctx-test",
                 context_window=context_window)


def _effort():
    return Effort("high", "HIGH", "#50fa7b", 1_048_576, 0.8, "high", "d")


# ==========================================================================
# A. error-body extraction keeps the context_length details
# ==========================================================================

class TestErrorBody:
    def test_nvidia_details_survive_extraction(self):
        msg = _extract_error_message(NVIDIA_BODY, 400)
        assert "context_length" in msg
        assert "1048933" in msg
        assert is_context_overflow(msg)

    def test_generic_invalid_request_is_not_overflow(self):
        msg = _extract_error_message(
            '{"message":"Invalid request","type":"Bad Request","code":400}',
            400)
        assert not is_context_overflow(msg)
        assert _parse_overflow(msg, 1000) is None

    def test_parse_recovers_window_and_input(self):
        msg = _extract_error_message(NVIDIA_BODY, 400)
        info = _parse_overflow(msg, sent_max_tokens=1_031_010)
        assert info["window"] == 1_048_576
        assert info["total"] == 1_048_933
        assert info["input_tokens"] == 1_048_933 - 1_031_010

    def test_parse_without_sent_max_tokens_has_no_input_guess(self):
        msg = _extract_error_message(NVIDIA_BODY, 400)
        info = _parse_overflow(msg)
        assert info["window"] == 1_048_576
        assert "input_tokens" not in info


# ==========================================================================
# B. chat_stream re-clamps and retries against the real error shape
# ==========================================================================

class TestOverflowRetry:
    def test_chat_stream_reclamps_instead_of_pruning(self):
        seen = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(n))
                seen.append(payload)
                sent_max = payload.get("max_tokens", 0)
                actual = ACTUAL_INPUT + sent_max
                if actual > WINDOW:
                    body = json.dumps({
                        "message": "Invalid request", "type": "Bad Request",
                        "code": 400,
                        "details": {"type": "context_length",
                                    "limit": WINDOW, "actual": actual},
                    }).encode()
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
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
        # learn_context_window persists to disk; keep tests hermetic
        saved_save = _client._save_calibration
        _client._save_calibration = lambda: None
        try:
            port = srv.server_address[1]
            prov = Provider(key="t", name="t",
                            base_url=f"http://127.0.0.1:{port}",
                            api_key="dummy", color="red")
            msgs = [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "do the thing"},
            ]
            res = chat_stream(prov, _model(), _effort(), msgs, None,
                              timeout=15)
        finally:
            _client._save_calibration = saved_save
            srv.shutdown()
            srv.server_close()

        assert res.content == "ok"
        assert len(seen) == 2, "must retry exactly once after re-clamping"
        first, second = seen
        # history was NOT pruned by the invalid-request recovery path
        assert len(second["messages"]) == len(first["messages"]) == len(msgs)
        # max_tokens was re-clamped from the provider's real numbers
        assert second["max_tokens"] < first["max_tokens"]
        assert second["max_tokens"] <= WINDOW - ACTUAL_INPUT


# ==========================================================================
# C. streaming usage + calibration hardening
# ==========================================================================

class TestUsageAndCalibration:
    def test_nvidia_stream_requests_usage(self):
        msgs = [{"role": "user", "content": "hi"}]
        streamed = build_payload(_model(), _effort(), msgs, None, stream=True)
        assert streamed["stream_options"] == {"include_usage": True}
        blocked = build_payload(_model(), _effort(), msgs, None, stream=False)
        assert "stream_options" not in blocked

    def test_other_providers_are_unchanged(self):
        msgs = [{"role": "user", "content": "hi"}]
        p = build_payload(_model(provider="kios"), _effort(), msgs, None,
                          stream=True)
        assert "stream_options" not in p

    def test_contradictory_sample_resets_calibration(self):
        key = _client._cal_key("cal-test-model")
        saved = {k: list(v) for k, v in _client._correction_samples.items()}
        saved_save = _client._save_calibration
        saved_bump = _client._bump_calibration_epoch
        _client._save_calibration = lambda: None
        _client._bump_calibration_epoch = lambda: None
        try:
            _client._correction_samples[key] = [0.4, 0.4, 0.4, 0.4]
            # 0.83 is >1.5x the mean: the old regime's samples are stale
            _client.learn_token_correction(100, 83, "cal-test-model")
            assert _client._correction_samples[key] == [0.83]
            # a consistent sample keeps accumulating
            _client.learn_token_correction(100, 84, "cal-test-model")
            assert len(_client._correction_samples[key]) == 2
        finally:
            _client._correction_samples.clear()
            _client._correction_samples.update(saved)
            _client._save_calibration = saved_save
            _client._bump_calibration_epoch = saved_bump


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
