
import sys as _sys
from pathlib import Path as _Path
_ROOT_DIR = str(_Path(__file__).resolve().parent.parent)
if _ROOT_DIR not in _sys.path:
    _sys.path.insert(0, _ROOT_DIR)

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


from fullagent import client, config
from fullagent.config import Model, Provider


def test_configured_providers():
    assert list(config.PROVIDERS) == ["kios", "gwarden", "sambanova"]
    assert {m.id for m in config.MODELS} == {
        "deepseek-v4.1-flash-1017", "gemini-4-argon",
        "deepseek-v4.1-flash-free", "minimax-m3.1-flash-free",
        "MiniMax-M3.1-Flash"}
    assert config.DEFAULT_MODEL_ID == "deepseek-v4.1-flash-1017"
    gwarden = config.PROVIDERS["gwarden"]
    assert gwarden.base_url == "https://gwarden.su/v1"
    assert gwarden.api_key, "gwarden must have embedded zero-config key"
    sambanova = config.PROVIDERS["sambanova"]
    assert sambanova.base_url == "https://api.sambanova.ai/v1"
    assert sambanova.api_key, "sambanova must have embedded zero-config key"


def test_kios_ships_embedded_api_key():
    provider = config.PROVIDERS["kios"]
    assert provider.api_key, "kios must have embedded zero-config key"
    headers = client._base_headers(provider, stream=False)
    assert headers["Authorization"].startswith("Bearer ")
    assert headers["Content-Type"] == "application/json"


def test_keyless_provider_still_requires_credentials():
    with pytest.raises(client.APIError, match="no API key configured"):
        client._check_api_key(
            Provider(key="kios", name="Kios API",
                     base_url="https://kiosapi.com/v1", api_key="",
                     color="#8be9fd"),
            Model("deepseek-v4.1-flash-free", "kios",
                  "DeepSeek V4.1 Flash Free"),
        )


def test_dead_model_error_is_not_retried(monkeypatch):
    """A retired model id must fail immediately, not burn 2+4+8s of backoff.

    Kios answers unknown/retired model ids with 503 model_not_found. That
    is permanent: retrying only made the UI feel frozen for ~14s."""
    attempts = []

    def fake_once(*_args, **_kwargs):
        attempts.append(True)
        raise client.APIError(
            "model_not_found: No available channel for model", status=503)

    monkeypatch.setattr(client, "_chat_stream_once", fake_once)
    with pytest.raises(client.APIError, match="model_not_found"):
        client._chat_stream_with_retries(
            "https://example.test", {}, {}, None, None, None, None, None, 5,
        )
    assert len(attempts) == 1


def test_retry_after_parses_seconds_and_ignores_invalid_values():
    assert client._parse_retry_after("3.5") == 3.5
    assert client._parse_retry_after("not a delay") is None
    assert client._parse_retry_after(None) is None


def test_streaming_429_retries_until_success(monkeypatch):
    attempts = []
    statuses = []
    expected = object()

    def fake_once(*_args, **_kwargs):
        attempts.append(True)
        if len(attempts) < 3:
            raise client.APIError("rate limited", status=429, retry_after=0)
        return expected

    monkeypatch.setattr(client, "_chat_stream_once", fake_once)
    result = client._chat_stream_with_retries(
        "https://example.test", {}, {}, None, None, None, None, None, 1,
        on_status=statuses.append,
    )

    assert result is expected
    assert len(attempts) == 3
    assert len(statuses) == 2
    assert all("429" in status for status in statuses)


def test_blocking_429_retries_and_closes_each_response(monkeypatch):
    class Response:
        def __init__(self, status, headers=None, body="rate limited", data=None):
            self.status_code = status
            self.headers = headers or {}
            self.text = body
            self._data = data
            self.closed = False

        def json(self):
            return self._data

        def close(self):
            self.closed = True

    first = Response(429, {"Retry-After": "0"})
    second = Response(200, data={"ok": True})

    class Session:
        def __init__(self):
            self.responses = [first, second]
            self.calls = 0

        def post(self, *_args, **_kwargs):
            self.calls += 1
            return self.responses.pop(0)

    session = Session()
    waits = []
    monkeypatch.setattr(client, "_http", lambda: session)
    monkeypatch.setattr(client, "sleep_cancellable",
                        lambda delay, _cancel: waits.append(delay))

    result = client._post_blocking(
        "https://example.test", {}, {}, timeout=1)

    assert result == {"ok": True}
    assert session.calls == 2
    assert first.closed and second.closed
    assert waits == [0.0]


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
