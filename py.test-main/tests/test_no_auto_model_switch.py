"""A provider error must never move a turn to a different model."""

import sys as _sys
from pathlib import Path as _Path
_ROOT_DIR = str(_Path(__file__).resolve().parent.parent)
if _ROOT_DIR not in _sys.path:
    _sys.path.insert(0, _ROOT_DIR)

from importlib import import_module
from types import SimpleNamespace

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


from fullagent import config
from fullagent.client import APIError

agent_module = import_module("fullagent.agent")


def test_provider_outage_does_not_auto_switch_or_retry_another_model(
        monkeypatch):
    agent_module._resolve_lazy(
        "APIError", "chat_blocking", "chat_stream",
        "is_context_overflow", "schemas_tokens")

    model_ids = [model.id for model in config.MODELS]
    selected = config.DEFAULT_MODEL_ID
    assert selected in model_ids
    assert all(m.provider in ("kios", "gwarden") for m in config.MODELS)
    alternate = "removed/provider-model"
    cfg = SimpleNamespace(
        model_id=selected,
        effort=config.DEFAULT_EFFORT,
        prompt="main",
        extra={"failover_model": alternate},
    )
    agent = agent_module.Agent.__new__(agent_module.Agent)
    agent.cfg = cfg
    agent.health = {"model_errors": {}, "failovers": 0}
    logged_events = []
    agent.log = SimpleNamespace(
        append=lambda *args, **kwargs: logged_events.append((args, kwargs)))
    agent.cassette = None
    agent._tool_schemas = lambda: []
    agent._prune_for_model = lambda reserved_tokens=0: []
    agent._overflow_shrink = lambda *args, **kwargs: None
    agent._flight_prewarmer = SimpleNamespace(launch=lambda _agent: None)
    agent._push_status = lambda _status: None
    # The real Agent always has Mastermind; this bare-bones test stub must
    # supply the wire gate that _complete() calls on every request.
    agent.mastermind = SimpleNamespace(
        gate=SimpleNamespace(seal_wire=lambda *_a, **_k: None))
    calls = []

    def fail_with_outage(*args, **kwargs):
        calls.append((args, kwargs))
        raise APIError("provider unavailable", status=503)

    monkeypatch.setattr(agent_module, "chat_stream", fail_with_outage,
                        raising=False)

    with pytest.raises(APIError, match="provider unavailable"):
        agent._complete(lambda _text: None, lambda _text: None,
                        lambda _status: None)

    assert len(calls) == 1
    assert cfg.model_id == selected
    assert agent.health["model_errors"] == {selected: 1}
    assert agent.health["failovers"] == 0
    assert logged_events == []


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
