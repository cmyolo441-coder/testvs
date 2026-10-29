"""Model picker helpers for the /model command.

Plain-text model list with fuzzy search, resolution helpers, and a
per-turn model override. Imports only from .config (never .agent / .tui).

The per-turn override is consumed inside Agent.run_turn (see the wiring
snippet in the module docstring): a /model <query> --once style command
calls set_turn_model(agent, id); run_turn picks it up via
get_effective_model() and clears it after that turn.
"""

from __future__ import annotations

import difflib

from .config import DEFAULT_MODEL_ID, MODELS, Model, model_by_id

__all__ = [
    "list_models",
    "format_model_list",
    "resolve_model",
    "set_turn_model",
    "clear_turn_model",
    "get_effective_model",
]


def _fmt_context(n: int) -> str:
    """262144 -> '262k', 1_048_576 -> '1.0m'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}m"
    return f"{n // 1_000}k"


def list_models(filter_text: str = "") -> list[dict]:
    """Return [{id, label, provider, tag, context_window, supports_tools}].

    Empty filter -> all models. Otherwise substring match on id/label
    plus difflib fuzzy matches on id+label.
    """
    q = (filter_text or "").strip().lower()
    if not q:
        return [_as_dict(m) for m in MODELS]
    seen: set[str] = set()
    out: list[dict] = []
    for m in MODELS:
        hay = f"{m.id} {m.label}".lower()
        if q in hay:
            seen.add(m.id)
            out.append(_as_dict(m))
    for match in difflib.get_close_matches(
        q, [m.id for m in MODELS] + [m.label for m in MODELS],
        # 0.6: high-confidence fuzzy only. Below this, near-strings of
        # REMOVED models (e.g. "space-bunny-free" -> "deepseek-v4.1-flash-free"
        # at 0.457) would silently resolve to the wrong model instead of
        # surfacing the clean "unknown model" error.
        n=10, cutoff=0.6,
    ):
        m = resolve_model(match)
        if m is not None and m.id not in seen:
            seen.add(m.id)
            out.append(_as_dict(m))
    return out


def _as_dict(m: Model) -> dict:
    return {
        "id": m.id,
        "label": m.label,
        "provider": m.provider,
        "tag": m.tag,
        "context_window": m.context_window,
        "supports_tools": m.supports_tools,
    }


def format_model_list(models: list[dict]) -> str:
    """Aligned columns: label | id | provider | tag | context | tools."""
    rows = [
        (m["label"], m["id"], m["provider"],
         m["tag"] or "-", _fmt_context(m["context_window"]),
         "✓" if m["supports_tools"] else "✗")
        for m in models
    ]
    header = ("LABEL", "ID", "PROVIDER", "TAG", "CTX", "TOOLS")
    widths = [max(len(c) for c in col) for col in zip(header, *rows)] \
        if rows else [len(c) for c in header]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    out = [line]
    for r in rows:
        out.append("  ".join(c.ljust(w) for c, w in zip(r, widths)))
    return "\n".join(out)


def resolve_model(query: str) -> Model | None:
    """Resolve a user query to a Model: exact id -> label match -> fuzzy."""
    q = (query or "").strip()
    if not q:
        return None
    # 1. exact id
    m = model_by_id(q)
    if m is not None:
        return m
    ql = q.lower()
    # 2. exact label (case-insensitive)
    for m in MODELS:
        if m.label.lower() == ql:
            return m
    # 3. substring of id or label -> best difflib score among those
    sub = [m for m in MODELS if ql in m.id.lower() or ql in m.label.lower()]
    if sub:
        return max(
            sub,
            key=lambda m: max(
                difflib.SequenceMatcher(None, ql, m.id.lower()).ratio(),
                difflib.SequenceMatcher(None, ql, m.label.lower()).ratio(),
            ),
        )
    # 4. fuzzy best over id + label (see cutoff note in list_models)
    cand = difflib.get_close_matches(
        q, [m.id for m in MODELS] + [m.label for m in MODELS],
        n=1, cutoff=0.6,
    )
    if cand:
        return resolve_model(cand[0])
    return None


# ---------------------------------------------------------------------------
# Per-turn override. The agent reads this at the start of run_turn and clears
# it after the turn, so "/model <q> --once" style flows only affect one turn.
# ---------------------------------------------------------------------------

def set_turn_model(agent: object, model_id: str) -> bool:
    """Store a one-turn model override. Returns False if id unknown."""
    m = model_by_id(model_id)
    if m is None:
        return False
    agent.turn_model_override = m.id  # type: ignore[attr-defined]
    return True


def clear_turn_model(agent: object) -> None:
    agent.turn_model_override = None  # type: ignore[attr-defined]


def get_effective_model(agent: object) -> Model:
    """The override if set (and known), else agent.model."""
    override = getattr(agent, "turn_model_override", None)
    if override:
        m = model_by_id(str(override))
        if m is not None:
            return m
    return agent.model  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest() -> None:
    checks: list[tuple[str, bool]] = []

    # 1. list count > 0
    all_models = list_models()
    checks.append(("list count > 0", len(all_models) > 0))
    checks.append(("empty filter == all", list_models("") == all_models))

    # 2. exact id, label and fuzzy free-route name
    checks.append(("exact id", (resolve_model(DEFAULT_MODEL_ID) or Model("", "", "")).id == DEFAULT_MODEL_ID))
    checks.append(("label match", (resolve_model("DeepSeek V4.1 Flash 1017") or Model("", "", "")).id == DEFAULT_MODEL_ID))
    checks.append(("fuzzy 'deepseek'", (resolve_model("deepseek") or Model("", "", "")).id == DEFAULT_MODEL_ID))
    checks.append(("unknown -> None", resolve_model("no-such-model-zzz") is None))
    # filter finds every kios free route (gwarden route is not "-free" tagged)
    filt = list_models("free")
    n_free = sum(1 for m in MODELS if m.id.endswith("-free"))
    checks.append(("substring filter finds free models",
                   len(filt) == n_free
                   and all(m["id"].endswith("-free") for m in filt)))

    # 3. format contains columns
    txt = format_model_list(all_models)
    cols_ok = all(h in txt for h in ("LABEL", "ID", "PROVIDER", "TAG", "CTX", "TOOLS"))
    checks.append(("format columns", cols_ok))
    checks.append(("format shows 1.0m", "1.0m" in txt))
    checks.append(("format tools tick", "✓" in txt))

    # 4. override set / get / clear
    class FakeAgent:
        model = model_by_id(DEFAULT_MODEL_ID)

    a = FakeAgent()
    checks.append(("set known id -> True", set_turn_model(a, DEFAULT_MODEL_ID)))
    checks.append(("get override", get_effective_model(a).id == DEFAULT_MODEL_ID))
    checks.append(("set unknown id -> False", not set_turn_model(a, "nope-zzz")))
    clear_turn_model(a)
    checks.append(("clear -> base model", get_effective_model(a).id == DEFAULT_MODEL_ID))
    # stale override id falls back to base model
    a.turn_model_override = "deleted-model-zzz"
    checks.append(("stale override falls back", get_effective_model(a).id == DEFAULT_MODEL_ID))

    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(("ok   " if ok else "FAIL ") + name)
    print(f"{len(checks) - len(failed)}/{len(checks)} checks passed")
    if failed:
        raise SystemExit("SELFTEST FAILED: " + ", ".join(failed))
    print("PASS")


if __name__ == "__main__":
    _selftest()
