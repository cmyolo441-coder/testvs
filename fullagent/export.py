"""`/export` command — export the current conversation to a file.

Claude Code CLI-style: dump the in-memory transcript
(``agent.messages``, a list of ``{"role", "content", ...}`` dicts) to
markdown or a self-contained HTML page. Tool calls are rendered from
the assistant messages' ``tool_calls`` entries, paired with the
following ``role: "tool"`` messages by ``tool_call_id``.

API-key-shaped strings (``sk-...`` / ``oc_sk_...``) are redacted with a
simple regex — exports never leak secrets that appear in content.

Public API:
    - :func:`register` -- attach an :class:`Exporter` on ``agent.exporter``
      (duck-typed, no imports of ``.agent``/``.tui`` at module level).
    - :func:`handle_export` -- TUI entry point: ``/export [md|markdown|html]
      [path]``. Default format markdown; default path
      ``~/fullagent-export-<session_id>-<timestamp>.md``.

``python3 -m fullagent.export`` runs the built-in self-test.
"""

from __future__ import annotations

import html as _html
import json
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

USAGE = (
    "usage:\n"
    "  /export                  export as markdown (default path)\n"
    "  /export html             export as a single-file HTML page\n"
    "  /export markdown <path>  export to a specific path"
)

MD_ARGS_LIMIT = 2000   # max chars for pretty-printed tool-call args
MD_RESULT_LIMIT = 4000  # max chars for a tool result

# Redact API-key-shaped strings before anything hits disk.
# NOTE: no leading \b — Python's \b uses Unicode word chars, so a key glued
# to CJK text ("密钥sk-...") had no word boundary and leaked unredacted.
_KEY_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9])oc_sk_[A-Za-z0-9_.\-]{8,}"),
    re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_.\-]{8,}"),
)
REDACTED = "[REDACTED]"


def scrub_keys(text: str) -> str:
    """Replace ``sk-...`` / ``oc_sk_...`` values with ``[REDACTED]``."""
    for pat in _KEY_PATTERNS:
        text = pat.sub(REDACTED, str(text))
    return text


def _truncate(text: str, limit: int) -> str:
    text = str(text or "")
    if len(text) > limit:
        return text[:limit] + f"\n…[truncated, {len(text)} chars total]"
    return text


def _content_text(content: Any) -> str:
    """Coerce a message's content (str or list of content parts) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                for key in ("text", "content"):
                    val = part.get(key)
                    if isinstance(val, str):
                        parts.append(val)
                        break
        return "\n".join(parts)
    return str(content)


def _pretty_args(fn: Dict[str, Any]) -> str:
    raw = fn.get("arguments", "")
    if isinstance(raw, dict):
        parsed: Any = raw
    elif isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            parsed = raw
    else:
        parsed = raw
    if isinstance(parsed, (dict, list)):
        return json.dumps(parsed, ensure_ascii=False, indent=2)
    return str(parsed)


# ---------------------------------------------------------------------------
# Exporter
# ---------------------------------------------------------------------------

class Exporter:
    """Render ``agent.messages`` as markdown or HTML."""

    def __init__(self, session_id: Optional[str] = None,
                 model_id: Optional[str] = None) -> None:
        self.session_id = session_id or "default"
        self.model_id = model_id or "?"

    # -- transcript access ------------------------------------------------
    def collect(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Pair tool calls with their results for rendering.

        Returns a list of sections:
        ``{"kind": "user"|"assistant"|"system"|"tool", ...}``.
        """
        # Map tool_call_id -> tool result content, first pass.
        tool_results: Dict[str, str] = {}
        for m in messages:
            if m.get("role") == "tool":
                tid = str(m.get("tool_call_id", "") or "")
                if tid:
                    tool_results[tid] = _content_text(m.get("content"))

        sections: List[Dict[str, Any]] = []
        for m in messages:
            role = str(m.get("role", "") or "")
            if role == "tool":
                continue  # folded into the matching tool call below
            text = _content_text(m.get("content"))
            calls: List[Dict[str, Any]] = []
            for tc in (m.get("tool_calls") or []):
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                name = str(fn.get("name", "") or "")
                tid = str(tc.get("id", "") or "")
                calls.append({
                    "name": name,
                    "args": _pretty_args(fn) if isinstance(fn, dict) else "",
                    "result": tool_results.get(tid, ""),
                })
            if role not in ("user", "assistant", "system", "developer"):
                role = "system"
            sections.append({"role": role, "text": text, "tool_calls": calls})
        return sections

    # -- markdown ----------------------------------------------------------
    def to_markdown(self, messages: List[Dict[str, Any]]) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            "# Conversation export",
            "",
            f"- **session:** `{self.session_id}`",
            f"- **model:** `{self.model_id}`",
            f"- **exported:** {now}",
            "",
        ]
        sections = self.collect(messages)
        for sec in sections:
            role = sec["role"]
            heading = {"user": "## User", "assistant": "## Assistant"}.get(
                role, "## System")
            lines.append(heading)
            lines.append("")
            text = sec["text"].strip()
            lines.append(text if text else "_(empty)_")
            lines.append("")
            for tc in sec["tool_calls"]:
                name = tc["name"] or "unknown"
                lines.append(f"### 🔧 Tool call: `{name}`")
                lines.append("")
                lines.append("**Arguments:**")
                lines.append("```json")
                lines.append(_truncate(tc["args"], MD_ARGS_LIMIT))
                lines.append("```")
                lines.append("")
                lines.append("**Result:**")
                lines.append("```")
                result = tc["result"]
                lines.append(_truncate(result, MD_RESULT_LIMIT)
                             if result else "_(no result recorded)_")
                lines.append("```")
                lines.append("")
        return scrub_keys("\n".join(lines).rstrip() + "\n")

    # -- html --------------------------------------------------------------
    def to_html(self, messages: List[Dict[str, Any]]) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        esc = _html.escape

        def block(label: str, cls: str, text: str) -> str:
            return (
                f'<section class="msg {cls}">'
                f'<h2 class="role">{esc(label)}</h2>'
                f'<pre class="body">{esc(text) if text.strip() else "(empty)"}'
                f"</pre></section>"
            )

        parts: List[str] = [block(
            {"user": "User", "assistant": "Assistant"}.get(
                s["role"], "System"),
            {"user": "user", "assistant": "asst"}.get(s["role"], "sys"),
            s["text"],
        ) for s in self.collect(messages)]

        # Interleave tool-call cards right after their assistant section.
        rendered: List[str] = []
        for sec, part in zip(self.collect(messages), parts):
            rendered.append(part)
            for tc in sec["tool_calls"]:
                name = esc(tc["name"] or "unknown")
                args = esc(_truncate(tc["args"], MD_ARGS_LIMIT))
                result = esc(_truncate(tc["result"], MD_RESULT_LIMIT)
                             ) if tc["result"] else "(no result recorded)"
                rendered.append(
                    f'<section class="tool">'
                    f'<h3 class="toolname">🔧 Tool call: <code>{name}</code></h3>'
                    f'<p class="toollabel">Arguments</p>'
                    f'<pre class="args">{args}</pre>'
                    f'<p class="toollabel">Result</p>'
                    f'<pre class="result">{result}</pre>'
                    f"</section>"
                )

        body = "\n".join(rendered) or '<p class="empty">No messages.</p>'
        page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Conversation export — {esc(self.session_id)}</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif;
       background: #f6f7f9; color: #1a1a1a; margin: 0; padding: 24px; }}
main {{ max-width: 860px; margin: 0 auto; }}
header {{ background: #fff; border: 1px solid #e2e5ea; border-radius: 8px;
          padding: 16px 20px; margin-bottom: 16px; }}
header h1 {{ font-size: 1.2rem; margin: 0 0 8px; }}
header dl {{ display: grid; grid-template-columns: 90px 1fr; gap: 4px 8px;
             margin: 0; font-size: 0.9rem; }}
header dt {{ color: #666; }}
header dd {{ margin: 0; font-family: monospace; }}
.msg {{ background: #fff; border: 1px solid #e2e5ea; border-radius: 8px;
        padding: 12px 16px; margin-bottom: 12px; }}
.role {{ font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.06em;
         margin: 0 0 8px; }}
.msg.user .role {{ color: #1a73e8; }}
.msg.asst .role {{ color: #188038; }}
.msg.sys .role {{ color: #888; }}
.msg .body {{ margin: 0; white-space: pre-wrap; word-break: break-word;
              font-size: 0.95rem; font-family: inherit; }}
.tool {{ background: #fffbe6; border: 1px solid #ecd98b; border-radius: 8px;
         padding: 12px 16px; margin: -4px 0 12px 24px; }}
.toolname {{ font-size: 0.9rem; margin: 0 0 8px; }}
.toollabel {{ font-size: 0.75rem; color: #666; margin: 8px 0 4px; }}
.tool pre {{ background: #1e1e1e; color: #dcdcdc; border-radius: 6px;
             padding: 10px 12px; overflow-x: auto; font-size: 0.8rem; }}
.empty {{ color: #888; }}
</style>
</head>
<body>
<main>
<header>
<h1>Conversation export</h1>
<dl>
<dt>session</dt><dd>{esc(self.session_id)}</dd>
<dt>model</dt><dd>{esc(self.model_id)}</dd>
<dt>exported</dt><dd>{esc(now)}</dd>
</dl>
</header>
{body}
</main>
</body>
</html>
"""
        return scrub_keys(page)

    # -- file output --------------------------------------------------------
    def default_path(self, fmt: str) -> str:
        suffix = "html" if fmt == "html" else "md"
        ts = time.strftime("%Y%m%d-%H%M%S")
        return os.path.join(
            os.path.expanduser("~"),
            f"fullagent-export-{self.session_id}-{ts}.{suffix}")

    def export_to_path(self, messages: List[Dict[str, Any]],
                       path: str, fmt: str = "markdown") -> str:
        fmt = (fmt or "markdown").lower()
        text = self.to_html(messages) if fmt == "html" \
            else self.to_markdown(messages)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path


def register(agent: Any) -> None:
    """Attach an :class:`Exporter` on ``agent.exporter`` (duck-typed)."""
    session_id = getattr(agent, "session_id", None)
    model = getattr(agent, "model", None)
    model_id = getattr(model, "id", None) if model is not None else None
    agent.exporter = Exporter(session_id=session_id, model_id=model_id)


# ---------------------------------------------------------------------------
# TUI entry point
# ---------------------------------------------------------------------------

_FORMATS = {"md", "markdown", "html"}


def handle_export(ui: Any, arg: str) -> None:
    """Handle ``/export [markdown|html] [path]``; prints via the TUI."""
    agent = getattr(ui, "agent", None)
    if agent is None:
        ui.print_error("/export needs the TUI host agent")
        return

    text = (arg or "").strip()
    tokens = text.split() if text else []
    fmt = "markdown"
    path = ""
    if tokens:
        first = tokens[0].lower()
        if first in _FORMATS:
            fmt = "html" if first == "html" else "markdown"
            path = " ".join(tokens[1:]).strip()
        else:
            path = text

    model = getattr(agent, "model", None)
    exporter = Exporter(
        session_id=getattr(agent, "session_id", None) or "default",
        model_id=getattr(model, "id", "?") if model is not None else "?")
    if not path:
        path = exporter.default_path(fmt)
    else:
        path = os.path.expanduser(path)

    messages = getattr(agent, "messages", None) or []
    if not messages:
        ui.print_error("nothing to export — the conversation is empty")
        return

    try:
        out = exporter.export_to_path(messages, path, fmt)
    except OSError as e:
        ui.print_error(f"cannot write export: {e}")
        return
    ui.print_info(f"✓ conversation exported → {out}")


# ---------------------------------------------------------------------------
# self-test: `python3 -m fullagent.export` → PASS
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    class _FakeModel:
        id = "test-model-v1"

    class _FakeAgent:
        def __init__(self):
            self.session_id = "abc123"
            self.model = _FakeModel()
            # user turn; assistant turn with a tool call + its result
            self.messages = [
                {"role": "system",
                 "content": "You are a helpful agent."},
                {"role": "user",
                 "content": "hello — use key sk-fakeREALkey1234567890abcdef"},
                {"role": "assistant",
                 "content": "Let me read that file.",
                 "tool_calls": [{
                     "id": "call_1",
                     "function": {
                         "name": "read",
                         "arguments": json.dumps(
                             {"path": "/tmp/x.py", "limit": 50}),
                     },
                 }],
                 "reasoning_content": ""},
                {"role": "tool",
                 "tool_call_id": "call_1",
                 "content": "print('hi')  # token oc_sk_FAKEkey9876543210zz"},
                {"role": "assistant",
                 "content": "Done, I read the file."},
            ]

    class _FakeUI:
        def __init__(self, agent):
            self.agent = agent
            self.infos: List[str] = []
            self.errors: List[str] = []

        def print_info(self, text: str, color: str | None = None) -> None:
            self.infos.append(text)

        def print_error(self, text: str) -> None:
            self.errors.append(text)

    agent = _FakeAgent()
    register(agent)
    assert isinstance(agent.exporter, Exporter)
    assert agent.exporter.session_id == "abc123"

    exp = Exporter(session_id="abc123", model_id="test-model-v1")

    # -- markdown body checks ------------------------------------------
    md = exp.to_markdown(agent.messages)
    assert "# Conversation export" in md, md[:200]
    assert "abc123" in md and "test-model-v1" in md
    assert "## User" in md and "## Assistant" in md and "## System" in md
    assert "Tool call: `read`" in md
    assert '"/tmp/x.py"' in md
    assert "print('hi')" in md
    # redaction of both key shapes
    assert "sk-fakeREALkey1234567890abcdef" not in md
    assert "oc_sk_FAKEkey9876543210zz" not in md
    assert md.count(REDACTED) >= 2, md

    # -- html body checks ----------------------------------------------
    ht = exp.to_html(agent.messages)
    assert "<!DOCTYPE html>" in ht and "</html>" in ht
    assert "abc123" in ht and "test-model-v1" in ht
    assert "<h2 class=\"role\">User</h2>" in ht
    assert "Tool call: <code>read</code>" in ht
    assert "sk-fakeREALkey1234567890abcdef" not in ht
    assert "oc_sk_FAKEkey9876543210zz" not in ht
    assert REDACTED in ht
    # inline CSS only — no external deps
    assert "<link" not in ht and "http" not in ht.replace("http://www.w3.org", "")

    # -- collect() pairing ---------------------------------------------
    secs = exp.collect(agent.messages)
    roles = [s["role"] for s in secs]
    assert roles == ["system", "user", "assistant", "assistant"], roles
    asst = secs[2]
    assert len(asst["tool_calls"]) == 1
    tc = asst["tool_calls"][0]
    assert tc["name"] == "read"
    assert tc["result"] == "print('hi')  # token oc_sk_FAKEkey9876543210zz"

    # -- truncation -----------------------------------------------------
    long_args = json.dumps({"blob": "x" * 5000})
    md2 = exp.to_markdown([{
        "role": "assistant", "content": "t",
        "tool_calls": [{"id": "c", "function":
                        {"name": "big", "arguments": long_args}}]}])
    assert "truncated" in md2
    long_result = "r" * 9000
    md3 = exp.to_markdown([
        {"role": "assistant", "content": "t",
         "tool_calls": [{"id": "c", "function":
                         {"name": "big", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c", "content": long_result},
    ])
    assert "truncated" in md3

    # -- handle_export via fake UI --------------------------------------
    ui = _FakeUI(agent)
    tmp = tempfile.mkdtemp(prefix="export_selftest_")
    md_path = os.path.join(tmp, "out.md")
    handle_export(ui, f"markdown {md_path}")
    assert os.path.exists(md_path), ui.errors
    assert any("out.md" in i for i in ui.infos), ui.infos
    assert "## Assistant" in open(md_path).read()

    html_path = os.path.join(tmp, "out.html")
    handle_export(ui, f"html {html_path}")
    assert os.path.exists(html_path)
    assert "<!DOCTYPE html>" in open(html_path).read()

    # empty conversation → error, no crash
    empty_agent = _FakeAgent()
    empty_agent.messages = []
    ui2 = _FakeUI(empty_agent)
    handle_export(ui2, f"markdown {os.path.join(tmp, 'no.md')}")
    assert ui2.errors and "empty" in ui2.errors[0].lower()

    # bad path → print_error, no raise
    ui3 = _FakeUI(agent)
    handle_export(ui3, "markdown /nonexistent-dir-xyz-123/out.md")
    assert ui3.errors and "cannot write" in ui3.errors[0].lower()

    # default path pattern (string only — don't write into $HOME)
    dp = exp.default_path("markdown")
    assert dp.endswith(".md") and "fullagent-export-abc123-" in dp, dp
    assert exp.default_path("html").endswith(".html")

    # list-style content parts coerce to text
    assert _content_text([{"type": "text", "text": "hi"}]) == "hi"
    assert _content_text(None) == ""

    print("export self-test PASSED")
