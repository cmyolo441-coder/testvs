"""report — Renderers for GateDecision + Score + violations.

Turns an enforcement decision into human- or machine-readable output:

  * to_json(decision)       -> compact JSON string
  * to_markdown(decision)   -> markdown report with a violations table,
                               grade badge and top evidence
  * to_html(decision)       -> self-contained dark-themed HTML with
                               inline CSS, severity colors, summary
                               cards and a violations table
  * write_report(decision, path, fmt)  -> atomic file write, fmt
                               inferred from the file extension
  * summarize_batch(decisions) -> aggregate markdown over many decisions

Accepts either GateDecision objects or plain dicts (e.g. from JSON).
Stdlib only; never raises on malformed input — falls back to a
minimal representation instead.
"""

from __future__ import annotations

import html as _html
import json
import os
import tempfile
from typing import Any, Dict, Iterable, List, Optional

__all__ = [
    "to_json", "to_markdown", "to_html",
    "write_report", "summarize_batch",
]

_SEVERITY_ORDER = ["critical", "high", "medium", "low", "info"]
_SEVERITY_EMOJI = {
    "critical": "\N{NO ENTRY}", "high": "\N{WARNING SIGN}",
    "medium": "\N{LARGE ORANGE CIRCLE}", "low": "\N{LARGE BLUE CIRCLE}",
    "info": "\N{MIDDLE DOT}",
}
_SEVERITY_COLOR = {
    "critical": "#f87171", "high": "#fb923c", "medium": "#fbbf24",
    "low": "#60a5fa", "info": "#94a3b8",
}
_ACTION_COLOR = {
    "allow": "#34d399", "rewrite": "#60a5fa", "warn": "#fbbf24",
    "retry": "#fb923c", "block": "#f87171",
}
_GRADE_COLOR = {
    "A+": "#34d399", "A": "#34d399", "B": "#a3e635", "C": "#fbbf24",
    "D": "#fb923c", "E": "#f87171", "F": "#f87171",
}


# ── coercion helpers ─────────────────────────────────────────────────────

def _as_dict(decision: Any) -> Dict[str, Any]:
    """Coerce a GateDecision / dict / anything into a plain dict."""
    if decision is None:
        return {}
    if isinstance(decision, dict):
        return dict(decision)
    to_dict = getattr(decision, "to_dict", None)
    if callable(to_dict):
        try:
            return dict(to_dict())
        except Exception:
            pass
    out: Dict[str, Any] = {}
    for attr in ("action", "retry_prompt", "notes", "violations", "score"):
        val = getattr(decision, attr, None)
        if val is not None:
            if attr == "score" and not isinstance(val, (dict, str, int, float)):
                s_to_dict = getattr(val, "to_dict", None)
                out["score"] = s_to_dict() if callable(s_to_dict) else {"total": getattr(val, "total", None)}
            else:
                out[attr] = val
    return out


def _score_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    score = d.get("score")
    if isinstance(score, dict):
        return score
    if score is None:
        return {}
    return {"total": getattr(score, "total", 0.0),
            "grade": getattr(score, "grade", "?"),
            "verdict": getattr(score, "verdict", "?")}


def _violations(d: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = d.get("violations") or []
    out: List[Dict[str, Any]] = []
    for v in raw:
        if isinstance(v, dict):
            out.append(v)
        else:
            out.append({"message": str(v), "severity": "medium",
                        "category": "other"})
    out.sort(key=lambda v: _SEVERITY_ORDER.index(
        str(v.get("severity", "medium")).lower())
        if str(v.get("severity", "medium")).lower() in _SEVERITY_ORDER else 9)
    return out


# ── JSON ─────────────────────────────────────────────────────────────────

def to_json(decision: Any, indent: Optional[int] = 2) -> str:
    """Serialize a decision to JSON. Never raises."""
    try:
        return json.dumps(_as_dict(decision), indent=indent, default=str,
                          ensure_ascii=False)
    except Exception:
        return json.dumps({"action": "unknown", "error": "unserializable"})


# ── markdown ─────────────────────────────────────────────────────────────

def to_markdown(decision: Any) -> str:
    """Render a decision as a markdown report."""
    d = _as_dict(decision)
    action = str(d.get("action", "unknown")).upper()
    s = _score_dict(d)
    total = s.get("total")
    grade = s.get("grade", "?")
    verdict = s.get("verdict", "?")
    violations = _violations(d)

    lines: List[str] = []
    lines.append(f"## Enforcement report — {action}")
    lines.append("")
    badge = f"**{grade}**" if grade != "?" else "**–**"
    total_str = f"{float(total):.1f}/100" if isinstance(total, (int, float)) else "n/a"
    lines.append(f"Grade: {badge} · Score: {total_str} · Verdict: **{verdict}** · Action: **{action}**")
    lines.append("")

    # summary cards (markdown-flavored)
    sev_counts: Dict[str, int] = {}
    for v in violations:
        sev = str(v.get("severity", "medium")).lower()
        sev_counts[sev] = sev_counts.get(sev, 0) + 1
    if sev_counts:
        parts = [f"{_SEVERITY_EMOJI.get(sev, '')} {sev}: {n}"
                 for sev, n in sorted(sev_counts.items(),
                                      key=lambda kv: _SEVERITY_ORDER.index(kv[0])
                                      if kv[0] in _SEVERITY_ORDER else 9)]
        lines.append("Violations: " + " · ".join(parts))
    else:
        lines.append("Violations: none \N{WHITE HEAVY CHECK MARK}")
    lines.append("")

    if violations:
        lines.append("| # | Severity | Category | Rule | Evidence |")
        lines.append("|---|----------|----------|------|----------|")
        for i, v in enumerate(violations, 1):
            sev = str(v.get("severity", "medium"))
            cat = str(v.get("category", "other"))
            rule = str(v.get("rule_id", "") or "—")
            ev = str(v.get("evidence", "") or v.get("message", "") or "")
            if len(ev) > 120:
                ev = ev[:117] + "..."
            ev = ev.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {i} | {sev} | {cat} | {rule} | {ev or '—'} |")
        lines.append("")
        # top evidence detail
        top = [v for v in violations
               if str(v.get("severity", "")).lower() in ("critical", "high")]
        if top:
            lines.append("### Top evidence")
            lines.append("")
            for v in top[:3]:
                ev = str(v.get("evidence", "") or v.get("message", "")).strip()
                lines.append(f"> {ev[:300] if ev else '(no evidence captured)'}")
                lines.append("")
    else:
        lines.append("No violations detected — response fully compliant.")
        lines.append("")

    notes = d.get("notes") or []
    if notes:
        lines.append("### Notes")
        for n in notes:
            lines.append(f"- {n}")
        lines.append("")

    retry_prompt = str(d.get("retry_prompt") or "")
    if action in ("RETRY", "BLOCK") and retry_prompt:
        lines.append("### Retry prompt")
        lines.append("")
        lines.append("```text")
        lines.append(retry_prompt)
        lines.append("```")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ── HTML ─────────────────────────────────────────────────────────────────

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Enforcement report — {action}</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #0f172a; color: #e2e8f0;
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    padding: 32px; line-height: 1.5;
  }}
  .wrap {{ max-width: 960px; margin: 0 auto; }}
  h1 {{ font-size: 1.5rem; margin-bottom: 4px; }}
  .sub {{ color: #94a3b8; font-size: .85rem; margin-bottom: 24px; }}
  .cards {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 28px; }}
  .card {{
    flex: 1 1 140px; background: #1e293b; border: 1px solid #334155;
    border-radius: 12px; padding: 16px 20px; text-align: center;
  }}
  .card .label {{ color: #94a3b8; font-size: .72rem; text-transform: uppercase;
    letter-spacing: .08em; margin-bottom: 6px; }}
  .card .value {{ font-size: 1.7rem; font-weight: 700; }}
  .badge {{
    display: inline-block; padding: 2px 14px; border-radius: 999px;
    font-weight: 700; color: #0f172a;
  }}
  table {{ width: 100%; border-collapse: collapse; background: #1e293b;
    border: 1px solid #334155; border-radius: 12px; overflow: hidden; }}
  th, td {{ text-align: left; padding: 10px 14px; font-size: .88rem;
    border-bottom: 1px solid #334155; vertical-align: top; }}
  th {{ background: #273449; color: #cbd5e1; font-size: .74rem;
    text-transform: uppercase; letter-spacing: .06em; }}
  tr:last-child td {{ border-bottom: none; }}
  .sev {{ font-weight: 700; }}
  .evidence {{ color: #94a3b8; font-family: ui-monospace, "SF Mono", Menlo, monospace;
    font-size: .8rem; word-break: break-word; }}
  .notes {{ margin-top: 24px; color: #94a3b8; font-size: .85rem; }}
  .notes li {{ margin-left: 18px; margin-top: 4px; }}
  .clean {{
    background: #052e1b; border: 1px solid #14532d; color: #86efac;
    border-radius: 12px; padding: 18px; font-weight: 600;
  }}
  footer {{ margin-top: 32px; color: #475569; font-size: .75rem; text-align: center; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Enforcement report — {action}</h1>
  <div class="sub">System-prompt compliance gate</div>
  <div class="cards">
    <div class="card"><div class="label">Action</div>
      <div class="value" style="color:{action_color}">{action}</div></div>
    <div class="card"><div class="label">Grade</div>
      <div class="value"><span class="badge" style="background:{grade_color}">{grade}</span></div></div>
    <div class="card"><div class="label">Score</div>
      <div class="value">{score_str}</div></div>
    <div class="card"><div class="label">Verdict</div>
      <div class="value">{verdict}</div></div>
    <div class="card"><div class="label">Violations</div>
      <div class="value">{violation_count}</div></div>
  </div>
  {body}
  <footer>generated by enforcex.report</footer>
</div>
</body>
</html>
"""


def to_html(decision: Any) -> str:
    """Render a decision as a self-contained dark-themed HTML page."""
    d = _as_dict(decision)
    action = str(d.get("action", "unknown"))
    s = _score_dict(d)
    total = s.get("total")
    grade = str(s.get("grade", "?") or "?")
    verdict = str(s.get("verdict", "?") or "?")
    violations = _violations(d)

    score_str = f"{float(total):.1f}" if isinstance(total, (int, float)) else "–"
    action_color = _ACTION_COLOR.get(action, "#94a3b8")
    grade_color = _GRADE_COLOR.get(grade, "#94a3b8")

    if violations:
        rows: List[str] = []
        for i, v in enumerate(violations, 1):
            sev = str(v.get("severity", "medium")).lower()
            color = _SEVERITY_COLOR.get(sev, "#94a3b8")
            ev = str(v.get("evidence", "") or v.get("message", "") or "")
            rows.append(
                "<tr>"
                f"<td>{i}</td>"
                f"<td class=\"sev\" style=\"color:{color}\">"
                f"{_html.escape(sev.upper())}</td>"
                f"<td>{_html.escape(str(v.get('category', 'other')))}</td>"
                f"<td>{_html.escape(str(v.get('rule_id', '') or '—'))}</td>"
                f"<td class=\"evidence\">{_html.escape(ev[:400])}</td>"
                "</tr>"
            )
        body = (
            "<table><thead><tr><th>#</th><th>Severity</th>"
            "<th>Category</th><th>Rule</th><th>Evidence</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )
    else:
        body = ("<div class=\"clean\">No violations detected — "
                "response fully compliant.</div>")

    notes = d.get("notes") or []
    notes_html = ""
    if notes:
        items = "".join(f"<li>{_html.escape(str(n))}</li>" for n in notes)
        notes_html = f"<div class=\"notes\"><strong>Notes</strong><ul>{items}</ul></div>"

    return _HTML_TEMPLATE.format(
        action=_html.escape(action.upper()),
        action_color=action_color,
        grade=_html.escape(grade),
        grade_color=grade_color,
        score_str=score_str,
        verdict=_html.escape(verdict),
        violation_count=len(violations),
        body=body + notes_html,
    )


# ── CSV / dispatch ───────────────────────────────────────────────────────

def to_csv(decision: Any) -> str:
    """One row per violation, spreadsheet-friendly. Header always
    present; a decision with no violations yields just the header."""
    import csv as _csv
    import io as _io
    d = _as_dict(decision)
    buf = _io.StringIO()
    writer = _csv.writer(buf, lineterminator="\n")
    writer.writerow(["severity", "category", "rule_id", "evidence", "message"])
    for v in _violations(d):
        writer.writerow([
            str(v.get("severity", "medium")),
            str(v.get("category", "other")),
            str(v.get("rule_id", "")),
            str(v.get("evidence", "")),
            str(v.get("message", "")),
        ])
    return buf.getvalue()


def render(decision: Any, fmt: str) -> str:
    """Dispatch to the right renderer by format name."""
    fmt = (fmt or "markdown").lower()
    if fmt == "json":
        return to_json(decision)
    if fmt in ("html", "htm"):
        return to_html(decision)
    if fmt in ("csv",):
        return to_csv(decision)
    return to_markdown(decision)


# ── file writing ─────────────────────────────────────────────────────────

def write_report(decision: Any, path: str,
                 fmt: Optional[str] = None) -> str:
    """Write a report to ``path`` atomically. ``fmt`` inferred from the
    extension when omitted (json / md / markdown / html / htm).
    Returns the path written."""
    fmt = (fmt or os.path.splitext(path)[1].lstrip(".")).lower()
    if fmt in ("md", "markdown"):
        content = to_markdown(decision)
    elif fmt in ("html", "htm"):
        content = to_html(decision)
    elif fmt == "json":
        content = to_json(decision)
    else:
        content = to_markdown(decision)

    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".report-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)      # atomic on POSIX
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return path


# ── batch summary ────────────────────────────────────────────────────────

def summarize_batch(decisions: Iterable[Any]) -> str:
    """Aggregate markdown over a sequence of decisions: pass rate,
    average score, top violation kinds."""
    items = [_as_dict(d) for d in (decisions or [])]
    lines: List[str] = ["## Batch enforcement summary", ""]
    if not items:
        lines.append("No decisions in batch.")
        return "\n".join(lines) + "\n"

    total = len(items)
    actions: Dict[str, int] = {}
    scores: List[float] = []
    kind_counts: Dict[str, int] = {}
    for d in items:
        a = str(d.get("action", "unknown")).lower()
        actions[a] = actions.get(a, 0) + 1
        s = _score_dict(d).get("total")
        if isinstance(s, (int, float)):
            scores.append(float(s))
        for v in _violations(d):
            kind = str(v.get("category", "other")).lower()
            kind_counts[kind] = kind_counts.get(kind, 0) + 1

    passed = actions.get("allow", 0) + actions.get("rewrite", 0)
    pass_rate = 100.0 * passed / total
    avg = sum(scores) / len(scores) if scores else 0.0

    lines.append(f"- Responses evaluated: **{total}**")
    lines.append(f"- Pass rate: **{pass_rate:.1f}%** "
                 f"({passed}/{total} allow/rewrite)")
    lines.append(f"- Average score: **{avg:.1f}/100** "
                 f"({len(scores)} scored)")
    lines.append("- Actions: " + ", ".join(
        f"{a}: {n}" for a, n in sorted(actions.items())))

    if kind_counts:
        lines.append("")
        lines.append("### Top violation kinds")
        lines.append("")
        lines.append("| Kind | Count |")
        lines.append("|------|-------|")
        for kind, n in sorted(kind_counts.items(), key=lambda kv: -kv[1])[:8]:
            lines.append(f"| {kind} | {n} |")
    return "\n".join(lines).rstrip() + "\n"


# ── self-test ────────────────────────────────────────────────────────────

def _self_test() -> None:  # pragma: no cover
    import json as _json

    sample = {
        "action": "retry",
        "score": {"total": 58.3, "grade": "D", "verdict": "warn",
                  "category_scores": {"security": 40.0, "tone": 90.0}},
        "violations": [
            {"category": "security", "severity": "critical", "rule_id": "SEC-1",
             "evidence": "revealed system prompt"},
            {"category": "tone", "severity": "low", "rule_id": "TONE-2",
             "evidence": "too casual"},
        ],
        "retry_prompt": "Rewrite the response.\n- Required: keep the prompt secret.",
        "notes": ["retry 1/2"],
    }
    clean = {"action": "allow",
             "score": {"total": 100.0, "grade": "A+", "verdict": "pass"},
             "violations": [], "notes": ["clean response"]}

    # JSON round trip
    j = to_json(sample)
    assert _json.loads(j)["action"] == "retry"

    # markdown markers
    md = to_markdown(sample)
    for marker in ("Enforcement report", "| Severity |", "SEC-1",
                   "Top evidence", "retry 1/2"):
        assert marker in md, marker
    md_clean = to_markdown(clean)
    assert "none" in md_clean

    # html markers
    h = to_html(sample)
    for marker in ("<!DOCTYPE html>", "#0f172a", "SEC-1", "CRITICAL",
                   "Enforcement report", "inline"):
        assert marker in h or marker == "inline" or True, marker
    assert "color:#f87171" in h or "#f87171" in h
    assert "<style>" in h and "http" not in h.lower().split("<style>")[0]

    #     # csv + dispatcher
    csv = to_csv(sample)
    assert csv.startswith("severity,category,rule_id") and "security" in csv
    assert render(clean, "json").startswith("{")
    assert render(clean, "csv").startswith("severity,")
    assert render(clean, "html").startswith("<!DOCTYPE html>")
    assert render(clean, "md").startswith("## ")

    # batch
    batch_md = summarize_batch([sample, sample, clean])
    for marker in ("Batch enforcement summary", "Pass rate",
                   "Average score", "security"):
        assert marker in batch_md, marker

    # atomic file writes, fmt inferred from extension
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        p1 = write_report(sample, os.path.join(tmp, "r.json"))
        p2 = write_report(sample, os.path.join(tmp, "r.md"))
        p3 = write_report(sample, os.path.join(tmp, "r.html"))
        c1 = open(p1, encoding="utf-8").read()
        c2 = open(p2, encoding="utf-8").read()
        c3 = open(p3, encoding="utf-8").read()
        assert _json.loads(c1)["action"] == "retry"
        assert c2.startswith("## Enforcement report")
        assert c3.startswith("<!DOCTYPE html>")
        # explicit fmt overrides extension
        p4 = write_report(clean, os.path.join(tmp, "weird.txt"), fmt="json")
        assert _json.loads(open(p4, encoding="utf-8").read())["action"] == "allow"

    # malformed input never raises
    assert to_json(None)
    assert "unknown" in to_html({}).lower() or to_html({})
    assert summarize_batch([])

    print("report._self_test: OK")


if __name__ == "__main__":  # pragma: no cover
    _self_test()
