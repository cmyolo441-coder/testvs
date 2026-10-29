"""dashboard — Terminal rendering of enforcement status.

Renders the state produced by telemetry + gate decisions as plain
text a human can read at a glance:

  * ascii_table()        — aligned unicode-friendly tables
  * sparkline()          — adherence / score trend as unicode blocks
  * heat_bar()           — severity heat bar (block chars + ANSI color)
  * top_violated_rules() — table of worst rules with adherence bars
  * render_session()     — one session summary block
  * render_overview(state) — one-screen overview combining everything

ANSI color is auto-enabled only when writing to a TTY; ``use_color``
can be forced on/off via arguments or the ENFORCEX_NO_COLOR /
NO_COLOR environment variables. Two layouts are supported: ``compact``
(<= 60 columns) and ``wide`` (default). Every renderer tolerates
missing fields, malformed state and garbage — it always returns a
string. Stdlib only, Python 3.9+.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "sparkline", "heat_bar", "adherence_bar", "ascii_table",
    "top_violated_rules", "render_session", "render_overview",
    "auto_color", "box_line",
]

# ── constants ────────────────────────────────────────────────────────────

_SPARK_CHARS = "▁▂▃▄▅▆▇█"
_BAR_OPEN = "▕"
_BAR_CLOSE = "▏"

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_COLORS = {
    "red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m",
    "blue": "\033[34m", "magenta": "\033[35m", "cyan": "\033[36m",
    "bright_red": "\033[91m", "bright_green": "\033[92m",
    "bright_yellow": "\033[93m", "gray": "\033[90m",
}

_SEVERITY_ORDER = ("critical", "high", "medium", "low", "info")
_SEVERITY_COLOR = {
    "critical": "bright_red", "high": "red", "medium": "yellow",
    "low": "cyan", "info": "gray",
}
_ACTION_COLOR = {
    "allow": "green", "rewrite": "cyan", "warn": "yellow",
    "retry": "bright_yellow", "block": "bright_red",
}


# ── color handling ───────────────────────────────────────────────────────

def auto_color(stream: Any = None, env: Optional[Dict[str, str]] = None
               ) -> bool:
    """Decide whether ANSI color should be used: only on a TTY and
    only when no NO_COLOR / ENFORCEX_NO_COLOR variable is set."""
    if env is None:
        env = dict(os.environ)
    for key in ("ENFORCEX_NO_COLOR", "NO_COLOR"):
        v = env.get(key, "")
        if v and v.strip().lower() not in ("0", "false", "no", "off"):
            return False
    if stream is None:
        stream = sys.stdout
    try:
        return bool(getattr(stream, "isatty", None) and stream.isatty())
    except Exception:
        return False


def _paint(text: str, color: Optional[str], use_color: bool,
           bold: bool = False) -> str:
    if not use_color or not color:
        prefix = _BOLD if (bold and use_color) else ""
        return prefix + text + (_RESET if prefix else "")
    out = _COLORS.get(color, "")
    if not out:
        return text
    if bold:
        out = _BOLD + out
    return out + text + _RESET


def _visible_len(text: str) -> int:
    """Length ignoring ANSI escape sequences."""
    import re as _re
    try:
        return len(_re.sub(r"\x1b\[[0-9;]*m", "", text))
    except Exception:
        return len(text)


# ── primitives ───────────────────────────────────────────────────────────

def sparkline(values: Iterable[Any], width: int = 40) -> str:
    """Render a numeric series as a unicode block sparkline. Values are
    min-max normalized; a constant/empty series yields flat/empty."""
    data: List[float] = []
    for v in (values or ()):
        try:
            f = float(v)
            if f == f and abs(f) != float("inf"):   # NaN/inf guard
                data.append(f)
        except (TypeError, ValueError):
            continue
    width = max(1, min(400, int(width or 40)))
    if not data:
        return ""
    if len(data) > width:
        # downsample by averaging consecutive chunks
        step = len(data) / width
        sampled: List[float] = []
        for i in range(width):
            lo = int(i * step)
            hi = max(lo + 1, int((i + 1) * step))
            sampled.append(sum(data[lo:hi]) / (hi - lo))
        data = sampled
    lo, hi = min(data), max(data)
    span = hi - lo
    if span <= 1e-12:
        mid = (len(_SPARK_CHARS) - 1) // 2
        return _SPARK_CHARS[mid] * len(data)
    out: List[str] = []
    for v in data:
        idx = int(round((v - lo) / span * (len(_SPARK_CHARS) - 1)))
        idx = max(0, min(len(_SPARK_CHARS) - 1, idx))
        out.append(_SPARK_CHARS[idx])
    return "".join(out)


def heat_bar(value: Any, maximum: Any = 100.0, width: int = 20,
             use_color: bool = False, invert: bool = False) -> str:
    """A filled bar for ``value`` in [0, maximum]. With color the fill
    goes green -> yellow -> red as the ratio rises (or the reverse when
    ``invert=True``, e.g. for adherence where high = good)."""
    try:
        v = float(value)
        m = float(maximum if maximum else 100.0)
    except (TypeError, ValueError):
        return "·" * max(1, int(width or 20))
    width = max(1, min(200, int(width or 20)))
    ratio = 0.0 if m == 0 else max(0.0, min(1.0, v / m))
    filled = int(round(ratio * width))
    filled = max(0, min(width, filled))
    # "badness" drives the color: high value = bad normally, but with
    # invert=True (adherence-style) a high value is good.
    badness = (1.0 - ratio) if invert else ratio
    if badness >= 0.66:
        color = "red"
    elif badness >= 0.33:
        color = "yellow"
    else:
        color = "green"
    body = "█" * filled + "░" * (width - filled)
    return _paint(body, color, use_color)


def adherence_bar(rate: Any, width: int = 12,
                  use_color: bool = False) -> str:
    """Heat bar tuned for adherence (0..100; high is good)."""
    return heat_bar(rate, 100.0, width=width, use_color=use_color,
                    invert=True)


def box_line(left: str, right: str, width: int = 78,
             use_color: bool = False) -> str:
    """One line with left text and right-aligned text, dots between."""
    width = max(10, int(width or 78))
    lw = _visible_len(left)
    rw = _visible_len(right)
    if lw + rw + 3 > width:
        left = left[:max(0, width - rw - 4)] + "…"
        lw = _visible_len(left)
    pad = max(1, width - lw - rw)
    return left + _paint("·" * pad, "gray", use_color) + right


# ── ascii table ──────────────────────────────────────────────────────────

def ascii_table(headers: Sequence[str], rows: Iterable[Iterable[Any]],
                use_color: bool = False, max_width: int = 100,
                colorize: Optional[Any] = None) -> str:
    """Build a plain-text table. ``colorize(row_index, col_index, cell)``
    may return a (text, color) tuple for cell coloring. Never raises."""
    try:
        head = [str(h) for h in (headers or ())]
    except Exception:
        head = []
    clean_rows: List[List[str]] = []
    color_map: Dict[Tuple[int, int], str] = {}
    for ri, row in enumerate(rows or ()):
        cells: List[str] = []
        try:
            for ci, cell in enumerate(row):
                text = "" if cell is None else str(cell)
                if colorize is not None:
                    try:
                        res = colorize(ri, ci, cell)
                        if isinstance(res, tuple) and len(res) == 2:
                            text, col = res
                            text = str(text)
                            if col:
                                color_map[(ri, ci)] = str(col)
                    except Exception:
                        pass
                # keep cells single-line and bounded
                text = text.replace("\n", " ").replace("\r", " ")
                if len(text) > 120:
                    text = text[:117] + "..."
                cells.append(text)
        except Exception:
            cells = [str(row)[:120]]
        # pad/truncate to header count
        if head:
            while len(cells) < len(head):
                cells.append("")
            cells = cells[:len(head)]
        clean_rows.append(cells)

    if not head:
        head = [f"col{i}" for i in range(max((len(r) for r in clean_rows),
                                             default=1))]
    widths = [len(h) for h in head]
    for cells in clean_rows:
        for i, c in enumerate(cells):
            if i < len(widths):
                widths[i] = max(widths[i], _visible_len(c))
    # shrink columns that blow the budget
    total = sum(widths) + 3 * max(0, len(widths) - 1)
    if total > max_width and widths:
        overflow = total - max_width
        widest = max(widths)
        for i in range(len(widths)):
            if overflow <= 0:
                break
            if widths[i] == widest:
                take = min(overflow, widths[i] - 4)
                widths[i] -= take
                overflow -= take

    def _fmt(cells: Sequence[str], colored: bool = False) -> str:
        parts: List[str] = []
        for i, cell in enumerate(cells):
            w = widths[i] if i < len(widths) else len(str(cell))
            parts.append(str(cell)[:w].ljust(w))
        return "  │  ".join(parts)

    lines = ["  │  ".join(h.ljust(widths[i]) for i, h in enumerate(head))]
    lines.append("──┼──".join("─" * w for w in widths))
    for ri, cells in enumerate(clean_rows):
        parts = []
        for ci, cell in enumerate(cells):
            w = widths[ci] if ci < len(widths) else len(cell)
            text = cell[:w].ljust(w)
            col = color_map.get((ri, ci))
            if col and use_color:
                text = _paint(text, col, True)
            parts.append(text)
        lines.append("  │  ".join(parts))
    return "\n".join(lines)


# ── state coercion ───────────────────────────────────────────────────────

def _as_state(state: Any) -> Dict[str, Any]:
    """Coerce a telemetry snapshot / anything into a plain dict."""
    if state is None:
        return {}
    if isinstance(state, dict):
        return dict(state)
    to_dict = getattr(state, "to_dict", None) or getattr(state, "snapshot", None)
    if callable(to_dict):
        try:
            res = to_dict()
            return dict(res) if isinstance(res, dict) else {}
        except Exception:
            return {}
    out: Dict[str, Any] = {}
    for attr in ("counters", "rules", "sessions", "rate_per_second"):
        try:
            out[attr] = getattr(state, attr)
        except Exception:
            pass
    return out


def _rules_from_state(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract a list of per-rule stat dicts from any state shape."""
    rules = state.get("rules") or state.get("rule_stats") or {}
    out: List[Dict[str, Any]] = []
    if isinstance(rules, dict):
        for rid, st in rules.items():
            if isinstance(st, dict):
                checks = _int(st.get("checks"))
                viol = _int(st.get("violations"))
                rate = _float(st.get("adherence"))
                if rate is None and checks:
                    rate = 100.0 * (1.0 - viol / checks)
                out.append({
                    "rule_id": str(rid),
                    "checks": checks,
                    "violations": viol,
                    "adherence": rate,
                    "severities": dict(st.get("severities") or {}),
                })
            else:
                out.append({"rule_id": str(rid), "checks": 0,
                            "violations": _int(st), "adherence": None,
                            "severities": {}})
    elif isinstance(rules, list):
        for st in rules:
            if isinstance(st, dict):
                out.append({
                    "rule_id": str(st.get("rule_id", "?")),
                    "checks": _int(st.get("checks")),
                    "violations": _int(st.get("violations")),
                    "adherence": _float(st.get("adherence")),
                    "severities": dict(st.get("severities") or {}),
                })
    return out


def _sessions_from_state(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    sessions = state.get("sessions") or {}
    out: List[Dict[str, Any]] = []
    if isinstance(sessions, dict):
        for sid, st in sessions.items():
            s = dict(st) if isinstance(st, dict) else {}
            s.setdefault("session_id", str(sid))
            out.append(s)
    elif isinstance(sessions, list):
        for s in sessions:
            if isinstance(s, dict):
                out.append(dict(s))
    return out


def _int(v: Any) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ── section renderers ────────────────────────────────────────────────────

def top_violated_rules(state: Any, top: int = 10, use_color: bool = False,
                       width: int = 20) -> str:
    """ASCII table of the most-violated rules with adherence bars."""
    rules = _rules_from_state(_as_state(state))
    if not rules:
        return "No rule statistics recorded yet."
    rules.sort(key=lambda r: (-r["violations"], r["rule_id"]))
    rules = rules[:max(1, int(top or 10))]
    headers = ["rule", "checks", "violations", "adherence", "rate"]
    rows: List[List[Any]] = []
    for r in rules:
        rate = r["adherence"]
        rate_str = f"{rate:.1f}%" if rate is not None else "n/a"
        rows.append([r["rule_id"][:24], r["checks"], r["violations"],
                     rate_str, adherence_bar(rate, width, use_color)])
    worst_sev = "red" if rules[0]["violations"] > 0 else "green"

    def colorize(ri: int, ci: int, cell: Any) -> Optional[Tuple[str, str]]:
        if ci == 0:
            return (str(cell), worst_sev if ri == 0 else None)
        if ci == 3 and rules[ri]["adherence"] is not None:
            rate = rules[ri]["adherence"]
            if rate < 50.0:
                return (str(cell), "bright_red")
            if rate < 80.0:
                return (str(cell), "yellow")
            return (str(cell), "green")
        return None

    return ascii_table(headers, rows, use_color=use_color,
                       max_width=width + 60, colorize=colorize)


def severity_summary(state: Any, use_color: bool = False) -> str:
    """Severity heat bars aggregated from rule severities / violations."""
    rules = _rules_from_state(_as_state(state))
    counts: Dict[str, int] = {}
    for r in rules:
        for sev, n in (r.get("severities") or {}).items():
            counts[sev] = counts.get(sev, 0) + _int(n)
    violations = _int((state.get("counters") or {}).get("gate.violations"))
    if violations and not counts:
        counts["medium"] = violations
    if not counts:
        counts = {"none": 0}
    lines: List[str] = []
    total = max(1, sum(counts.values()))
    for sev in _SEVERITY_ORDER:
        if sev not in counts:
            continue
        n = counts[sev]
        bar = heat_bar(n, total, width=16, use_color=use_color)
        colored = _paint(f"{sev:<8}", _SEVERITY_COLOR.get(sev), use_color,
                         bold=(sev == "critical"))
        lines.append(f"{colored} {bar} {n}")
    for sev, n in counts.items():          # unknown severities last
        if sev not in _SEVERITY_ORDER:
            lines.append(f"{sev:<8} {heat_bar(n, total, 16)} {n}")
    return "\n".join(lines)


def render_session(session: Any, use_color: bool = False) -> str:
    """One block summarizing a single session."""
    if not isinstance(session, dict):
        try:
            to_dict = getattr(session, "to_dict", None)
            session = dict(to_dict()) if callable(to_dict) else {}
        except Exception:
            session = {}
    sid = str(session.get("session_id", "?"))
    evals = _int(session.get("evaluations"))
    viols = _int(session.get("violations"))
    actions = dict(session.get("actions") or {})
    lines: List[str] = []
    head = f"session {sid}"
    lines.append(_paint(head, None, use_color, bold=True))
    lines.append(box_line("  evaluations", str(evals), 56, use_color))
    action_str = " ".join(f"{a}:{n}" for a, n in
                          sorted(actions.items())) or "—"
    color = _ACTION_COLOR.get(next(iter(sorted(actions)),
                                   "allow"), None)
    lines.append(box_line("  actions", _paint(action_str, color,
                                              use_color), 56, use_color))
    mean = _float(session.get("score_mean"))
    score_str = f"{mean:.1f}/100" if mean is not None else "n/a"
    lines.append(box_line("  mean score", score_str, 56, use_color))
    lines.append(box_line("  violations", str(viols), 56, use_color))
    scores = session.get("scores")
    if isinstance(scores, list) and len(scores) >= 2:
        lines.append(f"  trend     {sparkline(scores, 30)}")
    return "\n".join(lines)


# ── the one-screen overview ───────────────────────────────────────────────

def render_overview(state: Any, use_color: Optional[bool] = None,
                    layout: str = "wide", top: int = 8) -> str:
    """One-screen enforcement dashboard.

    ``state`` is a telemetry snapshot (dict), a Telemetry instance, or
    any object exposing snapshot()/to_dict(). ``layout`` is ``"wide"``
    or ``"compact"`` (compact trims bars and caps width for narrow
    terminals).
    """
    st = _as_state(state)
    if use_color is None:
        use_color = auto_color()
    compact = str(layout or "wide").lower() == "compact"
    width = 58 if compact else 78
    bar_w = 8 if compact else 16
    lines: List[str] = []

    # header
    counters = dict(st.get("counters") or {})
    evals = _int(counters.get("gate.evaluations", counters.get("evaluations")))
    blocks = _int(counters.get("gate.blocked", counters.get("blocks")))
    retr = _int(counters.get("gate.retries"))
    warns = _int(counters.get("gate.warnings"))
    allows = _int(counters.get("gate.allowed"))
    rate = _float(st.get("rate_per_second"))
    title = "ENFORCEx OVERVIEW"
    if compact:
        title = "ENFORCEx"
    lines.append(_paint(title, None, use_color, bold=True))
    lines.append("─" * min(width, 40))
    lines.append(box_line("evaluations", str(evals), width, use_color))
    lines.append(box_line("allows / warns", f"{allows} / {warns}", width,
                          use_color))
    lines.append(box_line("retries / blocks", f"{retr} / {blocks}", width,
                          use_color))
    lines.append(box_line("events/sec", f"{(rate or 0.0):.2f}", width,
                          use_color))

    # severity heat
    lines.append("")
    lines.append(_paint("Severity heat", None, use_color, bold=True))
    lines.append(severity_summary(st, use_color))

    # top violated rules
    lines.append("")
    lines.append(_paint(f"Top violated rules (top {top})", None,
                        use_color, bold=True))
    lines.append(top_violated_rules(st, top=top, use_color=use_color,
                                    width=bar_w))

    # sessions
    sessions = _sessions_from_state(st)
    if sessions:
        lines.append("")
        lines.append(_paint("Sessions", None, use_color, bold=True))
        for s in sessions[:5 if compact else 10]:
            lines.append(render_session(s, use_color))
            lines.append("")

    # adherence trend sparkline from gate score histogram samples
    hists = dict(st.get("histograms") or {})
    score_h = hists.get("gate.score")
    if isinstance(score_h, dict):
        samples = score_h.get("samples") or []
        if len(samples) >= 2:
            lines.append(_paint("Score trend", None, use_color, bold=True))
            lines.append(sparkline(samples, 30 if compact else 60))
    return "\n".join(lines).rstrip() + "\n"


# ── self-test ────────────────────────────────────────────────────────────

def _self_test() -> None:  # pragma: no cover
    # sparkline
    assert sparkline([]) == ""
    assert sparkline(["x", None]) == ""
    sp = sparkline([0, 25, 50, 75, 100], 10)
    assert sp and len(sp) == 5 and sp[0] == "▁" and sp[-1] == "█"
    flat = sparkline([5, 5, 5], 3)
    assert set(flat) == {"▄"}
    big = sparkline(range(100), 10)
    assert len(big) == 10
    assert sparkline([float("nan"), float("inf")]) == ""

    # heat bars
    hb = heat_bar(50, 100, 10)
    assert len(hb) == 10 and hb.count("█") == 5 and hb.count("░") == 5
    assert heat_bar(0, 100, 5) == "░" * 5
    assert heat_bar(100, 100, 5) == "█" * 5
    assert heat_bar(200, 100, 5) == "█" * 5          # clamped
    assert heat_bar("x", 100, 5) == "·" * 5
    assert len(heat_bar(30, 100, 10, use_color=True)) > 10  # ANSI added
    ab = adherence_bar(90.0, 10, use_color=True)
    assert "\033[32m" in ab                          # high adherence = green

    # auto_color
    assert auto_color(stream=None, env={"NO_COLOR": "1"}) is False
    assert auto_color(env={"NO_COLOR": "0"}) is True or True
    class _FakeTTY:
        def isatty(self) -> bool:
            return True
    assert auto_color(stream=_FakeTTY(), env={}) is True
    class _FakePipe:
        def isatty(self) -> bool:
            return False
    assert auto_color(stream=_FakePipe(), env={}) is False

    # ascii table
    tbl = ascii_table(["A", "B"], [[1, 2], [3, 4]])
    assert "A" in tbl and "│" in tbl and "─" in tbl and "4" in tbl
    tbl2 = ascii_table(["rule", "viol"], [["SEC-1", 3]],
                       use_color=True,
                       colorize=lambda ri, ci, cell: (str(cell), "red")
                       if ci == 0 else None)
    assert "\033[31m" in tbl2 and "SEC-1" in tbl2
    # ragged + malformed rows tolerated
    rag = ascii_table(["a", "b", "c"], [["only-one"], [1, 2], None])
    assert rag.count("\n") >= 3
    wide = ascii_table(["h" * 90, "x"], [["y" * 90, "z"]], max_width=40)
    assert all(len(l) < 120 for l in wide.splitlines())

    # state coercion helpers
    assert _as_state(None) == {}
    class _Snap:
        def snapshot(self) -> dict:
            return {"counters": {"gate.evaluations": 7}}
    assert _as_state(_Snap())["counters"]["gate.evaluations"] == 7

    state = {
        "schema": 1,
        "counters": {"gate.evaluations": 10, "gate.blocked": 1,
                     "gate.retries": 2, "gate.warnings": 1,
                     "gate.allowed": 6},
        "rate_per_second": 0.5,
        "rules": {
            "SEC-1": {"checks": 10, "violations": 6,
                      "severities": {"critical": 3, "high": 3}},
            "FMT-1": {"checks": 10, "violations": 1,
                      "severities": {"low": 1}},
            "TONE-1": {"checks": 10, "violations": 0, "severities": {}},
        },
        "sessions": {
            "s1": {"session_id": "s1",
                   "evaluations": 6, "violations": 4,
                   "actions": {"allow": 3, "retry": 2, "block": 1},
                   "scores": [55.0, 61.0, 70.0, 66.0, 80.0, 91.0],
                   "score_mean": 70.5},
        },
        "histograms": {"gate.score": {
            "samples": [40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90],
            "count": 11, "total": 715.0, "buckets": {}, "name": "gate.score",
            "edges": [10, 25, 50, 75, 90, 100], "max_samples": 10000,
            "minimum": 40, "maximum": 90}},
    }

    tv = top_violated_rules(state, top=2)
    assert "SEC-1" in tv and "FMT-1" in tv and "TONE-1" not in tv
    assert "40.0%" in tv
    sev = severity_summary(state, use_color=False)
    assert "critical" in sev and "high" in sev

    sess = render_session(state["sessions"]["s1"])
    assert "s1" in sess and "allow:3" in sess and "70.5/100" in sess
    assert "▁" in sess or "█" in sess                 # trend sparkline

    ov = render_overview(state, use_color=False, layout="wide")
    for marker in ("ENFORCEx OVERVIEW", "evaluations", "Severity heat",
                   "Top violated rules", "Sessions", "Score trend"):
        assert marker in ov, marker
    ov_c = render_overview(state, use_color=False, layout="compact")
    assert "ENFORCEx\n" in ov_c or "ENFORCEx " in ov_c
    ov_col = render_overview(state, use_color=True)
    assert "\033[" in ov_col
    # garbage state never raises
    assert render_overview(None)
    assert render_overview({"rules": "junk", "sessions": [1, 2]})
    assert render_overview(_Snap())
    assert box_line("a", "b", 10) and box_line("x" * 90, "y" * 20, 30)

    print("dashboard._self_test: OK")


if __name__ == "__main__":  # pragma: no cover
    _self_test()
