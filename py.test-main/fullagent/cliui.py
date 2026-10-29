"""Qoder-style CLI chrome for FullAgent.

Pure formatting helpers shared by the welcome screen and the always-on
prompt frame: a block-art logo, a "Tips for getting started" card, the
mode/skills HUD line, the model/agents status footer, and the live
parallel-subagent board.

Everything here is deterministic and side-effect free: callers pass in a
``colors`` mapping (``fullagent.tui.C``) and get back plain strings or
prompt_toolkit ``(style, text)`` fragment lists. That keeps the layout
testable headlessly — no TTY and no prompt_toolkit application needed.
"""

from __future__ import annotations

import unicodedata
from typing import Any, List, Sequence, Tuple

Fragment = Tuple[str, str]

# ---------------------------------------------------------------------------
# Brand mark — a 6-row "FA" monogram, same shape language as Qoder's logo.
# 17 columns wide so it fits next to a 40-column tips card in an 80-col TTY.
# ---------------------------------------------------------------------------

LOGO: Tuple[str, ...] = (
    " ███████╗ █████╗ ",
    " ██╔════╝██╔══██╗",
    " █████╗  ███████║",
    " ██╔══╝  ██╔══██║",
    " ██║     ██║  ██║",
    " ╚═╝     ╚═╝  ╚═╝",
)

LOGO_STYLES: Tuple[str, ...] = (
    "accent", "accent", "pink", "cyan", "cyan", "pink",
)

TIPS: Tuple[str, ...] = (
    "Use AGENTS.md files to customize behavior",
    "/help for more information",
    "Ask coding questions, edit code, run commands",
    "Be specific for the best results",
)

# ---------------------------------------------------------------------------
# Live subagent board
# ---------------------------------------------------------------------------

ACTIVE_STATES = ("queued", "running", "cancelling")

STATE_ICONS = {
    "queued": "○",
    "running": "◉",
    "cancelling": "◌",
    "done": "✓",
    "blocked": "!",
    "error": "×",
    "cancelled": "–",
}

STATE_COLOR_KEYS = {
    "queued": "dim",
    "running": "green",
    "cancelling": "yellow",
    "done": "cyan",
    "blocked": "orange",
    "error": "red",
    "cancelled": "dim",
}

# Short specialist chips, matching the way Qoder labels its agents
# ("Explore", "Code", "Test"…). Fall back to the title-cased role.
SHORT_ROLE = {
    "researcher": "Explore",
    "coder": "Code",
    "tester": "Test",
    "reviewer": "Review",
    "analyst": "Analyze",
    "architect": "Architect",
    "debugger": "Debug",
    "optimizer": "Optimize",
    "refactorer": "Refactor",
    "documenter": "Docs",
    "devops": "DevOps",
    "integrator": "Integrate",
    "planner": "Plan",
    "security": "Security",
}


def short_role(role: str) -> str:
    """One-word chip for a role name (never raises on unknown roles)."""
    role = str(role or "").strip().lower()
    if role in SHORT_ROLE:
        return SHORT_ROLE[role]
    return role.replace("_", " ").title() or "Agent"


# ---------------------------------------------------------------------------
# Width helpers (independent of fullagent.tui to avoid an import cycle)
# ---------------------------------------------------------------------------

_WIDE = ("W", "F")


def display_width(s: Any) -> int:
    """Terminal columns used by *s* (wide East-Asian glyphs count twice)."""
    total = 0
    for ch in str(s):
        total += 2 if unicodedata.east_asian_width(ch) in _WIDE else 1
    return total


def truncate(s: Any, max_cols: int) -> str:
    """Cut *s* to *max_cols* columns with an ellipsis; never split a glyph."""
    text = str(s)
    if max_cols <= 0:
        return ""
    if display_width(text) <= max_cols:
        return text
    out: List[str] = []
    used = 0
    for ch in text:
        cw = 2 if unicodedata.east_asian_width(ch) in _WIDE else 1
        if used + cw > max_cols - 1:
            break
        out.append(ch)
        used += cw
    return "".join(out) + "…"


def _pad(frags: Sequence[Fragment]) -> int:
    return sum(display_width(text) for _, text in frags)


def align(
    left: Sequence[Fragment],
    right: Sequence[Fragment],
    width: int,
    fill: str = " ",
) -> List[Fragment]:
    """Left + fill + right, padded to exactly *width* columns (never less).

    The right side wins when space runs out: the left side is dropped first
    so the informative trailing badge stays visible on narrow terminals.
    """
    left = list(left)
    right = list(right)
    width = max(1, int(width))
    if _pad(right) + 1 > width:
        # extreme narrow terminal: the badge itself must shrink, otherwise
        # prompt_toolkit wraps the row and the pinned frame corrupts
        room = max(0, width - 1)
        fitted_right: List[Fragment] = []
        used = 0
        for style, text in right:
            tw = display_width(text)
            if used + tw <= room:
                fitted_right.append((style, text))
                used += tw
                continue
            rest = room - used
            if rest > 0:
                fitted_right.append((style, truncate(text, rest)))
            break
        right = fitted_right
    if _pad(left) + _pad(right) + 1 > width:
        room = max(0, width - _pad(right) - 1)
        fitted: List[Fragment] = []
        used = 0
        for style, text in left:
            if style is None:
                continue
            tw = display_width(text)
            if used + tw <= room:
                fitted.append((style, text))
                used += tw
                continue
            rest = room - used
            if rest > 0:
                fitted.append((style, truncate(text, rest)))
            break
        left = fitted
    gap = max(1, width - _pad(left) - _pad(right))
    out: List[Fragment] = list(left)
    out.append(("", fill * gap))
    out.extend(right)
    return out


# ---------------------------------------------------------------------------
# Welcome screen
# ---------------------------------------------------------------------------

def welcome(
    *,
    width: int,
    version: str,
    model_label: str,
    effort_label: str,
    signed_in: str,
    session: str,
    cwd: str,
    colors: dict,
    app_name: str = "FullAgent",
    tips: Sequence[str] = TIPS,
):
    """Build the startup card: block logo + identity on the left, tips right.

    Returns a rich renderable. Falls back to a stacked layout under 78
    columns so nothing wraps or gets clipped.
    """
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    width = max(40, int(width))
    logo_w = max(display_width(line) for line in LOGO)

    logo = Text()
    for i, line in enumerate(LOGO):
        # v4 aurora polish: a true per-character rainbow gradient animates
        # each logo row, with a stable phase offset down the mark. Falls
        # back to the flat style key if rich/fx is unavailable.
        try:
            from .fx import RAINBOW, gradient_text
            palette = tuple(RAINBOW[i % len(RAINBOW):]
                            + RAINBOW[:i % len(RAINBOW)])
            logo.append_text(gradient_text(line, list(palette)))
            logo.append("\n")
        except Exception:
            key = LOGO_STYLES[i % len(LOGO_STYLES)]
            logo.append(line + "\n", style=f"bold {colors.get(key, '')}")

    identity = Text()
    identity.append(f"  {app_name} CLI ", style=f"bold {colors.get('fg', '')}")
    identity.append(f"v{version}", style=f"bold {colors.get('pink', '')}")
    identity.append("\n")
    identity.append(f"  {signed_in}", style=colors.get("green", ""))
    identity.append("\n")
    identity.append(f"  {truncate(model_label, max(12, width - 22))}",
                    style=colors.get("cyan", ""))
    identity.append("  ·  ", style=colors.get("dim", ""))
    identity.append(effort_label, style=f"bold {colors.get('yellow', '')}")
    identity.append("\n")
    identity.append(f"  {truncate(cwd, max(12, width - 34))}",
                    style=colors.get("dim", ""))
    identity.append("  ·  ", style=colors.get("dim", ""))
    sid = str(session)
    sid = sid if len(sid) <= 12 else sid[:8] + "…"
    identity.append(f"session {sid}", style=colors.get("dim", ""))

    card_w = min(48, max(34, width - logo_w - 10))
    inner = max(10, card_w - 4)
    tip_text = Text()
    for idx, tip in enumerate(tips, 1):
        tip_text.append(f"{idx}. ", style=f"bold {colors.get('cyan', '')}")
        tip_text.append(truncate(str(tip), inner - 3) + "\n",
                        style=colors.get("fg", ""))

    card = Panel(
        tip_text,
        title=f"[bold {colors.get('accent', '')}]Tips for getting started[/]",
        border_style=colors.get("border", ""),
        width=card_w,
        padding=(0, 1),
        expand=False,
    )

    if width >= logo_w + card_w + 8:
        grid = Table.grid(padding=(0, 3))
        grid.add_column(no_wrap=True)
        grid.add_column(no_wrap=True)
        grid.add_row(Group(logo, identity), card)
        return grid

    # Narrow terminals: stack logo over card, never wrap the block art.
    return Group(logo, identity, Text("\n"), card)


# ---------------------------------------------------------------------------
# Mode / skills HUD and status footer
# ---------------------------------------------------------------------------

# Short aliases for the REAL autonomy ladder in agent.AUTONOMY_LEVELS.
# No invented "YOLO": level 4 is Pilot (deletes still ask), level 5 is
# Autonomous, level 3 the default Collaborator.
_MODE_NAMES = {
    0: "OBSERVE",
    1: "ADVISOR",
    2: "ASSIST",
    3: "COLLAB",
    4: "PILOT",
    5: "AUTONOMOUS",
}


def _mode_level(autonomy: int) -> int:
    try:
        return max(0, min(5, int(autonomy)))
    except (TypeError, ValueError):
        return 3


def mode_badge(autonomy: int, colors: dict) -> List[Fragment]:
    """Mode chip naming the real autonomy level (L0–L5)."""
    level = _mode_level(autonomy)
    key = ("dim" if level <= 1 else "cyan" if level == 2
           else "green" if level == 3 else "yellow" if level == 4 else "red")
    return [(f"bold {colors.get(key, '')}", f" {_MODE_NAMES[level]} ")]


def hud_fragments(
    *,
    width: int,
    colors: dict,
    autonomy: int,
    skills: int,
    agents_running: int = 0,
    agents_queued: int = 0,
) -> List[Fragment]:
    """The row above the prompt: mode chip, ⇧Tab hint, skills counter."""
    level = _mode_level(autonomy)
    left: List[Fragment] = mode_badge(level, colors)
    hint = " ⇧Tab pilot " if level < 4 else " ⇧Tab exit pilot "
    left.append((colors.get("dim", ""), hint))
    if agents_running or agents_queued:
        left.append((f"bold {colors.get('pink', '')}",
                     f"· ◎ {int(agents_running)} running "
                     f"· {int(agents_queued)} queued "))
    right: List[Fragment] = [
        (f"bold {colors.get('cyan', '')}", f" {max(0, int(skills))} skills "),
    ]
    return align(left, right, width, fill="─")


def _compact_number(value: int) -> str:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return "?"
    if n >= 1_000_000:
        # %g keeps 1M as "1M", 1.5M as "1.5M" — no rounding up to 2M
        return f"{n / 1_000_000:g}M"
    if n >= 1000:
        return f"{n // 1000}k"
    return str(n)


def context_fragments(used_pct: int, colors: dict) -> List[Fragment]:
    """`ctx ███░░░░░░░ 30%` — ten cells, colour-graded like the border."""
    try:
        pct = max(0, min(100, int(used_pct)))
    except (TypeError, ValueError):
        pct = 0
    filled = int(round(pct / 10))
    key = "green" if pct < 60 else "yellow" if pct < 85 else "red"
    bar = "█" * filled + "░" * (10 - filled)
    return [
        (colors.get("dim", ""), "ctx "),
        (colors.get(key, ""), bar),
        (f"bold {colors.get(key, '')}", f" {pct}%"),
    ]


def status_fragments(
    *,
    width: int,
    colors: dict,
    model_label: str,
    effort_label: str,
    context_window: int,
    used_pct: int,
    cwd: str,
    branch: str = "",
    agents_running: int = 0,
    agents_queued: int = 0,
) -> List[Fragment]:
    """The footer under the prompt: identity, context meter, agent badge."""
    right: List[Fragment] = []
    if branch:
        right.append((f"bold {colors.get('accent', '')}",
                      f" ◎ {truncate(branch, 24)} "))
    # Agent badge is live state only: it counts the workers that are
    # running or queued right now — never retained history from an earlier
    # batch — and it disappears the moment nothing is live.
    running = max(0, int(agents_running))
    queued = max(0, int(agents_queued))
    live = running + queued
    if live:
        if right:
            right.append((colors.get("dim", ""), "· "))
        noun = "agent" if live == 1 else "agents"
        right.append((f"bold {colors.get('pink', '')}",
                      f"◎ {live} {noun}"))
        right.append((colors.get("dim", ""), f" | {running} running "))
        if queued:
            right.append((colors.get("dim", ""), f"· {queued} queued "))
    right_pad = _pad(right)

    # Left side is assembled by priority: identity + context meter are
    # always shown; the context-window label and the cwd are dropped (in
    # that order) when the terminal is too narrow for everything.
    head: List[Fragment] = [
        (f"bold {colors.get('cyan', '')}", f" ● {model_label} "),
        (colors.get("dim", ""), "· "),
        (f"bold {colors.get('yellow', '')}", f"{effort_label} "),
    ]
    window_seg: Fragment = (colors.get("dim", ""),
                            f"· {_compact_number(context_window)} context · ")
    meter = context_fragments(used_pct, colors)
    cwd_seg: Fragment = (colors.get("dim", ""),
                         f" · {truncate(cwd, 28)}")

    budget = max(0, int(width) - right_pad - 1)
    left: List[Fragment] = list(head)
    if _pad(left + [window_seg] + meter) <= budget:
        left.append(window_seg)
    left += meter
    if _pad(left + [cwd_seg]) <= budget:
        left.append(cwd_seg)
    return align(left, right, width)


# ---------------------------------------------------------------------------
# Live agent board (rendered above the prompt while a swarm is running)
# ---------------------------------------------------------------------------

PANEL_LIMIT = 6


def agent_board(
    snapshot: dict | None,
    *,
    width: int,
    colors: dict,
    limit: int = PANEL_LIMIT,
) -> List[List[Fragment]]:
    """Rows for the live subagent board.

    Returns a list of fragment rows (no newlines). Empty when there is no
    active worker — the board disappears as soon as the swarm settles, so
    finished results stay in the scrollback instead of eating screen space.
    """
    if not isinstance(snapshot, dict):
        return []
    # Only LIVE workers belong on the live board: retained history from
    # earlier batches must never inflate the header or the hidden count.
    workers = [w for w in (snapshot.get("workers") or [])
               if isinstance(w, dict)
               and w.get("state") in ACTIVE_STATES]
    if not workers:
        return []

    width = max(24, int(width))
    total = len(workers)
    noun = "Agent" if total == 1 else "Agents"
    counts: dict[str, int] = {}
    for w in workers:
        state = str(w.get("state") or "queued")
        counts[state] = counts.get(state, 0) + 1
    running = counts.get("running", 0)
    queued = counts.get("queued", 0)
    cancelling = counts.get("cancelling", 0)
    # Exact live breakdown — "8 running" only when 8 are genuinely running.
    parts = []
    if running:
        parts.append(f"{running} running")
    if queued:
        parts.append(f"{queued} queued")
    if cancelling:
        parts.append(f"{cancelling} cancelling")
    head: List[Fragment] = [
        (f"bold {colors.get('accent', '')}", f" ≡ {total} {noun} "),
        (colors.get("green", ""), "(" + " · ".join(parts) + ") "),
    ]
    # v4 aurora polish: tint the swarm header with the fx rainbow so a
    # running board reads as alive (falls back silently to the flat accent).
    try:
        from .fx import RAINBOW
        head = [
            ("bold " + RAINBOW[0], f" ≡ {total} {noun} "),
            ("bold " + RAINBOW[3], "(" + " · ".join(parts) + ") "),
        ]
    except Exception:
        pass
    tail: List[Fragment] = [(colors.get("dim", ""), " ↓ to manage ")]
    rows: List[List[Fragment]] = [align(head, tail, width, fill="─")]

    # Running first, then cancelling, then queued.
    order = {"running": 0, "cancelling": 1, "queued": 2}
    ranked = sorted(workers,
                    key=lambda w: (order.get(w.get("state"), 3),
                                   -float(w.get("elapsed_ms") or 0)))
    for row in ranked[:max(1, int(limit))]:
        state = str(row.get("state") or "queued")
        color = colors.get(STATE_COLOR_KEYS.get(state, "fg"), "")
        icon = STATE_ICONS.get(state, "·")
        chip = short_role(str(row.get("role") or ""))
        action = " ".join(str(row.get("current_action") or state).split())
        # While a worker is waiting on the model, current_action is the
        # generic "model step N/20"; the real assigned task says more.
        if (not action or action == "starting"
                or action.startswith("model step")):
            task = " ".join(str(row.get("task") or "").split())
            action = task or action or state
        prefix = f" {icon} "
        room = max(8, width - display_width(prefix) - len(chip) - 4)
        rows.append([
            (f"bold {color}", prefix),
            (f"bold {colors.get('cyan', '')}", chip),
            (colors.get("dim", ""), " · "),
            (colors.get("fg", ""), truncate(action, room)),
        ])
        # v4: live telemetry tail per worker — elapsed seconds and retry
        # count, so a long swarm shows what each agent is up against.
        try:
            elapsed_s = int((float(row.get("elapsed_ms") or 0)) / 1000)
            retries = int(row.get("retries") or 0)
            bits = []
            if elapsed_s >= 3:
                bits.append(f"{elapsed_s}s")
            if retries:
                bits.append(f"retry {retries}")
            if bits:
                rows[-1] = rows[-1] + [
                    (colors.get("dim", ""), "  " + " · ".join(bits)),
                ]
        except Exception:
            pass

    hidden = total - min(total, max(1, int(limit)))
    if hidden > 0:
        rows.append([(colors.get("dim", ""),
                      f"   … +{hidden} more · ↓ to manage")])
    return rows

# ---------------------------------------------------------------------------
# Module self-test: layout is headless and deterministic
# ---------------------------------------------------------------------------
def _cliui_selftest() -> None:
    import io

    from rich.console import Console

    colors = {
        "border": "#5b6eae", "accent": "#c084fc", "cyan": "#22d3ee",
        "green": "#34d399", "yellow": "#facc15", "orange": "#fb923c",
        "red": "#f87171", "pink": "#f472b6", "fg": "#f8fafc",
        "dim": "#64748b",
    }

    def width_of(frags) -> int:
        return sum(display_width(t) for _, t in frags)

    # align: exact width, never negative padding, right badge survives
    for w in (30, 60, 100):
        line = align([("", "left side")], [("", "right")], w)
        assert width_of(line) == w, (w, width_of(line))
    narrow = align([("", "x" * 90)], [("", "tail")], 40)
    assert width_of(narrow) == 40, width_of(narrow)
    assert "tail" in "".join(t for _, t in narrow)

    # hud + footer stay on one terminal line
    hud = hud_fragments(width=80, colors=colors, autonomy=5, skills=21,
                        agents_running=8, agents_queued=0)
    assert "AUTONOMOUS" in "".join(t for _, t in hud)
    assert "21 skills" in "".join(t for _, t in hud)
    assert "8 running" in "".join(t for _, t in hud)
    assert width_of(hud) == 80
    foot = status_fragments(width=90, colors=colors, model_label="Test",
                            effort_label="max", context_window=1_000_000,
                            used_pct=3, cwd="/tmp", branch="main",
                            agents_running=8, agents_queued=0)
    foot_text = "".join(t for _, t in foot)
    assert "1M context" in foot_text, foot_text
    assert "8 agents | 8 running" in foot_text, foot_text
    assert width_of(foot) == 90
    idle_foot = "".join(t for _, t in status_fragments(
        width=90, colors=colors, model_label="Test", effort_label="max",
        context_window=1_000_000, used_pct=3, cwd="/tmp", branch="main",
        agents_running=0, agents_queued=0))
    assert "agents" not in idle_foot, idle_foot

    # board: hidden when idle, bounded when a swarm is live
    assert agent_board({"workers": []}, width=80, colors=colors) == []
    snap = {"workers": [
        {"id": f"ag-{i}", "state": "running", "role": "researcher",
         "current_action": f"Read src/file{i}.py", "elapsed_ms": i * 1000,
         "summary": ""} for i in range(9)
    ]}
    board = agent_board(snap, width=80, colors=colors)
    assert len(board) == 1 + PANEL_LIMIT + 1, len(board)
    assert "9 Agents" in "".join(t for _, t in board[0])
    assert "Explore" in "".join(t for _, t in board[1])
    assert "+3 more" in "".join(t for _, t in board[-1])
    assert all(width_of(r) <= 80 for r in board)
    done = {"workers": [{"id": "a", "state": "done", "role": "coder",
                         "current_action": "", "summary": "ok",
                         "elapsed_ms": 5}]}
    assert agent_board(done, width=80, colors=colors) == []

    # welcome screen: logo, version and all four tips render at 100 cols
    buf = io.StringIO()
    console = Console(file=buf, width=100, color_system=None)
    console.print(welcome(width=100, version="9.9.9", model_label="Test Model",
                          effort_label="max", signed_in="Signed in · test",
                          session="abcd1234", cwd="~/work", colors=colors))
    out = buf.getvalue()
    assert "9.9.9" in out and "abcd1234" in out, out
    assert "Tips for getting started" in out, out
    for idx in range(1, 5):
        assert f"{idx}." in out, (idx, out)
    assert "╔" in out, "block logo missing"

    # narrow welcome stacks instead of clipping
    buf = io.StringIO()
    console = Console(file=buf, width=60, color_system=None)
    console.print(welcome(width=60, version="9.9.9", model_label="Test",
                          effort_label="max", signed_in="Signed in",
                          session="s", cwd="~", colors=colors))
    assert "Tips for getting started" in buf.getvalue()

    print("cliui self-test PASS: align/hud/footer/board/welcome")


if __name__ == "__main__":
    _cliui_selftest()
