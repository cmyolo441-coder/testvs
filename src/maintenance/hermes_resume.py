"""Name the Hermes session that closed when OMH's own TUI launch exited.

Bare ``omh`` runs the user's ``hermes`` as a child (``commands/bare_launch.py``,
``_launch_hermes_tui``), so control returns to this process when the TUI
exits. Hermes prints its own epilogue first and that epilogue names the other
binary -- ``hermes --tui --resume <id>`` -- so someone who came in through
``omh`` is told to come back through ``hermes``.

OMH cannot read the id Hermes used for that epilogue. Hermes creates
``HERMES_TUI_ACTIVE_SESSION_FILE`` itself with ``mkstemp`` and unlinks it in a
``finally`` before returning (``hermes_cli/main_tui_launch.py``), so by the
time this process runs the file is gone. The id is recovered instead from the
state database the TUI just wrote to: the TUI gateway calls
``db.end_session()`` on exit when it owns the session's lifecycle
(``tui_gateway/session_lifecycle.py``), which stamps ``sessions.ended_at`` as
the child exits.

A gateway-originated session, or one another backend holds a lease on, is
deliberately never ended there. It therefore produces no candidate here and
no line, which is the right outcome rather than a gap: OMH would have no way
to tell which of several live sessions the person was looking at.

That recovery is the fallback. The OMH TUI writes the session it has open
into a file of OMH's own (``tui/launch.py``, ``SESSION_FILE_ENV``), which
``omh`` reads before removing it; only when that file names nothing (Hermes'
own TUI opened instead, or the OMH TUI never got a session) does the
``ended_at`` timing above choose. The timing cannot name a resumed session the
person sent nothing in: Hermes mounts a resumed session read-only and leaves
its old ``ended_at`` as it was until a prompt reopens it
(``tui_gateway/methods_prompt.py``, ``_reopen_if_finalized``), so the row
never falls inside the child's lifetime.

Once the session is known, ``exit_card_lines`` reads its one row, by id, to
say what it was: its title, counts and model, never message content or the cwd.

Everything here is best effort and runs after the person's session has
already ended. A line naming the wrong session is worse than no line, so
every ambiguity resolves to ``None``.
"""

from __future__ import annotations

import json
import re
import shlex
import sqlite3
from pathlib import Path
from typing import NamedTuple

from ..plugin_bundle.omh.sqlite_readonly import connect_readonly

# Hermes' own profile id rule (``hermes_cli/main.py``, ``_PROFILE_NAME_RE``,
# which mirrors ``hermes_cli/profiles.py``). A name that could not identify a
# profile directory never becomes a path component here.
_PROFILE_NAME_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
# What a session id has to look like before a copyable line carries it.
# Hermes' ids are `YYYYMMDD_HHMMSS_<hex>`; anything a shell would read as more
# than one word, or as an option, is never printed.
_SESSION_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")

# A database the gateway is still finishing with is reported as no line, never
# as a wait: this runs in a shell that has just come back from the TUI.
_DB_BUSY_TIMEOUT_SECONDS = 0.25

# How close to the child's exit a session's ``ended_at`` has to sit before it
# reads as "the session that just closed", and how far apart the two newest
# candidates have to be before the later one is unambiguous.
#
# What it has to cover: the gateway finishing its shutdown once the TUI
# process goes away, and then Hermes' own launcher opening state.db to print
# its exit summary (``hermes_cli/main_tui_launch.py``,
# ``_print_tui_exit_summary``) and returning, all before this process sees the
# child's exit status at all.
#
# Why it is not larger: the same window is what makes another terminal's
# session look like this one. Someone with two TUIs open who closes both
# around the same moment is the case this has to stay narrow enough to refuse,
# and ``_chosen_candidate`` refuses it by comparing the two newest ``ended_at``
# values against this same value.
#
# The value is reasoned from that ordering, not measured. Measuring it would
# take a live TUI exit, which nothing in this repository runs.
RESUME_END_PROXIMITY_SECONDS = 5.0

# Only what choosing the session needs: the id to print, the end stamp the
# rules compare, and whether the session holds a conversation at all. No
# title, no message content, no cwd.
#
# ``cwd`` is deliberately absent as a filter too, not just as a column:
# ``hermes --resume`` restores the session's own working directory unless
# ``--no-restore-cwd`` is given, so a resumed session's ``cwd`` legitimately
# differs from the directory the launch happened in.
#
# ``ended_at IS NOT NULL`` is a stated precondition rather than a branch:
# removing it turns no test red, because ``NULL >= ?`` is NULL and the window
# comparison below already drops the row. It is kept because "a session
# nothing ended is not a candidate" is the rule the gateway's behaviour makes
# load-bearing, and a later change to the window must not silently take it
# with it.
_CANDIDATE_SELECT = (
    "SELECT id, ended_at, message_count FROM sessions "
    "WHERE source = 'tui' AND ended_at IS NOT NULL "
    "AND ended_at >= ? AND ended_at <= ? "
    "ORDER BY ended_at DESC LIMIT 2"
)


def _epoch(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def hermes_profile_home(hermes_home: str | Path, profile: str | None = None) -> Path | None:
    """The Hermes home of the default profile, or of one named profile.

    Hermes' default profile lives at the home itself and every named one under
    ``<home>/profiles/<name>``. A name this cannot use as a directory
    component returns ``None`` instead of a path built out of it; the launch
    still forwards the name to Hermes, which validates it itself.
    """
    home = Path(hermes_home)
    if profile is None:
        return home
    if not _PROFILE_NAME_RE.fullmatch(profile):
        return None
    return home / "profiles" / profile


def hermes_state_db_path(hermes_home: str | Path, profile: str | None = None) -> Path | None:
    """The state database for the default home, or for one named profile (`hermes_profile_home`)."""
    home = hermes_profile_home(hermes_home, profile)
    return None if home is None else home / "state.db"


def _chosen_candidate(rows: list[tuple], child_ended_at: float) -> str | None:
    if not rows:
        return None
    session_id, ended_at, message_count = rows[0]
    latest = _epoch(ended_at)
    if latest is None or child_ended_at - latest > RESUME_END_PROXIMITY_SECONDS:
        return None
    if len(rows) > 1:
        runner_up = _epoch(rows[1][1])
        if runner_up is None or latest - runner_up <= RESUME_END_PROXIMITY_SECONDS:
            return None
    try:
        if int(message_count) <= 0:
            return None
    except (TypeError, ValueError):
        return None
    return str(session_id or "").strip() or None


def resumable_session_id(
    db_path: str | Path | None,
    child_started_at: float,
    child_ended_at: float,
) -> str | None:
    """The id of the TUI session that ended with this child, or ``None``.

    The window is the child's own lifetime: a session that ended before the
    launch started is some earlier terminal's. Within it, the newest ended TUI
    session wins, and only when it ended close to the child's exit, no second
    session ended within :data:`RESUME_END_PROXIMITY_SECONDS` of it, and it
    holds at least one message.

    A missing, locked, corrupt or foreign database returns ``None`` and
    raises nothing: the caller prints no line, which is the outcome every
    unresolved case wants.
    """
    if db_path is None or child_ended_at < child_started_at:
        return None
    try:
        connection = connect_readonly(db_path, timeout=_DB_BUSY_TIMEOUT_SECONDS)
    except (sqlite3.Error, ValueError, OSError):
        return None
    try:
        rows = connection.execute(_CANDIDATE_SELECT, (child_started_at, child_ended_at)).fetchall()
    except (sqlite3.Error, ValueError):
        return None
    finally:
        try:
            connection.close()
        except sqlite3.Error:
            pass
    return _chosen_candidate(rows, child_ended_at)


def resume_command_line(session_id: str, profile: str | None = None) -> str:
    """The copyable line: the same door the person came in through.

    The profile is carried because a profile has its own ``state.db``. Without
    it the line would send the person back to the default home, where the id
    does not exist.
    """
    if profile:
        return f"omh -p {profile} --resume {session_id}"
    return f"omh --resume {session_id}"


# The one row the exit card reads, by the id already chosen, and the person's
# own prompts in it (rows a rewind or a retired queue entry turned inactive are
# not turns). No message content, no cwd.
_CARD_SELECT = (
    "SELECT title, model, message_count, input_tokens, output_tokens, cache_read_tokens, "
    "cache_write_tokens, reasoning_tokens, model_config FROM sessions WHERE id = ?"
)
# A `/branch` child's copied history was billed on the sessions it was cut from (`model_config._branched_from`);
# Hermes' summary sums the chain (`_branch_lineage` in `hermes_cli/main_tui_launch.py`), and so does the card.
_LINEAGE_SELECT = (
    "SELECT input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, reasoning_tokens, model_config "
    "FROM sessions WHERE id = ?"
)
_LINEAGE_LIMIT = 64
# Hermes' `-c <title>` (`resolve_session_by_title` in `hermes_state_titles.py`) takes the newest `<title> #N` row
# over the exact title, so the card offers `omh -c` only when no such row exists. "Bot Chat" is Hermes' one
# exception: its exact title always wins.
_NUMBERED_TITLE_SELECT = "SELECT 1 FROM sessions WHERE title LIKE ? ESCAPE '\\' LIMIT 1"
_EXACT_TITLE_WINS = "Bot Chat"
_TURNS_SELECT = "SELECT COUNT(*) FROM messages WHERE session_id = ? AND role = 'user' AND active = 1"

# What a session nothing was said in leaves: Hermes stores a session's row
# only once its first prompt is sent, and prints no epilogue for one without
# messages, so there is nothing to resume.
NOTHING_SAID_LINE = "omh: nothing was said in this session, so there is nothing to resume."
UNTITLED = "untitled session"
# Characters a double-quoted shell word does not keep as typed; a title with
# one is quoted with `shlex.quote` instead.
_DOUBLE_QUOTE_UNSAFE = frozenset('"$`\\!')


def _count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(value, 0)


def _duration_text(seconds: float) -> str:
    whole = max(int(seconds), 0)
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def continue_command_line(title: str, profile: str | None = None) -> str:
    """`omh -c "<title>"`: Hermes resolves `--continue NAME` by the session's title.

    "" for a title a shell line cannot carry as typed (a control character),
    and for one starting with `-`, which `omh`'s parser would read as options
    (`-c "-p work"` is `-c` with no name, then `-p work`).
    """
    if not title or not title.isprintable() or title.startswith("-"):
        return ""
    quoted = shlex.quote(title) if _DOUBLE_QUOTE_UNSAFE & set(title) else f'"{title}"'
    return f"omh -p {profile} -c {quoted}" if profile else f"omh -c {quoted}"


def _escape_like(text: str) -> str:
    """Hermes' `escape_like`: `%` and `_` in a title match literally under `ESCAPE '\\'`."""
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _branched_from(model_config: object) -> str:
    try:
        config = json.loads(model_config) if isinstance(model_config, str) and model_config else {}
    except (ValueError, RecursionError):
        return ""
    parent = config.get("_branched_from") if isinstance(config, dict) else None
    return parent if isinstance(parent, str) else ""


class _CardRow(NamedTuple):
    title: str
    model: str
    messages: int
    tokens: tuple[int, int, int, int, int]  # in, out, cache read, cache write, reasoning: summed over the lineage
    turns: int | None  # None when it cannot be read
    continues_here: bool  # Hermes' `-c <title>` opens this session


def _session_card_row(db_path: str | Path, session_id: str) -> _CardRow | None | bool:
    """The session's card row; None when it has no row; False when the database cannot be read at all."""
    try:
        connection = connect_readonly(db_path, timeout=_DB_BUSY_TIMEOUT_SECONDS)
    except (sqlite3.Error, ValueError, OSError):
        return False
    try:
        row = connection.execute(_CARD_SELECT, (session_id,)).fetchone()
        if row is None:
            return None
        try:
            turns_row = connection.execute(_TURNS_SELECT, (session_id,)).fetchone()
        except sqlite3.Error:
            turns_row = None
        tokens = [_count(value) for value in row[3:8]]
        seen, parent = {session_id}, _branched_from(row[8])
        while parent and parent not in seen and len(seen) < _LINEAGE_LIMIT:
            seen.add(parent)
            ancestor = connection.execute(_LINEAGE_SELECT, (parent,)).fetchone()
            if ancestor is None:
                break
            tokens = [total + _count(value) for total, value in zip(tokens, ancestor[:5])]
            parent = _branched_from(ancestor[5])
        title = " ".join(str(row[0] or "").split())
        continues_here = bool(title) and str(row[0]) == title and (
            title == _EXACT_TITLE_WINS
            or connection.execute(_NUMBERED_TITLE_SELECT, (f"{_escape_like(title)} #%",)).fetchone() is None
        )
    except (sqlite3.Error, ValueError):
        return False
    finally:
        try:
            connection.close()
        except sqlite3.Error:
            pass
    return _CardRow(
        title=title,
        model=" ".join(str(row[1] or "").split()),
        messages=_count(row[2]),
        tokens=(tokens[0], tokens[1], tokens[2], tokens[3], tokens[4]),
        turns=_count(turns_row[0]) if turns_row else None,
        continues_here=continues_here,
    )


def exit_card_lines(
    db_path: str | Path | None, session_id: str, seconds: float, profile: str | None = None
) -> list[str]:
    """What `omh` prints for `session_id` as the TUI closes: the card, `NOTHING_SAID_LINE`, or nothing.

    The card says what Hermes' own exit summary says (the session id, its
    title, its messages and its tokens, split as Hermes splits them and summed
    over the `/branch` chain as Hermes sums them), plus
    its turns, how long this launch ran (`seconds`) and its model, and then
    only the `omh` way back: `omh --resume <id>` and, for a titled session
    Hermes' `-c <title>` would open, `omh -c "<title>"`.

    A session with no row is one nothing was said in: Hermes stores the row
    with the first prompt. A database that cannot be read prints nothing,
    since it cannot tell those apart from a session that has messages.
    """
    if db_path is None or not _SESSION_ID_RE.fullmatch(session_id):
        return []
    card = _session_card_row(db_path, session_id)
    if card is False:
        return []
    if card is None or card.messages == 0:
        return [NOTHING_SAID_LINE]
    tokens_in, tokens_out, cache_read, cache_write, reasoning = card.tokens
    session = [f"session {session_id}"]
    if card.turns is not None:
        session.append(f"{card.turns} turn" if card.turns == 1 else f"{card.turns} turns")
    session += [f"{card.messages} message" if card.messages == 1 else f"{card.messages} messages",
                _duration_text(seconds)]
    usage = [
        f"{tokens_in + tokens_out + cache_read + cache_write + reasoning} tokens (in {tokens_in}, out {tokens_out}, "
        f"cache {cache_read + cache_write}, reasoning {reasoning})"
    ]
    if card.model and card.model.isprintable():
        usage.append(card.model)
    lines = [
        f"omh · {card.title if card.title and card.title.isprintable() else UNTITLED}",
        "  " + " · ".join(session),
        "  " + " · ".join(usage),
        "  " + resume_command_line(session_id, profile),
    ]
    continued = continue_command_line(card.title, profile) if card.continues_here else ""
    if continued:
        lines.append("  " + continued)
    return lines
