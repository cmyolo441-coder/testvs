"""Host lifecycle facts for `omh_team`: a helper started, a helper came back.

`omh_team` must not check a part while any of its helpers is still editing.
Whether a helper is out is a fact only the host has, so it comes from the
host's own `subagent_start` / `subagent_stop` callbacks, never from the model
saying so:

* `subagent_start` carries the child's goal. A team entry's goal ends with
  the attempt key OMH reserved (`[omh-team:<team>/<unit>/attempt-<n>]`; one
  handed out before an update opens with it), the last key in the goal is
  the one read, and only an attempt the team record already holds, still
  undispatched, binds to the child's session id. The child's subagent id is kept beside it: with
  Hermes' worktree isolation on, it names the branch the helper's work is on.
  So is this Hermes process (`host_process.this_process`), which the helper
  runs in and ends with: a restart fires no `subagent_stop`.
* `subagent_stop` carries that child session id; the attempt it bound is
  marked returned, which is what lets `team_reconcile` check it. Hermes
  reads the id off the child at stop time, so a helper its rotating
  compaction moved to a new session stops under the new id. The earlier
  segments that id continues are read from Hermes' own compaction links in
  `state.db` (`live_session.compaction_ancestors`) and matched exactly. The
  parent's id is read the same way at both calls, so a parent compacted in
  between names a later segment; its team is found under the earlier
  segments the same links name.

A goal without the marker is not a team helper and costs one substring test.
Nothing here raises: both are observers the host fires on the spawn path, and
a failure leaves the attempt undispatched, which `team_reconcile` hands back
and eventually blocks as `dispatch_not_observed` -- visible, never silent.
"""
from __future__ import annotations

import time

from . import runtime_paths
from .host_process import this_process
from .live_session import compaction_ancestors

_MARKER_PREFIX = "[omh-team:"


def observe_team_dispatch(kwargs: dict[str, object]) -> bool:
    goal = kwargs.get("child_goal")
    if not isinstance(goal, str) or _MARKER_PREFIX not in goal:
        return False
    try:
        from omh.workflows import team

        home, hermes, parent = _parent(kwargs)
        if not parent:
            return False
        return team.observe_dispatch(home, parent, child_session_id=kwargs.get("child_session_id"),
                                     goal=goal, now=time.time(), subagent_id=kwargs.get("child_subagent_id"),
                                     earlier=lambda: _earlier_segments(hermes, parent), host=this_process())
    except (ImportError, OSError, ValueError, RuntimeError):
        return False


def observe_team_return(kwargs: dict[str, object]) -> bool:
    child = kwargs.get("child_session_id")
    if not isinstance(child, str) or not child.strip():
        return False
    try:
        from omh.workflows import team

        # Most homes never start a team: one stat before any session lookup.
        if not (runtime_paths.plugin_home(kwargs.get("omh_home")) / "runtime" / "teams").is_dir():
            return False
        home, hermes, parent = _parent(kwargs)
        if not parent:
            return False
        return team.observe_return(home, parent, child_session_id=child, now=time.time(),
                                   lineage=lambda session_id: compaction_ancestors(str(hermes), session_id),
                                   earlier=lambda: _earlier_segments(hermes, parent))
    except (ImportError, OSError, ValueError, RuntimeError):
        return False


def _earlier_segments(hermes: object, session: str) -> list[str]:
    """The earlier segments of this session's conversation that every link up to them was verified for."""
    return [str(item) for item in compaction_ancestors(str(hermes), session).get("ancestors") or []]


def _parent(kwargs: dict[str, object]) -> tuple[object, object, str]:
    reading_session_id = _runtime_reader.reading_session_id

    home = runtime_paths.plugin_home(kwargs.get("omh_home"))
    hermes = runtime_paths.plugin_home(kwargs.get("hermes_home"), hermes=True)
    parent = kwargs.get("parent_session_id")
    return home, hermes, reading_session_id(hermes, parent) if isinstance(parent, str) else ""


# Module scope, not call time: Hermes can evict the bundle from sys.modules
# while its callbacks stay live (#1979). Last, so an import cycle finds every name.
from . import runtime_reader as _runtime_reader
