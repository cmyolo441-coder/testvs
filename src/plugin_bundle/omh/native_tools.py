"""Whether a native Hermes tool is present for a session: one rule, one wording.

A native tool is PRESENT when its name is in the session's tool list OR in
the tool_search catalog. Hermes defers a curated set of core tools behind the
tool_search bridge by default -- `todo_list`, `cronjob_manage` and
`image_generate` among them, already at the plugin's floor (tag ``v2026.9.7``,
``_DEFAULT_DEFERRED_TOOLS`` in ``tools/tool_search.py``) -- and a deferred
tool is reachable through ``tool_call``: it is present, not
missing. Hermes also peels the bridge before plugin hooks run, so a deferred
tool's call reaches ``pre_tool_call`` / ``post_tool_call`` under its own
name, never as ``tool_call``.

Presence is never read off a Hermes version number. A version says what a
release shipped; it does not say what this session was given -- toolset
scope, the platform, a tool the person disabled, or a deferral list they
edited all change the answer on the same build.

Every OMH surface states the rule through ``native_tool_presence_clause``
rather than restating it: awareness lines, tool descriptions, skill bodies.
Hermes hands a plugin hook neither name set (``pre_llm_call`` carries no tool
list), while the model sees its own tool list and can ask tool_search, so the
text states the rule in these words and the model evaluates it for the
session it is in. Code that comes to hold both sets adds the predicate here,
beside the clause, when it first needs one.

A surface that needs a native tool states what it does when the clause is
false, in the same sentence. The first user is the plan checklist: native
`todo_list` tracks steps when present, and `omh_todo` when it is not
(`native_todo`).

Stdlib only. This ships in the plugin bundle, which may not import ``omh``,
and the skill renderer imports it from here, so the wording is one string in
both places.
"""
from __future__ import annotations

NATIVE_TODO_TOOL = "todo_list"
# Hermes' cron tool. `automation-blueprint` prepares its `create` call and reads
# its `list` output as post-activation status; OMH never calls it.
NATIVE_CRON_TOOL = "cronjob_manage"
TOOL_SEARCH_TOOL = "tool_search"
# The `blocked_reason` a cancelled todo_list item carries in the OMH plan
# record (`native_todo`). Written by OMH, not the model, and worded as what the
# record shows rather than why it happened. Here, beside the tool's name, so
# the projection and the `omh_todo` tool read one string.
NATIVE_CANCELLED_REASON = "cancelled in todo_list"
# The plans only `omh_todo` can hold, because `plan_stage` and the template
# exist only there: the plans `native_todo._held_by_omh_todo` keeps from a
# native write. One string, so every surface that says which ONE tool holds
# the steps names the same three: the `omh_todo` tool, `todo-checklist`, the
# route hint's `plan_todo` line, and the awareness primer.
OMH_TODO_HELD_PLANS = "a planning run, an accepted plan or a code-story template"


def native_tool_presence_clause(name: str, *, code: bool = False) -> str:
    """The presence rule as a clause the model evaluates: ``<name> is in your tool list or tool_search finds it``.

    ``code`` wraps the named tool in backticks for markdown surfaces (skill
    bodies), which is the one shape every surface states the rule in:
    ``f"when {native_tool_presence_clause('clarify', code=True)}"`` reads
    "when `clarify` is in your tool list or tool_search finds it". Plain text
    surfaces (awareness lines, tool descriptions) leave the name bare, as the
    rest of their wording does.
    """
    tick = "`" if code else ""
    return f"{tick}{name}{tick} is in your tool list or {TOOL_SEARCH_TOOL} finds it"
