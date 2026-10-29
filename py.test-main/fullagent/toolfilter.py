"""Tool-schema compression + per-turn tool filtering (PERF: tool bloat).

The model used to receive ALL ~117 tool schemas on EVERY request —
~46k chars / ~12-15k tokens of JSON before a single word of real
context. This module shrinks that in two independent, safe ways:

1. COMPRESSION — every schema is re-serialized compactly: the
   description is trimmed to its first sentence (<=150 chars),
   per-parameter descriptions are truncated (<=60 chars), and verbose
   "examples" annotation blobs are dropped (never a parameter actually
   named "examples" — synthesize_tool requires one). Names, types, required lists, enums
   and defaults are preserved exactly, so the model still calls tools
   correctly.

2. FILTERING — per turn, only *relevant* tools are sent:
   - CORE tools (files, shell, search, web, todos, planning,
     reasoning backbone) are always included.
   - ADVANCED tools (docker, git, db, ssh, browser, jobs, cron, ...)
     are included only when the user's message mentions them
     (keyword match), or when they were used in the last 2 turns
     (conversation continuity).
   - Tools the filter does not recognise (e.g. user-synthesised tools,
     persisted skills) are FAIL-OPEN: always included, so nothing can
     silently disappear.

Nothing here deletes or renames any tool: ``agent.tools`` keeps all
117 entries and the executor still resolves by name from the full
registry. Only the *schemas sent to the model* shrink.

Env overrides:
  FULLAGENT_TOOL_FILTER=off       -> old behaviour (all tools, raw schemas)
  FULLAGENT_TOOL_FILTER=compress  -> all tools, compressed schemas only
  (default)                       -> filter + compress
"""

from __future__ import annotations

import json
import os
import re

# ---------------------------------------------------------------------------
# 1. Compression
# ---------------------------------------------------------------------------

DESC_MAX = 150       # max chars kept of a tool's description
PROP_DESC_MAX = 60   # max chars kept of a parameter's description

_compressed_cache: dict[str, dict] = {}


def _first_sentence(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    # cut at the first sentence boundary inside the limit, else hard cut
    m = re.search(r"[.!?]\s", text[:limit])
    if m:
        return text[: m.start() + 1]
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut + "…"


def compress_description(text: str) -> str:
    """Keep the first sentence of a tool description (<=150 chars)."""
    return _first_sentence(text, DESC_MAX)


def _compress_params(params: object, _in_properties: bool = False) -> object:
    """Recursively compress a JSON-schema parameters dict.

    Preserves: type, properties (names), required, enum, default.
    Truncates: per-property "description" (<=60 chars).
    Drops: "examples" *annotations* (verbose, never needed for a
    correct call) — but never a parameter actually *named* "examples"
    (synthesize_tool requires one).
    """
    if isinstance(params, dict):
        out: dict = {}
        for k, v in params.items():
            if k == "examples" and not _in_properties:
                continue  # annotation-level examples only
            if k == "description" and isinstance(v, str):
                out[k] = _first_sentence(v, PROP_DESC_MAX)
            elif k == "properties" and isinstance(v, dict):
                out[k] = {pk: _compress_params(pv, True)
                          for pk, pv in v.items()}
            else:
                out[k] = _compress_params(v, _in_properties)
        return out
    if isinstance(params, list):
        return [_compress_params(v, _in_properties) for v in params]
    return params


def compressed_schema(tool) -> dict:
    """Compact OpenAI function schema for one Tool (cached by name)."""
    cached = _compressed_cache.get(tool.name)
    if cached is not None:
        return cached
    schema = {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": compress_description(tool.description),
            "parameters": _compress_params(tool.parameters),
        },
    }
    _compressed_cache[tool.name] = schema
    return schema


def compressed_schemas(tools: dict) -> list[dict]:
    """Compressed schemas for every tool in the registry (unfiltered)."""
    return [compressed_schema(t) for t in tools.values()]


def raw_schemas(tools: dict) -> list[dict]:
    """Original uncompressed schemas (pre-fix behaviour, for comparison)."""
    return [t.openai_schema() for t in tools.values()]


def schemas_chars(schemas: list[dict]) -> int:
    return len(json.dumps(schemas, ensure_ascii=False))


# ---------------------------------------------------------------------------
# 2. Per-turn filtering
# ---------------------------------------------------------------------------

# Always sent: the general coding-assistant loop + reasoning backbone.
CORE_TOOLS = frozenset({
    # files
    "read_file", "write_file", "edit_file", "list_dir", "file_info",
    "create_directory", "copy_path", "move_path", "delete_path",
    "apply_patch", "MultiEdit",
    # shell
    "run_command", "live_shell", "live_shell_reset",
    "BashBG", "BashOutput",
    # search / code intelligence
    "search_files", "glob_files",
    "code_symbols", "code_impact",
    "analyze_code", "graph_index", "graph_query", "graph_impact",
    "measure_coverage",
    # web
    "web_fetch", "web_search",
    # planning / review
    "TodoWrite", "TodoRead", "PlanWrite", "ReviewChanges",
    # reasoning backbone
    "brain_recall", "inspect_why", "verify_plan", "predict_impact",
    "knowledge_ask",
    # sessions / misc small
    "SessionList", "SessionNew", "SessionSwitch",
    "ImageRead", "OutputStyle", "NotebookRead",
})

# keyword (word-boundary regex, lowercased) -> advanced tool names.
# A group is included when ANY keyword matches the user message.
KEYWORD_GROUPS: tuple[tuple[frozenset[str], frozenset[str]], ...] = (
    (frozenset({"docker", "container"}),
     frozenset({"docker_images", "docker_logs", "docker_ps",
                "docker_run", "docker_stop"})),
    (frozenset({"git", "commit", "branch", "push", "pull", "merge",
                "rebase", "repo", "checkout", "stash"}),
     frozenset({"GitStatus", "GitCommit", "GitBranch", "GitPush",
                "GitDiff", "GitLog"})),
    (frozenset({"database", "sql", "sqlite", "postgres", r"db\b"}),
     frozenset({"DbQuery", "DbSchema", "DbTables"})),
    (frozenset({"email", "mail", "smtp"}),
     frozenset({"EmailConfig", "EmailSend"})),
    (frozenset({"ssh", "remote host"}),
     frozenset({"SshRun", "SshTest"})),
    (frozenset({"browser", "playwright", "chromium", "headless"}),
     frozenset({"BrowserOpen", "BrowserClick", "BrowserSnapshot",
                "BrowserClose"})),
    (frozenset({r"\bjob\b", "background job", "daemon"}),
     frozenset({"JobStart", "JobStatus", "JobLogs", "JobCancel"})),
    (frozenset({"cron", "schedul"}),
     frozenset({"CronAdd", "CronList", "CronLog", "CronRemove"})),
    (frozenset({"watch", "file watcher"}),
     frozenset({"WatchAdd", "WatchList", "WatchRemove"})),
    (frozenset({"webhook"}),
     frozenset({"WebhookAdd", "WebhookList", "WebhookRemove",
                "WebhookTest"})),
    (frozenset({"notif"}),
     frozenset({"NotifySend", "NotifyTest"})),
    (frozenset({"voice", "audio", "transcrib", "speech", "microphone"}),
     frozenset({"VoiceRecord", "VoiceTranscribe"})),
    (frozenset({"sandbox"}),
     frozenset({"SandboxInfo", "SandboxedRun"})),
    (frozenset({"backup", "restore"}),
     frozenset({"BackupCreate", "BackupList", "BackupRestore"})),
    (frozenset({"supervis"}),
     frozenset({"SuperviseAdd", "SuperviseList", "SuperviseLogs",
                "SuperviseStop"})),
    (frozenset({"log tail", "follow log", "streaming log", "log stream"}),
     frozenset({"LogFollow", "LogList", "LogTail"})),
    (frozenset({"retry", "retr", "flaky"}),
     frozenset({"RetryRun", "RetryShell"})),
    (frozenset({"progress bar"}),
     frozenset({"ProgressStart", "ProgressUpdate", "ProgressDone"})),
    (frozenset({"notebook", "ipynb", "jupyter"}),
     frozenset({"NotebookEdit", "NotebookInsert", "NotebookDelete"})),
    (frozenset({r"\binit\b", "scaffold"}),
     frozenset({"InitProject"})),
    (frozenset({"pull request", r"\bpr\b"}),
     frozenset({"ReviewPR"})),
    (frozenset({"debate"}),
     frozenset({"run_debate"})),
    (frozenset({r"\bsubagents?\b", r"\bagents?\b", "parallel",
                "delegat", "task group", "specialists", "fan out",
                "team of", "independent tasks"}),
     frozenset({"Task", "TaskGroup", "TaskWriteGroup", "TaskOutput",
                "TaskCollect", "TaskList", "TaskStop"})),
    (frozenset({"mcts", "monte carlo"}),
     frozenset({"mcts_solve"})),
    (frozenset({"synthesi", "new tool", "create a tool"}),
     frozenset({"synthesize_tool"})),
    (frozenset({"fuzz"}),
     frozenset({"fuzz_target"})),
)

# Precompile one regex per group.
_GROUP_RES: tuple[tuple[re.Pattern, frozenset[str]], ...] = tuple(
    (re.compile(r"(?:%s)" % "|".join(sorted(kw))), tools)
    for kw, tools in KEYWORD_GROUPS
)

# Every tool name the filter knows about (core + all groups).
_KNOWN: frozenset[str] = CORE_TOOLS | frozenset(
    t for _, tools in KEYWORD_GROUPS for t in tools)


def select_tools(registry: dict, user_text: str,
                 recent: tuple[str, ...] = ()) -> list[str]:
    """Names of tools to send this turn, in registry order.

    CORE always + keyword-matched groups + recently used + any
    unrecognised tool (fail-open: synthesised tools and persisted
    skills can never vanish from the model's view).
    """
    text = (user_text or "").lower()
    want = set(CORE_TOOLS)
    for rx, tools in _GROUP_RES:
        if rx.search(text):
            want.update(tools)
    want.update(r for r in recent if r in registry)
    # fail-open for tools the filter map doesn't know (skills, etc.)
    want.update(n for n in registry if n not in _KNOWN)
    return [n for n in registry if n in want]


def mode() -> str:
    """'off' | 'compress' | 'filter' (default)."""
    return os.environ.get("FULLAGENT_TOOL_FILTER", "filter").lower()


def filter_enabled() -> bool:
    return mode() == "filter"


def compression_enabled() -> bool:
    return mode() in ("filter", "compress")


def filter_signature(user_text: str) -> str:
    """Cache key fragment: which groups the text activates."""
    text = (user_text or "").lower()
    hits = [str(i) for i, (rx, _) in enumerate(_GROUP_RES)
            if rx.search(text)]
    return ",".join(hits)


def selected_schemas(registry: dict, user_text: str,
                     recent: tuple[str, ...] = ()) -> list[dict]:
    """The actual per-turn payload: filtered + compressed schemas."""
    names = select_tools(registry, user_text, recent)
    return [compressed_schema(registry[n]) for n in names]


# ---------------------------------------------------------------------------
# 3. Progressive tool disclosure (ranking)
# ---------------------------------------------------------------------------
# Worker 12/20 (RELIABILITY): filtering *removes* irrelevant tools, but the
# model still saw ~40-60 tools in arbitrary registry order. Ranking *orders*
# the survivors so the tools most relevant to the current hint appear first
# in the model request. Schema order matters: models attend more to early
# function definitions, and a relevant-first order reduces wrong-tool
# confusion (the 512s loop had 117 tools in context during a simple edit).
#
# Data-driven: RANK_HINTS is a plain table of (keywords, ordered tools).
# To cover a new tool, add its name to the right tuple; to cover a new
# task family, add a new (keywords, tools) row. Nothing is hard-coded
# anywhere else. rank_tools() is stable: tools with equal relevance keep
# their input order, and an empty hint returns the input order unchanged.

# keyword (word-boundary regex, lowercased) -> tool names, MOST relevant
# first. A tool's relevance for a hint = its best (lowest) position across
# all groups whose keywords match.
RANK_HINTS: tuple[tuple[frozenset[str], tuple[str, ...]], ...] = (
    (frozenset({r"\bsubagents?\b", r"\bagents?\b", "parallel",
                "delegat", "task group", "specialists", "fan out",
                "team of", "independent tasks"}),
     ("TaskGroup", "TaskWriteGroup", "TaskCollect", "TaskOutput", "TaskList",
      "TaskStop", "Task")),
    (frozenset({"edit", "fix", "patch", "modify", "rewrite", "update",
                "refactor", "rename variable"}),
     ("MultiEdit", "edit_file", "apply_patch", "write_file", "read_file")),
    (frozenset({"file", "files", "write", "create file", "new file",
                "directory", "folder", "path"}),
     ("write_file", "read_file", "edit_file", "MultiEdit", "apply_patch",
      "list_dir", "file_info", "create_directory", "copy_path",
      "move_path", "delete_path")),
    (frozenset({"read", "show", "view", "open", "display", "print",
                "cat "}),
     ("read_file", "list_dir", "file_info", "ImageRead", "NotebookRead")),
    (frozenset({"run", "test", "execute", "pytest", "check", "build",
                "compile", "script"}),
     ("run_command", "live_shell", "BashBG",
      "BashOutput", "RetryRun", "RetryShell", "measure_coverage",
      "fuzz_target")),
    (frozenset({"git", "commit", "branch", "push", "pull", "merge",
                "rebase", "checkout", "stash", "diff"}),
     ("GitStatus", "GitDiff", "GitCommit", "GitBranch", "GitPush",
      "GitLog")),
    (frozenset({"search", "find", "grep", "locate", "where is",
                "which file"}),
     ("search_files", "glob_files", "code_symbols", "graph_query",
      "graph_index", "knowledge_ask")),
    (frozenset({"analy", "impact", "symbol", "coverage", "call graph",
                "dependenc"}),
     ("analyze_code", "code_symbols", "code_impact", "graph_impact",
      "graph_query", "graph_index", "measure_coverage", "inspect_why",
      "verify_plan", "predict_impact")),
    (frozenset({"move", "copy", "delete", "remove file", "rename",
                "mkdir"}),
     ("move_path", "copy_path", "delete_path", "create_directory")),
    (frozenset({"web", "fetch", "url", "http", "website", "download"}),
     ("web_fetch", "web_search", "BrowserOpen", "BrowserSnapshot",
      "BrowserClick", "BrowserClose")),
    (frozenset({"browser", "playwright", "chromium", "headless",
                "webpage", "click"}),
     ("BrowserOpen", "BrowserSnapshot", "BrowserClick", "BrowserClose")),
    (frozenset({"docker", "container"}),
     ("docker_ps", "docker_logs", "docker_run", "docker_stop",
      "docker_images")),
    (frozenset({"database", "sql", "sqlite", "postgres", r"db\b"}),
     ("DbQuery", "DbTables", "DbSchema")),
    (frozenset({"email", "mail", "smtp"}),
     ("EmailSend", "EmailConfig")),
    (frozenset({"ssh", "remote"}),
     ("SshRun", "SshTest")),
    (frozenset({"todo", "plan", "planning", "roadmap"}),
     ("TodoWrite", "TodoRead", "PlanWrite", "ReviewChanges",
      "verify_plan")),
    (frozenset({"review", "pull request", r"\bpr\b", "code review"}),
     ("ReviewPR", "ReviewChanges")),
    (frozenset({"cron", "schedul"}),
     ("CronAdd", "CronList", "CronLog", "CronRemove")),
    (frozenset({r"\bjob\b", "background", "daemon"}),
     ("JobStart", "JobStatus", "JobLogs", "JobCancel")),
    (frozenset({"log", "tail", "follow"}),
     ("LogTail", "LogFollow", "LogList")),
    (frozenset({"notif"}),
     ("NotifySend", "NotifyTest")),
    (frozenset({"voice", "audio", "transcrib", "speech"}),
     ("VoiceTranscribe", "VoiceRecord")),
    (frozenset({"sandbox"}),
     ("SandboxedRun", "SandboxInfo")),
    (frozenset({"backup", "restore"}),
     ("BackupCreate", "BackupList", "BackupRestore")),
    (frozenset({"supervis"}),
     ("SuperviseAdd", "SuperviseList", "SuperviseLogs",
      "SuperviseStop")),
    (frozenset({"watch", "file watcher"}),
     ("WatchAdd", "WatchList", "WatchRemove")),
    (frozenset({"webhook"}),
     ("WebhookAdd", "WebhookTest", "WebhookList", "WebhookRemove")),
    (frozenset({"progress bar"}),
     ("ProgressStart", "ProgressUpdate", "ProgressDone")),
    (frozenset({"retry", "retr", "flaky"}),
     ("RetryRun", "RetryShell")),
    (frozenset({"notebook", "ipynb", "jupyter"}),
     ("NotebookRead", "NotebookEdit", "NotebookInsert",
      "NotebookDelete")),
    (frozenset({r"\binit\b", "scaffold", "new project"}),
     ("InitProject",)),
    (frozenset({"image", "picture", "screenshot", "photo"}),
     ("ImageRead",)),
    (frozenset({"session"}),
     ("SessionList", "SessionNew", "SessionSwitch")),
    (frozenset({"brain", "memory", "recall", "remember"}),
     ("brain_recall", "knowledge_ask", "inspect_why")),
    (frozenset({"debate"}),
     ("run_debate",)),
    (frozenset({"mcts", "monte carlo"}),
     ("mcts_solve",)),
    (frozenset({"synthesi", "new tool", "create a tool"}),
     ("synthesize_tool",)),
    (frozenset({"config", "settings", "preferences"}),
     ("read_file", "edit_file", "list_dir")),
)

_RANK_RES: tuple[tuple[re.Pattern, tuple[str, ...]], ...] = tuple(
    (re.compile(r"(?:%s)" % "|".join(sorted(kw))), tools)
    for kw, tools in RANK_HINTS
)

_NO_RANK = 10 ** 9


def rank_tools(task_hint: str,
               names: list[str] | None = None) -> list[str]:
    """Order tool names by relevance to ``task_hint`` (progressive
    disclosure). Tools matching hint keywords sort first (best position
    across matched groups); everything else keeps its input order
    (stable). Empty hint -> input order unchanged.

    ``names`` defaults to the full registry in registry order.
    """
    if names is None:
        names = list(build_full_registry())
    if not task_hint:
        return list(names)
    text = task_hint.lower()
    best: dict[str, int] = {}
    for rx, tools in _RANK_RES:
        if rx.search(text):
            for i, tname in enumerate(tools):
                if i < best.get(tname, _NO_RANK):
                    best[tname] = i
    default_pos = {n: i for i, n in enumerate(names)}
    return sorted(names,
                  key=lambda n: (best.get(n, _NO_RANK),
                                 default_pos.get(n, _NO_RANK)))


def rank_signature(task_hint: str) -> str:
    """Cache key fragment: which rank groups the hint activates."""
    text = (task_hint or "").lower()
    hits = [str(i) for i, (rx, _) in enumerate(_RANK_RES)
            if rx.search(text)]
    return ",".join(hits)


# ---------------------------------------------------------------------------
# Registry reconstruction (for offline measurement / self-test)
# ---------------------------------------------------------------------------

def build_full_registry():
    """All tools the real Agent registers, without importing the
    (heavier, currently churning) agent/client modules."""
    import ast as _ast

    _REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _REPO not in __import__("sys").path:
        __import__("sys").path.insert(0, _REPO)

    from fullagent.tools import build_registry as _br
    registry = _br()
    mods = ("todos", "multiedit", "bgsh", "planmode", "codeinit",
            "review", "notebook", "permissions", "outstyle",
            "webfetch", "checkpoints", "hooks",
            "mcp", "skills", "doctor", "statuscmd",
            "permrules", "configcmd", "images", "export", "vimmode",
            "thinking", "undocmd", "diffpreview", "costtrack",
            "jsonout", "ctxmeter", "prreview", "smartctx",
            "termsetup", "plugins", "supervise", "turnresume",
            "cronsched", "watch", "sshops", "streamlog", "voicein",
            "browserauto", "multisess", "backup", "sandbox",
            "smartretry", "progress", "dockerops", "jobs", "dbtools",
            "emailops", "gitops", "notify", "webhook")

    class _Fake:
        def __init__(self, tools):
            self.tools = tools

    fake = _Fake(registry)
    for mod_name in mods:
        try:
            mod = __import__(f"fullagent.{mod_name}",
                             fromlist=["register"])
            mod.register(fake)
        except Exception:
            pass  # same tolerance as Agent._register_feature_modules
    from fullagent.roles import ROLES as _ROLES
    _role_names = sorted(_ROLES)
    _read_roles = sorted(role for role, spec in _ROLES.items()
                         if not spec.get("writes"))
    _task_item = {
        "type": "object",
        "properties": {
            "task": {"type": "string"},
            "role": {"type": "string", "enum": _role_names},
            "name": {"type": "string"},
            "context": {"type": "string"},
            "model_id": {"type": "string"},
        },
        "required": ["task"],
    }
    _write_item = {
        "type": "object",
        "properties": {
            "task": {"type": "string"},
            "name": {"type": "string"},
            "context": {"type": "string"},
            "write_scope": {
                "type": "array", "items": {"type": "string"},
                "minItems": 1, "maxItems": 16,
            },
        },
        "required": ["task", "write_scope"],
    }
    # Tool() literals inside Agent._register_*_tools (need real
    # agent internals to run, so extract statically instead)
    _ns = {"_STR": {"type": "string"}, "roles": _role_names,
           "read_roles": _read_roles, "task_item": _task_item,
           "write_item": _write_item}
    tree = _ast.parse(open(os.path.join(
        _REPO, "fullagent", "agent.py")).read())

    def _ev(node):
        if isinstance(node, _ast.Constant):
            return node.value
        if (isinstance(node, _ast.BinOp)
                and isinstance(node.op, _ast.Add)):
            return _ev(node.left) + _ev(node.right)
        if isinstance(node, _ast.Name):
            if node.id not in _ns:
                raise ValueError(f"non-literal name: {node.id}")
            return _ns[node.id]
        if isinstance(node, _ast.Dict):
            return {_ev(k): _ev(v)
                    for k, v in zip(node.keys, node.values)}
        if isinstance(node, _ast.List):
            return [_ev(e) for e in node.elts]
        if isinstance(node, _ast.Tuple):
            return tuple(_ev(e) for e in node.elts)
        raise ValueError("non-literal")

    from fullagent.tools import Tool as _Tool
    for node in _ast.walk(tree):
        if (isinstance(node, _ast.FunctionDef)
                and node.name.startswith("_register_")):
            for sub in _ast.walk(node):
                if (isinstance(sub, _ast.Call)
                        and isinstance(sub.func, _ast.Name)
                        and sub.func.id == "Tool"
                        and len(sub.args) >= 3):
                    try:
                        name = _ev(sub.args[0])
                        desc = _ev(sub.args[1])
                        params = _ev(sub.args[2])
                    except ValueError:
                        continue
                    if name not in registry:
                        registry[name] = _Tool(
                            name, desc, params,
                            lambda *a, **k: "stub")
    return registry



# ---------------------------------------------------------------------------
# Self-test (offline: no API key needed)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    reg = build_full_registry()
    n_total = len(reg)
    assert n_total >= 100, f"expected at least 100 supported tools, got {n_total}"
    removed = {"spawn_agent", "spawn_parallel", "send_to_agent",
               "wait_for_agents", "close_agent", "resume_agent",
               "run_market", "race_strategies"}
    assert removed.isdisjoint(reg), f"removed delegation tools remain: {removed & reg.keys()}"
    required_subagent_tools = {"Task", "TaskGroup", "TaskWriteGroup",
                               "TaskOutput", "TaskCollect", "TaskList",
                               "TaskStop"}
    assert required_subagent_tools <= set(reg), (
        f"subagent tools missing: {required_subagent_tools - set(reg)}")
    # every tool still registered AND callable
    for tname, tool in reg.items():
        assert callable(tool.handler), f"{tname}: handler not callable"

    raw = raw_schemas(reg)
    before_chars = schemas_chars(raw)
    before_names = len(raw)

    # 1) compression alone (all tools, like FULLAGENT_TOOL_FILTER=compress)
    comp_all = compressed_schemas(reg)
    comp_chars = schemas_chars(comp_all)
    assert len(comp_all) == n_total, "compression dropped tools!"
    # schema contract preserved for every tool
    for s in comp_all:
        fn = s["function"]
        assert fn["name"] and isinstance(fn["parameters"], dict)
    # required lists + types survive compression byte-identical
    for r, c in zip(raw, comp_all):
        rp, cp = r["function"]["parameters"], c["function"]["parameters"]
        assert rp.get("required", []) == cp.get("required", []), \
            f'{r["function"]["name"]}: required list changed!'
        rprops = rp.get("properties", {})
        cprops = cp.get("properties", {})
        assert set(rprops) == set(cprops), \
            f'{r["function"]["name"]}: properties changed!'
        for pk, pv in rprops.items():
            assert pv.get("type") == cprops[pk].get("type"), \
                f'{r["function"]["name"]}.{pk}: type changed!'

    # 2) filter + compress on a plain coding turn
    plain = "fix the bug in parser.py where empty input crashes"
    sel = select_tools(reg, plain)
    sel_schemas = selected_schemas(reg, plain)
    sel_chars = schemas_chars(sel_schemas)
    assert set(sel) <= set(reg), "filter invented unknown tools"
    assert CORE_TOOLS <= set(sel), "filter dropped CORE tools!"
    assert "docker_ps" not in sel and "GitCommit" not in sel, \
        "filter leaked advanced tools on a plain turn"

    # 3) keyword activation
    dk = select_tools(reg, "check docker containers and restart the db")
    assert {"docker_ps", "docker_run", "DbQuery"} <= set(dk), \
        f"keyword groups not activated: {sorted(dk)}"
    g = select_tools(reg, "commit and push my changes")
    assert {"GitCommit", "GitPush", "GitStatus"} <= set(g)

    # 4) recent-turn continuity
    cont = select_tools(reg, plain, recent=("docker_ps",))
    assert "docker_ps" in cont, "recent tools not carried over"

    # 5) every known advanced tool is reachable via SOME keyword
    for kw_set, tools in KEYWORD_GROUPS:
        probe = " ".join(sorted(kw_set)).replace(r"\b", "")
        got = select_tools(reg, probe)
        assert set(tools) <= set(got), \
            f"group unreachable: {sorted(set(tools) - set(got))}"

    def _tok(chars):
        return chars // 4

    # 6) progressive disclosure ranking
    all_names = list(reg)
    r_edit = rank_tools("edit the config file", all_names)
    assert set(r_edit) == set(all_names), "rank dropped/added tools"
    assert {"MultiEdit", "edit_file", "read_file"} <= set(r_edit[:5]), \
        f"edit hint not ranked first: {r_edit[:6]}"
    r_test = rank_tools("run the tests", all_names)
    assert r_test[0] == "run_command", \
        f"run hint: expected run_command first, got {r_test[0]}"
    assert {"live_shell", "BashBG"} <= set(r_test[:8]), \
        f"shell tools not near top: {r_test[:8]}"
    r_git = rank_tools("commit and push my changes", all_names)
    assert {"GitCommit", "GitPush", "GitStatus"} <= set(r_git[:6]), \
        f"git tools not near top: {r_git[:6]}"
    r_dock = rank_tools("check docker containers", all_names)
    assert "docker_ps" in r_dock[:4], \
        f"docker_ps not near top: {r_dock[:4]}"
    # empty hint -> stable default order
    assert rank_tools("", all_names) == all_names, \
        "empty hint changed default order"
    assert rank_tools(None, all_names) == all_names, \
        "None hint changed default order"
    # ranking never reorders ties against the default order
    r_plain = rank_tools("something with no keywords xyzzy", all_names)
    assert r_plain == all_names, "unmatched hint reordered tools"

    print(f"tools registered+callable : {n_total}")
    print(f"BEFORE  all schemas       : {before_chars:,} chars "
          f"(~{_tok(before_chars):,} tok)")
    print(f"AFTER   compress-only     : {comp_chars:,} chars "
          f"(~{_tok(comp_chars):,} tok, "
          f"{100 * comp_chars // before_chars}% of before)")
    print(f"AFTER   filter+compress   : {len(sel)} tools, "
          f"{sel_chars:,} chars (~{_tok(sel_chars):,} tok, "
          f"{100 * sel_chars // before_chars}% of before)")
    print(f"AFTER   docker/db turn    : {len(dk)} tools, "
          f"{schemas_chars(selected_schemas(reg, 'check docker containers and restart the db')):,} chars")
    print("TOOLFILTER SELF-TEST PASS")
