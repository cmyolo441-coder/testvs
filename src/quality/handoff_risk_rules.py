"""Small deterministic command vocabulary; never evaluate or execute brief text."""
from __future__ import annotations

from dataclasses import dataclass
import re
import shlex
from typing import Final

from .handoff_risk_model import Finding, FindingId, ScanError, Signal, finding

_NEGATED: Final = re.compile(r"^(?:please\s+)?(?:do\s+not|don['’]t|never|avoid|must\s+not)\b", re.I)
_EXAMPLE: Final = re.compile(r"^(?:for\s+example\b|examples?\s*:|illustration\s*:|explanation\s*:)", re.I)
_PREFIX: Final = re.compile(r"^(?:(?:please|run|execute|then|and)\s+)+", re.I)


@dataclass(frozen=True, slots=True)
class RuleContext:
    protected_branches: tuple[str, ...]
    current_branch: str | None = None


def _command_ids(tokens: list[str], context: RuleContext) -> set[FindingId]:
    """Classify parsed argv, not arbitrary matches inside argument strings."""
    ids: set[FindingId] = set()
    words = list(tokens)
    while words and words[0] in ("sudo", "command", "env"):
        _ = words.pop(0)
    while words and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", words[0]):
        _ = words.pop(0)
    if not words:
        return ids
    executable = words[0].rsplit("/", 1)[-1]
    args = words[1:]
    lower = [word.lower() for word in words]
    if executable == "git":
        # Git's global options with values must not become subcommands.
        while args and args[0].startswith("-"):
            width = 2 if args[0] in ("-C", "-c", "--git-dir", "--work-tree") else 1
            args = args[width:]
        if args:
            operation, *rest = args
            if (operation == "reset" and "--hard" in rest
                    or operation == "clean" and any(re.fullmatch(r"-[^-]*f[^-]*", a) for a in rest)
                    and not any(a in ("--dry-run", "-n") or re.fullmatch(r"-[^-]*n[^-]*", a) for a in rest)
                    or operation == "branch" and "-D" in rest):
                ids.add("destructive_git")
            targets: list[str] = []
            if operation == "push" and not any(a in ("--dry-run", "-n") for a in rest):
                positional = [a for a in rest if not a.startswith("-")]
                # First positional is the remote. Refs after it name destinations.
                targets = [a.lstrip("+").split(":")[-1].removeprefix("refs/heads/") for a in positional[1:]]
                if len(positional) <= 1 and context.current_branch:
                    targets.append(context.current_branch)
                if "--all" in rest or "--mirror" in rest:
                    targets.extend(context.protected_branches)
            if operation in ("commit", "merge", "rebase", "reset", "cherry-pick", "am") and context.current_branch:
                targets.append(context.current_branch)
            if operation in ("branch", "checkout", "switch") and any(a in ("-f", "-D", "-B", "-C", "--force") for a in rest):
                targets.extend(a for a in rest if not a.startswith("-"))
            if any(target in context.protected_branches for target in targets):
                ids.add("protected_branch_write")
    if executable in ("rm", "rmdir", "shred") and any(not a.startswith("-") for a in args):
        ids.add("destructive_filesystem")
        if any(re.search(r"(?:^|/)tests?(?:/|\.|$)", a, re.I) for a in args):
            ids.add("test_weakening")
    if (lower[0] in ("drop", "truncate") and len(lower) > 1 and lower[1] in ("table", "database", "schema")
            or lower[:2] == ["delete", "from"] and "where" not in lower):
        ids.add("destructive_database")
    if lower[0] in ("delete", "remove", "disable", "bypass", "weaken", "skip") and any(
        re.fullmatch(r"tests?|testing|verification", word) for word in lower[1:]
    ):
        ids.add("test_weakening")
    if executable in ("pytest", "python", "python3") and any(
        a.startswith(("--ignore", "--deselect")) for a in args
    ):
        ids.add("test_weakening")
    if executable in ("psql", "mysql", "sqlite3"):
        for index, arg in enumerate(args):
            if arg in ("-c", "--command", "-e", "--execute") and index + 1 < len(args):
                for nested in _segments(args[index + 1]):
                    ids.update(_command_ids(nested, context))
    return ids


def _segments(text: str) -> list[list[str]]:
    """Let shlex preserve quoted arguments while splitting shell control operators."""
    # Apostrophes inside prose words are not shell quote delimiters.
    lexical_text = re.sub(r"(?<=[A-Za-z])'(?=[A-Za-z])", "’", text)
    lexer = shlex.shlex(lexical_text, posix=True, punctuation_chars=";&|")
    lexer.whitespace_split = True
    lexer.commenters = "#"
    segments: list[list[str]] = [[]]
    try:
        for token in lexer:
            if token and all(char in ";&|" for char in token):
                segments.append([])
            else:
                segments[-1].append(token)
    except ValueError as exc:
        raise ScanError("command_syntax_invalid") from exc
    return segments


def brief_findings(brief: str, context: RuleContext) -> list[Finding]:
    """One finding per rule, with first actionable line offset in UTF-8 bytes."""
    found: dict[FindingId, Finding] = {}
    illustrative = False
    fenced = False
    offset = 0
    # Preserve byte offsets for ASCII continuation pairs, LF or CRLF: a brief
    # file written on Windows carries \<CR><LF>, which must join identically.
    text = brief.replace("\\\r\n", "   ").replace("\\\n", "  ")
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        line_bytes = line.encode("utf-8")
        start = offset
        offset += len(line_bytes)
        if stripped.startswith(("```", "~~~")):
            fenced = not fenced
            if not fenced:
                illustrative = False
            continue
        if not stripped and not fenced:
            illustrative = False
        if _EXAMPLE.match(stripped) or _NEGATED.match(stripped) and stripped.endswith(":"):
            illustrative = stripped.endswith(":")
            continue
        if illustrative or stripped.startswith(">") or not stripped:
            continue
        # Backticks wrapping each command survive shlex, which still preserves
        # quoted arguments and does not split their embedded semicolons.
        actionable = _PREFIX.sub("", stripped)
        for tokens in _segments(actionable):
            clause = " ".join(tokens)
            if _NEGATED.match(clause):
                continue
            prefix_free = _PREFIX.sub("", clause)
            prefix_count = len(clause.split()) - len(prefix_free.split())
            commands = tokens[prefix_count:]
            if commands and commands[0].startswith("`") and commands[-1].endswith("`"):
                commands[0] = commands[0][1:]
                commands[-1] = commands[-1][:-1]
            for kind in sorted(_command_ids(commands, context)):
                if kind not in found:
                    found[kind] = finding(Signal(kind, line_bytes, start))
    return list(found.values())
