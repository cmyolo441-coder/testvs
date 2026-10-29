"""Slash command argument parsing for fullagent.

Supports Claude Code style placeholders in slash-command templates:

    $ARGUMENTS   -> " ".join(args)
    $@           -> " ".join(args)
    $1, $2, ...  -> positional args (1-based)

Anything else (unknown placeholders, bare $) is left untouched.
Pure functions, stdlib only. Never raises on weird input: best effort.

Contract (used by slashcmds.py; keep these names/signatures exact):
    split_args(text: str) -> list[str]
    expand_arguments(template: str, args: list[str]) -> str
    parse_command_line(text: str) -> tuple[str, list[str]]
"""

import re
import shlex

__all__ = ["split_args", "expand_arguments", "parse_command_line"]

# Matches $ARGUMENTS, $@, $1..$99 as whole tokens/occurrences.
_PLACEHOLDER_RE = re.compile(r"\$(?:ARGUMENTS|@|[1-9][0-9]*)")


def split_args(text):
    """Shell-like split honoring quotes (stdlib shlex). Never raises."""
    if text is None:
        return []
    try:
        return shlex.split(text)
    except Exception:
        # Best effort fallback on unbalanced quotes etc.
        try:
            lexer = shlex.shlex(str(text), posix=True)
            lexer.whitespace_split = True
            return list(lexer)
        except Exception:
            s = str(text).replace('"', " ").replace("'", " ").strip()
            return s.split() if s else []


def expand_arguments(template, args):
    """Expand $ARGUMENTS/$@/$N placeholders in template.

    Unknown placeholders (e.g. $FOO, $0, $$) are left as-is.
    Missing positional args (e.g. $3 with only 2 args) are left as-is.
    Never raises.
    """
    if template is None:
        return ""
    if args is None:
        args = []
    try:
        args = list(args)
    except Exception:
        args = []

    def _repl(match):
        token = match.group(0)
        if token == "$ARGUMENTS" or token == "$@":
            return " ".join(str(a) for a in args)
        # Positional $1, $2, ...
        idx = int(token[1:]) - 1
        if 0 <= idx < len(args):
            return args[idx]
        return token  # missing positional: leave as-is

    try:
        return _PLACEHOLDER_RE.sub(_repl, str(template))
    except Exception:
        return str(template)


def parse_command_line(text):
    """Split "/cmd arg1 arg2" -> ("/cmd", ["arg1", "arg2"]).

    Returns ("", []) for empty/None input. Never raises.
    """
    parts = split_args(text)
    if not parts:
        return "", []
    return parts[0], parts[1:]


def _check(name, cond):
    print(("PASS" if cond else "FAIL"), "-", name)
    return cond


def _selftest():
    ok = True

    # split_args: quoted args
    ok &= _check("quoted args", split_args('a "b c" d') == ["a", "b c", "d"])
    ok &= _check("single quotes", split_args("a 'b c'") == ["a", "b c"])
    ok &= _check("escaped quote", split_args(r'a "b\"c"') == ["a", 'b"c'])
    ok &= _check("empty input", split_args("") == [])
    ok &= _check("None input", split_args(None) == [])
    ok &= _check("unbalanced quote", split_args('"abc') == ["abc"])
    ok &= _check("whitespace only", split_args("   ") == [])

    # expand_arguments: $ARGUMENTS
    ok &= _check(
        "$ARGUMENTS",
        expand_arguments("do $ARGUMENTS now", ["x", "y"]) == "do x y now",
    )
    ok &= _check(
        "$@",
        expand_arguments("do $@ now", ["x", "y"]) == "do x y now",
    )
    # $1/$2 positional
    ok &= _check(
        "$1 $2",
        expand_arguments("$1 then $2", ["a", "b"]) == "a then b",
    )
    ok &= _check(
        "multi-digit $10",
        expand_arguments("$10", ["a"] * 10) == "a",
    )
    # missing placeholders untouched
    ok &= _check(
        "missing $3 untouched",
        expand_arguments("$1 $3", ["a", "b"]) == "a $3",
    )
    ok &= _check(
        "$FOO untouched",
        expand_arguments("$FOO bar", ["a"]) == "$FOO bar",
    )
    ok &= _check(
        "$0 untouched",
        expand_arguments("$0", ["a"]) == "$0",
    )
    ok &= _check(
        "$$ untouched",
        expand_arguments("$$", ["a"]) == "$$",
    )
    ok &= _check(
        "bare $ untouched",
        expand_arguments("price $", ["a"]) == "price $",
    )
    # empty args / empty template
    ok &= _check(
        "empty args",
        expand_arguments("$ARGUMENTS", []) == "",
    )
    ok &= _check(
        "empty template",
        expand_arguments("", ["a"]) == "",
    )
    ok &= _check(
        "None template",
        expand_arguments(None, ["a"]) == "",
    )
    ok &= _check(
        "no placeholders",
        expand_arguments("plain text", ["a"]) == "plain text",
    )
    ok &= _check(
        "placeholder inside word",
        expand_arguments("pre$1post", ["X"]) == "preXpost",
    )

    # parse_command_line
    ok &= _check(
        "parse basic",
        parse_command_line("/cmd arg1 arg2") == ("/cmd", ["arg1", "arg2"]),
    )
    ok &= _check(
        "parse quoted",
        parse_command_line('/cmd "a b" c') == ("/cmd", ["a b", "c"]),
    )
    ok &= _check("parse empty", parse_command_line("") == ("", []))
    ok &= _check("parse None", parse_command_line(None) == ("", []))
    ok &= _check(
        "parse no args",
        parse_command_line("/help") == ("/help", []),
    )

    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys

    sys.exit(0 if _selftest() else 1)
