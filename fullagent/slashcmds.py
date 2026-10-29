"""Custom user slash commands from ~/.fullagent/commands/*.md (Claude Code style).

Each ``<name>.md`` file holds YAML-ish frontmatter followed by a prompt body:

    ---
    description: Summarize the given file
    ---
    Please summarize the following file:
    $ARGUMENTS

The body may contain ``$ARGUMENTS`` (and, via fullagent.cmdargs, ``$@`` /
``$1`` .. ``$N``) placeholders. Typed ``/name args`` expands the template
with the user's args and the result is submitted as a normal turn.

Pure stdlib. Never raises on weird input: best effort everywhere.
Contract used by the TUI wiring snippet (kept minimal on purpose):

    load_commands(dir) -> dict[str, {"description": str, "template": str}]
    expand(name, arg_text) -> str   # "" when the command is unknown
"""

import os
import re

try:
    from .cmdargs import split_args, expand_arguments
except ImportError:  # minimal local fallback (never import .tui)
    import shlex

    def split_args(text):
        if text is None:
            return []
        try:
            return shlex.split(text)
        except Exception:
            s = str(text).strip()
            return s.split() if s else []

    def expand_arguments(template, args):
        if template is None:
            return ""
        try:
            return str(template).replace("$ARGUMENTS", " ".join(args or []))
        except Exception:
            return str(template)


__all__ = ["default_commands_dir", "load_commands", "expand"]

_FRONTMATTER_RE = re.compile(r"^---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n(.*)$",
                             re.DOTALL)


def default_commands_dir():
    """Path of the user commands directory: ~/.fullagent/commands."""
    base = os.environ.get("FULLAGENT_HOME") or os.path.expanduser(os.path.join("~", ".fullagent"))
    return os.path.join(base, "commands")


def _parse_file(path):
    """Read one .md file -> (description, template). Never raises."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except Exception:
        return "", ""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        # No frontmatter: whole file is the template.
        return "", text
    description = ""
    for line in m.group(1).splitlines():
        key, _, value = line.partition(":")
        if key.strip().lower() == "description" and value.strip():
            description = value.strip()
            break
    return description, m.group(2)


def load_commands(commands_dir=None):
    """Load all custom commands from <commands_dir>/*.md.

    Returns {name: {"description": str, "template": str}} keyed by the
    lowercased filename stem (so "/Review" matches review.md).
    Missing/non-dir input -> {}. Never raises.
    """
    commands = {}
    if commands_dir is None:
        commands_dir = default_commands_dir()
    try:
        entries = sorted(os.listdir(commands_dir))
    except Exception:
        return commands
    for entry in entries:
        if not entry.lower().endswith(".md"):
            continue
        stem = entry[: -len(".md")]
        if not stem:
            continue
        description, template = _parse_file(os.path.join(commands_dir, entry))
        if not template.strip():
            continue  # skip empty command files
        commands[stem.lower()] = {"description": description,
                                  "template": template}
    return commands


def expand(name, arg_text, commands_dir=None):
    """Expand a custom command into the prompt text for the agent.

    Looks up ``name`` (case-insensitive) in the custom commands dir,
    expands $ARGUMENTS/$@/$N with split args of ``arg_text``.
    Returns "" for unknown commands or empty results. Never raises.
    """
    try:
        if not name:
            return ""
        commands = load_commands(commands_dir)
        cmd = commands.get(str(name).lower())
        if not cmd:
            return ""
        args = split_args(arg_text or "")
        return expand_arguments(cmd["template"], args).strip()
    except Exception:
        return ""


def _check(label, cond):
    print(("PASS" if cond else "FAIL"), "-", label)
    return cond


def _selftest():
    import tempfile

    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        body = ("---\n"
                "description: Summarize the given files\n"
                "---\n"
                "Please summarize the following files:\n"
                "$ARGUMENTS\n")
        with open(os.path.join(tmp, "summarize.md"), "w",
                  encoding="utf-8") as fh:
            fh.write(body)
        # non-md file must be ignored
        with open(os.path.join(tmp, "notes.txt"), "w",
                  encoding="utf-8") as fh:
            fh.write("hello")

        cmds = load_commands(tmp)
        ok &= _check("load_commands finds one command",
                     set(cmds) == {"summarize"})
        ok &= _check("description parsed",
                     cmds["summarize"]["description"] ==
                     "Summarize the given files")
        ok &= _check("$ARGUMENTS in template",
                     "$ARGUMENTS" in cmds["summarize"]["template"])

        out = expand("summarize", "a.py b.py", tmp)
        ok &= _check("expand substitutes args",
                     out == "Please summarize the following files:\na.py b.py")
        ok &= _check("expand case-insensitive",
                     expand("SUMMARIZE", "x", tmp).endswith("x"))
        ok &= _check("expand unknown name -> ''",
                     expand("nope", "x", tmp) == "")
        ok &= _check("expand quoted args",
                     expand("summarize", '"a b" c', tmp).endswith("a b c"))

        # no frontmatter: whole file is the template
        with open(os.path.join(tmp, "plain.md"), "w",
                  encoding="utf-8") as fh:
            fh.write("Just do $ARGUMENTS now.")
        cmds2 = load_commands(tmp)
        ok &= _check("no-frontmatter template",
                     cmds2["plain"]["template"] == "Just do $ARGUMENTS now."
                     and cmds2["plain"]["description"] == "")

    ok &= _check("missing dir -> {}", load_commands("/nonexistent-xyz-123") == {})
    ok &= _check("None dir (home) does not raise",
                 isinstance(load_commands(None), dict))

    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys

    sys.exit(0 if _selftest() else 1)
