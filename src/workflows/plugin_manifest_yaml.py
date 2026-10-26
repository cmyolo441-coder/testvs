"""Bounded reader for the top-level block structure of a Hermes `plugin.yaml`.

This is deliberately not a YAML implementation and is not a substitute for one.
It models exactly the shape a plugin manifest declares its lists in -- top-level
mapping keys, an inline scalar, or a block sequence of scalars -- and refuses to
guess at anything else. A key whose value uses a construct outside that subset
(a nested mapping, a block scalar, a flow collection, an anchor or a tag) is
reported as ``unmodeled`` rather than parsed into something the file does not
say. The caller decides whether an unmodeled value is a problem: for
``description: >-`` it is not, for ``provides_hooks`` it is.

The subset covers how manifests are actually written, because a construct this
reader refuses costs the audit the whole declaration. A block sequence may sit
at its parent key's indentation (what ``yaml.safe_dump(default_flow_style=False)``
emits, and the commonest way a manifest gets generated), one document's ``---``
and ``...`` markers are boundaries rather than errors, a key may carry a space
before its colon, and a key needs no ASCII identifier shape.

The width is checked by ``tests/test_plugin_manifest_subset_corpus.py``, which
is committed and re-runnable: one row per construct, each recording what this
reader does and what PyYAML does with the same bytes. **That table is the
standing criterion.** The subset is revisited when a row in it changes or when a
real manifest is refused in the field.

A differential against PyYAML 6.0.3 informed the table and is not a substitute
for it, and its coverage is part of its result: both runs are fragment
recombination, so neither generates a construct absent from its own fragment
list. The control characters that made line splitting manufacture structure were
found by targeted probing, not by either fuzz. On 2026-09-15 it reported no disagreement over 17,397 shared cases from
the fragment set used that day -- but the fuzz and its fragments are not
committed, so that number cannot be re-derived here and must not be cited as the
check. A later run over different fragments found 1,872 inputs this reader read
and PyYAML rejected, every one of them a real gap; they are rows in the table
now. A number nobody can re-run is evidence for a change, never a standing
guarantee.

Being *more* permissive than a real YAML parser is the dangerous direction, so
it is closed explicitly. ``|`` and ``>`` open a block scalar whose header this
reader checks, so ``description: >-`` is unmodeled while ``requires_hermes:
>=0.21.1`` -- which YAML rejects outright -- makes the manifest unreadable. An
anchor or an alias is refused for the same reason: resolving one is semantics
this reader does not implement, and reading on as though it had resolved is what
let ``<<: *base`` pass.

The subset is still incomplete, so the caller also carries one total rule: a
manifest that was not understood never reports a clean result. That covers what
the subset misses and what nobody has found yet, which is what a per-construct
widening cannot do. ``tests/test_plugin_manifest_subset_corpus.py`` keeps every
refusal a recorded decision rather than an accident.

The audit that consumes this reads an untrusted local directory, so every input
dimension is bounded before any of it is interpreted, and no error message ever
echoes manifest content.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Final, Literal, TypeAlias


# Line count and line length are hard bounds on the file: past them the reader
# refuses the manifest. The per-entry bounds below are not -- an entry that
# exceeds them is reported as unmodeled, so one oversized value the caller does
# not care about (a long `description`, a long `provides_tools`) never costs it
# the fields it does care about.
MAX_MANIFEST_LINES: Final = 512
MAX_MANIFEST_LINE_CHARS: Final = 4096
MAX_MANIFEST_SEQUENCE_ITEMS: Final = 128
MAX_MANIFEST_SCALAR_CHARS: Final = 1024

# Constructs this subset does not model. A value that opens with one of these is
# reported as unmodeled instead of being read as a plain scalar that happens to
# start with a punctuation character. `|`/`>` and `&`/`*` are handled before
# this tuple is consulted, in `_scalar`, because for those the reader has to
# separate "valid YAML I do not model" from "a document YAML itself rejects".
_UNMODELED_VALUE_PREFIXES: Final = ("!", "[", "{", "?", "%", "@", "`")
# A key is any run without whitespace or a colon, ASCII or not: the caller only
# ever compares a key against three known names, so one it cannot recognise
# costs nothing while refusing the file would cost the declaration. YAML allows
# a space before the colon.
_KEY = re.compile(r"[^\s:]{1,128}")
# In block context YAML requires the colon to be followed by a space or to end
# the line, and forbids a tab as separation. Accepting `weird:novalue` or
# `name\t: p` would invent a mapping entry out of input YAML rejects outright --
# the permissive direction, and the one that lets an unloadable manifest be
# reported as read.
_ENTRY_LINE = re.compile(rf"(?P<key>{_KEY.pattern}) *:(?P<rest>(?: .*)?)")
# An anchor or alias makes the document's meaning depend on resolution this
# reader does not do, so it is refused wherever it appears in a value position.
_REFERENCE_INDICATORS: Final = ("&", "*")
# YAML forbids C0 controls other than tab, newline and carriage return, and
# forbids DEL. `str.splitlines()` breaks on five of them anyway -- VT, FF, FS,
# GS, RS -- which turned a single-line scalar into two lines and let the reader
# manufacture a declaration out of a document YAML rejects outright. Refusing
# here, before any splitting, is what matches the host's own parser. The
# terminators YAML does accept (NEL, LS, PS, CR) are deliberately not listed:
# splitlines breaks on those too, and so does PyYAML.
_FORBIDDEN_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# A plain (unquoted) scalar may not contain a raw tab: YAML forbids it outright,
# and reading one was this reader succeeding where PyYAML raises. A quoted
# scalar may, and does not come through here.
#
# Deliberately NOT extended to a colon that ends the scalar or is followed by a
# space, though `name: a provides_hooks:` is also a reader-only success. That
# rule was written, measured, and withdrawn: a sequence item like `- name: brv`
# is a nested mapping, not a plain scalar, and refusing it broke 23 of the 105
# shipped manifests. The remaining gap is recorded as a corpus row instead.
_PLAIN_SCALAR_TAB = re.compile(r"\t")
_SEQUENCE_LINE = re.compile(r"-(?:\s+(?P<item>.*))?")
_DOCUMENT_START: Final = "---"
_DOCUMENT_END: Final = "..."
# `|` and `>` open a block scalar, whose header is the indicator plus an
# optional chomping and indentation indicator and nothing else. `>-` is a
# legitimate value this reader does not model, so it is `unmodeled`. `>=0.21.1`
# is not a header at all -- it is a document YAML itself rejects -- so calling
# it `unmodeled` would let a file the host cannot load pass as read.
_BLOCK_SCALAR_HEADER = re.compile(r"[|>](?:[0-9][+-]?|[+-][0-9]?)?")

ManifestValueKind: TypeAlias = Literal["scalar", "sequence", "unmodeled"]


class PluginManifestFormatError(ValueError):
    """The manifest is outside the bounded subset this reader models."""


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """One top-level mapping entry, in document order.

    Duplicates are preserved rather than collapsed: "declared twice" is a
    finding the caller has to be able to make, not something a reader may
    silently resolve to a last-writer-wins value.
    """

    key: str
    kind: ManifestValueKind
    scalar: str = ""
    sequence: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _Line:
    indent: int
    text: str


def read_plugin_manifest_entries(text: str) -> tuple[ManifestEntry, ...]:
    """Return the manifest's top-level entries, or raise on an unreadable file."""
    lines = _single_document_lines(_bounded_significant_lines(text))
    entries: list[ManifestEntry] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.indent != 0:
            raise PluginManifestFormatError("plugin manifest indents a line outside a top-level entry")
        match = _ENTRY_LINE.fullmatch(line.text)
        if match is None:
            raise PluginManifestFormatError("plugin manifest contains a top-level line that is not a mapping entry")
        # A block sequence may sit at its parent key's own indentation, which is
        # what `yaml.safe_dump(default_flow_style=False)` emits. Those items are
        # the key's value, not new top-level lines.
        body, index = _collect_block(lines, index + 1, allow_root_sequence=not match["rest"].strip(" "))
        entries.append(_entry(match["key"], match["rest"], body))
    return tuple(entries)


def entry_values(entries: tuple[ManifestEntry, ...], key: str) -> tuple[ManifestEntry, ...]:
    """Return every entry declared under *key*, in document order."""
    return tuple(entry for entry in entries if entry.key == key)


def _bounded_significant_lines(text: str) -> tuple[_Line, ...]:
    if _FORBIDDEN_CONTROL.search(text):
        raise PluginManifestFormatError("plugin manifest contains a control character YAML forbids")
    raw_lines = text.splitlines()
    if len(raw_lines) > MAX_MANIFEST_LINES:
        raise PluginManifestFormatError(f"plugin manifest exceeds {MAX_MANIFEST_LINES} lines")
    lines: list[_Line] = []
    for raw in raw_lines:
        if len(raw) > MAX_MANIFEST_LINE_CHARS:
            raise PluginManifestFormatError(f"plugin manifest line exceeds {MAX_MANIFEST_LINE_CHARS} characters")
        stripped = raw.lstrip(" \t")
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw) - len(stripped)
        if "\t" in raw[:indent]:
            # YAML forbids tab indentation outright; guessing a width would make
            # the reader disagree with the host about the document's structure.
            raise PluginManifestFormatError("plugin manifest indents with a tab")
        # `rstrip(" ")` and not `rstrip()`: a bare rstrip removes a trailing tab
        # before `_scalar` can see it, which left the plain-scalar tab rule
        # blind in exactly the position its own comment claims to cover.
        # `splitlines()` above already consumes every line terminator including
        # CR and CRLF, so no carriage return survives into a line here.
        lines.append(_Line(indent, stripped.rstrip(" ")))
    return tuple(lines)


def _single_document_lines(lines: tuple[_Line, ...]) -> tuple[_Line, ...]:
    """Strip one document's markers, and refuse a stream carrying several.

    A leading `---` and a trailing `...` are one document's boundaries, so they
    are dropped. A second `---` means the file holds more than one document, and
    reading the first while ignoring the rest would report a manifest the file
    does not have. That is a wrong answer rather than a missing one, so the
    stream is refused even though the markers themselves are understood.
    """
    body = lines[1:] if lines and lines[0].indent == 0 and lines[0].text == _DOCUMENT_START else lines
    if body and body[-1].indent == 0 and body[-1].text == _DOCUMENT_END:
        body = body[:-1]
    for line in body:
        if line.indent == 0 and line.text in (_DOCUMENT_START, _DOCUMENT_END):
            raise PluginManifestFormatError("plugin manifest holds more than one document")
    return body


def _collect_block(lines: tuple[_Line, ...], start: int, *, allow_root_sequence: bool) -> tuple[tuple[_Line, ...], int]:
    index = start
    while index < len(lines):
        line = lines[index]
        if line.indent > 0 or (allow_root_sequence and _SEQUENCE_LINE.fullmatch(line.text) is not None):
            index += 1
            continue
        break
    return lines[start:index], index


def _entry(key: str, rest: str, body: tuple[_Line, ...]) -> ManifestEntry:
    # `strip(" ")`, not `strip()`: the two places a value is trimmed before
    # `_scalar` sees it are the two places a trailing tab used to disappear,
    # which left the plain-scalar tab rule blind inside its own stated scope.
    inline = rest.strip(" ")
    if inline:
        # An inline value owns the entry; indented lines after it are a
        # continuation this subset does not model.
        if body:
            # Validate the value before abandoning it. `unmodeled` says "I did
            # not read this", which is only honest if the document is readable
            # at all -- and `notes: |\t` is not: the tab makes YAML reject the
            # file, and returning early meant the block-header and tab rules
            # never ran on the one line that breaks it.
            _scalar(inline)
            return ManifestEntry(key, "unmodeled")
        scalar = _scalar(inline)
        return ManifestEntry(key, "scalar", scalar) if scalar is not None else ManifestEntry(key, "unmodeled")
    if not body:
        # `key:` with nothing under it is YAML null, not an empty list. Reporting
        # it as an empty scalar keeps "declared with no value" distinguishable
        # from "declared as an empty sequence".
        return ManifestEntry(key, "scalar")
    return _block_entry(key, body)


def _block_entry(key: str, body: tuple[_Line, ...]) -> ManifestEntry:
    # Every line first, before any early return. The block is about to be
    # reported `unmodeled`, which says "I did not read this" -- but an anchor or
    # alias inside it means nobody can read the file without resolving a
    # reference, so the right answer is to refuse the manifest rather than to
    # carry on reading the keys around it. Missing that is how `<<: *base` was
    # refused at the top level and admitted one line down.
    _refuse_nested_references(body)
    sequence_indent = body[0].indent
    items: list[str] = []
    for line in body:
        if line.indent != sequence_indent:
            return ManifestEntry(key, "unmodeled")
        match = _SEQUENCE_LINE.fullmatch(line.text)
        if match is None:
            return ManifestEntry(key, "unmodeled")
        if len(items) >= MAX_MANIFEST_SEQUENCE_ITEMS:
            return ManifestEntry(key, "unmodeled")
        item = _scalar((match["item"] or "").strip(" "))
        if item is None:
            return ManifestEntry(key, "unmodeled")
        items.append(item)
    return ManifestEntry(key, "sequence", sequence=tuple(items))


def _refuse_nested_references(body: tuple[_Line, ...]) -> None:
    """Refuse a block whose nested value is an anchor or an alias.

    The scan has to know where a block scalar starts, because everything under
    one is text. `_entry` handles that for a top-level `description: |`, whose
    body never reaches here at all -- but a `|` nested inside a mapping does,
    and scanning its prose line by line matched `tom: &jerry are cats` as though
    it were structure and refused a legitimate manifest. So this tracks the
    indent of a key that opened a block scalar and skips everything below it.

    A nested `|` or `>` whose header is malformed is refused here rather than
    skipped, for the same reason `_block_scalar_value` refuses one at the top
    level: YAML rejects the document, so reading past it would be this reader
    succeeding where the host's parser fails.

    Both trims below take spaces only, and that is load-bearing rather than
    tidy: `value` is handed whole to `_block_scalar_value`, so a bare trim
    removed the tab from `notes: |\t` before the header check could see it and
    the nested path read a document the host rejects. It was classified as
    "leading character only" at the time, which stopped being true when this
    function gained that second consumer.
    """
    block_indent: int | None = None
    for line in body:
        if block_indent is not None:
            if line.indent > block_indent:
                continue
            block_indent = None
        entry = _ENTRY_LINE.fullmatch(line.text)
        if entry is not None:
            value = entry["rest"].strip(" ")
        else:
            item = _SEQUENCE_LINE.fullmatch(line.text)
            if item is None:
                continue
            value = (item["item"] or "").strip(" ")
        if value[:1] and value[0] in ("|", ">"):
            _block_scalar_value(value)
            block_indent = line.indent
            continue
        if value.startswith(_REFERENCE_INDICATORS):
            raise PluginManifestFormatError("plugin manifest uses an anchor or alias this reader cannot resolve")


def _scalar(value: str) -> str | None:
    """Return the scalar *value* denotes, or None when it is outside the subset."""
    if not value:
        return ""
    if value[0] in ("\"", "'"):
        return _quoted_scalar(value)
    if value[0] in ("|", ">"):
        return _block_scalar_value(value)
    if value[0] in _REFERENCE_INDICATORS:
        # An anchor or an alias makes the document's meaning depend on
        # resolution this reader does not do. Calling it `unmodeled` would leave
        # the rest of the file read as if the reference had been resolved -- and
        # an alias with no anchor is a document YAML rejects outright, which is
        # how `<<: *base` slipped through as a plain key with a value nobody
        # could resolve.
        raise PluginManifestFormatError("plugin manifest uses an anchor or alias this reader cannot resolve")
    if value.startswith(_UNMODELED_VALUE_PREFIXES):
        return None
    # The third and last trim before the tab rule runs. All three take spaces
    # only: a bare rstrip here removed the very character the next line looks
    # for, which is why the rule read as covering a position it could not see.
    plain = _without_trailing_comment(value).rstrip(" ")
    if _PLAIN_SCALAR_TAB.search(plain):
        raise PluginManifestFormatError("plugin manifest has a tab inside a plain scalar")
    return plain if len(plain) <= MAX_MANIFEST_SCALAR_CHARS else None


def _block_scalar_value(value: str) -> str | None:
    """Separate a block scalar this reader does not model from a broken document.

    `description: >-` is valid YAML holding a value the subset does not model,
    so it is `unmodeled` and costs the caller nothing. `requires_hermes:
    >=0.21.1` only looks similar: `>` opens a block scalar, `=0.21.1` is not a
    legal header, and YAML rejects the whole document. Reporting that as merely
    unmodeled would be this reader succeeding where the host's own parser fails,
    and would let a manifest Hermes cannot load be classified as read.
    """
    header = _without_trailing_comment(value).rstrip(" ")
    if _BLOCK_SCALAR_HEADER.fullmatch(header):
        return None
    raise PluginManifestFormatError("plugin manifest opens a block scalar with an unreadable header")


def _quoted_scalar(value: str) -> str | None:
    quote = value[0]
    closing = value.find(quote, 1)
    if closing < 0:
        return None
    if quote == "\"" and "\\" in value[1:closing]:
        # Escape sequences are a YAML feature this reader does not implement;
        # returning the raw bytes would report a value the file does not hold.
        return None
    trailing = _without_trailing_comment(value[closing + 1:]).strip(" ")
    if "\t" in trailing:
        # A tab after the closing quote is separation, and YAML forbids a raw
        # tab there. Reporting the value `unmodeled` would leave the rest of a
        # document the host cannot load being read around it.
        raise PluginManifestFormatError("plugin manifest has a tab after a quoted scalar")
    if trailing:
        return None
    inner = value[1:closing]
    return inner if len(inner) <= MAX_MANIFEST_SCALAR_CHARS else None


def _without_trailing_comment(value: str) -> str:
    """Drop a ` #` comment, which is where YAML ends an unquoted scalar."""
    if value.startswith("#"):
        return ""
    marker = value.find(" #")
    return value if marker < 0 else value[:marker]
