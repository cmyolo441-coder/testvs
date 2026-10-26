"""Syntax colour for fenced code, one source line at a time.

A fence streams in line by line and a line that left the live region is never
repainted, so highlighting cannot look ahead either. Each line is
tokenized on its own; the one thing it inherits is `carry`, the multi-line
construct the line before ended inside: a Python triple-quoted string, a
`/* block comment */`, a JavaScript template literal or a Go raw string. The
caller threads it through `MarkdownState.code_carry`, so a fence rendered line
by line is coloured exactly as the whole fence is.

Each language is a small table (`Language`): keywords, comment markers, string
quotes and which of them may span lines or take no backslash escapes. The
tokenizer paints keywords, strings, numbers and comments with the theme's
`syntax_*` roles and everything else as plain text. The generic languages (C,
Java, Ruby, SQL, ...) have no keywords: only their strings, numbers and
comments are coloured. A language with no table is not highlighted at all.
Nothing here raises on any input; a string or comment left open on a line
that cannot span lines simply ends with the line.
"""

from __future__ import annotations

from dataclasses import dataclass
import keyword

from .style import Span, Style
from .theme import Theme

PLAIN = "plain"
KEYWORD = "keyword"
STRING = "string"
NUMBER = "number"
COMMENT = "comment"

BLOCK_COMMENT = ("/*", "*/")
_ESCAPE = "\\"
_DIGITS = "0123456789"
_YAML_VALUE_OPENERS = frozenset("[{,:'")  # after a closing quote too: 'it''s' is one string
_MAX_CHAR_ESCAPE = 10  # the longest Rust char escape, '\u{10FFFF}'


@dataclass(frozen=True)
class Language:
    keywords: frozenset[str] = frozenset()
    line_comments: tuple[str, ...] = ()
    block_comments: bool = False  # /* ... */, which may span lines
    quotes: tuple[str, ...] = ('"', "'")  # longest first: a triple quote must be tried before a single one
    multiline_quotes: frozenset[str] = frozenset()  # strings that may run past the end of the line
    raw_quotes: frozenset[str] = frozenset()  # strings with no backslash escapes
    comment_at_word_start: bool = False  # shell, YAML: `#` opens a comment only at the start of a word
    quote_at_value_start: bool = False  # YAML: a quote inside a plain scalar (don't) is a letter
    char_literals: bool = False  # Rust: `'` opens a char literal only when it closes at once ('a' vs 'a)


_JS_KEYWORDS = frozenset(
    "break case catch class const continue debugger default delete do else export extends finally for function "
    "if import in instanceof new return super switch this throw try typeof var void while with yield let static "
    "async await of null undefined true false from as interface type enum implements namespace declare readonly "
    "private protected public abstract keyof satisfies".split()
)
_SHELL_KEYWORDS = frozenset(
    "if then else elif fi for while until do done case esac in function select return local export readonly "
    "declare unset break continue exit source".split()
)
_GO_KEYWORDS = frozenset(
    "break case chan const continue default defer else fallthrough for func go goto if import interface map "
    "package range return select struct switch type var true false nil iota".split()
)
_RUST_KEYWORDS = frozenset(
    "as async await break const continue crate dyn else enum extern false fn for if impl in let loop match mod "
    "move mut pub ref return self Self static struct super trait true type unsafe use where while".split()
)
_YAML_KEYWORDS = frozenset("true false null yes no on off True False TRUE FALSE Null NULL Yes No".split())

_PYTHON = Language(
    keywords=frozenset(keyword.kwlist),
    line_comments=("#",),
    quotes=('"""', "'''", '"', "'"),
    multiline_quotes=frozenset({'"""', "'''"}),
)
_JAVASCRIPT = Language(
    keywords=_JS_KEYWORDS,
    line_comments=("//",),
    block_comments=True,
    quotes=('"', "'", "`"),
    multiline_quotes=frozenset({"`"}),
)
_JSON = Language(keywords=frozenset({"true", "false", "null"}), quotes=('"',))
_SHELL = Language(
    keywords=_SHELL_KEYWORDS,
    line_comments=("#",),
    raw_quotes=frozenset({"'"}),
    comment_at_word_start=True,
)
_GO = Language(
    keywords=_GO_KEYWORDS,
    line_comments=("//",),
    block_comments=True,
    quotes=('"', "'", "`"),
    multiline_quotes=frozenset({"`"}),
    raw_quotes=frozenset({"`"}),
)
_RUST = Language(
    keywords=_RUST_KEYWORDS,
    line_comments=("//",),
    block_comments=True,
    quotes=('"',),
    char_literals=True,
)
_YAML = Language(
    keywords=_YAML_KEYWORDS,
    line_comments=("#",),
    raw_quotes=frozenset({"'"}),
    comment_at_word_start=True,
    quote_at_value_start=True,
)
_GENERIC_SLASH = Language(line_comments=("//",), block_comments=True)
_GENERIC_HASH = Language(line_comments=("#",), comment_at_word_start=True)
_GENERIC_DASH = Language(line_comments=("--",))


def _aliases(language: Language, *names: str) -> dict[str, Language]:
    return {name: language for name in names}


LANGUAGES: dict[str, Language] = {
    **_aliases(_PYTHON, "python", "py", "python3", "py3"),
    **_aliases(_JAVASCRIPT, "javascript", "js", "jsx", "mjs", "cjs", "typescript", "ts", "tsx", "mts", "cts"),
    **_aliases(_JSON, "json"),
    **_aliases(_SHELL, "bash", "sh", "shell", "zsh"),
    **_aliases(_GO, "go", "golang"),
    **_aliases(_RUST, "rust", "rs"),
    **_aliases(_YAML, "yaml", "yml"),
    **_aliases(
        _GENERIC_SLASH,
        "c", "h", "cpp", "c++", "cc", "cxx", "hpp", "java", "kotlin", "kt", "swift", "scala", "cs", "csharp", "dart",
        "php",
    ),
    **_aliases(_GENERIC_HASH, "ruby", "rb", "toml", "perl", "pl", "r", "dockerfile", "makefile", "make", "elixir",
               "ex", "exs", "powershell", "ps1"),
    **_aliases(_GENERIC_DASH, "sql", "lua", "haskell", "hs"),
}


def highlights(lang: str) -> bool:
    """Whether a fence labelled `lang` (lowercased) is coloured at all."""
    return lang in LANGUAGES


# ── scanning ──────────────────────────────────────────────────────────────


def _closer(opener: str) -> str:
    return BLOCK_COMMENT[1] if opener == BLOCK_COMMENT[0] else opener


def _close(text: str, start: int, opener: str, language: Language) -> tuple[int, bool]:
    """(index after the construct `opener` began, whether it closed on this line), scanning from `start`."""
    closer = _closer(opener)
    escapes = opener != BLOCK_COMMENT[0] and opener not in language.raw_quotes
    index = start
    while index < len(text):
        if escapes and text[index] == _ESCAPE:
            index += 2
            continue
        if text.startswith(closer, index):
            return index + len(closer), True
        index += 1
    return len(text), False


def _at_word_start(text: str, index: int) -> bool:
    return index == 0 or text[index - 1].isspace()


def _line_comment_at(text: str, index: int, language: Language) -> bool:
    for marker in language.line_comments:
        if text.startswith(marker, index):
            return not language.comment_at_word_start or _at_word_start(text, index)
    return False


def _quote_at(text: str, index: int, language: Language) -> str:
    """The quote that opens a string at `index`, or ""."""
    for quote in language.quotes:
        if text.startswith(quote, index):
            if language.quote_at_value_start and not (
                _at_word_start(text, index) or text[index - 1] in _YAML_VALUE_OPENERS
            ):
                return ""
            return quote
    return ""


def _char_literal_end(text: str, index: int) -> int:
    """The index after a Rust char literal opening at `index` ('a', '\\n', '\\u{1F600}'), or 0 for a lifetime."""
    if text[index + 1 : index + 2] == _ESCAPE:
        close = text.find("'", index + 3, index + 3 + _MAX_CHAR_ESCAPE)
        return close + 1 if close > 0 else 0
    if len(text) > index + 2 and text[index + 2] == "'":
        return index + 3
    return 0


def _is_word_start(ch: str) -> bool:
    return ch.isalpha() or ch in "_$"


def _word_end(text: str, index: int) -> int:
    while index < len(text) and (text[index].isalnum() or text[index] in "_$"):
        index += 1
    return index


def _number_end(text: str, index: int) -> int:
    """Digits and what follows them in a literal: 0x1F, 1_000, 3.14, 1e10, 10u32."""
    while index < len(text) and (text[index].isascii() and (text[index].isalnum() or text[index] in "_.")):
        index += 1
    return index


def _tokens(text: str, language: Language, carry: str) -> tuple[list[tuple[str, str]], str]:
    """(kind, text) pieces covering `text` exactly, and the construct the line ends inside ("" for none)."""
    tokens: list[tuple[str, str]] = []
    index = 0
    if carry:
        end, closed = _close(text, 0, carry, language)
        tokens.append((COMMENT if carry == BLOCK_COMMENT[0] else STRING, text[:end]))
        if not closed:
            return tokens, carry
        index = end
    while index < len(text):
        ch = text[index]
        if _line_comment_at(text, index, language):
            tokens.append((COMMENT, text[index:]))
            break
        if language.block_comments and text.startswith(BLOCK_COMMENT[0], index):
            end, closed = _close(text, index + 2, BLOCK_COMMENT[0], language)
            tokens.append((COMMENT, text[index:end]))
            if not closed:
                return tokens, BLOCK_COMMENT[0]
            index = end
            continue
        if language.char_literals and ch == "'":
            end = _char_literal_end(text, index) or index + 1
            tokens.append((STRING if end > index + 1 else PLAIN, text[index:end]))
            index = end
            continue
        quote = _quote_at(text, index, language)
        if quote:
            end, closed = _close(text, index + len(quote), quote, language)
            tokens.append((STRING, text[index:end]))
            if not closed and quote in language.multiline_quotes:
                return tokens, quote
            index = end
            continue
        if _is_word_start(ch):
            end = _word_end(text, index)
            word = text[index:end]
            after_dot = index > 0 and text[index - 1] == "."
            tokens.append((KEYWORD if word in language.keywords and not after_dot else PLAIN, word))
            index = end
            continue
        if ch in _DIGITS:
            end = _number_end(text, index)
            tokens.append((NUMBER, text[index:end]))
            index = end
            continue
        tokens.append((PLAIN, ch))
        index += 1
    return tokens, ""


def _style(kind: str, theme: Theme) -> Style:
    if kind == KEYWORD:
        return theme.syntax_keyword
    if kind == STRING:
        return theme.syntax_string
    if kind == NUMBER:
        return theme.syntax_number
    if kind == COMMENT:
        return theme.syntax_comment
    return theme.text


def highlight_line(text: str, lang: str, carry: str, theme: Theme) -> tuple[list[Span], str]:
    """Spans covering `text` exactly (adjacent pieces of one style merged), and the carry for the next line.

    A language without a table is one plain code span and never carries.
    """
    language = LANGUAGES.get(lang)
    if language is None:
        return ([Span(text, theme.code)] if text else []), ""
    tokens, carry = _tokens(text, language, carry)
    spans: list[Span] = []
    for kind, piece in tokens:
        if not piece:
            continue
        style = _style(kind, theme)
        if spans and spans[-1].style == style:
            spans[-1] = Span(spans[-1].text + piece, style)
        else:
            spans.append(Span(piece, style))
    return spans, carry
