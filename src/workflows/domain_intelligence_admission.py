from __future__ import annotations

import re
import unicodedata

from ..plugin_bundle.omh._governance_safety import classify_memory_admission


MAX_MAPPINGS = 40
MAX_PHRASE_CHARS = 80
MAX_CANONICAL_CHARS = 80
MAX_HINTS = 20
MAPPING_KEYS = {"phrase", "canonical"}
_PROMPTISH_WORDS = {"message", "prompt", "raw", "text", "body", "content", "transcript", "hidden_reasoning"}
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,79}$")
_WORKFLOW_HINT = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,79}$")
_ROLE_MARKER = re.compile(
    r"^[^\S\r\n]*(?:user|assistant|system|developer|human|agent)[^\S\r\n]*:(?!\d+\Z)",
    flags=re.IGNORECASE | re.MULTILINE,
)
_SENSITIVE_ASSIGNMENT = re.compile(
    r"(?<![a-z0-9])(?:ssn|social[-_ ]?security(?:[-_ ]?number)?|api[-_ ]?key|password|passwd|pwd|token|"
    r"access[-_ ]?token|authorization|auth[-_ ]?header|bearer)[^\S\r\n]*[:=][^\S\r\n]*\S+",
    flags=re.IGNORECASE,
)
_SSN = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_EMAIL = re.compile(
    r"(?<![a-z0-9._%+-])[a-z0-9][a-z0-9._%+-]{0,63}@"
    r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.){1,8}[a-z]{2,63}"
    r"(?![a-z0-9.-])",
    flags=re.IGNORECASE,
)
_CREDENTIAL_SHAPE = re.compile(
    r"(?<![a-z0-9_-])(?:"
    r"gh[pousr]_[a-z0-9]{36,255}|"
    r"github_pat_[a-z0-9_]{20,255}|"
    r"(?:AKIA|ASIA)[A-Z0-9]{16}|"
    r"sk-(?:proj-)?[a-z0-9_-]{20,255}|"
    r"xox[baprs]-[a-z0-9-]{20,255}|"
    r"eyJ[a-z0-9_-]{8,}\.eyJ[a-z0-9_-]{8,}\.[a-z0-9_-]{16,}"
    r")(?![a-z0-9_-])",
    flags=re.IGNORECASE,
)
_PROMPT_REQUEST = re.compile(r"^[^\S\r\n]*(?:please|kindly)(?:[^\S\r\n]+\S+){4,}", flags=re.IGNORECASE)
_BIDI_CONTROLS = {"LRE", "RLE", "LRO", "RLO", "PDF", "LRI", "RLI", "FSI", "PDI"}
_IDENTIFIER_SEPARATORS = frozenset("._:-")


def normalize_mappings(mappings: list[tuple[str, str]]) -> list[dict[str, str]]:
    if not isinstance(mappings, list) or not mappings or len(mappings) > MAX_MAPPINGS:
        raise ValueError("invalid_mapping_count")
    seen: set[str] = set()
    normalized: list[dict[str, str]] = []
    for item in mappings:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError("invalid_mapping")
        phrase_value = normalize_phrase(item[0])
        phrase_key = phrase_value.casefold()
        if phrase_key in seen:
            raise ValueError("duplicate_phrase")
        seen.add(phrase_key)
        canonical_value = normalize_vocabulary_identifier(item[1])
        normalized.append({"phrase": phrase_value, "canonical": canonical_value})
    return sorted(normalized, key=lambda item: (item["phrase"].casefold(), item["canonical"]))


def normalize_mappings_from_value(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError("vocabulary_mappings must be a list")
    pairs: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != MAPPING_KEYS:
            raise ValueError("mapping_schema_mismatch")
        phrase = item.get("phrase")
        canonical = item.get("canonical")
        if not isinstance(phrase, str) or not isinstance(canonical, str):
            raise ValueError("invalid_mapping_value")
        pairs.append((phrase, canonical))
    return normalize_mappings(pairs)


def normalize_phrase(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("invalid_phrase")
    _ensure_vocabulary_safe(value)
    phrase = " ".join(unicodedata.normalize("NFKC", value).strip().split())
    if not phrase or len(phrase) > MAX_PHRASE_CHARS:
        raise ValueError("invalid_phrase")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in phrase):
        raise ValueError("phrase_contains_control_character")
    if phrase.lower() in _PROMPTISH_WORDS:
        raise ValueError("promptish_phrase")
    if phrase.lstrip().startswith(("{", "[", "<")):
        raise ValueError("promptish_structured_phrase")
    return phrase


def normalize_vocabulary_identifier(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("invalid_canonical_term")
    ensure_safe_identifier_content(value, "canonical_term")
    normalized = unicodedata.normalize("NFKC", value.strip().lower())
    if not _IDENTIFIER.fullmatch(normalized) or len(normalized) > MAX_CANONICAL_CHARS:
        raise ValueError("invalid_canonical_term")
    _ensure_vocabulary_safe(normalized.replace("-", " "))
    return normalized


def normalize_workflow_hints(values: object) -> list[str]:
    if not isinstance(values, list):
        raise ValueError("workflow_hints must be a list")
    normalized = sorted({_normalize_workflow_hint(value) for value in values})
    if len(normalized) > MAX_HINTS:
        raise ValueError("too_many_workflow_hints")
    return normalized


def _normalize_workflow_hint(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("invalid_workflow_hint")
    ensure_safe_identifier_content(value, "workflow_hint")
    normalized = unicodedata.normalize("NFKC", value.strip().lower()).replace(" ", "-")
    if not _WORKFLOW_HINT.fullmatch(normalized):
        raise ValueError("invalid_workflow_hint")
    return normalized


def _ensure_vocabulary_safe(value: str) -> None:
    """Refuse domain vocabulary that carries protected or bulk-history content.

    This is the only gate that matches the raw-log and transcript SHAPE, and
    deliberately so: `classify_memory_admission` is a shared primitive whose
    callers include action gates and handoff manifests, several of which turn
    any non-safe verdict into a raise, so a line-shape rule there fails an
    ordinary note whose lines open with "Error" or "User:". Keep the markers
    here rather than moving them up; see that function's module docstring.
    """
    normalized = unicodedata.normalize("NFKC", value)
    lowered = normalized.lower()
    ensure_safe_phrase_content(value)
    raw_log = any(
        marker in lowered
        for marker in ("traceback (most recent call last)", "\nstderr", "\nstdout", "[error]", "exception:", "raw log", "full log")
    )
    timestamps = len(re.findall(r"^\d{4}-\d{2}-\d{2}[ t]\d{2}:\d{2}:\d{2}", normalized, flags=re.MULTILINE))
    transcript = "full transcript" in lowered or "chat transcript" in lowered or _ROLE_MARKER.search(normalized)
    if raw_log or timestamps >= 3 or transcript or classify_memory_admission(value).get("status") != "safe":
        raise ValueError("unsafe_domain_vocabulary")


def ensure_safe_phrase_content(value: str) -> None:
    _ensure_sensitive_content_safe(value, "unsafe_domain_vocabulary", reject_prompt_request=True)


def ensure_safe_identifier_content(value: str, label: str) -> None:
    _ensure_sensitive_content_safe(value, f"invalid_{label}")


def ensure_safe_opaque_ref_content(value: str, label: str) -> None:
    _ensure_sensitive_content_safe(value, f"unsafe_{label}")


def _ensure_sensitive_content_safe(value: str, error: str, *, reject_prompt_request: bool = False) -> None:
    normalized = unicodedata.normalize("NFKC", value)
    if (
        _has_unsafe_unicode(value, normalized)
        or _ROLE_MARKER.search(normalized)
        or _SENSITIVE_ASSIGNMENT.search(normalized)
        or _SSN.search(normalized)
        or _EMAIL.search(normalized)
        or _CREDENTIAL_SHAPE.search(normalized)
        or (reject_prompt_request and _PROMPT_REQUEST.search(normalized))
    ):
        raise ValueError(error)


def _has_unsafe_unicode(value: str, normalized: str) -> bool:
    if any(
        unicodedata.category(ch) in {"Cc", "Cf", "Cs", "Zl", "Zp"}
        or unicodedata.bidirectional(ch) in _BIDI_CONTROLS
        for ch in value + normalized
    ):
        return True
    return any(
        ch not in _IDENTIFIER_SEPARATORS
        and unicodedata.normalize("NFKC", ch) in _IDENTIFIER_SEPARATORS
        for ch in value
    )
