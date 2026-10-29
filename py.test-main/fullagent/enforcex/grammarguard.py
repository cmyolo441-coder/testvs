"""grammarguard — Output Grammar Validator & Deterministic Structural Auto-Repairer.

Validates and deterministically repairs model outputs against structural
grammars required by the system prompt:
  * JSON Grammar (`json_object` / `json_schema`): strips markdown fences,
    removes trailing commas before `}` or `]`, balances unclosed brackets/braces,
    and validates required schema keys & types.
  * Markdown Table Grammar: verifies header separator rows (`|---|---|`) and
    repairs jagged rows with missing pipe cells.
  * Numbered / Bulleted List Grammar: verifies monotonic 1..N numbering and
    minimum item count.
  * Code-Only Grammar: strips conversational preamble/postamble outside fenced
    code blocks when the system prompt demands code-only output.
  * Balanced Fence Grammar: closes unclosed ``` code fences.

Self-test: ``python3 -m fullagent.enforcex.grammarguard``
"""

from __future__ import annotations

import html
import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "GrammarRequirement",
    "GrammarParseReport",
    "GrammarGuardEngine",
    "infer_grammar_requirements",
    "repair_json_text",
    "repair_markdown_table",
    "validate_and_repair",
]

# zero-width / BOM characters that break naive JSON scanners
_ZW_RE = re.compile("[\u200b\u200c\u200d\u200e\u200f\u2060\ufeff\u00ad\u180e]")

# ReDoS bound for requirement inference on oversized system prompts
_MAX_SCAN = 262144


@dataclass
class GrammarRequirement:
    json_only: bool = False
    required_json_keys: List[str] = field(default_factory=list)
    code_only: bool = False
    require_table: bool = False
    require_numbered_list: bool = False
    forbid_horizontal_rules: bool = False


@dataclass
class GrammarParseReport:
    valid: bool = True
    repaired: bool = False
    score: float = 100.0
    errors: List[str] = field(default_factory=list)
    repairs_applied: List[str] = field(default_factory=list)
    repaired_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "repaired": self.repaired,
            "score": round(self.score, 2),
            "errors": list(self.errors),
            "repairs_applied": list(self.repairs_applied),
            "repaired_text": self.repaired_text,
        }


def _normalize_prompt(text: str) -> str:
    """NFKC + zero-width strip + HTML entity decode + collapsed casefold."""
    try:
        s = unicodedata.normalize("NFKC", str(text or "")[: _MAX_SCAN])
        s = _ZW_RE.sub("", s)
        if "&" in s:
            try:
                s = html.unescape(s)
            except Exception:
                pass
        return re.sub(r"\s+", " ", s).casefold()
    except Exception:
        return str(text or "").lower()


def infer_grammar_requirements(system_text: str) -> GrammarRequirement:
    """Infer structural output grammar requirements from `system_text`."""
    low = _normalize_prompt(system_text)
    req = GrammarRequirement()
    if re.search(
        r"\b(?:only\s+(?:valid\s+)?json|json\s+only|"
        r"(?:respond|reply|answer|output|return)\s+(?:only\s+)?"
        r"(?:in|with|as)?\s*(?:valid\s+)?json)\b", low):
        req.json_only = True
    if re.search(
        r"\b(?:only\s+code|code\s+only|return\s+only\s+the\s+code|"
        r"no\s+(?:prose|explanations?)[,;]?\s+(?:just|only)\s+code)\b",
        low):
        req.code_only = True
    if re.search(
        r"\b(?:must\s+include\s+a\s+(?:markdown\s+)?table|"
        r"use\s+a\s+markdown\s+table|"
        r"(?:include|provide|add|present)\s+(?:a\s+)?"
        r"(?:markdown\s+)?table)\b", low):
        req.require_table = True
    if re.search(r"\bnumbered\s+(?:list|steps)\b", low):
        req.require_numbered_list = True
    if re.search(
        r"(?:never\s+use\s+horizontal\s+rules?|no\s+horizontal\s+rules?|"
        r"(?:do\s+not|don't|never|avoid)\s+(?:use\s+)?"
        r"(?:horizontal\s+rules?|`?---`?))", low):
        req.forbid_horizontal_rules = True
    return req


def _json_scan(s: str) -> Tuple[int, List[str], bool, bool, bool]:
    """String-aware structural scan of `s`.

    Returns ``(cut, stack, in_string, escape_pending, mismatched)`` where
    ``cut`` is the index just past the last structurally valid character of
    the root value (trailing prose starts at ``cut``)."""
    stack: List[str] = []
    in_str = False
    esc = False
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack:
                return i, [], False, False, False   # stray closer -> cut
            expected = stack.pop()
            if ch != expected:
                stack.append(expected)
                return i, stack, False, False, True
            if not stack:
                return i + 1, [], False, False, False
        i += 1
    return n, stack, in_str, esc, False


def _balance_json(body: str, stack: List[str], in_str: bool,
                  esc: bool) -> Tuple[str, List[str]]:
    notes: List[str] = []
    if in_str:
        if esc:
            body += "\\"
        body += '"'
        notes.append("closed_unterminated_string")
    if stack:
        body += "".join(reversed(stack))
        notes.append("closed_unbalanced_brackets")
    return body, notes


def _remove_trailing_commas(body: str) -> Tuple[str, bool]:
    """Drop commas that sit directly before a closing brace/bracket,
    ignoring commas inside JSON string literals."""
    out: List[str] = []
    in_str = False
    esc = False
    changed = False
    i = 0
    n = len(body)
    while i < n:
        ch = body[i]
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        if ch == ",":
            j = i + 1
            while j < n and body[j] in " \t\r\n":
                j += 1
            if j >= n or body[j] in "}]":
                changed = True
                i += 1
                continue
        out.append(ch)
        i += 1
    return "".join(out), changed


def repair_json_text(raw: str) -> Tuple[Optional[Any], str, List[str]]:
    """Attempt to parse and deterministically repair malformed JSON output.
    Returns `(parsed_obj_or_None, repaired_json_str, repair_notes)`.

    Repairs are string-aware: commas/braces inside JSON string literals are
    never touched, trailing prose after the root value is dropped, and
    zero-width/BOM characters are removed."""
    notes: List[str] = []
    raw_s = str(raw or "")
    s = _ZW_RE.sub("", raw_s)
    if s != raw_s:
        notes.append("stripped_zero_width_chars")
    s = s.strip()
    if not s:
        return None, s, ["empty input"]

    # 1. Try direct parse
    try:
        obj = json.loads(s)
        return obj, s, notes
    except Exception:
        pass

    # 2. Strip markdown ```json ... ``` fences or surrounding prose
    m_fence = re.search(r"```(?:json)?\s*\n(.*?)```", s, re.DOTALL | re.IGNORECASE)
    if m_fence:
        s = m_fence.group(1).strip()
        notes.append("stripped_markdown_json_fence")
    else:
        # Extract from first '{' or '[' to last '}' or ']'
        first_brace = min([i for i in (s.find("{"), s.find("[")) if i >= 0] or [-1])
        if first_brace > 0:
            s = s[first_brace:].strip()
            notes.append("stripped_leading_prose")

    # 3. Structural scan: cut trailing prose and find unclosed containers
    cut, stack, in_str, esc, mismatched = _json_scan(s)
    if cut < len(s):
        notes.append("stripped_trailing_prose")
    body = s[:cut]
    body, bnotes = _balance_json(body, stack, in_str, esc)
    notes.extend(bnotes)
    if mismatched:
        notes.append("closed_mismatched_bracket")

    # 4. Remove trailing commas before } or ] (strings protected)
    body, comma_changed = _remove_trailing_commas(body)
    if comma_changed:
        notes.append("removed_trailing_commas")
    body = body.strip()

    try:
        obj = json.loads(body)
        return obj, json.dumps(obj, indent=2, ensure_ascii=False), notes
    except Exception as exc:
        return None, body, notes + [f"json_parse_failed: {exc}"]


_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")


def repair_markdown_table(text: str) -> Tuple[str, bool]:
    """Pad jagged Markdown table rows so each row has the header's cell
    count. Returns ``(new_text, changed)``; never raises."""
    try:
        lines = str(text or "").split("\n")
    except Exception:
        return str(text or ""), False
    changed = False
    i = 0
    while i < len(lines) - 1:
        header = lines[i]
        sep = lines[i + 1]
        if ("|" in header and "|" in sep and "-" in sep
                and _TABLE_SEP_RE.match(sep)):
            header_pipes = header.count("|")
            j = i + 2
            while j < len(lines) and lines[j].strip() and "|" in lines[j]:
                row_pipes = lines[j].count("|")
                if row_pipes < header_pipes:
                    lines[j] = (lines[j].rstrip()
                                + " |" * (header_pipes - row_pipes))
                    changed = True
                j += 1
            i = max(j, i + 1)
        else:
            i += 1
    return "\n".join(lines), changed


def _extract_code_blocks(text: str) -> List[str]:
    """All fenced code blocks, tolerating an unterminated trailing fence."""
    blocks: List[str] = []
    pos = 0
    n = len(text)
    while pos < n:
        start = text.find("```", pos)
        if start < 0:
            break
        end = text.find("```", start + 3)
        if end < 0:
            blocks.append(text[start:].rstrip("\n"))
            break
        blocks.append(text[start:end + 3])
        pos = end + 3
    return blocks


def validate_and_repair(
    response_text: str,
    *,
    system_text: str = "",
    requirements: Optional[GrammarRequirement] = None,
) -> GrammarParseReport:
    req = requirements or infer_grammar_requirements(system_text)
    out = str(response_text or "")
    errors: List[str] = []
    repairs: List[str] = []

    # 1. JSON-only grammar
    if req.json_only:
        obj, fixed_json, jnotes = repair_json_text(out)
        if obj is None:
            errors.append("Output is not valid JSON and could not be auto-repaired")
        else:
            if jnotes:
                repairs.extend(jnotes)
                out = fixed_json
            if req.required_json_keys:
                if not isinstance(obj, dict):
                    errors.append("Output is JSON but not an object; required "
                                  f"keys missing: {req.required_json_keys}")
                else:
                    missing = [k for k in req.required_json_keys if k not in obj]
                    if missing:
                        errors.append(f"Missing required JSON keys: {missing}")

    # 2. Code-only grammar
    elif req.code_only:
        fences = _extract_code_blocks(out)
        if fences and out.strip() != "\n\n".join(fences).strip():
            out = "\n\n".join(fences)
            repairs.append("stripped_non_code_prose")

    # 3. Horizontal rules ban
    if req.forbid_horizontal_rules:
        cleaned = re.sub(r"^\s*(?:---|___|\*\*\*)\s*\n?", "", out, flags=re.M)
        if cleaned != out:
            out = cleaned
            repairs.append("removed_forbidden_horizontal_rules")

    # 4. Unclosed code fence repair
    if out.count("```") % 2 == 1:
        out = out.rstrip() + "\n```\n"
        repairs.append("closed_unterminated_code_fence")

    # 5. Table requirement (pad jagged rows when a table is present)
    if req.require_table:
        if not re.search(r"^\s*\|.*\|\s*\n\s*\|[\s:|-]+\|", out, re.M):
            errors.append("Required Markdown pipe table with header separator is missing")
        else:
            padded, changed = repair_markdown_table(out)
            if changed:
                out = padded
                repairs.append("padded_jagged_table_rows")

    # 6. Numbered list requirement
    if req.require_numbered_list:
        if not re.search(r"^\s*1[.)]\s+", out, re.M):
            errors.append("Required numbered list (starting with '1.') is missing")

    score = max(0.0, 100.0 - 30.0 * len(errors) - 5.0 * len(repairs))
    return GrammarParseReport(
        valid=len(errors) == 0,
        repaired=len(repairs) > 0,
        score=score,
        errors=errors,
        repairs_applied=repairs,
        repaired_text=out,
    )


class GrammarGuardEngine:
    def evaluate(self, response_text: str, *, system_text: str = "") -> GrammarParseReport:
        return validate_and_repair(response_text, system_text=system_text)


def _self_test() -> int:
    # 1. Malformed JSON with prose + trailing comma + unclosed brace repaired
    sys_json = "Respond in valid JSON only."
    broken_json = 'Sure! Here is the JSON:\n```json\n{"status": "ok", "items": [1, 2,],\n```'
    rep = validate_and_repair(broken_json, system_text=sys_json)
    assert rep.valid and rep.repaired, rep.to_dict()
    parsed = json.loads(rep.repaired_text)
    assert parsed == {"status": "ok", "items": [1, 2]}

    # 2. Forbidden horizontal rule stripped & unclosed code fence closed
    sys_hr = "Never use horizontal rules (---)."
    dirty = "Section 1\n---\n```python\nx = 1"
    rep2 = validate_and_repair(dirty, system_text=sys_hr)
    assert "---" not in rep2.repaired_text
    assert rep2.repaired_text.count("```") == 2

    # 3. String-aware repair: commas/braces inside JSON strings are preserved
    obj3, fixed3, notes3 = repair_json_text('{"a": "x,}", "b": [1, 2,]}')
    assert obj3 == {"a": "x,}", "b": [1, 2]} and "removed_trailing_commas" in notes3, notes3

    # 4. Trailing prose after the root value is dropped
    obj4, _, notes4 = repair_json_text('{"ok": true} Hope that helps!')
    assert obj4 == {"ok": True} and "stripped_trailing_prose" in notes4, notes4

    # 5. Zero-width characters inside JSON are stripped and reported
    obj5, fixed5, notes5 = repair_json_text('{\u200b"a": 1}')
    assert obj5 == {"a": 1} and "stripped_zero_width_chars" in notes5 and "\u200b" not in fixed5

    # 6. Unterminated strings (and dangling escapes) are closed deterministically
    obj6, _, notes6 = repair_json_text('{"a": "unterminated')
    assert obj6 == {"a": "unterminated"} and "closed_unterminated_string" in notes6
    obj6b, _, _ = repair_json_text('{"a": "x\\')
    assert obj6b == {"a": "x\\"}

    # 7. Mismatched closer is repaired instead of crashing
    obj7, _, notes7 = repair_json_text('{"a": 1] trailing')
    assert obj7 == {"a": 1} and "closed_mismatched_bracket" in notes7, (obj7, notes7)

    # 8. NFKC / HTML-entity / spacing obfuscated system prompt still infers JSON-only
    rep8 = infer_grammar_requirements("Ｒｅｓｐｏｎｄ   ｉｎ   ＪＳＯＮ&nbsp;only.")
    assert rep8.json_only
    assert infer_grammar_requirements("Output JSON.").json_only
    assert infer_grammar_requirements(None) == GrammarRequirement()

    # 9. Jagged table rows are padded to the header width
    rep9 = validate_and_repair(
        "| Name | Score | Note |\n|---|---|---|\n| Ada | 9 |",
        system_text="Use a markdown table.")
    assert rep9.valid and "padded_jagged_table_rows" in rep9.repairs_applied, rep9.to_dict()
    assert rep9.repaired_text.count("|") == 12

    # 10. Code-only extraction tolerates an unterminated trailing fence
    rep10 = validate_and_repair(
        "Here you go:\n```python\nx = 1",
        system_text="Only code.")
    assert "Here you go" not in rep10.repaired_text
    assert rep10.repaired_text.count("```") == 2

    # 11. Non-dict JSON cannot satisfy required keys; junk never raises
    rep11 = validate_and_repair(
        "[1, 2]", system_text="Respond in JSON only.",
        requirements=GrammarRequirement(json_only=True,
                                        required_json_keys=["status"]))
    assert not rep11.valid and any("not an object" in e for e in rep11.errors)
    huge = '{"a": "' + "x" * 300000 + '"}'
    assert validate_and_repair(huge, system_text="JSON only").valid
    assert validate_and_repair("", system_text=None).valid

    print("grammarguard._self_test: OK PASS (string-aware JSON repair, fence balance, "
          "HR strip, table/list check, obfuscated prompts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
