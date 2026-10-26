"""Executor-name and coding-delivery phrase groups, owned once by the bundle.

The control plane (`omh.routing.executor_cues`) re-exports these names from
here; the bundle's awareness route hints read them directly. Until 2026-10-06
the bundle carried a vendored copy behind an `ImportError` fallback, and since
Hermes loads the bundle as a top-level package that copy was what the live
plugin always ran, held equal only by a parity test. One definition, no copy.
Stdlib only.
"""

from __future__ import annotations

from .pi_family_names import PI_DISTRIBUTION_HOST, RUNTIME_USER_PHRASE

# Explicit coding-agent / coding-runtime names.
#
# Bare "claude" and bare "gemini" are intentionally absent: they are external-advisor
# names as often as executor names, and treating them as executor selection would
# break advisor routing. Only unambiguous executor names belong here. Product names
# stay in Latin script because ja/zh users write them that way too.
#
# These names are distinctive enough for plain substring containment on the
# folded message; the pi-family-runtime family below is not and gets its own group.
SUBSTRING_NAMED_CODING_AGENT_PHRASES: tuple[str, ...] = (
    "codex",
    "코덱스",
    "claude code",
    "claude-code",
    "claudecode",
    "클로드 코드",
    "클로드코드",
    "hermes coding",
    "헤르메스 코딩",
    "헤르메스가 코딩",
    "헤르메스한테 코딩",
)

# pi-family runtime host CLIs. Bare "pi" stays out on purpose: the token is
# claimed by Raspberry-Pi physical-device routing, and as a substring it
# hides inside "api", "pip", and "pipeline". Pi is named only through
# delegation phrasings or attached agent particles. "with pi" and "pi로"
# are deliberately absent: "with pip install" and "api로" would both
# match them as substrings.
#
# Even the bounded forms hide inside ordinary words as raw substrings:
# "api한테" contains "pi한테", and a word ending in the runtime's short name
# contains `RUNTIME_USER_PHRASE`. Every
# consumer of this group must match it with `contains_boundary_phrase`,
# never plain containment.
PI_FAMILY_CODING_AGENT_PHRASES: tuple[str, ...] = (
    PI_DISTRIBUTION_HOST,
    "opencode",
    RUNTIME_USER_PHRASE,
    "have pi implement",
    "ask pi to",
    "tell pi to",
    "delegate to pi",
    "pi한테",
    "pi에게",
)

NAMED_CODING_AGENT_PHRASES: tuple[str, ...] = (
    *SUBSTRING_NAMED_CODING_AGENT_PHRASES,
    *PI_FAMILY_CODING_AGENT_PHRASES,
)


# Multi-word or non-tokenizable coding-delivery requests.
#
# Japanese entries avoid dakuten/handakuten characters because `normalized_phrase`
# strips combining marks, so only mark-free forms survive on both routing surfaces.
#
# "해줘" and "맡겨" are broad on their own -- most Korean requests end in "해줘",
# and "맡겨" alone can report an unrelated hand-off -- but every consumer of this
# tuple only calls it after an explicit named coding-agent phrase already matched
# (see `named_coding_agent_delivery_requested` in `routing/policy.py` and
# `_named_coding_agent_delivery_signal` in `plugin_bundle/omh/awareness.py`), so
# the combination stays unambiguous.
CODING_DELIVERY_REQUEST_PHRASES: tuple[str, ...] = (
    "open a pr",
    "open the pr",
    "raise a pr",
    "send a pr",
    "write the code",
    "until tests pass",
    "해결",
    "고쳐",
    "고치",
    "구현",
    "수정",
    "만들어",
    "작성",
    "추가",
    "개선",
    "테스트",
    "짜줘",
    "짜 줘",
    "처리해",
    "작업해",
    "맡겨",
    "해줘",
    "実装",
    "修正",
    "解決",
    "対応",
    "直して",
    "テスト",
    "实现",
    "修复",
    "解决",
    "测试",
)

# Single English tokens that are unambiguous coding-delivery requests once an
# explicit coding-agent name is already present in the same message.
CODING_DELIVERY_REQUEST_TOKENS: frozenset[str] = frozenset(
    {
        "fix",
        "fixes",
        "implement",
        "implementation",
        "patch",
        "pr",
        "resolve",
        "solve",
        "test",
        "tests",
    }
)
