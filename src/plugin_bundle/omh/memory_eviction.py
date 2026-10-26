"""What could be freed when Hermes' memory file has no room left.

Hermes enforces its cap by refusing the write. There is no eviction policy
behind it: the file fills, and the next thing worth remembering is simply lost.
MemGPT -- the architecture Letta grew from -- handles the same pressure with a
queue manager that evicts a fixed share of the oldest messages at a numeric
threshold, which is mechanical enough for a wrapper to reproduce.

Reproducing *oldest-first* is the part this deliberately does not do. Hermes
memory entries carry no timestamp, so "oldest" is unavailable, and position in
the file is not age. What is available is redundancy: two entries that restate
each other cost twice and are worth once. Those are proposed here.

Nothing else is. An entry that no OMH record explains is not thereby wrong --
it is unexplained, which is a reason to ask and not a reason to delete -- so it
is counted and left alone. The plan reports how far short the file is and what
is provably redundant; choosing what actually goes is the executor's, through
Hermes' own memory tool.

Metadata only: indices, character counts, similarity scores. Never entry text.
"""

from __future__ import annotations

from typing import Any

from .hermes_memory import DUPLICATE_SIMILARITY_THRESHOLD, HERMES_MEMORY_DELIMITER, similarity

EVICTION_PLAN_SCHEMA_VERSION = "omh_memory_eviction_plan/v1"


def build_eviction_plan(
    entries: tuple[str, ...] | list[str],
    *,
    cap: int,
    cap_source: str = "default",
    required_chars: int = 0,
    threshold: float = DUPLICATE_SIMILARITY_THRESHOLD,
) -> dict[str, object]:
    """What is redundant, and whether shedding it would make room.

    ``required_chars`` is what a caller wants to write. Pass 0 to ask only
    whether the file is redundant, not whether a specific write would fit.
    """
    items = [str(entry) for entry in entries]
    chars = len(HERMES_MEMORY_DELIMITER.join(items)) if items else 0
    headroom = max(0, cap - chars)
    # The delimiter Hermes inserts before an appended entry is part of the cost.
    needed = required_chars + len(HERMES_MEMORY_DELIMITER) if required_chars > 0 else 0
    shortfall = max(0, needed - headroom)

    clusters = _duplicate_clusters(items, threshold)
    reclaimable = sum(int(cluster["reclaimable_chars"]) for cluster in clusters)

    return {
        "schema_version": EVICTION_PLAN_SCHEMA_VERSION,
        "cap": cap,
        "cap_source": cap_source,
        "chars": chars,
        "entry_count": len(items),
        "headroom_chars": headroom,
        "required_chars": needed,
        "shortfall_chars": shortfall,
        "duplicate_clusters": clusters,
        "reclaimable_chars": reclaimable,
        "sufficient": shortfall == 0 or reclaimable >= shortfall,
        "unexplained_entries_are_not_candidates": True,
        "redaction_policy": "metadata_only",
        "claim_boundary": (
            "An eviction plan proposes what is redundant. It is not a deletion, not evidence "
            "that Hermes memory changed, and not a judgement that an unexplained entry is wrong."
        ),
        "next_action": (
            "Ask Hermes to merge or drop one entry per cluster through its own memory tool; "
            "OMH cannot write to Hermes memory."
        ),
    }


def _duplicate_clusters(entries: list[str], threshold: float) -> list[dict[str, object]]:
    """Groups of entries that restate one another.

    Single-link grouping: an entry joins a cluster when it is close enough to
    any member, so a chain of rewordings lands in one group rather than several
    overlapping pairs. The character count kept is the longest member's, on the
    assumption that the fullest statement is the one worth keeping -- which is a
    proposal for a reader, not a decision taken here.
    """
    clusters: list[list[int]] = []
    for index, entry in enumerate(entries):
        for cluster in clusters:
            if any(similarity(entry, entries[member]) >= threshold for member in cluster):
                cluster.append(index)
                break
        else:
            clusters.append([index])

    rows: list[dict[str, object]] = []
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        sizes = [len(entries[member]) for member in cluster]
        keep = max(sizes)
        rows.append(
            {
                "entry_indices": list(cluster),
                "entry_chars": sizes,
                "keep_chars": keep,
                # Dropping every member but the longest, delimiters included.
                "reclaimable_chars": sum(sizes) - keep + (len(cluster) - 1) * len(HERMES_MEMORY_DELIMITER),
                "min_similarity": round(_min_pairwise_similarity(entries, cluster), 2),
            }
        )
    return rows


def _min_pairwise_similarity(entries: list[str], cluster: list[int]) -> float:
    scores = [
        similarity(entries[left], entries[right])
        for position, left in enumerate(cluster)
        for right in cluster[position + 1 :]
    ]
    return min(scores) if scores else 0.0


def eviction_plan_summary(plan: dict[str, Any]) -> str:
    """One line an operator can read without opening the payload."""
    if not isinstance(plan, dict):
        return "No eviction plan."
    shortfall = int(plan.get("shortfall_chars", 0) or 0)
    reclaimable = int(plan.get("reclaimable_chars", 0) or 0)
    clusters = len(plan.get("duplicate_clusters", []) or [])
    if shortfall == 0:
        if clusters == 0:
            return "Memory has room and no redundant entries."
        return f"Memory has room; {clusters} redundant group(s) worth {reclaimable} chars remain."
    if reclaimable >= shortfall:
        return f"Short {shortfall} chars; {clusters} redundant group(s) could free {reclaimable}."
    return f"Short {shortfall} chars; only {reclaimable} chars are provably redundant."
