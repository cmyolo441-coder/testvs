from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
import json
from typing import Final


CATALOG_SCHEMA_VERSION = "awesome_hermes_agent_catalog/v1"
COVERAGE_SCHEMA_VERSION = "awesome_hermes_agent_coverage/v1"
SOURCE_REPO = "0xNyk/awesome-hermes-agent"
PLUGIN_SUBSECTION = "Plugins"
UPSTREAM_SOURCE_URL: Final = "https://github.com/0xNyk/awesome-hermes-agent"
UPSTREAM_SOURCE_COMMIT: Final = "27389ad544f923ee67b455457c214c679f26ad8a"
UPSTREAM_README_SHA256: Final = "33d58901d6f8a96f1801406c293d5c3061dce2e23a0525796ee22639bed01bcf"
UPSTREAM_ITEM_COUNT: Final = 216
UPSTREAM_PLUGIN_COUNT: Final = 35

#: What this catalog and the selected-outcome matrix are, said in the payloads
#: themselves rather than only in a doc. Both are projections of one upstream
#: revision packaged into an OMH release; neither observes the catalog a host
#: is running today, and a reader who mistakes one for the other believes OMH
#: covered plugins that were admitted, revised, or withdrawn since.
PACKAGED_COVERAGE_SCOPE: Final = "snapshot_limited"
PACKAGED_COVERAGE_SCOPE_NOTE: Final = (
    "This is the upstream catalog snapshot packaged with this OMH revision, not the active host catalog. "
    "It cannot report entries the host admitted, revised, or removed after the pinned revision. For current "
    "host coverage, reconcile a host-supplied plugin_catalog_snapshot/v1 with "
    "`omh ecosystem plugin-catalog coverage`."
)


class AwesomeHermesCatalogError(ValueError):
    pass


@dataclass(frozen=True)
class AwesomeHermesSource:
    repo: str
    url: str
    default_branch: str
    commit: str
    readme_sha256: str
    retrieved_at: str
    claim_boundary: str

    def to_dict(self) -> dict[str, str]:
        return {
            "repo": self.repo,
            "url": self.url,
            "default_branch": self.default_branch,
            "commit": self.commit,
            "readme_sha256": self.readme_sha256,
            "retrieved_at": self.retrieved_at,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class AwesomeHermesItem:
    id: str
    name: str
    url: str
    author: str
    author_url: str
    maturity: str
    section: str
    subsection: str
    summary: str
    readme_line: int

    def to_dict(self) -> dict[str, str | int]:
        return {
            "id": self.id,
            "name": self.name,
            "url": self.url,
            "author": self.author,
            "author_url": self.author_url,
            "maturity": self.maturity,
            "section": self.section,
            "subsection": self.subsection,
            "summary": self.summary,
            "readme_line": self.readme_line,
        }


@dataclass(frozen=True)
class CoverageRule:
    terms: tuple[str, ...]
    status: str
    priority: str
    surfaces: tuple[str, ...]
    note: str
    rule_id: str


@dataclass(frozen=True)
class AwesomeHermesCoverageSummary:
    item_count: int
    plugin_count: int
    status_counts: dict[str, int]
    priority_counts: dict[str, int]
    subsection_counts: dict[str, int]

    def to_dict(self) -> dict[str, int | dict[str, int]]:
        return {
            "item_count": self.item_count,
            "plugin_count": self.plugin_count,
            "status_counts": self.status_counts,
            "priority_counts": self.priority_counts,
            "subsection_counts": self.subsection_counts,
        }


@dataclass(frozen=True)
class AwesomeHermesCoverage:
    item: AwesomeHermesItem
    status: str
    priority: str
    omh_surfaces: tuple[str, ...]
    notes: tuple[str, ...]
    rule_set_version: str
    matched_rule_id: str

    def to_dict(self) -> dict[str, str | int | list[str]]:
        payload = self.item.to_dict()
        payload.update(
            {
                "coverage_status": self.status,
                "adoption_priority": self.priority,
                "omh_surfaces": list(self.omh_surfaces),
                "coverage_notes": list(self.notes),
                "rule_set_version": self.rule_set_version,
                "matched_rule_id": self.matched_rule_id,
            }
        )
        return payload


@dataclass(frozen=True)
class AwesomeHermesCatalog:
    source: AwesomeHermesSource
    items: tuple[AwesomeHermesItem, ...]

    def to_dict(self) -> dict[str, str | int | dict[str, str] | list[dict[str, str | int]]]:
        return {
            "schema_version": CATALOG_SCHEMA_VERSION,
            "source": self.source.to_dict(),
            "item_count": len(self.items),
            "items": [item.to_dict() for item in self.items],
        }


def awesome_hermes_catalog() -> AwesomeHermesCatalog:
    return _awesome_hermes_catalog_cached()


def awesome_hermes_items() -> tuple[AwesomeHermesItem, ...]:
    return awesome_hermes_catalog().items


def awesome_hermes_item(item_id: str) -> AwesomeHermesCoverage:
    normalized = item_id.strip().lower()
    for coverage in awesome_hermes_coverage():
        if coverage.item.id == normalized or coverage.item.name.lower() == normalized:
            return coverage
    raise AwesomeHermesCatalogError(f"unknown awesome-hermes-agent item: {item_id}")


def awesome_hermes_coverage(
    *,
    section: str = "",
    subsection: str = "",
    maturity: str = "",
    status: str = "",
) -> tuple[AwesomeHermesCoverage, ...]:
    from .awesome_hermes_agent_rules import RULE_SET_VERSION

    filters = {
        "section": section.strip().lower(),
        "subsection": subsection.strip().lower(),
        "maturity": maturity.strip().lower(),
        "status": status.strip().lower(),
    }
    coverage = _awesome_hermes_coverage_cached(awesome_hermes_items(), RULE_SET_VERSION)
    return tuple(item for item in coverage if _coverage_matches(item, filters))


def awesome_hermes_summary() -> dict[str, str | int | dict[str, int]]:
    summary = _summarize_coverage(awesome_hermes_coverage())
    return {
        "schema_version": COVERAGE_SCHEMA_VERSION,
        "source_repo": SOURCE_REPO,
        "source_commit": awesome_hermes_catalog().source.commit,
        "coverage_scope": PACKAGED_COVERAGE_SCOPE,
        "coverage_scope_note": PACKAGED_COVERAGE_SCOPE_NOTE,
        **summary.to_dict(),
    }


def awesome_hermes_coverage_payload(
    *,
    section: str = "",
    subsection: str = "",
    maturity: str = "",
    status: str = "",
) -> dict[str, str | int | dict[str, str] | dict[str, int] | list[dict[str, str | int | list[str]]]]:
    coverage = awesome_hermes_coverage(section=section, subsection=subsection, maturity=maturity, status=status)
    return {
        "schema_version": COVERAGE_SCHEMA_VERSION,
        "source": awesome_hermes_catalog().source.to_dict(),
        "filters": {
            "section": section,
            "subsection": subsection,
            "maturity": maturity,
            "status": status,
        },
        "item_count": len(coverage),
        "summary": _summarize_coverage(coverage).to_dict(),
        "catalog_summary": awesome_hermes_summary(),
        "coverage_scope": PACKAGED_COVERAGE_SCOPE,
        "coverage_scope_note": PACKAGED_COVERAGE_SCOPE_NOTE,
        "items": [item.to_dict() for item in coverage],
        "claim_boundary": (
            "Coverage means OMH has a routing, review, readiness, or handoff surface. It is not external "
            "plugin installation, safety approval, live connector evidence, or feature parity."
        ),
    }


@lru_cache(maxsize=1)
def _awesome_hermes_coverage_cached(
    items: tuple[AwesomeHermesItem, ...],
    _rule_set_version: str,
) -> tuple[AwesomeHermesCoverage, ...]:
    return tuple(_coverage_for_item(item) for item in items)


@lru_cache(maxsize=1)
def _awesome_hermes_catalog_cached() -> AwesomeHermesCatalog:
    resource = files("omh.catalogs").joinpath("awesome_hermes_agent_catalog.json")
    raw = json.loads(resource.read_text(encoding="utf-8"))
    return _parse_catalog(raw)


def _parse_catalog(raw: object) -> AwesomeHermesCatalog:
    if not isinstance(raw, dict):
        raise AwesomeHermesCatalogError("awesome Hermes catalog must be a JSON object")
    if raw.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise AwesomeHermesCatalogError("unsupported awesome Hermes catalog schema")
    source = _parse_source(raw.get("source"))
    _validate_pinned_source(source)
    raw_items = raw.get("items")
    if not isinstance(raw_items, list):
        raise AwesomeHermesCatalogError("awesome Hermes catalog items must be a list")
    items = tuple(_parse_item(item) for item in raw_items)
    if raw.get("item_count") != len(items):
        raise AwesomeHermesCatalogError("awesome Hermes catalog item_count does not match items")
    if len(items) != UPSTREAM_ITEM_COUNT:
        raise AwesomeHermesCatalogError("awesome Hermes catalog item count deviates from pinned upstream source")
    plugin_count = sum(item.subsection == PLUGIN_SUBSECTION for item in items)
    if plugin_count != UPSTREAM_PLUGIN_COUNT:
        raise AwesomeHermesCatalogError("awesome Hermes catalog plugin count deviates from pinned upstream source")
    return AwesomeHermesCatalog(source, items)


def _summarize_coverage(coverage: tuple[AwesomeHermesCoverage, ...]) -> AwesomeHermesCoverageSummary:
    return AwesomeHermesCoverageSummary(
        item_count=len(coverage),
        plugin_count=sum(1 for item in coverage if item.item.subsection == PLUGIN_SUBSECTION),
        status_counts=dict(sorted(Counter(item.status for item in coverage).items())),
        priority_counts=dict(sorted(Counter(item.priority for item in coverage).items())),
        subsection_counts=dict(sorted(Counter(item.item.subsection for item in coverage).items())),
    )


def _parse_source(raw: object) -> AwesomeHermesSource:
    if not isinstance(raw, dict):
        raise AwesomeHermesCatalogError("awesome Hermes catalog source must be an object")
    return AwesomeHermesSource(
        repo=_required_str(raw, "repo"),
        url=_required_str(raw, "url"),
        default_branch=_required_str(raw, "default_branch"),
        commit=_required_str(raw, "commit"),
        readme_sha256=_required_str(raw, "readme_sha256"),
        retrieved_at=_required_str(raw, "retrieved_at"),
        claim_boundary=_required_str(raw, "claim_boundary"),
    )


def _validate_pinned_source(source: AwesomeHermesSource) -> None:
    if (
        source.repo != SOURCE_REPO
        or source.url != UPSTREAM_SOURCE_URL
        or source.default_branch != "main"
        or source.commit != UPSTREAM_SOURCE_COMMIT
        or source.readme_sha256 != UPSTREAM_README_SHA256
    ):
        raise AwesomeHermesCatalogError("awesome Hermes catalog deviates from pinned upstream source")


def _parse_item(raw: object) -> AwesomeHermesItem:
    if not isinstance(raw, dict):
        raise AwesomeHermesCatalogError("awesome Hermes item must be an object")
    return AwesomeHermesItem(
        id=_required_str(raw, "id"),
        name=_required_str(raw, "name"),
        url=_required_str(raw, "url"),
        author=_required_str(raw, "author"),
        author_url=_required_str(raw, "author_url"),
        maturity=_required_str(raw, "maturity"),
        section=_required_str(raw, "section"),
        subsection=_required_str(raw, "subsection"),
        summary=_required_str(raw, "summary"),
        readme_line=_required_int(raw, "readme_line"),
    )


def _required_str(raw: dict[object, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str):
        raise AwesomeHermesCatalogError(f"awesome Hermes catalog field must be a string: {key}")
    return value


def _required_int(raw: dict[object, object], key: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int):
        raise AwesomeHermesCatalogError(f"awesome Hermes catalog field must be an integer: {key}")
    return value


def _coverage_for_item(item: AwesomeHermesItem) -> AwesomeHermesCoverage:
    from .awesome_hermes_agent_rules import coverage_for_item

    return coverage_for_item(item)


def _coverage_matches(coverage: AwesomeHermesCoverage, filters: dict[str, str]) -> bool:
    item = coverage.item
    return (
        (not filters["section"] or item.section.lower() == filters["section"])
        and (not filters["subsection"] or item.subsection.lower() == filters["subsection"])
        and (not filters["maturity"] or item.maturity == filters["maturity"])
        and (not filters["status"] or coverage.status == filters["status"])
    )
