"""Deterministic, offline review surface for picker trigger loss and collisions.

Two things about routing triggers were previously invisible. First, a trigger
defined in the catalog does not always reach the picker frontmatter: it may be
unsafe as an unquoted YAML scalar, duplicate an alias, or fall past the
description budget. Second, several triggers normalize onto the same identity
across skills, and some of that sharing is deliberate while some is accidental.

This module reports both from the existing catalog. It adds no second catalog,
rewrites no trigger, changes no routing behavior, and emits no score, grade, or
badge. A collision is NOT presented as a defect; the report exposes the group
and its declared ownership so a human can judge, and the gate only fails when a
group's sharing has never been reviewed at all.
"""

from __future__ import annotations

from collections import Counter

from ..routing.localization import normalized_phrase
from .catalog import SkillDefinition, builtin_definitions
from .trigger_collisions import INTENTIONAL_COLLISIONS, CollisionDeclaration
from .render import (
    BUDGET_OVERFLOW,
    DUPLICATE_OF_ALIAS,
    ROUTER_CARVE_OUT,
    UNSAFE_FOR_FRONTMATTER,
    frontmatter_trigger_emission,
)

TRIGGER_REVIEW_SCHEMA_VERSION = "skill_trigger_review/v1"

# Closed reason set. `frontmatter_trigger_emission()` is the only producer, so a
# reason outside this tuple means the emission rule grew a case the review
# surface has not been taught to explain.
OMISSION_REASONS: tuple[str, ...] = (
    ROUTER_CARVE_OUT,
    UNSAFE_FOR_FRONTMATTER,
    DUPLICATE_OF_ALIAS,
    BUDGET_OVERFLOW,
)

DECLARED = "declared"
UNDECLARED = "undeclared"
COLLISION_STATUS_VALUES: tuple[str, ...] = (DECLARED, UNDECLARED)

_DECLARATION_HINT = (
    "add a CollisionDeclaration(identity=..., owners=(...), rationale_id=...) entry to "
    "INTENTIONAL_COLLISIONS in src/skills/trigger_collisions.py"
)


def trigger_omissions(definitions: list[SkillDefinition] | None = None) -> list[dict[str, str]]:
    """List every catalog-defined trigger that never reached picker frontmatter."""
    entries: list[dict[str, str]] = []
    for definition in _definitions(definitions):
        _, omitted = frontmatter_trigger_emission(definition)
        for trigger, reason in omitted:
            entries.append({"skill": definition.name, "trigger": trigger, "reason": reason})
    return entries


def collision_owner_map(definitions: list[SkillDefinition] | None = None) -> dict[str, list[tuple[str, str, str]]]:
    """Map each shared normalized identity to its sorted `(skill, source, phrase)` owners.

    Aliases participate because the picker reads them from the same description
    line: an alias colliding with another skill's trigger is the same review
    question as two triggers colliding. This is the one place ownership is
    derived, so the report and the gate can never disagree about who owns a
    group.
    """
    owners_by_identity: dict[str, set[tuple[str, str, str]]] = {}
    for definition in _definitions(definitions):
        for phrase in definition.triggers:
            _record_owner(owners_by_identity, phrase, definition.name, "trigger")
        for phrase in definition.aliases:
            _record_owner(owners_by_identity, phrase, definition.name, "alias")

    return {
        identity: sorted(owners)
        for identity, owners in sorted(owners_by_identity.items())
        if len({owner[0] for owner in owners}) > 1
    }


def collision_groups(definitions: list[SkillDefinition] | None = None) -> list[dict[str, object]]:
    """Render the shared-identity map as the report's collision-group rows."""
    return [
        {
            "identity": identity,
            "owners": [
                {"skill": skill, "source": source, "phrase": phrase} for skill, source, phrase in owners
            ],
        }
        for identity, owners in collision_owner_map(definitions).items()
    ]


def validate_collision_declarations(
    definitions: list[SkillDefinition] | None = None,
    *,
    declarations: tuple[CollisionDeclaration, ...] | None = None,
) -> dict[str, object]:
    """Fail closed unless every group's declared owners equal its observed owners.

    The comparison runs in both directions, because either mismatch hides an
    unreviewed collision. An owner observed but not declared is sharing nobody
    approved. An owner declared but not observed is a pre-approval: it signs off
    on a group that does not exist yet, so when that owner really does join, the
    declaration already covers it and no review is ever triggered.

    Failing states, each naming the exact edit that resolves it: an undeclared
    group, a declaration whose group no longer collides (stale), a group whose
    owner set grew past what was approved, a declaration naming an owner the
    catalog does not share, and two declarations for one identity.
    """
    approved = _declarations(declarations)
    observed = {
        identity: sorted({owner[0] for owner in owners})
        for identity, owners in collision_owner_map(definitions).items()
    }
    declared_by_identity = {declaration.identity: declaration for declaration in approved}
    errors: list[str] = []

    counted = Counter(declaration.identity for declaration in approved)
    for identity in sorted(identity for identity, count in counted.items() if count > 1):
        errors.append(
            f"duplicate trigger collision declaration {identity!r}: one identity is reviewed once, "
            f"so merge its {counted[identity]} entries into a single CollisionDeclaration "
            f"in INTENTIONAL_COLLISIONS"
        )

    for identity in sorted(observed):
        declaration = declared_by_identity.get(identity)
        if declaration is None:
            errors.append(
                f"undeclared trigger collision {identity!r} shared by "
                f"{', '.join(observed[identity])}: {_DECLARATION_HINT}"
            )
            continue
        approved_owners = ", ".join(sorted(set(declaration.owners)))
        expanded = sorted(set(observed[identity]) - set(declaration.owners))
        if expanded:
            errors.append(
                f"owner-expanded trigger collision {identity!r} adds {', '.join(expanded)} "
                f"beyond the approved owners {approved_owners}: {_DECLARATION_HINT}"
            )
        phantom = sorted(set(declaration.owners) - set(observed[identity]))
        if phantom:
            errors.append(
                f"unobserved owner in trigger collision declaration {identity!r}: "
                f"{', '.join(phantom)} does not share this identity, so approving "
                f"{approved_owners} pre-approves sharing nobody reviewed; narrow its owners to "
                f"{', '.join(observed[identity])} in INTENTIONAL_COLLISIONS"
            )

    for identity in sorted(declared_by_identity):
        if identity not in observed:
            errors.append(
                f"stale trigger collision declaration {identity!r}: the catalog no longer shares "
                f"this identity, so remove its entry from INTENTIONAL_COLLISIONS"
            )

    return {"ok": not errors, "errors": errors}


def skill_trigger_review_payload(definitions: list[SkillDefinition] | None = None) -> dict[str, object]:
    """Build the full `skill_trigger_review/v1` payload.

    Deterministic and offline: catalog order for skills, normalized-identity
    order for collisions, and no timestamps, hostnames, or environment reads.
    """
    resolved = _definitions(definitions)
    declared_by_identity = {declaration.identity: declaration for declaration in _declarations(None)}

    skills: list[dict[str, object]] = []
    for definition in resolved:
        emitted, omitted = frontmatter_trigger_emission(definition)
        skills.append(
            {
                "skill": definition.name,
                "defined_triggers": list(definition.triggers),
                "aliases": list(definition.aliases),
                "emitted_triggers": emitted,
                "omitted_triggers": [{"trigger": trigger, "reason": reason} for trigger, reason in omitted],
            }
        )

    collisions: list[dict[str, object]] = []
    for group in collision_groups(resolved):
        declaration = declared_by_identity.get(str(group["identity"]))
        entry = dict(group)
        entry["status"] = DECLARED if declaration is not None else UNDECLARED
        entry["rationale_id"] = declaration.rationale_id if declaration is not None else ""
        collisions.append(entry)

    return {
        "schema_version": TRIGGER_REVIEW_SCHEMA_VERSION,
        "review_boundary": (
            "A shared trigger is not asserted to be a defect. This report shows which defined "
            "triggers reached the picker description and which normalized identities are shared, "
            "so a maintainer can decide; judgment is reserved to the reviewer."
        ),
        "omission_reasons": list(OMISSION_REASONS),
        "skills": skills,
        "omissions": trigger_omissions(resolved),
        "collisions": collisions,
        "declaration_gate": validate_collision_declarations(resolved),
    }


def _record_owner(
    owners_by_identity: dict[str, set[tuple[str, str, str]]],
    phrase: str,
    skill: str,
    source: str,
) -> None:
    identity = normalized_phrase(phrase)
    if not identity:
        return
    owners_by_identity.setdefault(identity, set()).add((skill, source, phrase))


def _definitions(definitions: list[SkillDefinition] | None) -> list[SkillDefinition]:
    return builtin_definitions() if definitions is None else list(definitions)


def _declarations(
    declarations: tuple[CollisionDeclaration, ...] | None,
) -> tuple[CollisionDeclaration, ...]:
    return INTENTIONAL_COLLISIONS if declarations is None else declarations
