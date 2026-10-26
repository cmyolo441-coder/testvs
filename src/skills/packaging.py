from __future__ import annotations

from functools import lru_cache

from .catalog import installable_skill_definitions
from .render import SkillReferenceTemplate, SkillTemplate, router_skill
from .skill_record import reference_templates_in_producer_order, skill_record
from .support_files import builtin_skill_support_templates as builtin_skill_support_templates


def builtin_skill_templates() -> list[SkillTemplate]:
    return list(_builtin_skill_templates_cached())


def builtin_skill_reference_templates() -> list[SkillReferenceTemplate]:
    # The producer list is data in `skill_record.REFERENCE_PRODUCERS`, where
    # each record also reads its own references from it.
    return reference_templates_in_producer_order()


def _skill_template_for(name: str) -> SkillTemplate:
    return skill_record(name).body()


@lru_cache(maxsize=1)
def _builtin_skill_templates_cached() -> tuple[SkillTemplate, ...]:
    names = [definition.name for definition in installable_skill_definitions()]
    return (router_skill(), *[_skill_template_for(name) for name in names if name != "oh-my-hermes"])
