"""Generated ULW projections for the site and the English README.

Both surfaces derive from `ulw_inventory_payload()` -- the single producer for
the ULW engine inventory -- and are byte-checked by `omh docs ulw-site --check`
and `omh docs ulw-inventory --check`. The generators emit whole marked regions
(markers included), so a hand edit anywhere inside a region fails its check.

Neither template hardcodes an engine count: the README count word and the site
lead sentence spell the producer's canonical count. Localized README tables and
the `site/i18n.js` locale strings stay hand-maintained by decision (plan Q4);
only row counts and numerals are gated there, in `tests/test_ulw_inventory.py`.
"""

from __future__ import annotations

from ..skills.catalog import ulw_inventory_payload

ULW_README_REGION_BEGIN = (
    "<!-- omh:ulw-inventory:begin "
    "(generated: uv run python -m omh.cli docs ulw-inventory; source: src/skills/catalog.py) -->"
)
ULW_README_REGION_END = "<!-- omh:ulw-inventory:end -->"

ULW_SITE_REGION_BEGIN = (
    "<!-- omh:ulw-site:begin "
    "(generated: uv run python -m omh.cli docs ulw-site; source: src/skills/catalog.py) -->"
)
ULW_SITE_REGION_END = "<!-- omh:ulw-site:end -->"

_SITE_INDENT = "      "

# English count words for the sizes this table can plausibly reach. Falling
# back to the numeral keeps an out-of-range count honest instead of raising in
# a docs generator.
_COUNT_WORDS = {
    0: "Zero",
    1: "One",
    2: "Two",
    3: "Three",
    4: "Four",
    5: "Five",
    6: "Six",
    7: "Seven",
    8: "Eight",
    9: "Nine",
    10: "Ten",
    11: "Eleven",
    12: "Twelve",
    13: "Thirteen",
    14: "Fourteen",
    15: "Fifteen",
    16: "Sixteen",
    17: "Seventeen",
    18: "Eighteen",
    19: "Nineteen",
    20: "Twenty",
}


def _count_word(count: int) -> str:
    return _COUNT_WORDS.get(count, str(count))


def ulw_readme_region() -> str:
    """The English README ULW block, markers included, no trailing newline."""
    payload = ulw_inventory_payload()
    engines = payload["canonical_engines"]
    lines = [
        ULW_README_REGION_BEGIN,
        f"{_count_word(len(engines))} `ulw-` workflows. Say the trigger in chat — Hermes routes the",
        "rest. Full catalog: [Workflow Reference](docs/WORKFLOWS.md).",
        "",
        "| Workflow&nbsp;command | What it does |",
        "| --- | --- |",
    ]
    for engine in engines:
        lines.append(f"| ⚡ `{engine['display_name']}` | {engine['summary']} |")
    lines.append(ULW_README_REGION_END)
    return "\n".join(lines)


def ulw_site_region() -> str:
    """The site ULW section, markers included, indented for `site/index.html`."""
    payload = ulw_inventory_payload()
    engines = payload["canonical_engines"]
    lead = f"{_count_word(len(engines))} long-horizon lanes."
    lines = [
        ULW_SITE_REGION_BEGIN,
        '<section class="section section--rail" id="workflows" aria-labelledby="ulw-title">',
        '  <div class="section__head" data-reveal>',
        '    <p class="kicker" data-i18n="ulw.kicker">Flagship workflows</p>',
        '    <h2 id="ulw-title" data-i18n="ulw.title">The ulw-* family.</h2>',
        f'    <p class="section__lead" data-i18n="ulw.lead">{lead}</p>',
        "  </div>",
        "",
        '  <ol class="ulw-list">',
    ]
    for engine in engines:
        site = engine["site"]
        stem = site["i18n_stem"]
        cues = " ".join(f"<q>{cue}</q>" for cue in site["cues"])
        lines.extend(
            [
                '    <li class="glass-card ulw-row" data-reveal>',
                '      <div class="ulw-row__head">',
                f'        <code class="ulw-row__slug">{engine["display_name"]}</code>',
                f'        <span class="ulw-row__tag" data-i18n="ulw.{stem}.tag">{site["tag"]}</span>',
                "      </div>",
                '      <div class="ulw-row__body">',
                f'        <h3 data-i18n="ulw.{stem}.title">{site["title"]}</h3>',
                f'        <p data-i18n="ulw.{stem}.body">{site["body"]}</p>',
                f'        <p class="ulw-row__say"><em data-i18n="ulw.trigger">Say</em> {cues}</p>',
                "      </div>",
                "    </li>",
            ]
        )
    lines.extend(["  </ol>", "</section>", ULW_SITE_REGION_END])
    indented = [
        f"{_SITE_INDENT}{line}" if line else line
        for line in lines
    ]
    # The first marker line is spliced in after the file's own indentation, so
    # it must not carry its own.
    indented[0] = lines[0]
    return "\n".join(indented)


def _spliced(text: str, region: str, begin: str, end: str, *, label: str) -> str:
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError(
            f"{label} must contain the markers {begin!r} and {end!r} exactly once each; "
            "restore them from git or re-add the region by hand once"
        )
    start = text.index(begin)
    stop = text.index(end) + len(end)
    if stop < start:
        raise ValueError(f"{label} has its ULW end marker before the begin marker")
    return text[:start] + region + text[stop:]


def readme_with_generated_region(text: str) -> str:
    return _spliced(
        text,
        ulw_readme_region(),
        ULW_README_REGION_BEGIN,
        ULW_README_REGION_END,
        label="README.md",
    )


def site_with_generated_region(text: str) -> str:
    return _spliced(
        text,
        ulw_site_region(),
        ULW_SITE_REGION_BEGIN,
        ULW_SITE_REGION_END,
        label="site/index.html",
    )
