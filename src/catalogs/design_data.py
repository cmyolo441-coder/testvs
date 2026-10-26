"""Curated design reference data behind a deterministic local query.

The rows here are authored design knowledge, not a scrape: palette role
tokens, font-pairing stacks, and UX guidelines that an executor can query
before choosing tokens for a target project. Everything is a plain Python
structure so the lookup stays offline, dependency-free, and byte-stable.

The data informs a design contract; it does not replace one. A DESIGN.md
contract still gates implementation, and a queried row is prepared reference
material, never observed visual evidence.
"""

from __future__ import annotations

from dataclasses import dataclass


DESIGN_DATA_SCHEMA_VERSION = "design_reference_data/v1"
DESIGN_DATA_EVIDENCE_BOUNDARY = (
    "Design reference rows are prepared local data. They are not a design contract, implementation, "
    "rendered UI, accessibility measurement, or visual-QA PASS evidence."
)

DESIGN_DATA_KINDS = ("palette", "font", "ux")

DESIGN_DATA_CONTEXTS = (
    "dashboard",
    "data-viz",
    "dev-tool",
    "docs",
    "ecommerce",
    "editorial",
    "education",
    "fintech",
    "healthcare",
    "landing",
    "mobile",
    "portfolio",
    "public-sector",
    "saas",
)

PALETTE_ROLE_KEYS = (
    "background",
    "surface",
    "text",
    "muted",
    "border",
    "primary",
    "on_primary",
    "accent",
    "danger",
)


@dataclass(frozen=True)
class ColorPalette:
    """One product-context palette expressed as named role tokens."""

    name: str
    mode: str
    contexts: tuple[str, ...]
    roles: tuple[tuple[str, str], ...]
    note: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "mode": self.mode,
            "contexts": list(self.contexts),
            "roles": {role: value for role, value in self.roles},
            "note": self.note,
        }


@dataclass(frozen=True)
class FontPairing:
    """One display/body stack pairing with fallbacks and CJK guidance."""

    name: str
    contexts: tuple[str, ...]
    display_stack: str
    body_stack: str
    cjk_note: str
    note: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "contexts": list(self.contexts),
            "display_stack": self.display_stack,
            "body_stack": self.body_stack,
            "cjk_note": self.cjk_note,
            "note": self.note,
        }


@dataclass(frozen=True)
class UxGuideline:
    """One UX heuristic with the contexts it applies to and why it holds."""

    name: str
    contexts: tuple[str, ...]
    guideline: str
    rationale: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "contexts": list(self.contexts),
            "guideline": self.guideline,
            "rationale": self.rationale,
        }


_COLOR_PALETTES = (
    ColorPalette(
        name="Archive Sepia",
        mode="light",
        contexts=("docs", "editorial"),
        roles=(
            ("background", "#FAF6EE"),
            ("surface", "#FFFDF7"),
            ("text", "#2B2318"),
            ("muted", "#6E6252"),
            ("border", "#E6DCC9"),
            ("primary", "#5B4326"),
            ("on_primary", "#FDFBF6"),
            ("accent", "#2F5D50"),
            ("danger", "#98301F"),
        ),
        note="Warm paper ground for archival documentation; keep code blocks on the cooler surface token.",
    ),
    ColorPalette(
        name="Chalk Classroom",
        mode="light",
        contexts=("docs", "education"),
        roles=(
            ("background", "#FCFCF7"),
            ("surface", "#FFFFFF"),
            ("text", "#1F2933"),
            ("muted", "#616E7C"),
            ("border", "#E4E7EB"),
            ("primary", "#2C5FAE"),
            ("on_primary", "#FFFFFF"),
            ("accent", "#C77700"),
            ("danger", "#B02A1E"),
        ),
        note="Low-saturation ground for long reading sessions; the amber accent marks progress, not errors.",
    ),
    ColorPalette(
        name="Civic Navy",
        mode="light",
        contexts=("docs", "public-sector"),
        roles=(
            ("background", "#FFFFFF"),
            ("surface", "#F4F6F9"),
            ("text", "#10233F"),
            ("muted", "#4E5F78"),
            ("border", "#D3DAE4"),
            ("primary", "#1B3E7A"),
            ("on_primary", "#FFFFFF"),
            ("accent", "#8A5A00"),
            ("danger", "#A32118"),
        ),
        note="Conservative institutional palette; pair every color signal with a text label for compliance work.",
    ),
    ColorPalette(
        name="Clinical Calm",
        mode="light",
        contexts=("dashboard", "healthcare"),
        roles=(
            ("background", "#F6F9FC"),
            ("surface", "#FFFFFF"),
            ("text", "#12303F"),
            ("muted", "#5A7484"),
            ("border", "#D8E4EC"),
            ("primary", "#1A7A8C"),
            ("on_primary", "#FFFFFF"),
            ("accent", "#4B6FB5"),
            ("danger", "#B4281F"),
        ),
        note="Desaturated ground so clinical alerts are the only saturated element on screen.",
    ),
    ColorPalette(
        name="Data Prism",
        mode="light",
        contexts=("dashboard", "data-viz"),
        roles=(
            ("background", "#FFFFFF"),
            ("surface", "#F7F8FA"),
            ("text", "#1A1D23"),
            ("muted", "#5B6472"),
            ("border", "#DDE1E7"),
            ("primary", "#3355D1"),
            ("on_primary", "#FFFFFF"),
            ("accent", "#C2277A"),
            ("danger", "#C2261A"),
        ),
        note="Neutral chrome that leaves hue budget for a separate categorical series scale.",
    ),
    ColorPalette(
        name="Deep Ocean",
        mode="dark",
        contexts=("dashboard", "saas"),
        roles=(
            ("background", "#0C1A26"),
            ("surface", "#13293A"),
            ("text", "#E4EEF5"),
            ("muted", "#8FA6B5"),
            ("border", "#21394C"),
            ("primary", "#3EA6D8"),
            ("on_primary", "#06131C"),
            ("accent", "#F0A868"),
            ("danger", "#E4635A"),
        ),
        note="Dark product shell with a warm accent reserved for the single primary call to action.",
    ),
    ColorPalette(
        name="Editorial Ink",
        mode="light",
        contexts=("editorial", "portfolio"),
        roles=(
            ("background", "#FFFDF9"),
            ("surface", "#FFFFFF"),
            ("text", "#14110E"),
            ("muted", "#6B625A"),
            ("border", "#E7E0D6"),
            ("primary", "#1F2937"),
            ("on_primary", "#FFFFFF"),
            ("accent", "#9A3412"),
            ("danger", "#A61B1B"),
        ),
        note="Near-black ink primary keeps typography dominant; the rust accent marks links and pull quotes only.",
    ),
    ColorPalette(
        name="Midnight Trading",
        mode="dark",
        contexts=("dashboard", "fintech"),
        roles=(
            ("background", "#0B0F14"),
            ("surface", "#141B23"),
            ("text", "#E6EDF3"),
            ("muted", "#8A9AA8"),
            ("border", "#253039"),
            ("primary", "#2DD4A0"),
            ("on_primary", "#06120D"),
            ("accent", "#4C8DFF"),
            ("danger", "#FF5C5C"),
        ),
        note="Gain/loss reads through primary and danger; never let those two carry any non-directional meaning.",
    ),
    ColorPalette(
        name="Paper Ledger",
        mode="light",
        contexts=("docs", "fintech"),
        roles=(
            ("background", "#FBFBF8"),
            ("surface", "#FFFFFF"),
            ("text", "#1B1F23"),
            ("muted", "#5C6670"),
            ("border", "#E1E4E8"),
            ("primary", "#0B6E4F"),
            ("on_primary", "#FFFFFF"),
            ("accent", "#B45309"),
            ("danger", "#B91C1C"),
        ),
        note="Document-first fintech ground; reserve the green primary for confirmed state, not for decoration.",
    ),
    ColorPalette(
        name="Pocket Mint",
        mode="light",
        contexts=("mobile", "saas"),
        roles=(
            ("background", "#FFFFFF"),
            ("surface", "#F4F8F6"),
            ("text", "#14231C"),
            ("muted", "#5A6E66"),
            ("border", "#DCE7E1"),
            ("primary", "#128A5E"),
            ("on_primary", "#FFFFFF"),
            ("accent", "#2F6FB5"),
            ("danger", "#C42B22"),
        ),
        note="High-contrast text on a white ground survives outdoor mobile viewing better than a tinted background.",
    ),
    ColorPalette(
        name="Signal Teal",
        mode="light",
        contexts=("landing", "saas"),
        roles=(
            ("background", "#F7FAFA"),
            ("surface", "#FFFFFF"),
            ("text", "#12232B"),
            ("muted", "#55707A"),
            ("border", "#DCE7E9"),
            ("primary", "#0E7C86"),
            ("on_primary", "#FFFFFF"),
            ("accent", "#F26B3A"),
            ("danger", "#C02718"),
        ),
        note="Cool primary with a warm accent gives a landing page one unmistakable conversion color.",
    ),
    ColorPalette(
        name="Slate Console",
        mode="dark",
        contexts=("dashboard", "dev-tool"),
        roles=(
            ("background", "#0F172A"),
            ("surface", "#1E293B"),
            ("text", "#E2E8F0"),
            ("muted", "#94A3B8"),
            ("border", "#334155"),
            ("primary", "#38BDF8"),
            ("on_primary", "#0F172A"),
            ("accent", "#A78BFA"),
            ("danger", "#F87171"),
        ),
        note="Two-step dark surface separation keeps log panes readable without borders on every element.",
    ),
    ColorPalette(
        name="Soft Studio",
        mode="light",
        contexts=("landing", "portfolio"),
        roles=(
            ("background", "#F5F3EF"),
            ("surface", "#FFFFFF"),
            ("text", "#1A1A1A"),
            ("muted", "#6E6A64"),
            ("border", "#E2DED7"),
            ("primary", "#2E2A26"),
            ("on_primary", "#F5F3EF"),
            ("accent", "#B08968"),
            ("danger", "#9E2B25"),
        ),
        note="Warm neutral ground lets uploaded work images carry the color; chrome stays intentionally quiet.",
    ),
    ColorPalette(
        name="Sunrise Landing",
        mode="light",
        contexts=("ecommerce", "landing"),
        roles=(
            ("background", "#FFFBF5"),
            ("surface", "#FFFFFF"),
            ("text", "#201A17"),
            ("muted", "#6B5C53"),
            ("border", "#F0E2D4"),
            ("primary", "#E4572E"),
            ("on_primary", "#FFFFFF"),
            ("accent", "#17607D"),
            ("danger", "#9E1B1B"),
        ),
        note="Warm primary and cool accent stay distinguishable in grayscale, which matters for print receipts.",
    ),
    ColorPalette(
        name="Terminal Amber",
        mode="dark",
        contexts=("dev-tool",),
        roles=(
            ("background", "#12100E"),
            ("surface", "#1C1916"),
            ("text", "#EDE6DA"),
            ("muted", "#A29684"),
            ("border", "#2E2A25"),
            ("primary", "#E8A33D"),
            ("on_primary", "#12100E"),
            ("accent", "#7FB77E"),
            ("danger", "#E05A4E"),
        ),
        note="Warm dark shell for terminal-adjacent tooling; the green accent is success output, not a brand color.",
    ),
    ColorPalette(
        name="Warm Commerce",
        mode="light",
        contexts=("ecommerce", "landing"),
        roles=(
            ("background", "#FFF9F4"),
            ("surface", "#FFFFFF"),
            ("text", "#24160F"),
            ("muted", "#6F5A4C"),
            ("border", "#EEDFD2"),
            ("primary", "#C2410C"),
            ("on_primary", "#FFFFFF"),
            ("accent", "#166534"),
            ("danger", "#9F1239"),
        ),
        note="Green accent is stock and delivery state; keeping it out of the buy button avoids a false success read.",
    ),
)


_FONT_PAIRINGS = (
    FontPairing(
        name="Compact Data Tables",
        contexts=("dashboard", "data-viz"),
        display_stack="'IBM Plex Sans Condensed', 'Roboto Condensed', system-ui, sans-serif",
        body_stack="'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif",
        cjk_note="Condensed Latin faces have no CJK counterpart; fall back to a normal-width CJK face and accept wider columns.",
        note="Condensed headers buy column width; keep the body face normal-width so digits stay scannable.",
    ),
    FontPairing(
        name="Display Commerce",
        contexts=("ecommerce", "editorial"),
        display_stack="'Playfair Display', Georgia, 'Times New Roman', serif",
        body_stack="Lato, 'Helvetica Neue', Arial, sans-serif",
        cjk_note="Pair with a serif CJK face such as Noto Serif KR for headings; keep Korean body at 14px or larger.",
        note="High-contrast serif display against a neutral sans body; needs generous heading line-height.",
    ),
    FontPairing(
        name="Geometric Product",
        contexts=("education", "landing"),
        display_stack="Poppins, 'Century Gothic', system-ui, sans-serif",
        body_stack="'Open Sans', system-ui, -apple-system, sans-serif",
        cjk_note="Geometric Latin next to CJK looks mismatched at small sizes; raise CJK body to 15px and reduce letter-spacing to 0.",
        note="Friendly geometric headings; avoid geometric faces for dense body copy where aperture matters.",
    ),
    FontPairing(
        name="Inter Duo",
        contexts=("dashboard", "dev-tool", "saas"),
        display_stack="Inter, system-ui, -apple-system, 'Segoe UI', sans-serif",
        body_stack="Inter, system-ui, -apple-system, 'Segoe UI', sans-serif",
        cjk_note="Inter has no CJK coverage; append 'Pretendard', 'Noto Sans KR' and hold Korean body text at a 14px floor.",
        note="One family across display and body; separate the levels with weight and size, not with a second face.",
    ),
    FontPairing(
        name="Ledger Numerals",
        contexts=("data-viz", "fintech"),
        display_stack="Inter, system-ui, -apple-system, sans-serif",
        body_stack="Inter, system-ui, sans-serif",
        cjk_note="Set tabular figures before adding a CJK fallback so the fallback cannot reintroduce proportional digits.",
        note="Enable font-variant-numeric: tabular-nums and pair with 'IBM Plex Mono' for raw identifiers only.",
    ),
    FontPairing(
        name="Plex Humanist",
        contexts=("docs", "education"),
        display_stack="'IBM Plex Serif', Georgia, serif",
        body_stack="'IBM Plex Sans', system-ui, -apple-system, sans-serif",
        cjk_note="IBM Plex ships Sans KR and Sans JP; use the matching family rather than a generic CJK fallback.",
        note="Serif headings signal document structure while the sans body keeps long instructions legible.",
    ),
    FontPairing(
        name="Pretendard Korean-First",
        contexts=("dashboard", "mobile", "saas"),
        display_stack="Pretendard, 'Apple SD Gothic Neo', system-ui, sans-serif",
        body_stack="Pretendard, 'Apple SD Gothic Neo', 'Malgun Gothic', system-ui, sans-serif",
        cjk_note="Designed for mixed Korean and Latin runs; hold Korean body at 14px minimum and line-height 1.6 or looser.",
        note="Use when Korean is the primary content language, so Latin and Hangul share one metric set.",
    ),
    FontPairing(
        name="Public Sans Civic",
        contexts=("healthcare", "public-sector"),
        display_stack="'Public Sans', system-ui, -apple-system, sans-serif",
        body_stack="'Public Sans', system-ui, -apple-system, sans-serif",
        cjk_note="For Korean forms append 'Noto Sans KR' and keep the 14px body floor; form labels must not drop below it.",
        note="Single neutral family with wide weights; suits accessibility-audited forms and long legal copy.",
    ),
    FontPairing(
        name="Source Serif Editorial",
        contexts=("docs", "editorial"),
        display_stack="'Source Serif 4', Georgia, 'Times New Roman', serif",
        body_stack="'Source Sans 3', system-ui, -apple-system, sans-serif",
        cjk_note="Use 'Noto Serif KR' for Korean headings; Korean serif needs more leading than the Latin heading value.",
        note="Reading-first pairing for long-form articles and reference documentation.",
    ),
    FontPairing(
        name="Space Grotesk Landing",
        contexts=("landing", "portfolio"),
        display_stack="'Space Grotesk', 'Helvetica Neue', system-ui, sans-serif",
        body_stack="Inter, system-ui, -apple-system, sans-serif",
        cjk_note="The distinctive display face has no CJK match; set CJK headings in a plain sans and rely on size for hierarchy.",
        note="Characterful display face for a short hero line; keep it away from paragraphs and form labels.",
    ),
    FontPairing(
        name="System Native",
        contexts=("mobile", "saas"),
        display_stack="system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif",
        body_stack="system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif",
        cjk_note="System stacks resolve to the platform CJK face automatically; still set the 14px Korean body floor explicitly.",
        note="Zero webfont payload and no flash of unstyled text; the correct default when load time outranks character.",
    ),
    FontPairing(
        name="Terminal Pair",
        contexts=("dev-tool",),
        display_stack="'JetBrains Mono', 'SFMono-Regular', Menlo, monospace",
        body_stack="Inter, system-ui, -apple-system, sans-serif",
        cjk_note="Monospace CJK doubles cell width; never set Korean UI labels in the mono display face.",
        note="Mono headings echo the terminal, but body copy stays proportional so paragraphs remain readable.",
    ),
)


_UX_GUIDELINES = (
    UxGuideline(
        name="Chart baselines start at zero",
        contexts=("data-viz", "fintech"),
        guideline="Bar and area charts comparing magnitude start the value axis at zero; truncate only on an explicitly labelled axis.",
        rationale="A truncated baseline exaggerates differences the reader cannot see is exaggerated.",
    ),
    UxGuideline(
        name="Clinical data shows raw values first",
        contexts=("data-viz", "healthcare"),
        guideline="Show the measured value as the primary number and the change as secondary text.",
        rationale="Deltas hide the absolute level that a safety decision actually depends on.",
    ),
    UxGuideline(
        name="Code blocks label language and offer copy",
        contexts=("dev-tool", "docs"),
        guideline="Every code block carries a language label and a copy control.",
        rationale="Readers of documentation transcribe code; an unlabeled block loses both syntax context and a working copy path.",
    ),
    UxGuideline(
        name="Color is never the only signal",
        contexts=("data-viz", "healthcare", "public-sector"),
        guideline="Pair every color-coded state with an icon, label, or shape difference.",
        rationale="Color-blind users and grayscale printouts lose a color-only distinction entirely.",
    ),
    UxGuideline(
        name="Confirm only what cannot be undone",
        contexts=("dashboard", "dev-tool"),
        guideline="Add a confirmation step to irreversible actions; give reversible ones an undo instead.",
        rationale="Confirmations on reversible actions train users to dismiss the ones that matter.",
    ),
    UxGuideline(
        name="Destructive actions carry an undo path",
        contexts=("dashboard", "dev-tool", "fintech"),
        guideline="Provide undo, a recovery window, or an explicit restore route for every destructive action.",
        rationale="Recovery costs less design effort than the support load of unrecoverable mistakes.",
    ),
    UxGuideline(
        name="Empty states are first-run task lists",
        contexts=("dashboard", "mobile", "saas"),
        guideline="Replace empty collections with the two or three actions that populate them.",
        rationale="A blank panel gives a new user no way to learn what the surface is for.",
    ),
    UxGuideline(
        name="Error copy names the action and the next step",
        contexts=("dev-tool", "fintech", "healthcare"),
        guideline="State what failed, and what the user can do now, in that order.",
        rationale="An error without a next step converts a recoverable failure into an abandoned session.",
    ),
    UxGuideline(
        name="Focus order follows visual order",
        contexts=("dashboard", "docs", "public-sector"),
        guideline="Keep DOM order aligned with visual order and keep the focus ring visible.",
        rationale="Reordering with CSS alone leaves keyboard users navigating an invisible layout.",
    ),
    UxGuideline(
        name="Hero states the offer in one sentence",
        contexts=("landing", "portfolio"),
        guideline="The first screen names who it is for and what it does before any decorative element.",
        rationale="A visitor who cannot restate the offer after one screen has no reason to scroll.",
    ),
    UxGuideline(
        name="Korean body text holds a 14px floor",
        contexts=("docs", "editorial", "mobile"),
        guideline="Set Korean and CJK body text at 14px or larger with line-height 1.6 or looser.",
        rationale="Hangul and CJK glyphs carry more strokes per em, so a Latin-tuned 12px body becomes unreadable.",
    ),
    UxGuideline(
        name="Loading states reserve final layout",
        contexts=("dashboard", "landing", "saas"),
        guideline="Use skeletons sized to the loaded content instead of centered spinners.",
        rationale="Layout that shifts after load costs the user the click they had already aimed.",
    ),
    UxGuideline(
        name="Long-form lines cap near 70 characters",
        contexts=("docs", "editorial"),
        guideline="Constrain body measure to roughly 60-75 characters per line.",
        rationale="Beyond that width the eye loses the line return and re-reads the same row.",
    ),
    UxGuideline(
        name="Number formatting is explicit",
        contexts=("data-viz", "fintech"),
        guideline="Fix currency, decimal precision, grouping, and timezone per field rather than per locale default.",
        rationale="An implicit locale silently changes what a financial figure means between two readers.",
    ),
    UxGuideline(
        name="One primary action per view",
        contexts=("ecommerce", "landing", "mobile"),
        guideline="Give a screen exactly one primary-weight button; demote the rest.",
        rationale="Two equally-weighted primary actions make the user decide before they understand either.",
    ),
    UxGuideline(
        name="Preserve input across failure",
        contexts=("ecommerce", "education", "public-sector"),
        guideline="Restore entered values after a validation error, navigation, or session timeout.",
        rationale="Re-entry is the most common abandonment point in any multi-step form.",
    ),
    UxGuideline(
        name="Progress appears past one second",
        contexts=("dev-tool", "ecommerce", "saas"),
        guideline="Show determinate progress for operations over one second and a stage label past five.",
        rationale="Without feedback the user retries, which duplicates the work that was already running.",
    ),
    UxGuideline(
        name="Respect reduced motion",
        contexts=("editorial", "landing", "portfolio"),
        guideline="Gate transitions over 200ms and any parallax behind prefers-reduced-motion.",
        rationale="Vestibular triggers are an accessibility failure, not a taste disagreement.",
    ),
    UxGuideline(
        name="Search recovers from zero results",
        contexts=("docs", "ecommerce", "saas"),
        guideline="A zero-result search offers relaxed filters, corrected spelling, or nearby matches.",
        rationale="An empty list reads as a broken feature rather than as an unmatched query.",
    ),
    UxGuideline(
        name="Table density is a setting",
        contexts=("dashboard", "data-viz"),
        guideline="Ship comfortable spacing by default and expose a compact mode.",
        rationale="Scanning and reading want opposite row heights, and only the user knows which task is active.",
    ),
    UxGuideline(
        name="Tap targets stay at least 44px",
        contexts=("ecommerce", "mobile"),
        guideline="Give touch controls a 44px minimum hit area even when the visible control is smaller.",
        rationale="Below that size the miss rate rises sharply on real thumbs and moving devices.",
    ),
    UxGuideline(
        name="Validate on blur, not per keystroke",
        contexts=("education", "fintech", "public-sector"),
        guideline="Run field validation when the field loses focus, then revalidate live once it has errored.",
        rationale="Per-keystroke errors mark correct input as wrong while the user is still typing it.",
    ),
)


def design_data_contexts() -> list[str]:
    """Return the closed context vocabulary shared by every design row."""
    return list(DESIGN_DATA_CONTEXTS)


def design_data_kinds() -> list[str]:
    """Return the queryable data kinds."""
    return list(DESIGN_DATA_KINDS)


def color_palettes() -> tuple[ColorPalette, ...]:
    return _COLOR_PALETTES


def font_pairings() -> tuple[FontPairing, ...]:
    return _FONT_PAIRINGS


def ux_guidelines() -> tuple[UxGuideline, ...]:
    return _UX_GUIDELINES


_ROWS_BY_KIND = {
    "palette": _COLOR_PALETTES,
    "font": _FONT_PAIRINGS,
    "ux": _UX_GUIDELINES,
}


def query_design_data(kind: str, context: str = "") -> dict[str, object]:
    """Return the design reference rows for one kind, optionally filtered by context.

    Ordering is by row name so repeated queries print byte-identical output.
    """
    normalized_kind = (kind or "").strip().lower()
    if normalized_kind not in _ROWS_BY_KIND:
        raise ValueError(f"unknown design data kind: {kind or '(empty)'}; expected one of {', '.join(DESIGN_DATA_KINDS)}")
    normalized_context = (context or "").strip().lower()
    if normalized_context and normalized_context not in DESIGN_DATA_CONTEXTS:
        raise ValueError(
            f"unknown design context: {context}; expected one of {', '.join(DESIGN_DATA_CONTEXTS)}"
        )
    rows = [row for row in _ROWS_BY_KIND[normalized_kind] if not normalized_context or normalized_context in row.contexts]
    rows.sort(key=lambda row: row.name)
    return {
        "schema_version": DESIGN_DATA_SCHEMA_VERSION,
        "kind": normalized_kind,
        "context": normalized_context,
        "available_contexts": list(DESIGN_DATA_CONTEXTS),
        "count": len(rows),
        "rows": [row.to_dict() for row in rows],
        "evidence_boundary": DESIGN_DATA_EVIDENCE_BOUNDARY,
    }


__all__ = [
    "ColorPalette",
    "DESIGN_DATA_CONTEXTS",
    "DESIGN_DATA_EVIDENCE_BOUNDARY",
    "DESIGN_DATA_KINDS",
    "DESIGN_DATA_SCHEMA_VERSION",
    "FontPairing",
    "PALETTE_ROLE_KEYS",
    "UxGuideline",
    "color_palettes",
    "design_data_contexts",
    "design_data_kinds",
    "font_pairings",
    "query_design_data",
    "ux_guidelines",
]
