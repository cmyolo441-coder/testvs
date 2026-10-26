from __future__ import annotations

import argparse
from pathlib import Path

from ..installer import OmhError
from ..visual_summary import (
    ASPECT_RATIO_CHOICES,
    CAPABILITY_STATES,
    LANGUAGE_MODES,
    OBSERVATION_TYPES,
    POSTER_ARCHETYPE_CHOICES,
    SOURCE_KINDS,
    VISUAL_FORMAT_CHOICES,
    build_visual_observation,
    build_visual_prompt_card,
    list_visual_observations,
    normalize_observation_type,
    parse_section_arg,
    summarize_visual_observation,
    write_visual_observation,
)
from ..workflows.visual_generation_receipts import (
    CREDENTIAL_CLASSES,
    FAILURE_STAGES,
    OPERATIONS,
    RECEIPT_OUTCOMES,
    ROUTE_FIELDS,
    UNKNOWN,
    USAGE_METRIC_NAMES,
    VISUAL_GENERATION_RECEIPT_SCHEMA_VERSION,
    VisualGenerationReceiptError,
    bind_visual_generation_receipt,
    latest_receipt_for_effect,
    parse_usage_arg,
    read_visual_generation_receipt,
    read_visual_generation_receipts,
    record_visual_generation_receipt,
    route_evidence,
    route_status_warnings,
)
from .common import _paths, _print_json


def cmd_visual_prompt_card(args: argparse.Namespace) -> int:
    try:
        source_text = _source_text_from_args(args)
        sections = [parse_section_arg(value) for value in args.section or []]
        card = build_visual_prompt_card(
            kind=args.kind,
            headline=args.headline,
            audience=args.audience,
            language=args.language,
            aspect_ratio=args.aspect_ratio,
            visual_format=args.visual_format,
            poster_archetype=args.poster_archetype,
            sections=sections,
            source_text=source_text,
            capability_state=args.capability_state,
        )
    except (OSError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    if args.json:
        _print_json(card)
    else:
        _print_prompt_card_summary(card)
    return 0


def cmd_visual_observe(args: argparse.Namespace) -> int:
    paths = _paths(args)
    try:
        observation = build_visual_observation(
            card_id=args.card_id,
            observation_type=args.type,
            path_or_uri=args.path,
            mime_type=args.mime_type,
            evidence_summary=args.summary,
            observer=args.observer,
        )
        if args.receipt_id:
            receipt = read_visual_generation_receipt(paths, args.receipt_id)
            if not receipt:
                raise ValueError(f"--receipt-id names no recorded receipt: {args.receipt_id}")
            # A superseded receipt is a retracted report. Binding one would let a
            # later `failed` report sit in the store while an observation still
            # claims the earlier success, so the attempt is judged through the
            # store's one selection rule rather than by receipt id alone.
            effect_id = str(receipt.get("effect_id", ""))
            current = latest_receipt_for_effect(
                read_visual_generation_receipts(paths, effect_id=effect_id), effect_id
            )
            if str(current.get("receipt_id", "")) != str(receipt.get("receipt_id", "")):
                raise ValueError(
                    f"--receipt-id names a superseded receipt: {args.receipt_id}; "
                    f"the attempt now reports {current.get('outcome', 'nothing')} "
                    f"in {current.get('receipt_id', 'no receipt')}"
                )
            observation = bind_visual_generation_receipt(observation, receipt)
        written = write_visual_observation(paths, observation)
    except (OSError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    if args.json:
        _print_json(written)
    else:
        _print_observation_summary(written)
    return 0


def cmd_visual_receipt(args: argparse.Namespace) -> int:
    """Record what a producer attested about one image attempt."""
    try:
        usage = dict(parse_usage_arg(value) for value in args.usage or [])
        record, minted = record_visual_generation_receipt(
            _paths(args),
            card_id=args.card_id,
            card_digest=args.card_digest,
            action_id=args.action_id,
            attempt_id=args.attempt_id,
            producer=args.producer,
            outcome=args.outcome,
            failure_stage=args.failure_stage,
            requested_route=_route_from_args(args, "requested"),
            observed_route=_route_from_args(args, "observed"),
            attested_route_fields=_attested_from_args(args),
            artifact=_artifact_from_args(args),
            input_lineage={
                "source_image_count": args.source_image_count,
                "source_image_digests": list(args.source_image_digest or []),
                "edit_constraints": list(args.edit_constraint or []),
            },
            usage=usage,
            external_effect_ref=args.external_effect_ref,
            evidence_refs=list(args.evidence_ref or []),
            summary=args.summary,
        )
    except (OSError, ValueError, VisualGenerationReceiptError) as exc:
        raise OmhError(str(exc)) from exc
    if args.json:
        _print_json({"minted": minted, "receipt": record, "route_evidence": route_evidence(record)})
    else:
        _print_receipt_summary(record, minted=minted)
    return 0


def cmd_visual_status(args: argparse.Namespace) -> int:
    """Show requested versus observed route for one card's recorded attempts."""
    paths = _paths(args)
    try:
        receipts = read_visual_generation_receipts(paths, card_id=args.card_id)
        observations = [
            summarize_visual_observation(record)
            for record in list_visual_observations(paths, card_id=args.card_id)
        ]
    except (OSError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    # A retracted report stays in the store, so the render says which receipt
    # still speaks for each attempt rather than showing two and letting a reader
    # pick the flattering one.
    superseded = {
        str(receipt.get("supersedes_receipt_ref", "")) for receipt in receipts if receipt.get("supersedes_receipt_ref")
    }
    attempts = []
    for receipt in receipts:
        evidence = route_evidence(receipt)
        evidence["superseded"] = str(receipt.get("receipt_id", "")) in superseded
        attempts.append(evidence)
    payload = {
        "card_id": args.card_id,
        "card_digest": args.card_digest,
        "receipt_schema": VISUAL_GENERATION_RECEIPT_SCHEMA_VERSION,
        "attempts": attempts,
        "store_warnings": route_status_warnings(receipts, card_digest=args.card_digest),
        "observations": observations,
        "not_evidence": ["visual QA", "attachment", "posting", "sharing", "delivery"],
    }
    if args.json:
        _print_json(payload)
    else:
        _print_status_summary(payload)
    return 0


def _route_from_args(args: argparse.Namespace, prefix: str) -> dict[str, str]:
    return {field: str(getattr(args, f"{prefix}_{field}", "") or "") for field in ROUTE_FIELDS}


def _attested_from_args(args: argparse.Namespace) -> list[str]:
    """Which observed fields the producer actually named.

    Derived from which `--observed-*` values were supplied rather than from a
    separate flag, so a caller cannot claim an attestation it did not pass a
    value for, and a value it did pass cannot go unattested and silently read as
    unknown.
    """
    return [
        field
        for field in ROUTE_FIELDS
        if str(getattr(args, f"observed_{field}", "") or "").strip() not in ("", UNKNOWN)
    ]


def _artifact_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "artifact_ref": args.artifact_ref,
        "content_sha256": args.content_sha256,
        "mime_type": args.mime_type,
        "byte_size": args.byte_size,
        "provider_response_ref": args.provider_response_ref,
    }


def _source_text_from_args(args: argparse.Namespace) -> str:
    parts: list[str] = []
    if args.from_file:
        path = Path(args.from_file).expanduser()
        if not path.is_file():
            raise ValueError(f"--from-file must point to a readable local file: {path}")
        parts.append(path.read_text(encoding="utf-8"))
    if args.source_text:
        parts.append(" ".join(args.source_text).strip())
    return "\n".join(part for part in parts if part.strip())


def _print_prompt_card_summary(card: dict[str, object]) -> None:
    capability = card.get("capability_detection", {})
    state = capability.get("state", "unknown") if isinstance(capability, dict) else "unknown"
    print("Visual prompt card prepared.")
    print(f"Card: {card['card_id']}")
    print(f"Card digest: {card.get('card_digest', '')}")
    print(f"Kind: {card['source_kind']}")
    print(f"Status: {card['status']}")
    print(f"Copy mode: {card['copy_mode']}")
    print(f"Language: {', '.join(card.get('languages', []))}")
    print(f"Visual format: {card['visual_format']}")
    theme = card.get("visual_theme", {})
    if isinstance(theme, dict) and theme.get("label"):
        domain_key = str(theme.get("domain_key", "")).strip()
        suffix = f" ({domain_key})" if domain_key else ""
        print(f"Visual domain: {theme['label']}{suffix}")
    print(f"Poster archetype: {card['poster_archetype']}")
    print(f"Aspect ratio: {card['aspect_ratio']}")
    print(f"Image generator: {state}")
    print("")
    print(str(card["image_text"]["headline"]))
    for section in card.get("sections", []):
        print(f"- {section['title']}: {section['image_text']}")
    if card.get("requires_human_or_hermes_review"):
        print("")
        print("Needs review before public use:")
        for item in card.get("missing_structured_inputs", []):
            print(f"- {item}")
    print("")
    print("Next actions:")
    if "generate_visual_image" in card.get("available_actions", []):
        print("- Generate image in the connected wrapper or image tool.")
    else:
        setup = card.get("capability_setup", {})
        if isinstance(setup, dict) and setup.get("required"):
            print("- Choose and set up an image tool before generation.")
            options = setup.get("options", [])
            if isinstance(options, list) and options:
                labels = [
                    str(item.get("label", "")).strip()
                    for item in options
                    if isinstance(item, dict) and str(item.get("label", "")).strip()
                ]
                if labels:
                    print(f"- Options: {', '.join(labels)}.")
            print("- Setup is capability preparation only; it is not generated image evidence.")
        print("- Copy the prompt into the image tool selected by the user or wrapper.")
    print("- Record generated image, visual QA, and delivery only after observed evidence exists.")
    print("")
    print("Not evidence yet: image generated, visual QA passed, delivered.")
    print("For machine-readable output, rerun with `--json`.")


def _print_observation_summary(record: dict[str, object]) -> None:
    artifact = record.get("artifact", {})
    print("Visual observation recorded.")
    print(f"Observation: {record['observation_id']}")
    print(f"Card: {record['visual_card_id']}")
    print(f"Type: {record['observation_type']}")
    print(f"Status: {record['status']}")
    if isinstance(artifact, dict):
        print(f"Artifact: {artifact.get('path_or_uri', '')}")
        print(f"MIME: {artifact.get('mime_type', '')}")
    binding = record.get("generation_receipt")
    if isinstance(binding, dict) and binding:
        print(f"Generation receipt: {binding.get('receipt_id', '')}")
        print(f"Attempt: {binding.get('attempt_id', '')}")
        print(f"Content digest: {binding.get('content_sha256', '')}")
    else:
        print("Generation receipt: none (route, model, and attempt are unknown)")
    print("")
    if record.get("does_not_prove"):
        print("Does not prove:")
        for item in record["does_not_prove"]:
            print(f"- {item}")
    print("For machine-readable output, rerun with `--json`.")


def _print_receipt_summary(record: dict[str, object], *, minted: bool) -> None:
    evidence = route_evidence(record)
    print("Visual generation receipt recorded." if minted else "Visual generation receipt already recorded.")
    print(f"Receipt: {record['receipt_id']}")
    print(f"Card: {record['card_id']}")
    print(f"Attempt: {record['attempt_id']}")
    print(f"Producer: {record['producer']}")
    print(f"Outcome: {record['outcome']} (stage {record['failure_stage']})")
    print("")
    _print_route_lines(evidence)
    print("")
    print("Does not prove:")
    for item in record.get("does_not_prove", []):
        print(f"- {item}")
    print("For machine-readable output, rerun with `--json`.")


def _print_status_summary(payload: dict[str, object]) -> None:
    attempts = payload.get("attempts", [])
    observations = payload.get("observations", [])
    print("Visual generation route evidence.")
    print(f"Card: {payload.get('card_id', '')}")
    print(f"Attempts recorded: {len(attempts) if isinstance(attempts, list) else 0}")
    if not attempts:
        print("- No visual_generation_receipt/v1 recorded; requested and observed route are both unknown.")
    for attempt in attempts if isinstance(attempts, list) else []:
        print("")
        marker = " [superseded]" if attempt.get("superseded") else ""
        print(f"Attempt {attempt.get('attempt_id', '')} ({attempt.get('receipt_id', '')}){marker}")
        print(f"Outcome: {attempt.get('outcome', '')} (stage {attempt.get('failure_stage', '')})")
        _print_route_lines(attempt)
    store_warnings = payload.get("store_warnings", [])
    if isinstance(store_warnings, list) and store_warnings:
        print("")
        print("Store warnings:")
        for warning in store_warnings:
            print(f"- {warning}")
    print("")
    print(f"Observations recorded: {len(observations) if isinstance(observations, list) else 0}")
    for observation in observations if isinstance(observations, list) else []:
        receipt_id = str(observation.get("generation_receipt_id", "")) or "unbound"
        print(f"- {observation.get('observation_type', '')}: {observation.get('artifact', '')} [{receipt_id}]")
    print("")
    print("Not evidence: visual QA, attachment, posting, sharing, delivery.")
    print("For machine-readable output, rerun with `--json`.")


def _print_route_lines(evidence: dict[str, object]) -> None:
    requested = evidence.get("requested_route", {})
    observed = evidence.get("observed_route", {})
    if isinstance(requested, dict):
        print(f"Requested route: {_route_line(requested)}")
    if isinstance(observed, dict):
        print(f"Observed route:  {_route_line(observed)}")
    warnings = evidence.get("warnings", [])
    if isinstance(warnings, list) and warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")


def _route_line(route: dict[str, object]) -> str:
    return " ".join(f"{field}={route.get(field, UNKNOWN)}" for field in ROUTE_FIELDS)


def _add_visual_commands(sub) -> None:
    _add_visual_parser(
        sub,
        "img-summary",
        help="Prepare img-summary prompt cards and record observed visual evidence.",
    )
    _add_visual_parser(
        sub,
        "visual",
        help=argparse.SUPPRESS,
        hidden=True,
    )


def _add_visual_parser(sub, name: str, *, help: str, hidden: bool = False) -> None:
    visual = sub.add_parser(
        name,
        help=help,
    )
    if hidden:
        sub._choices_actions = [action for action in sub._choices_actions if action.dest != name]
    visual_sub = visual.add_subparsers(dest=f"{name.replace('-', '_')}_command", required=True)

    prompt_card = visual_sub.add_parser(
        "prompt-card",
        help="Prepare a visual_prompt_card/v1 for a meeting, PR, issue, research, or release summary image.",
    )
    prompt_card.add_argument("source_text", nargs="*", help="Raw source text for extractive draft mode.")
    prompt_card.add_argument("--kind", required=True, help=f"Source kind or strict alias. Canonical values: {', '.join(SOURCE_KINDS)}.")
    prompt_card.add_argument("--headline", default="")
    prompt_card.add_argument("--audience", default="")
    prompt_card.add_argument("--language", choices=LANGUAGE_MODES, default="source")
    prompt_card.add_argument("--aspect-ratio", choices=ASPECT_RATIO_CHOICES, default="auto")
    prompt_card.add_argument("--visual-format", choices=VISUAL_FORMAT_CHOICES, default="auto")
    prompt_card.add_argument("--poster-archetype", choices=POSTER_ARCHETYPE_CHOICES, default="auto")
    prompt_card.add_argument("--section", action="append", metavar="ROLE:TITLE:TEXT")
    prompt_card.add_argument("--from-file", default="")
    prompt_card.add_argument("--capability-state", choices=CAPABILITY_STATES, default="unknown")
    prompt_card.add_argument("--json", action="store_true")
    prompt_card.set_defaults(func=cmd_visual_prompt_card)

    observe = visual_sub.add_parser(
        "observe",
        help="Record supplied visual_observation/v1 metadata for generated image, visual QA, or delivery evidence.",
    )
    observe.add_argument("--card-id", required=True)
    observe.add_argument(
        "--type",
        required=True,
        help=f"Observation type or alias. Canonical values: {', '.join(OBSERVATION_TYPES)}.",
    )
    observe.add_argument("--path", required=True, help="Absolute local path or URI for the observed image artifact.")
    observe.add_argument("--mime-type", default="")
    observe.add_argument("--summary", required=True, help="Short observed evidence summary.")
    observe.add_argument("--observer", default="wrapper_or_user")
    observe.add_argument(
        "--receipt-id",
        default="",
        help=(
            "Bind this observation to a recorded visual_generation_receipt/v1. "
            "Only a succeeded receipt for the same card can bind generated-image evidence."
        ),
    )
    observe.add_argument("--json", action="store_true")
    observe.set_defaults(func=cmd_visual_observe)

    receipt = visual_sub.add_parser(
        "receipt",
        help=(
            "Record a visual_generation_receipt/v1 for one image attempt, keeping requested route "
            "separate from the route the producer attested."
        ),
    )
    receipt.add_argument("--card-id", required=True)
    receipt.add_argument("--card-digest", required=True, help="sha256 digest of the visual_prompt_card/v1 revision used.")
    receipt.add_argument("--action-id", required=True, help="Opaque identity of the accepted generation action.")
    receipt.add_argument("--attempt-id", required=True, help="Opaque identity of this attempt.")
    receipt.add_argument("--producer", required=True, help="Opaque identity of the host or connector that reported this attempt.")
    receipt.add_argument("--outcome", required=True, choices=RECEIPT_OUTCOMES)
    receipt.add_argument(
        "--failure-stage",
        default="none",
        choices=FAILURE_STAGES,
        help="Where a non-succeeded attempt stopped. `none` belongs to a succeeded attempt only.",
    )
    receipt.add_argument("--operation", dest="requested_operation", required=True, choices=OPERATIONS, help="What the request asked for.")
    receipt.add_argument("--requested-provider", default="")
    receipt.add_argument("--requested-model", default="")
    receipt.add_argument("--requested-quality", default="")
    receipt.add_argument("--requested-dimensions", default="")
    receipt.add_argument("--requested-credential-class", default=UNKNOWN, choices=CREDENTIAL_CLASSES)
    for field in ("provider", "model", "quality", "dimensions"):
        receipt.add_argument(
            f"--observed-{field}",
            default="",
            help=f"Observed {field}, only when the producer attests it. Omit to leave it unknown.",
        )
    receipt.add_argument("--observed-operation", default="", choices=("", *OPERATIONS))
    receipt.add_argument(
        "--observed-credential-class",
        default="",
        choices=("", *(value for value in CREDENTIAL_CLASSES if value != UNKNOWN)),
    )
    receipt.add_argument("--artifact-ref", default="", help="Opaque handle for the returned artifact.")
    receipt.add_argument("--content-sha256", default="", help="sha256 digest of the returned bytes.")
    receipt.add_argument("--mime-type", default="")
    receipt.add_argument("--byte-size", type=int, default=None)
    receipt.add_argument("--provider-response-ref", default="")
    receipt.add_argument("--source-image-count", type=int, default=0)
    receipt.add_argument("--source-image-digest", action="append", metavar="SHA256")
    receipt.add_argument("--edit-constraint", action="append", choices=("preserve", "remove", "replace"))
    receipt.add_argument(
        "--usage",
        action="append",
        metavar="NAME=VALUE:MEASUREMENT",
        help=(
            f"Producer-attributed usage. Names: {', '.join(USAGE_METRIC_NAMES)}. "
            "Measurement is measured or estimated; omit a metric that is unavailable."
        ),
    )
    receipt.add_argument("--external-effect-ref", default="", help="Reuse an identity the wrapper already minted for this action.")
    receipt.add_argument("--evidence-ref", action="append")
    receipt.add_argument("--summary", default="", help="Short bounded metadata line.")
    receipt.add_argument("--json", action="store_true")
    receipt.set_defaults(func=cmd_visual_receipt)

    status = visual_sub.add_parser(
        "status",
        help="Show requested versus observed route, unknown route fields, and route warnings for one card.",
    )
    status.add_argument("--card-id", required=True)
    status.add_argument(
        "--card-digest",
        default="",
        help="Current card digest, so receipts filed against an earlier revision are reported as stale.",
    )
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=cmd_visual_status)


__all__ = [
    "_add_visual_commands",
    "cmd_visual_observe",
    "cmd_visual_prompt_card",
    "cmd_visual_receipt",
    "cmd_visual_status",
    "normalize_observation_type",
]
