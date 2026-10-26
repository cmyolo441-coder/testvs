from __future__ import annotations

from typing import Any


CHAT_SOURCES = ("generic", "discord", "slack", "telegram", "hermes", "github")
# The sources that are a messaging gateway's own channel. Only there does
# Hermes put a shared session's `[<display name>] ` before the person's text
# (`gateway/run_inbound.py`, `_prefix_inbound_sender_context`); `hermes` names
# no platform, so it is not one of them.
MESSAGING_CHAT_SOURCES = frozenset({"discord", "slack", "telegram"})
SOURCE_METADATA_KEYS = (
    "source_event_id",
    "project_ref",
    "channel_ref",
    "thread_ref",
    "user_ref",
    "timestamp",
    "agent_ref",
    "target_ref",
    "runtime_ref",
    "workflow_ref",
    "executor_ref",
    "hermes_home",
    "agent_count",
    "target_count",
    "render_profile",
)

_EVENT_TEXT_PATHS = (
    ("message", "content"),
    ("message", "text"),
    ("event", "text"),
    ("body", "text"),
    ("body", "content"),
    ("data", "text"),
    ("data", "content"),
    ("content",),
    ("text",),
    ("prompt",),
    ("input",),
)

_SOURCE_METADATA_PATHS: dict[str, tuple[tuple[str, ...], ...]] = {
    "source_event_id": (
        ("id",),
        ("event_id",),
        ("update_id",),
        ("message", "id"),
        ("message", "message_id"),
        ("event", "id"),
        ("event", "client_msg_id"),
    ),
    "project_ref": (("project_ref",), ("project", "id"), ("repository", "full_name"), ("repo", "full_name")),
    "channel_ref": (
        ("channel",),
        ("channel_id",),
        ("message", "channel"),
        ("message", "chat", "id"),
        ("event", "channel"),
        ("channel", "id"),
    ),
    "thread_ref": (("thread_ref",), ("thread", "id"), ("message", "thread_ts"), ("event", "thread_ts")),
    "user_ref": (
        ("user",),
        ("user_id",),
        ("author", "id"),
        ("message", "author", "id"),
        ("message", "from", "id"),
        ("event", "user"),
    ),
    "timestamp": (
        ("timestamp",),
        ("created_at",),
        ("ts",),
        ("message", "timestamp"),
        ("message", "date"),
        ("event", "ts"),
        ("event", "event_ts"),
    ),
    "agent_ref": (("agent_ref",), ("agent", "id"), ("bot", "id"), ("message", "agent", "id"), ("event", "agent", "id")),
    "target_ref": (("target_ref",), ("target", "id"), ("workspace", "id"), ("team", "id"), ("guild_id",)),
    "runtime_ref": (("runtime_ref",), ("runtime", "id"), ("hermes", "runtime_id"), ("hermes_runtime", "id")),
    "workflow_ref": (("workflow_ref",), ("workflow", "id"), ("workflow", "name")),
    "executor_ref": (("executor_ref",), ("executor", "id"), ("executor", "profile")),
    "hermes_home": (("hermes_home",), ("runtime", "hermes_home"), ("hermes", "home")),
    "agent_count": (("agent_count",), ("agents_count",), ("target", "agent_count"), ("runtime", "agent_count"), ("hermes", "agent_count")),
    "target_count": (("target_count",), ("targets_count",), ("runtime", "target_count"), ("hermes", "target_count")),
    "render_profile": (("render_profile",), ("rendering", "profile"), ("surface", "render_profile"), ("platform", "render_profile")),
}


# Where chat platforms list the files a message carried.  Discord posts
# `attachments` rows (`filename`, `content_type`), Slack posts `files` rows
# (`name`, `mimetype`), Telegram nests one `document` object (`file_name`,
# `mime_type`) under the message.  Only names and declared media types are
# copied out; URLs, sizes, ids, and bytes never leave the event.
_EVENT_ATTACHMENT_LIST_PATHS = (
    ("attachments",),
    ("files",),
    ("message", "attachments"),
    ("message", "files"),
    ("event", "attachments"),
    ("event", "files"),
    ("data", "attachments"),
    ("data", "files"),
)
_EVENT_ATTACHMENT_OBJECT_PATHS = (("document",), ("message", "document"))
_ATTACHMENT_NAME_KEYS = ("name", "filename", "file_name")
_ATTACHMENT_MEDIA_TYPE_KEYS = ("media_type", "content_type", "mimetype", "mime_type")
# Per-row text is bounded; the row count is not.  The event is already parsed
# in memory, so reading every row costs less than parsing it did, and a cap
# would leave a media file behind a run of text attachments unseen -- the
# one thing the declaration exists to prevent.  A bounded NAME keeps its
# suffix: the suffix is what classifies a file whose platform sent no media
# type, so cutting it off would turn a long-named PDF into text.
_MAX_ATTACHMENT_TEXT = 160
_MAX_ATTACHMENT_SUFFIX = 16


def extract_message_text(event: dict[str, Any] | str) -> str:
    if isinstance(event, str):
        return event.strip()
    if not isinstance(event, dict):
        raise ValueError("chat event must be an object or string")
    for path in _EVENT_TEXT_PATHS:
        value = value_at_path(event, path)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("chat event does not contain a supported text field")


def extract_source_metadata(event: dict[str, Any] | str) -> dict[str, str]:
    if not isinstance(event, dict):
        return {}
    metadata: dict[str, str] = {}
    for output_key, paths in _SOURCE_METADATA_PATHS.items():
        for path in paths:
            value = value_at_path(event, path)
            if isinstance(value, (str, int, float)) and str(value).strip():
                metadata[output_key] = str(value).strip()
                break
    return metadata


def extract_event_attachments(event: dict[str, Any] | str) -> list[dict[str, str]]:
    """Name and declared media type of every file the chat event carried.

    Metadata only: each row is `{"name", "media_type"}` with both values
    bounded, and a row that names neither is dropped.  Every listed file is
    returned, however many there are.  A string message has no attachments.
    """
    if not isinstance(event, dict):
        return []
    rows: list[dict[str, str]] = []
    candidates: list[Any] = []
    for path in _EVENT_ATTACHMENT_LIST_PATHS:
        value = value_at_path(event, path)
        if isinstance(value, list):
            candidates.extend(value)
    for path in _EVENT_ATTACHMENT_OBJECT_PATHS:
        value = value_at_path(event, path)
        if isinstance(value, dict):
            candidates.append(value)
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        row = {
            "name": _bounded_attachment_name(_bounded_attachment_text(candidate, _ATTACHMENT_NAME_KEYS, bound=None)),
            "media_type": _bounded_attachment_text(candidate, _ATTACHMENT_MEDIA_TYPE_KEYS),
        }
        if row["name"] or row["media_type"]:
            rows.append(row)
    return rows


def _bounded_attachment_text(
    candidate: dict[str, Any],
    keys: tuple[str, ...],
    *,
    bound: int | None = _MAX_ATTACHMENT_TEXT,
) -> str:
    for key in keys:
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            text = value.strip()
            return text if bound is None else text[:bound]
    return ""


def _bounded_attachment_name(name: str) -> str:
    """Bound a file name to the row limit without losing its suffix."""
    if len(name) <= _MAX_ATTACHMENT_TEXT:
        return name
    stem, separator, suffix = name.rpartition(".")
    if not separator or not stem or not suffix or len(suffix) > _MAX_ATTACHMENT_SUFFIX or any(ch.isspace() for ch in suffix):
        return name[:_MAX_ATTACHMENT_TEXT]
    return stem[: _MAX_ATTACHMENT_TEXT - len(suffix) - 1] + "." + suffix


def compact_source_metadata(metadata: Any) -> dict[str, str]:
    if not isinstance(metadata, dict):
        return {}
    return {key: str(metadata[key]) for key in SOURCE_METADATA_KEYS if key in metadata and str(metadata[key])}


def value_at_path(event: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = event
    for part in path:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current
