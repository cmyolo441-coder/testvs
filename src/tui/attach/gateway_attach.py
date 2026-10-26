"""The OMH TUI's gateway child: Hermes' own gateway, reached the way Hermes' Ink TUI reaches it.

This file never runs on OMH's interpreter and nothing in `omh` imports it.
`omh.tui.gateway.gateway_spawn` runs it as a script on Hermes' interpreter
(`HERMES_PYTHON`, with the import path Hermes' launcher handed the frontend),
where `python -m tui_gateway.entry` ran before. Its directory holds no module
named like one of Hermes', so the script's directory on the import path
shadows nothing. It speaks to the OMH TUI over stdio as the stdio entry does
(newline JSON-RPC), and picks how to reach Hermes by what this Hermes install
has, never by what an error said:

* A Hermes with the shared-gateway attach API (`ensure_gateway_runtime`,
  `gateway_ws_target`, `connect_private`, and the `websockets` client it
  ships) is attached to as Ink attaches (`ui-tui/scripts/gateway_bootstrap.py`,
  `gatewayClient.ts`): the profile's gateway is ensured (started with idle
  exit when none runs) or found, a session ticket is minted over its private
  control socket, and its loopback WebSocket is dialled with that ticket.
  Since Hermes 9595c3f345 the stdio entry refuses a home a running gateway
  owns, so on such a Hermes this is the way in, gateway running or not, as it
  is for Ink.
* A Hermes without that API runs its stdio entry in this process
  (`tui_gateway.entry.main`), which is what the TUI spawned before.

Attached, the owner speaks Hermes' canonical session wire, which differs from
the stdio entry's in a closed set of places. `Adapter` translates those, after
`canonicalGateway.ts` and Ink's shared-control and image handling as of Hermes
4d04303818 (2026-10-21), and passes every other frame through unchanged:

* `session.create` carries the launch policy the stdio entry read from its
  own environment (`launch_options`: model, provider, toolsets, skills, the
  launcher's client environment, `--yolo` and the other launch flags) with
  `source: tui` and a fresh `request_id`, checked against the owner's
  `runtime.describe` contract. Those variables leave this process before a
  gateway may be started from it, as Hermes' own bootstrap removes them.
* `session.resume` asks for the display projection; a create or resume
  answer's `display_messages` are its `messages`, and an approval or clarify
  still open in it is sent as a request.
* `session.close` is the owner's `session.detach` of this connection's view.
* `prompt.submit` gets an `input_id` and the images queued for its session;
  one that truncates (a rewind) first reads the session's revision and
  rewinds through the owner's revision-fenced `session.mutate`.
* `config.set model` (a `/model` switch of this session) is the owner's
  revision-fenced `session.mutate model`, read and sent as Ink's
  `canonicalSessionControls.ts` does, and answered in the stdio entry's
  shape: the model, `deferred` while a turn runs, or `confirm_required` and
  Hermes' message for a guarded target; the owner's one-time token for that
  target is kept and rides the same switch sent again with
  `confirm_expensive_model`. Every other `config.set` passes.
* `session.steer` gets the execution generation of the turn it steers,
  `shell.exec` the session it runs in, and `session.list` is the owner's live
  listing followed by its stored one.
* `session.interrupt` gets the execution generation of the turn it stops, as
  Ink's `interruptTurn` sends it; the owner refuses any other generation
  (`stale_generation`). Refused so, or with no generation known, it reads the
  session (`session.resume`) and sends once more at the generation read, only
  while that read says a turn runs.
* `image.attach`, an image `input.detect_drop` and `clipboard.paste` stage
  the image with Hermes' own `image.attach_bytes` (or Hermes' clipboard
  helper, into Hermes' image staging directory) and queue it for the
  session's next prompt, as the stdio entry queued it; `image.detach` takes it
  out of that queue. A drop that is not an image is answered as no drop, as
  Ink answers it, so the TUI pastes the text.
* An approval or one-question clarify arrives as an event and is answered
  with `approval.respond` / `clarify.respond`; here it is the server request
  the TUI answers, and its `.settled` event, or the session's detach, a
  withdrawal.
* A lifecycle event (`session.info`, `message.start`, `message.complete`, a
  stamped `error`) from an older execution generation or another owner
  epoch is dropped, as Ink's `fenceLifecycleEvent` drops it, so an
  interrupted turn's late end does not end the next one.

It keeps no frame beyond the call it is translating and writes nothing itself;
the only file written for it is a pasted clipboard image, by Hermes' own
clipboard helper, where Hermes stages images. The only connections it opens
are the ones Hermes' own Ink TUI opens: Hermes' private control socket and
the gateway's loopback WebSocket, both of the person's own Hermes home.
"""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import signal
import sys
import threading
from typing import Any
import uuid

TO_CLIENT = "client"
TO_OWNER = "owner"
Action = tuple[str, dict[str, Any]]

# What `ui-tui/scripts/gateway_bootstrap.py` removes before it may start a gateway: launch policy travels in
# session.create, and a gateway started from this process must not take it as every later session's default.
LAUNCH_POLICY_ENV = (
    "HERMES_MODEL", "HERMES_INFERENCE_MODEL", "HERMES_TUI_PROVIDER", "HERMES_INFERENCE_PROVIDER",
    "HERMES_TUI_TOOLSETS", "HERMES_TUI_SKILLS", "HERMES_CWD", "TERMINAL_CWD", "HERMES_YOLO",
    "HERMES_ACCEPT_HOOKS", "HERMES_TUI_CHECKPOINTS", "HERMES_TUI_PASS_SESSION_ID", "HERMES_KANBAN_BOARD",
    "HERMES_TUI_CLIENT_ENV",
)
# `canonicalGateway.ts` `localCreationOptions`: launch flags that ride session.create as `true`.
LAUNCH_FLAG_ENV = (
    ("yolo", "HERMES_YOLO_MODE"), ("ignore_rules", "HERMES_IGNORE_RULES"), ("safe_mode", "HERMES_SAFE_MODE"),
    ("ignore_user_config", "HERMES_IGNORE_USER_CONFIG"), ("checkpoints", "HERMES_TUI_CHECKPOINTS"),
    ("pass_session_id", "HERMES_TUI_PASS_SESSION_ID"), ("accept_hooks", "HERMES_ACCEPT_HOOKS"),
)
TRUTHY = frozenset({"1", "true", "yes", "on"})
TOOL_PROGRESS_MODES = frozenset({"off", "new", "all", "verbose"})
TOOL_PROGRESS_ENV = "HERMES_TUI_TOOL_PROGRESS"

APPROVAL_CHOICES = frozenset({"once", "session", "always", "deny"})
# The events Ink fences by execution generation and owner epoch (`gatewayCanonicalEvents.ts` `fenceLifecycleEvent`).
LIFECYCLE_EVENTS = frozenset({"session.info", "message.start", "message.complete", "error"})
# `useSessionLifecycle.ts`: the stamp of a view the TUI closes for good, applied once no other viewer is left.
CLOSE_END_REASON = "tui_close"
REFUSED = 4001  # the owner's own refusal code; not a drift code (4000, -32601), so the TUI keeps the session
PARAM_REJECTED = 4000
STALE_GENERATION = "stale_generation"  # the owner's refusal of a control sent for a generation it has moved past
NOT_FOUND = 4016  # the stdio entry's "image not found" / "unsupported image"
ANSWER_NOTICE_KEY = "omh-attach-answer"
ANSWER_NOTICE_TTL_MS = 15000
# The images the owner admits (`gateway/session_ingress_media.py`), by their first bytes, and Ink's size cap.
IMAGE_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "image/png"), (b"\xff\xd8\xff", "image/jpeg"), (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)
MAX_IMAGE_BYTES = 20 * 1024 * 1024
TICKET_TIMEOUT_SECONDS = 5
# The `/model` flags the owner's model switch takes, as `canonicalSessionControls.ts` `modelPayload` reads them:
# `--session` (and Ink's `--tui-session`) say what a switch here always is, `--provider <slug>` names the route
# and `--reasoning <level>` commits a level with the pick. Any other flag (`--global`, `--once`, `--refresh`) is
# refused with the owner's own reason before anything is sent: a session cannot write every new session's default.
MODEL_SESSION_FLAGS = frozenset({"--session", "--tui-session"})
MODEL_REASONING_LEVELS = frozenset({
    "none", "false", "disabled", "minimal", "low", "medium", "high", "xhigh", "max", "ultra",
})
UNSUPPORTED_MODEL_OPTIONS = "unsupported_model_options"
ENSURE_TIMEOUT_SECONDS = 30
OPEN_TIMEOUT_SECONDS = 10
# After the TUI closes its input: how long the answers to calls already sent (the last session's detach) may take.
# The TUI waits 0.5 s, then terminates, then waits 2 s more before the kill (`app.CLOSE_GRACE_SECONDS`,
# `app.TERMINATE_GRACE_SECONDS`): this ends inside that, and a loopback detach answers in milliseconds.
DRAIN_SECONDS = 1.5


def _whole(value: object) -> bool:
    """An int that is not a bool, as a JSON integer arrives."""
    return isinstance(value, int) and not isinstance(value, bool)


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in TRUTHY


def launch_options(env: Mapping[str, str]) -> dict[str, Any]:
    """The session.create parameters this launch asks for, read as `canonicalGateway.ts` reads them."""
    options: dict[str, Any] = {}
    for name, variable in (("model", "HERMES_MODEL"), ("provider", "HERMES_TUI_PROVIDER"), ("cwd", "HERMES_CWD")):
        if env.get(variable):
            options[name] = env[variable]
    if env.get("HERMES_TUI_TOOLSETS"):
        options["toolsets"] = env["HERMES_TUI_TOOLSETS"].split(",")
    skills: list[str] = []
    for skill in env.get("HERMES_TUI_SKILLS", "").replace("\n", ",").split(","):
        if skill.strip() and skill.strip() not in skills:
            skills.append(skill.strip())
    if skills:
        options["skills"] = skills
    client_env = _client_env(env.get("HERMES_TUI_CLIENT_ENV", ""))
    if client_env is not None:
        options["client_env"] = client_env
    turns = env.get("HERMES_TUI_MAX_TURNS", "")
    if turns:
        try:
            options["max_turns"] = int(turns)
        except ValueError:
            options["max_turns"] = turns.strip()
    for name, variable in LAUNCH_FLAG_ENV:
        if _truthy(env.get(variable)):
            options[name] = True
    return options


def _client_env(raw: str) -> dict[str, str] | None:
    """The launcher's allowlisted shell environment; an unreadable hand-off sends nothing."""
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except ValueError:
        return None
    if isinstance(value, dict) and all(isinstance(item, str) for item in value.values()):
        return value
    return None


def launch_tool_progress(env: Mapping[str, str]) -> str | None:
    """`hermes --tui -v` / `-Q`: the session's tool-progress mode, pinned right after creation.

    Raises on a value Hermes does not have, as Ink refuses the launch rather than drop it.
    """
    raw = env.get(TOOL_PROGRESS_ENV, "").strip().lower()
    if not raw:
        return None
    if raw not in TOOL_PROGRESS_MODES:
        raise ValueError(f"{TOOL_PROGRESS_ENV}={raw} is not one of {', '.join(sorted(TOOL_PROGRESS_MODES))}")
    return raw


def image_mime(data: bytes) -> str | None:
    """The image type the owner admits for these bytes, or None."""
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return next((mime for signature, mime in IMAGE_SIGNATURES if data.startswith(signature)), None)


def _row_text(content: object) -> object:
    """A stored multipart row as display text (`canonicalGateway.ts` `rowText`)."""
    if not isinstance(content, list):
        return content
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and isinstance(part.get("text"), str):
            parts.append(part["text"])
        elif isinstance(part, dict) and part.get("type"):
            kind = str(part["type"])
            parts.append("[image]" if "image" in kind else f"[{kind}]")
    return "\n".join(part for part in parts if part)


def snapshot_result(value: dict[str, Any]) -> dict[str, Any]:
    """A create or resume answer as the stdio entry shaped it: `messages` are display rows."""
    stored = value.get("messages")
    if isinstance(stored, list):
        stored = [
            {**row, "text": _row_text(row.get("content")), "name": row.get("tool_name")} if isinstance(row, dict) else row
            for row in stored
        ]
    result = {key: item for key, item in value.items() if key != "display_messages"}
    display = value.get("display_messages")
    result["messages"] = display if isinstance(display, list) else stored
    result["stored_messages"] = stored
    return result


@dataclass(frozen=True)
class LocalImage:
    """An image file the person named, read for staging."""

    path: Path
    remainder: str
    data: bytes
    mime: str


class HermesFiles:
    """The person's image files, found with Hermes' own path helpers (`hermes_cli.cli_terminal_input`).

    Each method raises `ValueError` with the reason the TUI shows.
    """

    def image(self, raw: str, cwd: str = "") -> LocalImage:
        """`image.attach`'s path, with the text after it (`/image shot.png what is this?`).

        A relative path is read against `cwd`, the session's workspace: Hermes'
        helper reads it against `TERMINAL_CWD`, which this process removed
        before it could start a gateway (`LAUNCH_POLICY_ENV`), so it is set for
        the call only.
        """
        try:
            from hermes_cli.cli_terminal_input import _resolve_attachment_path, _split_path_input
        except ImportError as error:
            raise ValueError(f"this Hermes has no attachment path helper ({error})") from None
        previous = os.environ.get("TERMINAL_CWD")
        if cwd:
            os.environ["TERMINAL_CWD"] = cwd
        try:
            path, remainder = _resolve_attachment_path(raw), ""
            if path is None:
                token, remainder = _split_path_input(raw)
                path = _resolve_attachment_path(token)
        finally:
            if previous is None:
                os.environ.pop("TERMINAL_CWD", None)
            else:
                os.environ["TERMINAL_CWD"] = previous
        if path is None:
            raise ValueError(f"image not found: {raw.strip()}")
        return self.read(path, remainder)

    def dropped_image(self, text: str, cwd: str = "") -> LocalImage | None:
        """The image a pasted path names, as `cli._detect_file_drop` finds it; None for anything else."""
        stripped = text.strip()
        unquoted = stripped[1:] if stripped[:1] in ("'", '"') else stripped
        if not unquoted.startswith(("/", "~", "./", "../", "file://")):
            return None
        try:
            return self.image(stripped, cwd)
        except ValueError:
            return None

    def read(self, path: Path, remainder: str = "") -> LocalImage:
        try:
            size = path.stat().st_size
            if size > MAX_IMAGE_BYTES:
                raise ValueError(f"image too large: {path.name} ({size} bytes; {MAX_IMAGE_BYTES} at most)")
            data = path.read_bytes()
        except OSError as error:
            raise ValueError(f"image not readable: {path.name} ({error.strerror or error})") from None
        mime = image_mime(data)
        if mime is None:
            raise ValueError(f"unsupported image: {path.name} (PNG, JPEG, GIF or WebP)")
        return LocalImage(path, remainder, data, mime)

    def clipboard(self) -> tuple[Path, str] | None:
        """The clipboard's image, saved by Hermes' clipboard helper where Hermes stages images; None without one."""
        try:
            from gateway.platforms.base import get_image_cache_dir
            from hermes_cli.clipboard import save_clipboard_image
        except ImportError as error:
            raise ValueError(f"this Hermes has no clipboard helper ({error})") from None
        path = Path(get_image_cache_dir()) / f"clip_{uuid.uuid4().hex}.png"
        if not save_clipboard_image(path):
            return None
        return path, "image/png"


@dataclass
class _Shown:
    """An owner prompt the TUI is answering as a server request."""

    kind: str
    session_id: str
    prompt_id: str
    generation: object


@dataclass
class _Session:
    subscription_id: str = ""
    generation: object = None  # the execution generation it runs, from its snapshot and moved on by its events
    epoch: object = None  # the owner epoch of its snapshot
    images: list[dict[str, str]] = field(default_factory=list)  # [{path, mime}] for the next prompt
    cwd: str = ""  # its workspace, which a relative image path is read against


def _reply(request_id: object, result: dict[str, Any]) -> Action:
    return TO_CLIENT, {"jsonrpc": "2.0", "id": request_id, "result": result}


def _refuse(request_id: object, code: int, message: str, data: dict[str, Any] | None = None) -> Action:
    error: dict[str, Any] = {"code": code, "message": message, **({"data": data} if data else {})}
    return TO_CLIENT, {"jsonrpc": "2.0", "id": request_id, "error": error}


def _withdrawal(session_id: object, request_id: str) -> Action:
    """The stdio entry's withdrawal of a request it asked (`request.cancel`), matched by the TUI by id."""
    return TO_CLIENT, _event("request.cancel", session_id, {"id": request_id, "request_ids": [request_id]})


def _refusal(error: dict[str, Any]) -> str:
    """An owner refusal's reason code (`error.data.reason`, else its message)."""
    data = error.get("data") if isinstance(error.get("data"), dict) else {}
    return str(data.get("reason") or error.get("message") or "")


def _reason(error: dict[str, Any]) -> str:
    """An owner refusal in words: its reason, and the detail it gave."""
    data = error.get("data") if isinstance(error.get("data"), dict) else {}
    reason = str(error.get("message") or error.get("code") or "refused")
    detail = data.get("detail")
    return f"{reason}: {detail}" if isinstance(detail, str) and detail else reason


def model_payload(value: str) -> dict[str, str] | None:
    """The owner's `session.mutate model` payload for a `/model` value; None for a form the owner does not take."""
    words = value.split()
    if not words or words[0].startswith("-"):
        return None
    payload, rest = {"model": words[0]}, words[1:]
    while rest:
        flag = rest.pop(0)
        if flag in MODEL_SESSION_FLAGS:
            continue
        if flag == "--provider" and rest and not rest[0].startswith("-"):
            payload["provider"] = rest.pop(0)
        elif flag == "--reasoning" and rest and rest[0].lower() in MODEL_REASONING_LEVELS:
            payload["reasoning"] = rest.pop(0).lower()
        else:
            return None
    return payload


def _event(kind: str, session_id: object, payload: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "method": "event", "params": {"type": kind, "session_id": session_id, "payload": payload}}


class Adapter:
    """Translation between the stdio entry's wire (the TUI's side) and the owner's canonical wire.

    `from_client` and `from_owner` each take one frame and return the frames
    to send, in order, each addressed to the TUI or to the owner. The calls
    this adapter makes itself carry string ids it never reuses, so they meet
    neither the TUI's integer ids nor the owner's server-request ids. The only
    I/O is `files`, which reads an image the person named.
    """

    def __init__(self, options: dict[str, Any], tool_progress: str | None = None,
                 files: HermesFiles | None = None, new_id: Callable[[], str] = lambda: uuid.uuid4().hex,
                 cwd: str = "") -> None:
        self.options = options
        self.tool_progress = tool_progress
        self.cwd = cwd  # the launch's workspace, for a session whose own is not known
        self.files = files or HermesFiles()
        self.new_id = new_id
        self.contract: dict[str, Any] | None = None  # `runtime.describe`'s session_create, once answered
        self.held: list[dict[str, Any]] = []  # session.create calls that wait for it
        self.sessions: dict[str, _Session] = {}
        self.current = ""  # the session the TUI last opened
        self.shown: dict[str, _Shown] = {}  # request id the TUI answers -> the owner prompt it stands for
        self.calls: dict[object, str] = {}  # TUI call id -> method, for the calls whose answer is translated
        self.own: dict[str, Callable[[dict[str, Any]], list[Action]]] = {}  # this adapter's calls -> continuation
        self.model_confirms: dict[tuple[object, str], str] = {}  # (session, model value) -> the owner's confirm token
        self._serial = 0

    # ── what the owner is asked first ──

    def describe(self) -> list[Action]:
        """Ask the owner for its session contract; session.create waits for the answer."""
        return [self._call("runtime.describe", {}, self._on_described)]

    def _on_described(self, frame: dict[str, Any]) -> list[Action]:
        result = frame.get("result")
        contract = result.get("session_create") if isinstance(result, dict) else None
        if isinstance(contract, dict):
            self.contract = contract
        else:
            error = frame.get("error") if isinstance(frame.get("error"), dict) else {}
            self.contract = {"refused": _reason(error) if error else "no session contract"}
        held, self.held = self.held, []
        return [action for call in held for action in self.from_client(call)]

    # ── the TUI's frames ──

    def from_client(self, frame: dict[str, Any]) -> list[Action]:
        method = frame.get("method")
        request_id = frame.get("id")
        if not isinstance(method, str):
            if isinstance(request_id, str) and request_id in self.shown:
                return self._answer_shown(self.shown.pop(request_id), frame)
            return [(TO_OWNER, frame)]  # an answer to one of the owner's own requests (sudo, secret, ...)
        params = frame.get("params")
        params = dict(params) if isinstance(params, dict) else {}
        translate = {
            "session.create": self._create,
            "session.resume": self._resume,
            "session.close": self._close,
            "prompt.submit": self._submit,
            "session.steer": self._steer,
            "session.interrupt": self._interrupt,
            "config.set": self._config_set,
            "shell.exec": self._shell,
            "session.list": self._list,
            "image.attach": self._attach,
            "image.detach": self._detach_image,
            "input.detect_drop": self._drop,
            "clipboard.paste": self._paste,
        }.get(method)
        if translate is None:
            return [(TO_OWNER, frame)]
        return translate(request_id, params)

    def _forward(self, request_id: object, method: str, params: dict[str, Any], translated: bool = True) -> Action:
        if translated:
            self.calls[request_id] = method
        return TO_OWNER, {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}

    def _create(self, request_id: object, params: dict[str, Any]) -> list[Action]:
        if self.contract is None:
            self.held.append({"jsonrpc": "2.0", "id": request_id, "method": "session.create", "params": params})
            return []
        refused = self.contract.get("refused")
        if refused:
            return [_refuse(request_id, REFUSED, f"Hermes' gateway did not describe its session contract ({refused})")]
        sources = self.contract.get("sources")
        accepted = self.contract.get("parameters")
        accepted = set(accepted) if isinstance(accepted, list) else set()
        if not isinstance(sources, list) or "tui" not in sources:
            return [_refuse(request_id, REFUSED, "Hermes' gateway takes no session from a TUI")]
        params.pop("cols", None)
        for name, value in self.options.items():
            params.setdefault(name, value)  # the TUI's own cwd wins over the launcher's
        if "client_env" not in accepted:
            params.pop("client_env", None)  # a gateway predating it runs the session with its own environment
        params["source"] = "tui"
        params["request_id"] = self.new_id()
        unsupported = sorted(name for name in params if name not in accepted)
        if unsupported:
            return [_refuse(request_id, PARAM_REJECTED,
                            f"Hermes' gateway does not take the launch options {', '.join(unsupported)}")]
        return [self._forward(request_id, "session.create", params)]

    def _resume(self, request_id: object, params: dict[str, Any]) -> list[Action]:
        params.pop("cols", None)
        params["display_projection"] = True
        return [self._forward(request_id, "session.resume", params)]

    def _close(self, request_id: object, params: dict[str, Any]) -> list[Action]:
        session_id = params.get("session_id")
        state = self._state(params)
        if state is None or not state.subscription_id:
            # This connection holds no view of it: nothing to leave, and nothing it asked can still be answered.
            return [*self._withdraw_session(session_id), _reply(request_id, {})]
        detach = {"session_id": session_id, "subscription_id": state.subscription_id, "end_reason": CLOSE_END_REASON}
        return [self._forward(request_id, "session.detach", detach)]

    def _withdraw_session(self, session_id: object) -> list[Action]:
        """Withdraw every prompt of a session this connection no longer views.

        The owner stops sending it that session's `.settled` events once it
        detaches, so a question left open here would wait behind the next
        session's, and a decline sent for it later would stop that session's turn.
        """
        withdrawn = [request_id for request_id, shown in self.shown.items() if shown.session_id == session_id]
        for request_id in withdrawn:
            del self.shown[request_id]
        return [_withdrawal(session_id, request_id) for request_id in withdrawn]

    def _submit(self, request_id: object, params: dict[str, Any]) -> list[Action]:
        row = params.pop("truncate_before_row_id", None)
        truncating = params.pop("confirm_truncate", None) is not None or row is not None
        params["input_id"] = self.new_id()
        if not truncating:
            self._take_images(params)
            return [self._forward(request_id, "prompt.submit", params)]
        return [self._call("session.resume", {"session_id": params.get("session_id")},
                           lambda snapshot: self._rewind(request_id, row, params, snapshot))]

    def _take_images(self, params: dict[str, Any]) -> list[dict[str, str]]:
        """Move the session's queued images into this prompt, as the stdio entry sent its queue with the turn."""
        state = self._state(params)
        if state is None or not state.images:
            return []
        params["attachments"], state.images = state.images, []
        return params["attachments"]

    def _rewind(self, request_id: object, row: object, submit: dict[str, Any], snapshot: dict[str, Any]) -> list[Action]:
        """Cut the session before `row` at the revision just read (`session.mutate` rewind), then send the prompt.

        The queued images stay queued until the prompt goes, and come back to
        the queue if Hermes refuses it. A prompt refused after the cut says
        so (`data.rewound`), so the TUI does not report the history as kept.
        """
        result = snapshot.get("result")
        if not isinstance(result, dict):
            error = snapshot.get("error") if isinstance(snapshot.get("error"), dict) else {}
            return [_refuse(request_id, REFUSED, f"Hermes did not read the session to rewind it ({_reason(error)})")]
        self._note_view(result, snapshot=True)
        mutate = {
            "session_id": submit.get("session_id"), "request_id": self.new_id(),
            "expected_revision": result.get("revision"), "expected_generation": result.get("execution_generation"),
            "operation": "rewind", "payload": {"target_message_id": row},
        }

        def on_rewound(reply: dict[str, Any]) -> list[Action]:
            error = reply.get("error")
            if isinstance(error, dict):
                return [_refuse(request_id, REFUSED, f"Hermes did not rewind the session ({_reason(error)})")]
            images = self._take_images(submit)

            def on_submitted(answer: dict[str, Any]) -> list[Action]:
                refused = answer.get("error")
                if not isinstance(refused, dict):
                    return self._answered("prompt.submit", {**answer, "id": request_id})
                state = self._state(submit)
                if state is not None:
                    state.images = images + state.images
                return [_refuse(request_id, REFUSED, f"Hermes cut the session, then refused the prompt ({_reason(refused)})",
                                {"rewound": True})]

            return [self._call("prompt.submit", submit, on_submitted)]

        return [self._call("session.mutate", mutate, on_rewound)]

    def _steer(self, request_id: object, params: dict[str, Any]) -> list[Action]:
        state = self._state(params)
        if state is None or not _whole(state.generation):
            return [_reply(request_id, {"status": "rejected"})]  # no turn known to steer: the TUI queues the text
        params["execution_generation"] = state.generation
        return [self._forward(request_id, "session.steer", params, translated=False)]

    def _interrupt(self, request_id: object, params: dict[str, Any]) -> list[Action]:
        state = self._state(params)
        if state is None or not _whole(state.generation):
            return [self._interrupt_read(request_id, params)]
        return [self._interrupt_at(request_id, params, state.generation, retry=True)]

    def _interrupt_at(self, request_id: object, params: dict[str, Any], generation: int, *, retry: bool) -> Action:
        """Stop the turn of `generation`; a stale one, the first time, is read afresh and sent once more."""

        def on_answer(reply: dict[str, Any]) -> list[Action]:
            error = reply.get("error")
            if retry and isinstance(error, dict) and _refusal(error) == STALE_GENERATION:
                return [self._interrupt_read(request_id, params)]
            return [(TO_CLIENT, {**reply, "id": request_id})]

        return self._call("session.interrupt", {**params, "execution_generation": generation}, on_answer)

    def _interrupt_read(self, request_id: object, params: dict[str, Any]) -> Action:
        """Read the generation the session runs, then stop that turn.

        Ink reports a stale stop instead of retrying it. The TUI only sends a
        stop while it shows a turn running, so here the read names the turn
        to stop. A read that finds no turn running answers the stop as done:
        the turn ended on its own, and nothing else is stopped.
        """
        session_id = params.get("session_id")

        def on_read(snapshot: dict[str, Any]) -> list[Action]:
            result = snapshot.get("result")
            if not isinstance(result, dict):
                error = snapshot.get("error") if isinstance(snapshot.get("error"), dict) else {}
                return [_refuse(request_id, REFUSED, f"Hermes did not read the session to interrupt it ({_reason(error)})")]
            self._note_view(result, snapshot=True)
            generation = result.get("execution_generation")
            if result.get("running") is not True or not _whole(generation):
                return [_reply(request_id, {"session_id": session_id, "execution_generation": generation,
                                            "running": False})]
            return [self._interrupt_at(request_id, params, generation, retry=False)]

        return self._call("session.resume", {"session_id": session_id}, on_read)

    def _config_set(self, request_id: object, params: dict[str, Any]) -> list[Action]:
        """`config.set model` switches the session through the owner's `session.mutate model`; any other key passes.

        The revision and generation come from a fresh `session.resume`, as
        Ink reads them. The answer is the stdio entry's: `value` is the model
        the session now runs, or will at its next turn (`deferred`), and a
        guarded target (cost, data policy, a large context) is
        `confirm_required` with Hermes' message and no value, the switch not
        made; its token is kept, and the same switch sent again with
        `confirm_expensive_model` carries it, once, as Ink sends `confirm`. A
        refusal is the owner's own, its reason kept.
        """
        if params.get("key") != "model":
            return [self._forward(request_id, "config.set", params, translated=False)]
        session_id, value = params.get("session_id"), str(params.get("value") or "")
        payload = model_payload(value)
        if payload is None:
            return [_refuse(request_id, REFUSED, UNSUPPORTED_MODEL_OPTIONS, {"reason": UNSUPPORTED_MODEL_OPTIONS})]
        token = self.model_confirms.pop((session_id, value), None)
        if params.get("confirm_expensive_model") is True and token:
            payload["confirm"] = token

        def on_read(snapshot: dict[str, Any]) -> list[Action]:
            result = snapshot.get("result")
            if not isinstance(result, dict):
                return [(TO_CLIENT, {**snapshot, "id": request_id})]
            self._note_view(result, snapshot=True)
            mutate = {
                "session_id": session_id, "request_id": self.new_id(),
                "expected_revision": result.get("revision"), "expected_generation": result.get("execution_generation"),
                "operation": "model", "payload": payload,
            }
            return [self._call("session.mutate", mutate, on_mutated)]

        def on_mutated(reply: dict[str, Any]) -> list[Action]:
            result = reply.get("result")
            if not isinstance(result, dict):
                return [(TO_CLIENT, {**reply, "id": request_id})]
            if result.get("status") == "confirmation_required":
                if isinstance(result.get("confirm"), str) and result["confirm"]:
                    self.model_confirms[(session_id, value)] = result["confirm"]
                return [_reply(request_id, {"key": "model", "value": "", "confirm_required": True,
                                            "confirm_message": str(result.get("confirm_message") or "")})]
            self._note_generation(session_id, result.get("execution_generation"))
            return [_reply(request_id, {"key": "model", "value": result.get("model") or "", "scope": "session",
                                        "deferred": result.get("status") == "deferred"})]

        return [self._call("session.resume", {"session_id": session_id}, on_read)]

    def _shell(self, request_id: object, params: dict[str, Any]) -> list[Action]:
        if self.current:
            params.setdefault("session_id", self.current)  # the owner runs it in a session's workspace
        return [self._forward(request_id, "shell.exec", params, translated=False)]

    def _list(self, request_id: object, params: dict[str, Any]) -> list[Action]:
        """The owner's live sessions, then its stored ones, as one answer (Ink's resume picker reads both)."""
        limit = params.get("limit")

        def on_live(live: dict[str, Any]) -> list[Action]:
            if not isinstance(live.get("result"), dict):
                return [(TO_CLIENT, {**live, "id": request_id})]
            rows = [row for row in live["result"].get("sessions") or [] if isinstance(row, dict)]

            def on_stored(stored: dict[str, Any]) -> list[Action]:
                result = stored.get("result")
                seen = {row.get("id") for row in rows}
                for row in (result.get("sessions") or []) if isinstance(result, dict) else []:
                    if isinstance(row, dict) and row.get("id") not in seen:
                        rows.append(row)
                kept = rows[:limit] if _whole(limit) and limit > 0 else rows
                return [_reply(request_id, {"sessions": kept})]

            return [self._call("session.list", {**params, "scope": "stored"}, on_stored)]

        return [self._call("session.list", params, on_live)]

    # ── images ──

    def _cwd(self, params: Mapping[str, Any]) -> str:
        """The workspace a relative image path is read against: the session's, else the launch's."""
        state = self._state(params)
        return (state.cwd if state is not None else "") or self.cwd

    def _attach(self, request_id: object, params: dict[str, Any]) -> list[Action]:
        try:
            image = self.files.image(str(params.get("path") or ""), self._cwd(params))
        except ValueError as error:
            return [_refuse(request_id, NOT_FOUND, str(error))]
        return [self._stage(request_id, params.get("session_id"), image, lambda staged: {
            "attached": True, "remainder": image.remainder,
            "text": image.remainder or f"[User attached image: {image.path.name}]", **staged})]

    def _drop(self, request_id: object, params: dict[str, Any]) -> list[Action]:
        image = self.files.dropped_image(str(params.get("text") or ""), self._cwd(params))
        if image is None:
            return [_reply(request_id, {"matched": False})]
        return [self._stage(request_id, params.get("session_id"), image, lambda staged: {
            "matched": True, "is_image": True,
            "text": image.remainder or f"[User attached image: {image.path.name}]", **staged})]

    def _paste(self, request_id: object, params: dict[str, Any]) -> list[Action]:
        state = self._state(params)
        if state is None:
            return [_refuse(request_id, REFUSED, "no session to attach the image to")]
        try:
            saved = self.files.clipboard()
        except ValueError as error:
            return [_refuse(request_id, REFUSED, str(error))]
        if saved is None:
            return [_reply(request_id, {"attached": False, "message": "No image found in clipboard"})]
        path, mime = saved
        state.images.append({"path": str(path), "mime": mime})
        return [_reply(request_id, {"attached": True, "path": str(path), "name": path.name, "count": len(state.images)})]

    def _stage(self, request_id: object, session_id: object, image: LocalImage,
               answer: Callable[[dict[str, Any]], dict[str, Any]]) -> Action:
        """Hand the image's bytes to the owner's staging (`image.attach_bytes`), then queue its staged path."""
        stage = {"session_id": session_id, "filename": image.path.name,
                 "content_base64": base64.b64encode(image.data).decode("ascii")}

        def on_staged(reply: dict[str, Any]) -> list[Action]:
            result = reply.get("result")
            state = self.sessions.get(session_id) if isinstance(session_id, str) else None
            if not isinstance(result, dict) or not isinstance(result.get("path"), str) or state is None:
                error = reply.get("error") if isinstance(reply.get("error"), dict) else {}
                return [_refuse(request_id, REFUSED, f"Hermes did not stage the image ({_reason(error)})")]
            state.images.append({"path": result["path"], "mime": str(result.get("mime") or image.mime)})
            staged = {"path": result["path"], "name": image.path.name, "count": len(state.images)}
            return [_reply(request_id, answer(staged))]

        return self._call("image.attach_bytes", stage, on_staged)

    def _detach_image(self, request_id: object, params: dict[str, Any]) -> list[Action]:
        state = self._state(params)
        images = state.images if state is not None else []
        kept = [image for image in images if image["path"] != params.get("path")]
        if state is not None:
            state.images = kept
        return [_reply(request_id, {"detached": len(kept) != len(images), "count": len(kept)})]

    def _answer_shown(self, shown: _Shown, frame: dict[str, Any]) -> list[Action]:
        """The TUI's answer to an approval or clarify that the owner asked as an event."""
        result = frame.get("result") if isinstance(frame.get("result"), dict) else {}
        base = {"session_id": shown.session_id, "execution_generation": shown.generation, "prompt_id": shown.prompt_id}
        if shown.kind == "approval":
            choice = result.get("choice")
            method, params = "approval.respond", {**base, "choice": choice if choice in APPROVAL_CHOICES else "deny"}
        else:
            answers = result.get("answers")
            answer = answers.get(shown.prompt_id) if isinstance(answers, dict) else None
            # An empty answer is the owner's skip, which a cancel and a skipped question both are.
            method, params = "clarify.respond", {**base, "answer": answer if isinstance(answer, str) else ""}

        def on_answered(reply: dict[str, Any]) -> list[Action]:
            error = reply.get("error")
            if not isinstance(error, dict):
                return []
            return [(TO_CLIENT, _event("notification.show", shown.session_id, {
                "text": f"Hermes did not take that answer ({_reason(error)}).", "level": "error", "kind": "ttl",
                "ttl_ms": ANSWER_NOTICE_TTL_MS, "key": ANSWER_NOTICE_KEY,
            }))]

        return [self._call(method, params, on_answered)]

    # ── the owner's frames ──

    def from_owner(self, frame: dict[str, Any]) -> list[Action]:
        method = frame.get("method")
        request_id = frame.get("id")
        if method == "event":
            return self._event(frame)
        if method is not None:
            return [(TO_CLIENT, frame)]  # a server request the TUI answers itself
        if isinstance(request_id, str) and request_id in self.own:
            return self.own.pop(request_id)(frame)
        method = self.calls.pop(request_id, None)
        if method is None:
            return [(TO_CLIENT, frame)]
        return self._answered(method, frame)

    def _answered(self, method: str, frame: dict[str, Any]) -> list[Action]:
        """The owner's answer to a TUI call this adapter translated, as the TUI reads it."""
        result = frame.get("result")
        if not isinstance(result, dict):
            return [(TO_CLIENT, frame)]
        if method in ("session.create", "session.resume"):
            return self._opened(method, frame, result)
        if method == "session.detach":
            return [*self._forget_view(result), _reply(frame.get("id"), {})]
        if method == "prompt.submit" and isinstance(result.get("ref"), dict):
            self._note_generation(result["ref"].get("session_id"), result.get("execution_generation"))
        return [(TO_CLIENT, frame)]

    def _opened(self, method: str, frame: dict[str, Any], result: dict[str, Any]) -> list[Action]:
        session_id = result.get("session_id")
        self._note_view(result, snapshot=True)
        previous = self.current
        if isinstance(session_id, str) and session_id:
            self.current = session_id
        answer = {**frame, "result": snapshot_result(result)}
        asked = [self._ask(session_id, prompt) for prompt in result.get("prompts") or []
                 if isinstance(session_id, str) and isinstance(prompt, dict)]
        actions: list[Action] = [(TO_CLIENT, answer), *(request for request in asked if request is not None)]
        if method == "session.create" and self.tool_progress and isinstance(session_id, str):
            pin = {"key": "verbose", "session_id": session_id, "value": self.tool_progress}

            def on_pinned(reply: dict[str, Any]) -> list[Action]:
                if not isinstance(reply.get("error"), dict):
                    return actions
                # The create is refused, so the session it made must not stay open on the shared gateway, stay the
                # one `shell.exec` runs in, or keep a question open in the TUI.
                reason = _reason(reply["error"])
                state = self.sessions.pop(session_id, None)
                if self.current == session_id:
                    self.current = previous
                leave = {"session_id": session_id, "subscription_id": state.subscription_id if state else "",
                         "end_reason": CLOSE_END_REASON}
                return [*self._withdraw_session(session_id), self._call("session.detach", leave, lambda _answer: []),
                        _refuse(frame.get("id"), REFUSED, f"Hermes did not take the launch's tool progress ({reason})")]

            return [self._call("config.set", pin, on_pinned)]  # before the TUI can send its first prompt
        return actions

    def _note_view(self, result: dict[str, Any], *, snapshot: bool = False) -> None:
        """A create or resume snapshot: the connection's view of that session, its epoch and the generation it runs.

        Only a snapshot replaces the epoch and the generation, as in Ink; events and receipts only move it on.
        """
        session_id = result.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            return
        state = self.sessions.setdefault(session_id, _Session())
        if isinstance(result.get("subscription_id"), str):
            state.subscription_id = result["subscription_id"]
        if snapshot and _whole(result.get("execution_generation")):
            state.generation = result["execution_generation"]
        if snapshot and _whole(result.get("authority_epoch")):
            state.epoch = result["authority_epoch"]
        info = result.get("info")
        if isinstance(info, dict) and isinstance(info.get("cwd"), str) and info["cwd"]:
            state.cwd = info["cwd"]

    def _forget_view(self, result: dict[str, Any]) -> list[Action]:
        """A detach answer: the view is gone, and with it every prompt of that session still open here."""
        session_id = result.get("session_id")
        if not isinstance(session_id, str):
            return []
        state = self.sessions.get(session_id)
        if result.get("detached") is True and state is not None and state.subscription_id == result.get("subscription_id"):
            del self.sessions[session_id]
        return self._withdraw_session(session_id)

    def _note_generation(self, session_id: object, generation: object) -> None:
        state = self.sessions.get(session_id) if isinstance(session_id, str) else None
        if state is not None and _whole(generation) and (not _whole(state.generation) or generation > state.generation):
            state.generation = generation

    def _stale(self, session_id: object, kind: object, params: Mapping[str, Any], payload: Mapping[str, Any]) -> bool:
        """Whether a lifecycle event belongs to a generation or owner epoch this session has moved past.

        Ink's `fenceLifecycleEvent`: a late `message.complete` or `error`
        from an interrupted turn must not end the turn that replaced it. An
        `error` with no stamp (a build or call failure) is never fenced, nor
        is an event whose generation is unstamped (Ink drops that too; here
        the TUI still sees it, which keeps an owner that stamps less working).
        """
        if kind not in LIFECYCLE_EVENTS:
            return False
        state = self.sessions.get(session_id) if isinstance(session_id, str) else None
        generation = params.get("execution_generation", payload.get("execution_generation"))
        epoch = params.get("authority_epoch", payload.get("authority_epoch"))
        if state is None:
            return False
        if _whole(epoch) and _whole(state.epoch) and epoch != state.epoch:
            return True
        return _whole(generation) and _whole(state.generation) and generation < state.generation

    def _state(self, params: Mapping[str, Any]) -> _Session | None:
        session_id = params.get("session_id")
        return self.sessions.get(session_id) if isinstance(session_id, str) else None

    def _event(self, frame: dict[str, Any]) -> list[Action]:
        params = frame.get("params") if isinstance(frame.get("params"), dict) else {}
        kind = params.get("type")
        session_id = params.get("session_id")
        payload = params.get("payload") if isinstance(params.get("payload"), dict) else {}
        if self._stale(session_id, kind, params, payload):
            return []
        self._note_generation(session_id, params.get("execution_generation", payload.get("execution_generation")))
        if kind in ("approval.request", "clarify.request") and isinstance(session_id, str):
            request = self._ask(session_id, {**payload, "kind": kind.split(".")[0]})
            return [request] if request is not None else []
        if kind in ("approval.settled", "clarify.settled"):
            withdrawn = [
                request_id for request_id, shown in self.shown.items()
                if shown.prompt_id == payload.get("prompt_id") and shown.session_id == session_id
            ]
            for request_id in withdrawn:
                del self.shown[request_id]
            return [_withdrawal(session_id, request_id) for request_id in withdrawn]
        return [(TO_CLIENT, frame)]

    def _ask(self, session_id: str, prompt: dict[str, Any]) -> Action | None:
        """An owner prompt as the server request the stdio entry sent for it; None for one already asked."""
        kind, prompt_id = prompt.get("kind"), prompt.get("prompt_id")
        if kind not in ("approval", "clarify") or not isinstance(prompt_id, str) or not prompt_id:
            return None
        request_id = f"omh-{kind}-{prompt_id}"
        if request_id in self.shown:
            return None
        self.shown[request_id] = _Shown(kind, session_id, prompt_id, prompt.get("execution_generation"))
        if kind == "approval":
            params = {"session_id": session_id, "request_id": prompt_id, "command": prompt.get("command") or "",
                      "description": prompt.get("description") or "", "choices": prompt.get("choices") or []}
        else:
            question = {"qid": prompt_id, "question": prompt.get("question") or "",
                        "choices": prompt.get("choices") or None, "multi_select": prompt.get("multi_select") is True}
            params = {"session_id": session_id, "questions": [question]}
        return TO_CLIENT, {"jsonrpc": "2.0", "id": request_id, "method": kind, "params": params}

    def _call(self, method: str, params: dict[str, Any], then: Callable[[dict[str, Any]], list[Action]]) -> Action:
        self._serial += 1
        request_id = f"omh-attach-{self._serial}"
        self.own[request_id] = then
        return TO_OWNER, {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}

    def waiting(self) -> bool:
        """Whether a call sent to the owner still waits for its answer."""
        return bool(self.own or self.calls)


# ── reaching the owner ──


@dataclass(frozen=True)
class AttachApi:
    """The parts of a Hermes install the attach needs; `attach_api` finds them or says there are none."""

    grant: Callable[[], tuple[str, list[str]]]
    dial: Callable[..., Any]  # `websockets.asyncio.client.connect`
    protocol: str
    errors: tuple[type[Exception], ...]  # the WebSocket client's own failures (a refused upgrade, a bad URI)


def attach_api() -> AttachApi | None:
    """This Hermes' shared-gateway attach, or None when it has only the stdio entry.

    Decided by what imports, never by what a refusal said: the functions
    Hermes' own Ink bootstrap calls, and the WebSocket client Hermes ships.
    """
    try:
        from hermes_cli.gateway_client import GATEWAY_WS_PROTOCOL, gateway_ws_target
        from hermes_cli.gateway_runtime import attachable, control_home_for, ensure_gateway_runtime
        from hermes_cli.gateway_runtime_discovery import connect_private
        from hermes_constants import get_hermes_home
        from websockets.asyncio.client import connect
        from websockets.exceptions import WebSocketException
    except ImportError:
        return None

    def grant() -> tuple[str, list[str]]:
        """`gateway_bootstrap.bootstrap(start=True)`: ensure the owner, mint a session ticket, name the socket."""
        home = get_hermes_home().resolve()
        receipt = ensure_gateway_runtime(home, timeout=ENSURE_TIMEOUT_SECONDS, idle_exit=True)
        if not attachable(receipt):
            raise RuntimeError(failure_message(receipt))
        endpoint = receipt.endpoint
        request = json.dumps({"protocol": 1, "id": 1, "verb": "session-ticket", "params": {
            "profile_id": endpoint.profile_id, "instance_id": endpoint.instance_id,
            "purpose": "interactive"}}).encode() + b"\n"
        # POSIX only, as the OMH TUI is (termios).
        with connect_private(control_home_for(home, endpoint), TICKET_TIMEOUT_SECONDS) as peer:
            peer.sendall(request)
            with peer.makefile("rb") as stream:
                raw = stream.readline(65537)
        answer = json.loads(raw)
        granted = answer.get("result") if isinstance(answer, dict) else None
        if (not isinstance(granted, dict) or answer.get("ok") is not True or answer.get("id") != 1
                or granted.get("instance_id") != endpoint.instance_id
                or granted.get("profile_id") != endpoint.profile_id
                or not isinstance(granted.get("ticket"), str)):
            raise RuntimeError("Hermes' gateway refused the session ticket")
        url, protocols = gateway_ws_target(endpoint, granted["ticket"])
        return url, list(protocols)

    return AttachApi(grant=grant, dial=connect, protocol=GATEWAY_WS_PROTOCOL, errors=(WebSocketException,))


def failure_message(receipt: object) -> str:
    """Why there is no owner to attach, worded as Hermes' own bootstrap words it."""
    state = getattr(receipt, "state", "unavailable")
    head = f"gateway {state}: {getattr(receipt, 'reason_code', None) or 'not ready'}"
    detail = getattr(receipt, "detail", None)
    lines = [line.strip() for line in str(detail or "").splitlines() if line.strip()]
    return "\n".join([head, *lines])


# ── the relay ──


def _frames(text: str) -> Iterable[dict[str, Any] | str]:
    """Each non-empty line of a WebSocket message, parsed; a line that is not a JSON object stays text."""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            frame = json.loads(line)
        except ValueError:
            yield line
            continue
        yield frame if isinstance(frame, dict) else line


def _start_stdin_reader(loop: asyncio.AbstractEventLoop, lines: asyncio.Queue[bytes | None]) -> None:
    """Read the TUI's lines on a thread; end of input arrives as None."""

    def read() -> None:
        for raw in sys.stdin.buffer:
            loop.call_soon_threadsafe(lines.put_nowait, raw)
        loop.call_soon_threadsafe(lines.put_nowait, None)

    threading.Thread(target=read, name="omh-attach-stdin", daemon=True).start()


def _write_client(frame: dict[str, Any] | str) -> None:
    text = frame if isinstance(frame, str) else json.dumps(frame)
    sys.stdout.buffer.write(text.encode("utf-8") + b"\n")
    sys.stdout.buffer.flush()


async def relay(api: AttachApi, url: str, protocols: list[str], adapter: Adapter) -> int:
    """Relay the TUI's stdio and the owner's WebSocket through `adapter` until either side ends.

    0 when the TUI closed its input (the person left); 1 when the owner's side
    ended first, which the TUI reads as a gateway that exited on its own.
    """
    async with api.dial(url, subprotocols=protocols, open_timeout=OPEN_TIMEOUT_SECONDS,
                           max_size=None, proxy=None) as socket:
        if socket.subprotocol != api.protocol:
            _say(f"Hermes' gateway answered with protocol {socket.subprotocol!r}, not {api.protocol!r}")
            return 1

        async def send(actions: list[Action]) -> None:
            for target, frame in actions:
                if target == TO_CLIENT:
                    _write_client(frame)
                else:
                    await socket.send(json.dumps(frame))

        await send(adapter.describe())
        lines: asyncio.Queue[bytes | None] = asyncio.Queue()
        _start_stdin_reader(asyncio.get_running_loop(), lines)

        async def from_owner() -> None:
            async for message in socket:
                text = message.decode("utf-8", "replace") if isinstance(message, bytes) else message
                for frame in _frames(text):
                    if isinstance(frame, str):
                        _write_client(frame)  # the TUI reports what it cannot read
                    else:
                        await send(adapter.from_owner(frame))

        async def from_client() -> None:
            while (raw := await lines.get()) is not None:
                try:
                    frame = json.loads(raw)
                except ValueError:
                    continue
                if isinstance(frame, dict):
                    await send(adapter.from_client(frame))

        owner = asyncio.create_task(from_owner())
        client = asyncio.create_task(from_client())
        done, _ = await asyncio.wait({owner, client}, return_when=asyncio.FIRST_COMPLETED)
        if client in done and client.exception() is None:
            # The TUI is leaving: the answers it still waits for (the last session's detach, which carries its
            # close stamp) may arrive, then the socket closes. The TUI terminates a child still running after its
            # close grace, so the drain does not end on that SIGTERM; it ends at DRAIN_SECONDS, inside the TUI's
            # terminate grace, before the kill.
            loop = asyncio.get_running_loop()
            try:
                loop.add_signal_handler(signal.SIGTERM, lambda: None)
            except NotImplementedError:  # Windows' event loop has no signal handlers; there SIGTERM ends the drain
                pass
            deadline = loop.time() + DRAIN_SECONDS
            while adapter.waiting() and not owner.done() and loop.time() < deadline:
                await asyncio.sleep(0.05)
            owner.cancel()
            return 0
        # The owner's side ended first, or a send to it failed: either way the connection is gone.
        for task in (owner, client):
            if not task.done():
                task.cancel()
        error = next((task.exception() for task in done if task.exception() is not None), None)
        _say(f"the connection to Hermes' gateway ended ({error or f'close code {socket.close_code}'})")
        return 1


def _say(text: str) -> None:
    sys.stderr.write(f"omh-attach: {text}\n")
    sys.stderr.flush()


def run_stdio_entry() -> int:
    """A Hermes without the attach API: its stdio entry, in this process, as `python -m tui_gateway.entry` ran it."""
    from tui_gateway.entry import main as stdio_main

    code = stdio_main()
    return code if isinstance(code, int) else 0


def main() -> int:
    try:
        # First, as in every Hermes entry point: importing it selects Hermes' dependency environment. Hermes from
        # 2026-05-08 to 2026-06-23 has no `harden_import_path` (the import still ran the module); later ones also
        # run it on import.
        from hermes_bootstrap import harden_import_path
    except ImportError:
        pass
    else:
        harden_import_path()
    api = attach_api()
    if api is None:
        return run_stdio_entry()
    try:
        launch = dict(os.environ)  # a copy: what reads it cannot change this process's environment
        options, tool_progress = launch_options(launch), launch_tool_progress(launch)
    except ValueError as error:
        _say(str(error))
        return 1
    cwd = next((os.environ[name] for name in ("HERMES_TUI_CWD", "HERMES_CWD", "TERMINAL_CWD") if os.environ.get(name)),
               os.getcwd())
    for name in LAUNCH_POLICY_ENV:
        os.environ.pop(name, None)
    try:
        url, protocols = api.grant()
    except (OSError, RuntimeError, ValueError, LookupError, AttributeError, TypeError) as error:
        # The failures Hermes' own bootstrap reports: discovery, socket I/O and timeouts, bad JSON, a refused grant.
        _say(str(error))
        return 1
    try:
        return asyncio.run(relay(api, url, protocols, Adapter(options, tool_progress, cwd=cwd)))
    except (OSError, TimeoutError, *api.errors) as error:  # a refused upgrade or ticket is the client's own error
        _say(f"could not reach Hermes' gateway: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
