"""The agent loop: stream, call tools in parallel, repeat."""

from __future__ import annotations

import platform
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from .config import Config
from .errors import OmhError, StreamInterrupted
from .providers.base import ChatMessage, ToolCall, Usage
from .router import ModelRouter
from .session import Session
from .skills import Skill, skill_prompt
from .tools.base import Approver, ToolRegistry, ToolResult, validate_arguments


@dataclass
class AgentEvent:
    kind: str
    text: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


class _Cancelled(Exception):
    pass


def estimate_cost(profile: Any, usage: Optional[Usage]) -> float:
    if profile is None or usage is None:
        return 0.0
    return (
        usage.prompt_tokens * float(getattr(profile, "input_cost", 0.0))
        + usage.completion_tokens * float(getattr(profile, "output_cost", 0.0))
    ) / 1_000_000.0


class Agent:
    def __init__(
        self,
        config: Config,
        router: ModelRouter,
        registry: ToolRegistry,
        session: Session,
        *,
        skill: Optional[Skill] = None,
        approver: Optional[Approver] = None,
    ) -> None:
        self.config = config
        self.router = router
        self.registry = registry
        self.session = session
        self.skill = skill
        self.approver = approver
        self.cancelled = threading.Event()
        self.turn_tokens = 0
        self.turn_cost = 0.0
        self.last_model = ""
        self.steps = 0
        self._notes_seen = 0

    def cancel(self) -> None:
        self.cancelled.set()

    # -- prompt ----------------------------------------------------------
    def system_prompt(self) -> str:
        parts = [
            self.config.system_prompt,
            "",
            "Environment:",
            f"- workspace: {self.config.workspace}",
            f"- platform: {platform.system()} {platform.release()} · python {platform.python_version()}",
            f"- date: {date.today().isoformat()}",
            f"- approval: {self.config.approval_mode}",
            "- every path you pass to a file tool is relative to the workspace",
        ]
        if self.config.tools_enabled:
            parts.append("- tools: " + ", ".join(self.registry.names()))
        entries = self._workspace_snapshot()
        if entries:
            parts.extend(["", "Top level of the workspace:", entries])
        if self.skill is not None:
            parts.extend(["", skill_prompt(self.skill)])
        return "\n".join(parts)

    def _workspace_snapshot(self) -> str:
        try:
            children = sorted(
                (
                    item
                    for item in Path(self.config.workspace).iterdir()
                    if not item.name.startswith(".")
                ),
                key=lambda item: (not item.is_dir(), item.name.lower()),
            )[:18]
        except OSError:
            return ""
        return "\n".join(
            f"- {item.name}{'/' if item.is_dir() else ''}" for item in children
        )


    # -- loop ------------------------------------------------------------
    def run(self, user_input: str) -> Generator[AgentEvent, None, None]:
        self.cancelled.clear()
        self.turn_tokens = 0
        self.turn_cost = 0.0
        self.steps = 0
        self._notes_seen = len(self.router.failover_notes)

        self.session.add(ChatMessage.user(user_input))
        messages: List[ChatMessage] = [ChatMessage.system(self.system_prompt())]
        messages.extend(self._history_tail())
        messages = self._trim(messages)

        for step in range(1, max(1, self.config.max_steps) + 1):
            if self.cancelled.is_set():
                yield AgentEvent("done", data={"reason": "cancelled", "steps": self.steps})
                return
            self.steps = step
            yield AgentEvent(
                "step", data={"step": step, "max_steps": self.config.max_steps}
            )

            text_parts: List[str] = []
            calls: List[ToolCall] = []
            usage: Optional[Usage] = None
            model = ""
            try:
                for chunk in self.router.stream(
                    messages,
                    tools=self._tool_schema(),
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                ):
                    if self.cancelled.is_set():
                        raise _Cancelled()
                    if chunk.model:
                        model = chunk.model
                    if chunk.text:
                        text_parts.append(chunk.text)
                        yield AgentEvent("text", chunk.text, {"model": model})
                    if chunk.tool_calls:
                        calls = chunk.tool_calls
                    if chunk.usage:
                        usage = chunk.usage
            except (_Cancelled, KeyboardInterrupt):
                self.cancel()
                self._commit_assistant(text_parts, [], usage, model)
                yield AgentEvent("done", data={"reason": "cancelled", "steps": step})
                return
            except StreamInterrupted as exc:
                self._commit_assistant(text_parts, [], usage, model)
                yield AgentEvent("error", data={"error": exc})
                yield AgentEvent("done", data={"reason": "error", "steps": step})
                return
            except OmhError as exc:
                yield AgentEvent("error", data={"error": exc})
                yield AgentEvent("done", data={"reason": "error", "steps": step})
                return
            except Exception as exc:  # never dump a traceback at the user
                yield AgentEvent(
                    "error", data={"error": OmhError(f"{type(exc).__name__}: {exc}")}
                )
                yield AgentEvent("done", data={"reason": "error", "steps": step})
                return

            for index, call in enumerate(calls):
                if not call.id:
                    call.id = f"call_{step}_{index}"

            assistant = ChatMessage.assistant("".join(text_parts), calls)
            if assistant.content or assistant.tool_calls:
                self.session.add(assistant)
                messages.append(assistant)
            self._account(usage, model)

            yield AgentEvent(
                "usage",
                data={
                    "model": self.last_model or model,
                    "usage": usage,
                    "cost": self.turn_cost,
                    "tokens": self.turn_tokens,
                    "step": step,
                },
            )
            for note in self.router.failover_notes[self._notes_seen :]:
                yield AgentEvent("info", data={"message": f"failover · {note}"})
            self._notes_seen = len(self.router.failover_notes)

            if self.cancelled.is_set():
                yield AgentEvent("done", data={"reason": "cancelled", "steps": step})
                return
            if not calls:
                yield AgentEvent("done", data={"reason": "complete", "steps": step})
                return

            for call in calls:
                spec = self.registry.get(call.name)
                summary = ""
                if spec is not None:
                    try:
                        summary = spec.summary(call.parsed())
                    except ValueError:
                        summary = ""
                yield AgentEvent(
                    "tool_start",
                    data={
                        "call": call,
                        "summary": summary,
                        "spec": spec,
                        "model": model,
                    },
                )

            denied = self._pre_approve(calls)
            interrupted = False
            try:
                results = self._execute(calls, denied)
            except KeyboardInterrupt:
                self.cancel()
                interrupted = True
                results = dict(denied)
                for call in calls:
                    results.setdefault(
                        call.id,
                        ToolResult.error("cancelled by user", cancelled=True),
                    )
            for call in calls:
                result = results[call.id]
                tool_message = ChatMessage.tool(
                    call.id, self._tool_output(result), call.name
                )
                messages.append(tool_message)
                self.session.add(tool_message)
                yield AgentEvent("tool_result", data={"call": call, "result": result})
            if interrupted:
                yield AgentEvent(
                    "done", data={"reason": "cancelled", "steps": step}
                )
                return

        yield AgentEvent(
            "done",
            data={
                "reason": "max_steps",
                "steps": self.steps,
                "message": f"stopped after {self.steps} steps",
            },
        )


    # -- helpers ---------------------------------------------------------
    def _tool_schema(self) -> Optional[List[Dict[str, Any]]]:
        if not self.config.tools_enabled or not self.registry.specs:
            return None
        return self.registry.schema()

    def _history_tail(self) -> List[ChatMessage]:
        history = [message for message in self.session.messages if message.role != "system"]
        while history and history[0].role != "user":
            history.pop(0)
        return self._repair(history)

    @staticmethod
    def _repair(messages: List[ChatMessage]) -> List[ChatMessage]:
        """Drop orphan tool messages and assistant turns with missing results.

        A cancelled or killed run can leave a tool call without its result;
        providers reject that, so history is repaired before it is sent.
        """
        repaired: List[ChatMessage] = []
        index = 0
        while index < len(messages):
            message = messages[index]
            if message.role == "tool":
                index += 1
                continue
            if message.role == "assistant" and message.tool_calls:
                wanted = sorted(call.id for call in message.tool_calls)
                following: List[ChatMessage] = []
                cursor = index + 1
                while cursor < len(messages) and messages[cursor].role == "tool":
                    following.append(messages[cursor])
                    cursor += 1
                if sorted(result.tool_call_id for result in following) == wanted:
                    repaired.append(message)
                    repaired.extend(following)
                elif message.content:
                    repaired.append(ChatMessage.assistant(message.content))
                index = cursor
                continue
            repaired.append(message)
            index += 1
        return repaired

    def _trim(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        if not messages:
            return messages
        system, tail = messages[0], messages[1:]
        limit = max(4, self.config.history_limit)
        if len(tail) <= limit:
            return messages
        tail = tail[-limit:]
        while tail and tail[0].role != "user":
            tail.pop(0)
        return [system, *tail]

    def _commit_assistant(
        self,
        text_parts: List[str],
        calls: List[ToolCall],
        usage: Optional[Usage],
        model: str,
    ) -> None:
        assistant = ChatMessage.assistant("".join(text_parts), calls)
        if assistant.content or assistant.tool_calls:
            self.session.add(assistant)
        self._account(usage, model)

    def _account(self, usage: Optional[Usage], model: str) -> None:
        profile = self.router.profile_for(model) or self.router.last_profile
        cost = estimate_cost(profile, usage)
        if usage is not None:
            self.turn_tokens += usage.total
        self.turn_cost += cost
        self.session.add_usage(usage, cost)
        self.last_model = model or (profile.id if profile else self.last_model)
        if profile is not None:
            self.session.model = profile.id

    def _tool_output(self, result: ToolResult) -> str:
        status = "ok" if result.ok else "error"
        if not result.output:
            return status
        return f"{status}\n{result.output}"

    def _pre_approve(self, calls: List[ToolCall]) -> Dict[str, ToolResult]:
        denied: Dict[str, ToolResult] = {}
        if self.config.approval_mode == "auto":
            return denied
        for call in calls:
            spec = self.registry.get(call.name)
            if spec is None or not spec.requires_approval:
                continue
            try:
                arguments = call.parsed()
                validate_arguments(spec, arguments)
            except (ValueError, OmhError):
                continue  # the registry will report the bad arguments
            summary = spec.summary(arguments)
            if self.config.approval_mode == "deny":
                denied[call.id] = ToolResult.error(
                    f"{call.name} blocked by approval_mode=deny", denied=True
                )
                continue
            if self.approver is None:
                denied[call.id] = ToolResult.error(
                    f"{call.name} needs approval; rerun with --yes to allow tools",
                    denied=True,
                )
                continue
            try:
                allowed = self.approver(spec, arguments, summary)
            except (EOFError, KeyboardInterrupt):
                allowed = False
            if not allowed:
                denied[call.id] = ToolResult.error(
                    f"{call.name} denied by user", denied=True
                )
        return denied

    def _execute(
        self, calls: List[ToolCall], denied: Dict[str, ToolResult]
    ) -> Dict[str, ToolResult]:
        results: Dict[str, ToolResult] = dict(denied)
        pending = [call for call in calls if call.id not in results]
        if not pending:
            return results
        workers = min(max(1, self.config.router.max_parallel_tools), len(pending))
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="omh-tool")
        try:
            futures = {
                pool.submit(self.registry.execute, call, None, approved=True): call
                for call in pending
            }
            for future in as_completed(futures):
                call = futures[future]
                try:
                    results[call.id] = future.result()
                except Exception as exc:  # pragma: no cover - registry catches
                    results[call.id] = ToolResult.error(
                        f"{call.name} crashed: {type(exc).__name__}: {exc}"
                    )
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        pool.shutdown(wait=True)
        return results


