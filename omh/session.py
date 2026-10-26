"""Session persistence: one small JSON file per conversation."""

from __future__ import annotations

import json
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import Config
from .errors import OmhError
from .providers.base import ChatMessage, Usage


@dataclass
class Session:
    id: str
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    messages: List[ChatMessage] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    cost: float = 0.0
    model: str = ""
    skill: str = ""
    title: str = ""

    def add(self, message: ChatMessage) -> None:
        self.messages.append(message)
        if message.role == "user" and not self.title and message.content.strip():
            self.title = " ".join(message.content.strip().split())[:80]
        self.updated = time.time()

    def clear(self) -> None:
        self.messages.clear()
        self.usage = Usage()
        self.cost = 0.0
        self.title = ""
        self.updated = time.time()

    def add_usage(self, usage: Optional[Usage], cost: float = 0.0) -> None:
        self.usage.add(usage)
        self.cost += max(0.0, cost)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "created": self.created,
            "updated": self.updated,
            "model": self.model,
            "skill": self.skill,
            "title": self.title,
            "cost": self.cost,
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
            },
            "messages": [message.to_dict() for message in self.messages],
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Session":
        usage_raw = raw.get("usage") or {}
        return cls(
            id=str(raw.get("id") or "unknown"),
            created=float(raw.get("created") or time.time()),
            updated=float(raw.get("updated") or time.time()),
            messages=[ChatMessage.from_dict(item) for item in raw.get("messages") or []],
            usage=Usage(
                prompt_tokens=int(usage_raw.get("prompt_tokens", 0) or 0),
                completion_tokens=int(usage_raw.get("completion_tokens", 0) or 0),
            ),
            cost=float(raw.get("cost", 0.0) or 0.0),
            model=str(raw.get("model") or ""),
            skill=str(raw.get("skill") or ""),
            title=str(raw.get("title") or ""),
        )


class SessionStore:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).expanduser()

    def ensure(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)

    def new(self, model: str = "", skill: str = "") -> Session:
        session_id = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
        return Session(id=session_id, model=model, skill=skill)

    def path_for(self, session_id: str) -> Path:
        safe = "".join(char for char in session_id if char.isalnum() or char in "-_")
        if not safe:
            raise OmhError(f"invalid session id: {session_id!r}")
        return self.directory / f"{safe}.json"

    def save(self, session: Session) -> Path:
        self.ensure()
        target = self.path_for(session.id)
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(session.to_dict(), ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(target)
        return target

    def load(self, session_id: str) -> Session:
        target = self.path_for(session_id)
        if not target.exists():
            raise OmhError(f"session not found: {session_id}")
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise OmhError(f"cannot read session {session_id}: {exc}") from exc
        return Session.from_dict(raw)

    def summaries(self, limit: int = 20) -> List[Dict[str, Any]]:
        if not self.directory.is_dir():
            return []
        rows: List[Dict[str, Any]] = []
        for path in sorted(self.directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            rows.append(
                {
                    "id": str(raw.get("id") or path.stem),
                    "title": str(raw.get("title") or "(empty)"),
                    "model": str(raw.get("model") or ""),
                    "updated": float(raw.get("updated") or 0.0),
                    "messages": len(raw.get("messages") or []),
                }
            )
            if len(rows) >= limit:
                break
        return rows

    def latest(self) -> Optional[Session]:
        summaries = self.summaries(1)
        if not summaries:
            return None
        return self.load(summaries[0]["id"])

    def delete(self, session_id: str) -> bool:
        target = self.path_for(session_id)
        if target.exists():
            target.unlink()
            return True
        return False


def store_for(config: Config) -> SessionStore:
    return SessionStore(config.session_dir)
