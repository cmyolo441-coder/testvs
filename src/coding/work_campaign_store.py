"""Bounded, locked read-modify-write campaign records. No scheduler."""
from contextlib import contextmanager
import json
from pathlib import Path
import re

from ..system.local_store import atomic_write_text, file_lock

MAX_CAMPAIGNS = 64
MAX_BYTES = 65536


class CampaignStore:
    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def path(self, campaign_id):
        if not re.fullmatch(r"campaign-[0-9a-f]{24}", campaign_id):
            raise ValueError("invalid_campaign_id")
        path = self.directory / (campaign_id + ".json")
        if self.directory.is_symlink() or path.is_symlink():
            raise ValueError("campaign_store_symlink")
        return path

    def read(self, campaign_id):
        with self.path(campaign_id).open("rb") as stream:
            raw = stream.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError("campaign_record_cap")
        record = json.loads(raw)
        if not isinstance(record, dict) or record.get("campaign_id") != campaign_id:
            raise ValueError("campaign_record_identity_mismatch")
        return record

    def save(self, record):
        content = json.dumps(record, indent=2, sort_keys=True) + "\n"
        if len(content.encode("utf-8")) > MAX_BYTES:
            raise ValueError("campaign_record_cap")
        atomic_write_text(self.path(record["campaign_id"]), content, private=True)

    def create(self, record):
        with file_lock(self.directory / "index", private=True) as lock:
            if not lock["enforced"]:
                raise ValueError("campaign_lock_unavailable")
            path = self.path(record["campaign_id"])
            if path.exists():
                return self.read(record["campaign_id"])
            if sum(1 for _ in self.directory.glob("campaign-*.json")) >= MAX_CAMPAIGNS:
                raise ValueError("campaign_store_cap")
            self.save(record)
        return record

    @contextmanager
    def locked(self, campaign_id):
        path = self.path(campaign_id)
        if not path.exists():
            raise ValueError("unknown_campaign")
        with file_lock(path, private=True) as lock:
            if not lock["enforced"]:
                raise ValueError("campaign_lock_unavailable")
            yield self.read(campaign_id)
