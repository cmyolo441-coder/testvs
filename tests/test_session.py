"""Session persistence."""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from omh.errors import OmhError
from omh.providers.base import ChatMessage, ToolCall, Usage
from omh.session import Session, SessionStore


class SessionTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = SessionStore(Path(self._tmp.name) / "sessions")
        self.session = self.store.new(model="model-a")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_new_session_ids_are_unique(self) -> None:
        other = self.store.new()
        self.assertNotEqual(self.session.id, other.id)
        self.assertIn("-", self.session.id)

    def test_messages_title_and_usage(self) -> None:
        self.session.add(ChatMessage.user("  hello   world  "))
        self.session.add(ChatMessage.assistant("hi"))
        self.assertEqual(self.session.title, "hello world")
        self.session.add_usage(Usage(10, 5), cost=0.001)
        self.session.add_usage(None)
        self.assertEqual(self.session.usage.total, 15)
        self.assertAlmostEqual(self.session.cost, 0.001)

    def test_save_load_roundtrip(self) -> None:
        self.session.skill = "review"
        self.session.model = "model-a"
        self.session.add(ChatMessage.user("question"))
        call = ToolCall("c1", "read_file", '{"path": "x"}')
        self.session.add(ChatMessage.assistant("looking", [call]))
        self.session.add(ChatMessage.tool("c1", "ok", "read_file"))
        self.session.add_usage(Usage(100, 20), cost=0.002)
        path = self.store.save(self.session)
        self.assertTrue(path.exists())

        loaded = self.store.load(self.session.id)
        self.assertEqual(loaded.title, "question")
        self.assertEqual(loaded.skill, "review")
        self.assertEqual(len(loaded.messages), 3)
        self.assertEqual(loaded.messages[1].tool_calls[0].parsed(), {"path": "x"})
        self.assertEqual(loaded.usage.total, 120)
        self.assertAlmostEqual(loaded.cost, 0.002)

    def test_summaries_and_latest(self) -> None:
        first = self.store.new()
        first.add(ChatMessage.user("first"))
        self.store.save(first)
        time.sleep(0.01)
        second = self.store.new()
        second.add(ChatMessage.user("second"))
        self.store.save(second)

        rows = self.store.summaries()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["id"], second.id)
        self.assertEqual(self.store.latest().id, second.id)

    def test_delete(self) -> None:
        self.store.save(self.session)
        self.assertTrue(self.store.delete(self.session.id))
        self.assertFalse(self.store.delete(self.session.id))
        self.assertIsNone(self.store.latest())

    def test_clear_keeps_id(self) -> None:
        self.session.add(ChatMessage.user("x"))
        self.session.add_usage(Usage(1, 1), cost=1.0)
        session_id = self.session.id
        self.session.clear()
        self.assertEqual(self.session.id, session_id)
        self.assertEqual(self.session.messages, [])
        self.assertEqual(self.session.usage.total, 0)
        self.assertEqual(self.session.cost, 0.0)
        self.assertEqual(self.session.title, "")

    def test_invalid_ids_are_rejected(self) -> None:
        with self.assertRaises(OmhError):
            self.store.path_for("   ")
        for bad in ("../../etc/passwd", "a/b"):
            with self.assertRaises(OmhError):
                self.store.load(bad)

    def test_missing_session_raises(self) -> None:
        with self.assertRaises(OmhError):
            self.store.load("nope-123")

    def test_from_dict_defaults(self) -> None:
        session = Session.from_dict({"id": "abc"})
        self.assertEqual(session.id, "abc")
        self.assertEqual(session.messages, [])
        self.assertEqual(session.usage.total, 0)


if __name__ == "__main__":
    unittest.main()
