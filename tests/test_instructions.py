"""AGENTS.md project instructions are loaded, composed and really used.

Drives the real loader (``fullagent.instructions``), the real composer
(``fullagent.mastermind.CoherenceComposer``) and the real
``Agent._context_sections`` wiring against a temporary project tree and
stub subsystems. No mocks of the code under test.
"""

import sys as _sys
from pathlib import Path as _Path
_ROOT_DIR = str(_Path(__file__).resolve().parent.parent)
if _ROOT_DIR not in _sys.path:
    _sys.path.insert(0, _ROOT_DIR)

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fullagent import instructions
from fullagent.agent import Agent
from fullagent.mastermind import CoherenceComposer


class InstructionLoadingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="fa-agentsmd-")
        self.root = Path(self._tmp.name)
        self.user = self.root / "home"
        self.user.mkdir()
        self.project = self.root / "project"
        self.nested = self.project / "src" / "pkg"
        self.nested.mkdir(parents=True)
        (self.user / "AGENTS.md").write_text("global: be kind\n",
                                             encoding="utf-8")
        (self.project / "AGENTS.md").write_text("project: run pytest\n",
                                                encoding="utf-8")
        (self.nested / "AGENTS.md").write_text("nearest: use tabs\n",
                                               encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def test_nearest_file_wins_and_every_source_is_included(self):
        text = instructions.collect(self.nested, user_dir=self.user)
        self.assertIn("global: be kind", text)
        self.assertIn("project: run pytest", text)
        self.assertIn("nearest: use tabs", text)
        self.assertLess(text.index("global: be kind"),
                        text.index("project: run pytest"))
        self.assertLess(text.index("project: run pytest"),
                        text.index("nearest: use tabs"))
        # the real path is the heading, so the model can cite the source
        self.assertIn(str((self.nested / "AGENTS.md").resolve()), text)

    def test_no_instruction_files_means_no_section(self):
        bare = self.root / "bare"
        bare.mkdir()
        self.assertEqual(
            instructions.collect(bare, user_dir=self.root / "empty"), "")

    def test_edit_is_seen_on_the_next_read(self):
        target = self.nested / "AGENTS.md"
        self.assertIn("use tabs", instructions.collect(self.nested,
                                                       user_dir=self.user))
        target.write_text("nearest: use two-space indent\n", encoding="utf-8")
        text = instructions.collect(self.nested, user_dir=self.user)
        self.assertIn("two-space indent", text)
        self.assertNotIn("use tabs", text)

    def test_cap_is_enforced(self):
        (self.nested / "AGENTS.md").write_text("x" * 5000, encoding="utf-8")
        text = instructions.collect(self.nested, user_dir=self.user,
                                    limit=900)
        self.assertLessEqual(len(text), 900)
        self.assertTrue(text.endswith("(truncated)"))

    def test_describe_lists_real_files_only(self):
        paths = instructions.describe(self.nested, user_dir=self.user)
        self.assertEqual(len(paths), 3)
        for path in paths:
            self.assertTrue(Path(path).is_file(), path)


class ComposerTests(unittest.TestCase):
    def test_project_and_output_style_sections_reach_the_model(self):
        composer = CoherenceComposer()
        doc = composer.compose("SEALED PROMPT", {
            "project": "repo rule: never touch main",
            "output_style": "concise",
            "goal": "ship it",
        })
        self.assertIn("repo rule: never touch main", doc)
        self.assertIn("PROJECT INSTRUCTIONS", doc)
        self.assertIn("OUTPUT STYLE", doc)
        self.assertLess(doc.index("PROJECT INSTRUCTIONS"),
                        doc.index("LIVE CONTEXT"))
        self.assertEqual(
            composer.manifest({"project": "p", "output_style": "o",
                               "goal": "g"}),
            ["project", "output_style", "goal"])


def _bare_agent():
    """An Agent with only the subsystems _context_sections touches."""
    agent = Agent.__new__(Agent)
    agent.log = SimpleNamespace(head=lambda: 1)
    agent.oracle = SimpleNamespace(read_constitution=lambda: "")
    agent.goal = SimpleNamespace(
        status=lambda: SimpleNamespace(active=False))
    agent.memory = SimpleNamespace(context_block=lambda: "")
    agent.semantic = SimpleNamespace(recall_block=lambda q, k=3: "")
    agent.brain = SimpleNamespace(context_block=lambda q, k=3: "")
    agent._compact_digests = []
    return agent


class AgentWiringTests(unittest.TestCase):
    def test_context_sections_carry_real_agents_md(self):
        with tempfile.TemporaryDirectory(prefix="fa-agentsmd-agent-") as d:
            project = Path(d)
            (project / "AGENTS.md").write_text(
                "always run the full suite before claiming done\n",
                encoding="utf-8")
            old = os.getcwd()
            os.chdir(project)
            try:
                sections = _bare_agent()._context_sections()
            finally:
                os.chdir(old)
        self.assertIn("project", sections)
        self.assertIn("always run the full suite", sections["project"])

    def test_no_agents_md_means_no_project_section(self):
        with tempfile.TemporaryDirectory(prefix="fa-agentsmd-empty-") as d:
            old = os.getcwd()
            os.chdir(d)
            try:
                sections = _bare_agent()._context_sections()
            finally:
                os.chdir(old)
        self.assertNotIn("project", sections)


if __name__ == "__main__":
    unittest.main()
