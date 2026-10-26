"""Tool implementations, validation and approvals."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from omh.providers.base import ToolCall
from omh.tools import default_registry
from omh.tools.base import ToolContext, truncate_output


def call(name: str, **arguments) -> ToolCall:
    return ToolCall("id", name, json.dumps(arguments))


class ToolTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.registry = default_registry(self.root, approval_mode="auto")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def run_tool(self, name: str, **arguments):
        return self.registry.execute(call(name, **arguments))


class FileToolsTest(ToolTestCase):
    def test_read_file_numbers_lines_and_slices(self) -> None:
        (self.root / "a.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
        result = self.run_tool("read_file", path="a.txt", start_line=2, end_line=3)
        self.assertTrue(result.ok)
        self.assertIn("two", result.output)
        self.assertIn("three", result.output)
        self.assertNotIn("one", result.output)
        self.assertIn("lines 2-3 of 3", result.output)

    def test_read_missing_directory_and_binary(self) -> None:
        self.assertFalse(self.run_tool("read_file", path="missing.txt").ok)
        (self.root / "dir").mkdir()
        self.assertFalse(self.run_tool("read_file", path="dir").ok)
        (self.root / "bin.dat").write_bytes(b"\x00\x01\x02")
        result = self.run_tool("read_file", path="bin.dat")
        self.assertFalse(result.ok)
        self.assertIn("binary", result.output)

    def test_read_rejects_escape(self) -> None:
        result = self.run_tool("read_file", path="../../etc/passwd")
        self.assertFalse(result.ok)
        self.assertIn("escapes the workspace", result.output)

    def test_write_creates_parents_and_content(self) -> None:
        result = self.run_tool("write_file", path="a/b/c.txt", content="hello\n")
        self.assertTrue(result.ok)
        self.assertEqual((self.root / "a/b/c.txt").read_text(), "hello\n")
        self.assertIn("created", result.output)

    def test_write_requires_approval(self) -> None:
        registry = default_registry(self.root, approval_mode="ask")
        denied = registry.execute(call("write_file", path="x.txt", content="x"))
        self.assertFalse(denied.ok)
        self.assertIn("not approved", denied.output)
        allowed = registry.execute(
            call("write_file", path="x.txt", content="x"),
            lambda spec, args, summary: True,
        )
        self.assertTrue(allowed.ok)
        self.assertTrue((self.root / "x.txt").exists())

        blocking = default_registry(self.root, approval_mode="deny")
        blocked = blocking.execute(call("write_file", path="y.txt", content="y"))
        self.assertFalse(blocked.ok)
        self.assertIn("approval_mode=deny", blocked.output)

    def test_edit_file_returns_diff(self) -> None:
        target = self.root / "code.py"
        target.write_text("value = 1\n", encoding="utf-8")
        result = self.run_tool(
            "edit_file", path="code.py", old_string="value = 1", new_string="value = 2"
        )
        self.assertTrue(result.ok)
        self.assertIn("+value = 2", result.output)
        self.assertIn("-value = 1", result.output)
        self.assertEqual(target.read_text(), "value = 2\n")

    def test_edit_requires_unique_match(self) -> None:
        (self.root / "dup.txt").write_text("x\nx\n", encoding="utf-8")
        ambiguous = self.run_tool(
            "edit_file", path="dup.txt", old_string="x", new_string="y"
        )
        self.assertFalse(ambiguous.ok)
        self.assertIn("appears 2 times", ambiguous.output)
        replaced = self.run_tool(
            "edit_file",
            path="dup.txt",
            old_string="x",
            new_string="y",
            replace_all=True,
        )
        self.assertTrue(replaced.ok)
        self.assertEqual((self.root / "dup.txt").read_text(), "y\ny\n")

    def test_edit_missing_string_and_file(self) -> None:
        (self.root / "a.txt").write_text("hello", encoding="utf-8")
        self.assertFalse(
            self.run_tool("edit_file", path="a.txt", old_string="nope", new_string="x").ok
        )
        self.assertFalse(
            self.run_tool("edit_file", path="ghost", old_string="a", new_string="b").ok
        )

    def test_list_dir_skips_hidden_and_sorts_dirs(self) -> None:
        (self.root / "sub").mkdir()
        (self.root / "file.txt").write_text("x", encoding="utf-8")
        (self.root / ".hidden").write_text("x", encoding="utf-8")
        result = self.run_tool("list_dir")
        lines = result.output.splitlines()
        self.assertIn("sub/", result.output)
        self.assertNotIn(".hidden", result.output)
        self.assertLess(result.output.index("sub/"), result.output.index("file.txt"))
        self.assertEqual(lines[0], "./  (2 entries)")


class SearchToolsTest(ToolTestCase):
    def setUp(self) -> None:
        super().setUp()
        (self.root / "a.py").write_text("Hello world\nprint('x')\n", encoding="utf-8")
        (self.root / "b.txt").write_text("hello again\n", encoding="utf-8")
        (self.root / ".git").mkdir()
        (self.root / ".git" / "hidden.py").write_text("hello\n", encoding="utf-8")

    def test_grep_regex_case_and_glob(self) -> None:
        result = self.run_tool("grep", pattern="hello", ignore_case=True)
        self.assertTrue(result.ok)
        self.assertIn("a.py:1", result.output)
        self.assertIn("b.txt:1", result.output)
        self.assertNotIn(".git", result.output)

        filtered = self.run_tool("grep", pattern="hello", ignore_case=True, glob="*.py")
        self.assertIn("a.py", filtered.output)
        self.assertNotIn("b.txt", filtered.output)

    def test_grep_no_match_and_bad_regex(self) -> None:
        missing = self.run_tool("grep", pattern="zzz")
        self.assertTrue(missing.ok)
        self.assertIn("no matches", missing.output)
        bad = self.run_tool("grep", pattern="[")
        self.assertFalse(bad.ok)
        self.assertIn("invalid regex", bad.output)

    def test_glob_finds_and_skips_git(self) -> None:
        result = self.run_tool("glob", pattern="**/*.py")
        self.assertTrue(result.ok)
        self.assertIn("a.py", result.output)
        self.assertNotIn(".git", result.output)

    def test_glob_rejects_escape(self) -> None:
        self.assertFalse(self.run_tool("glob", pattern="../*").ok)


class ShellTest(ToolTestCase):
    def test_shell_captures_output_and_exit(self) -> None:
        result = self.run_tool("run_shell", command="printf 'hi'")
        self.assertTrue(result.ok)
        self.assertIn("hi", result.output)
        self.assertIn("exit 0", result.output)

        failed = self.run_tool("run_shell", command="echo bad >&2; exit 3")
        self.assertFalse(failed.ok)
        self.assertIn("bad", failed.output)
        self.assertIn("exit 3", failed.output)

    def test_shell_timeout(self) -> None:
        result = self.run_tool(
            "run_shell", command="python3 -c 'import time; time.sleep(5)'", timeout=1
        )
        self.assertFalse(result.ok)
        self.assertIn("timed out", result.output)

    def test_shell_requires_approval(self) -> None:
        registry = default_registry(self.root, approval_mode="ask")
        result = registry.execute(call("run_shell", command="echo hi"))
        self.assertFalse(result.ok)
        self.assertIn("not approved", result.output)


class HtmlHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = (
            b"<html><head><style>.x{}</style><script>bad()</script></head>"
            b"<body><h1>Title</h1><p>Hello <b>world</b></p></body></html>"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        return


class WebTest(ToolTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), HtmlHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def test_fetch_strips_html_and_scripts(self) -> None:
        result = self.run_tool("web_fetch", url=f"http://127.0.0.1:{self.port}/")
        self.assertTrue(result.ok)
        self.assertIn("Title", result.output)
        self.assertIn("Hello world", result.output)
        self.assertNotIn("bad()", result.output)
        self.assertIn("HTTP 200", result.output)

    def test_fetch_limit_and_scheme(self) -> None:
        limited = self.run_tool(
            "web_fetch", url=f"http://127.0.0.1:{self.port}/", max_chars=10
        )
        self.assertIn("truncated at 10", limited.output)
        self.assertFalse(self.run_tool("web_fetch", url="file:///etc/passwd").ok)


class RegistryTest(ToolTestCase):
    def test_unknown_tool(self) -> None:
        result = self.registry.execute(call("teleport"))
        self.assertFalse(result.ok)
        self.assertIn("unknown tool", result.output)

    def test_missing_and_unknown_arguments(self) -> None:
        missing = self.registry.execute(call("read_file"))
        self.assertFalse(missing.ok)
        self.assertIn("missing required argument", missing.output)
        extra = self.registry.execute(call("list_dir", wat=1))
        self.assertFalse(extra.ok)
        self.assertIn("unknown argument", extra.output)

    def test_wrong_argument_type(self) -> None:
        result = self.registry.execute(call("read_file", path=12))
        self.assertFalse(result.ok)
        self.assertIn("must be string", result.output)

    def test_bad_json_arguments(self) -> None:
        result = self.registry.execute(ToolCall("id", "read_file", "{oops"))
        self.assertFalse(result.ok)
        self.assertIn("invalid JSON", result.output)

    def test_output_truncation(self) -> None:
        (self.root / "big.txt").write_text("x" * 2000, encoding="utf-8")
        registry = default_registry(self.root, max_output=600)
        result = registry.execute(call("read_file", path="big.txt"))
        self.assertIn("characters truncated", result.output)
        self.assertLess(len(result.output), 1000)
        self.assertEqual(truncate_output("short", 600), "short")

    def test_workspace_resolution(self) -> None:
        context = ToolContext(workspace=self.root)
        self.assertEqual(context.resolve("a/../b"), (self.root / "b").resolve())
        with self.assertRaises(Exception):
            context.resolve("/etc/passwd")


if __name__ == "__main__":
    unittest.main()


