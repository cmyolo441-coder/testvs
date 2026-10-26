"""ANSI theme, markdown streaming and table rendering."""

from __future__ import annotations

import unittest

from omh.theme import (
    MarkdownStream,
    Theme,
    colorize_diff,
    render_markdown,
    table,
    truncate,
    visible_width,
)


class ThemeTest(unittest.TestCase):
    def test_colour_can_be_disabled(self) -> None:
        theme = Theme("hermes", enabled=False)
        self.assertEqual(theme.accent("hi"), "hi")
        self.assertEqual(theme.bold("hi"), "hi")

    def test_truecolor_codes_are_used(self) -> None:
        theme = Theme("hermes", enabled=True)
        painted = theme.accent("hi")
        self.assertTrue(painted.startswith("\x1b[38;2;0;206;209m"))
        self.assertTrue(painted.endswith("\x1b[0m"))

    def test_palettes_are_available(self) -> None:
        for name in ("hermes", "amber", "mono"):
            theme = Theme(name, enabled=False)
            self.assertEqual(theme.palette.name, name)

    def test_unknown_palette_falls_back(self) -> None:
        self.assertEqual(Theme("nope", enabled=False).palette.name, "hermes")


class MarkdownTest(unittest.TestCase):
    def test_inline_markup_is_stripped_when_colour_is_off(self) -> None:
        theme = Theme(enabled=False)
        rendered = render_markdown("**bold** and `code` and [link](http://x)", theme)
        self.assertIn("bold", rendered)
        self.assertIn("code", rendered)
        self.assertIn("link", rendered)
        self.assertNotIn("**", rendered)
        self.assertNotIn("`", rendered)

    def test_streaming_fence_across_chunk_boundaries(self) -> None:
        theme = Theme(enabled=False)
        stream = MarkdownStream(theme, width=60)
        chunks = ["```py", "thon\n", "print('hi')\n", "``", "`\n", "done"]
        output = "".join(stream.feed(chunk) for chunk in chunks) + stream.flush()
        self.assertIn("python", output)
        self.assertIn("print('hi')", output)
        self.assertIn("done", output)
        self.assertNotIn("```", output)
        self.assertFalse(stream.in_code)

    def test_headings_bullets_and_quotes(self) -> None:
        theme = Theme(enabled=False)
        rendered = render_markdown(
            "# Title\n\n- one\n- two\n\n1. first\n\n> quoted\n", theme
        )
        self.assertIn("Title", rendered)
        self.assertIn("• one", rendered)
        self.assertIn("1. first", rendered)
        self.assertIn("│ quoted", rendered)
        self.assertNotIn("# Title", rendered)

    def test_unclosed_fence_is_closed_on_flush(self) -> None:
        theme = Theme(enabled=False)
        stream = MarkdownStream(theme)
        first = stream.feed("```sh\nls\n")
        tail = stream.flush()
        self.assertIn("┌─", first)
        self.assertIn("└", tail)

    def test_long_line_without_newline_is_flushed(self) -> None:
        stream = MarkdownStream(Theme(enabled=False))
        self.assertEqual(stream.feed("partial"), "")
        self.assertIn("partial", stream.flush())


class LayoutTest(unittest.TestCase):
    def test_visible_width_counts_wide_characters(self) -> None:
        self.assertEqual(visible_width("abc"), 3)
        self.assertEqual(visible_width("你好"), 4)
        self.assertEqual(visible_width("\x1b[31mred\x1b[0m"), 3)

    def test_truncate_respects_width(self) -> None:
        self.assertEqual(truncate("abcdef", 4), "abc…")
        self.assertEqual(truncate("abc", 4), "abc")

    def test_table_pads_columns(self) -> None:
        theme = Theme(enabled=False)
        rendered = table(theme, ["a", "bb"], [["1", "2"], ["long", "x"]], max_width=40)
        lines = rendered.splitlines()
        self.assertEqual(len(lines), 4)
        self.assertIn("long", lines[3])
        self.assertEqual(visible_width(lines[0]), visible_width(lines[3]))

    def test_diff_colouring(self) -> None:
        theme = Theme(enabled=False)
        self.assertIn("+added", colorize_diff("+added", theme))
        self.assertIn("-removed", colorize_diff("-removed", theme))


if __name__ == "__main__":
    unittest.main()
