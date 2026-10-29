"""NotebookEdit tool: read and edit Jupyter ``.ipynb`` notebooks at cell level.

Stdlib only (json). The notebook is treated as JSON: only the ``cells``
list is ever rewritten; every other key (nbformat, nbformat_minor,
metadata, ...) is preserved untouched.

Cell sources may be stored as ``str`` or ``list[str]`` in the file —
they are normalized on read, and written back as a list of lines.

Public API::

    read_notebook(path)                       -> str   (formatted cells)
    edit_cell(path, index, new_source)        -> str
    insert_cell(path, index, cell_type, source) -> str
    delete_cell(path, index)                  -> str
    add_cell(path, cell_type, source)         -> str   (append)

``register(agent)`` exposes the tools ``NotebookRead``, ``NotebookEdit``,
``NotebookInsert`` and ``NotebookDelete`` on ``agent.tools``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .tools import Tool, RISK_CONFIRM, RISK_SAFE

CELL_TYPES = {"code", "markdown", "raw"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load(path: str) -> dict:
    """Load and validate a notebook. Returns the raw JSON dict."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Notebook not found: {path}")
    if p.suffix.lower() != ".ipynb":
        raise ValueError(f"Not a .ipynb file: {path}")
    try:
        nb = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid notebook JSON: {e}") from e
    if not isinstance(nb, dict) or not isinstance(nb.get("cells"), list):
        raise ValueError("Not a valid nbformat notebook: missing 'cells' list")
    return nb


def _save(path: str, nb: dict) -> None:
    """Write the notebook back atomically (everything except cells untouched)."""
    for cell in nb["cells"]:
        if isinstance(cell.get("source"), str):
            cell["source"] = cell["source"].splitlines(keepends=True)
    p = Path(path)
    tmp = p.with_suffix(".ipynb.tmp")
    tmp.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    tmp.replace(p)


def _source_text(cell: dict) -> str:
    src = cell.get("source", [])
    if isinstance(src, str):
        return src
    return "".join(src)


def _source_lines(source: str | list[str]) -> list[str]:
    """Normalize a new source into a list of lines for storage."""
    if isinstance(source, str):
        return source.splitlines(keepends=True)
    if isinstance(source, (list, tuple)) and all(isinstance(s, str) for s in source):
        return list(source)
    raise TypeError("source must be str or list[str]")


def _check_index(nb: dict, index: int, allow_end: bool = False) -> None:
    if not isinstance(index, int) or isinstance(index, bool):
        raise TypeError(f"cell index must be int, got {type(index).__name__}")
    n = len(nb["cells"])
    hi = n if allow_end else n - 1
    if index < 0 or index > hi:
        raise IndexError(f"cell index {index} out of range (0..{hi}, "
                         f"{n} cells)")


def _new_cell(cell_type: str, source: str | list[str]) -> dict:
    if cell_type not in CELL_TYPES:
        raise ValueError(f"cell_type must be one of {sorted(CELL_TYPES)}, "
                         f"got {cell_type!r}")
    cell: dict[str, Any] = {
        "cell_type": cell_type,
        "metadata": {},
        "source": _source_lines(source),
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def read_notebook(path: str) -> str:
    """Return a formatted view of every cell in the notebook."""
    nb = _load(path)
    parts = []
    for i, cell in enumerate(nb["cells"]):
        ctype = cell.get("cell_type", "code")
        parts.append(f"Cell {i} [{ctype}]:\n{_source_text(cell)}")
    return "\n---\n".join(parts)


def edit_cell(path: str, index: int, new_source: str | list[str]) -> str:
    """Replace the source of cell ``index``. Metadata preserved."""
    nb = _load(path)
    _check_index(nb, index)
    cell = nb["cells"][index]
    cell["source"] = _source_lines(new_source)
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    _save(path, nb)
    return f"Edited cell {index}"


def insert_cell(path: str, index: int, cell_type: str,
                source: str | list[str]) -> str:
    """Insert a new cell before ``index`` (``index`` may equal len(cells))."""
    nb = _load(path)
    _check_index(nb, index, allow_end=True)
    nb["cells"].insert(index, _new_cell(cell_type, source))
    _save(path, nb)
    return f"Inserted {cell_type} cell at index {index}"


def delete_cell(path: str, index: int) -> str:
    """Delete cell ``index``."""
    nb = _load(path)
    _check_index(nb, index)
    nb["cells"].pop(index)
    _save(path, nb)
    return f"Deleted cell {index}"


def add_cell(path: str, cell_type: str, source: str | list[str]) -> str:
    """Append a new cell at the end of the notebook."""
    nb = _load(path)
    nb["cells"].append(_new_cell(cell_type, source))
    _save(path, nb)
    return f"Appended {cell_type} cell at index {len(nb['cells']) - 1}"


# ---------------------------------------------------------------------------
# tool registration
# ---------------------------------------------------------------------------

def _path_schema() -> dict:
    return {"type": "string", "description": "Path to the .ipynb notebook"}


def register(agent) -> None:
    """Register the notebook tools on ``agent.tools``."""
    agent.tools["NotebookRead"] = Tool(
        "NotebookRead",
        "Read a Jupyter .ipynb notebook, formatted cell by cell.",
        {"type": "object", "properties": {"path": _path_schema()},
         "required": ["path"]},
        lambda path: _safe(lambda: read_notebook(path)),
        risk=RISK_SAFE,
    )
    agent.tools["NotebookEdit"] = Tool(
        "NotebookEdit",
        "Replace the source of one notebook cell by index.",
        {"type": "object",
         "properties": {"path": _path_schema(),
                        "cell_index": {"type": "integer",
                                       "description": "Cell index (0-based)"},
                        "new_source": {"type": "string",
                                       "description": "New cell source"}},
         "required": ["path", "cell_index", "new_source"]},
        lambda path, cell_index, new_source: _safe(
            lambda: edit_cell(path, cell_index, new_source)),
        risk=RISK_CONFIRM,
    )
    agent.tools["NotebookInsert"] = Tool(
        "NotebookInsert",
        "Insert a new notebook cell before the given index "
        "(index may equal the cell count to append).",
        {"type": "object",
         "properties": {"path": _path_schema(),
                        "index": {"type": "integer",
                                  "description": "Insert position (0-based)"},
                        "cell_type": {"type": "string",
                                      "enum": ["code", "markdown", "raw"]},
                        "source": {"type": "string",
                                   "description": "New cell source"}},
         "required": ["path", "index", "cell_type", "source"]},
        lambda path, index, cell_type, source: _safe(
            lambda: insert_cell(path, index, cell_type, source)),
        risk=RISK_CONFIRM,
    )
    agent.tools["NotebookDelete"] = Tool(
        "NotebookDelete",
        "Delete one notebook cell by index.",
        {"type": "object",
         "properties": {"path": _path_schema(),
                        "cell_index": {"type": "integer",
                                       "description": "Cell index (0-based)"}},
         "required": ["path", "cell_index"]},
        lambda path, cell_index: _safe(lambda: delete_cell(path, cell_index)),
        risk=RISK_CONFIRM,
    )


def _safe(fn) -> str:
    try:
        return fn()
    except (OSError, ValueError, IndexError, TypeError) as e:
        return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    def check(name: str, cond: bool) -> None:
        print(("PASS" if cond else "FAIL"), "-", name)
        if not cond:
            raise SystemExit(f"self-test failed: {name}")

    with tempfile.TemporaryDirectory() as d:
        nb_path = str(Path(d) / "nb.ipynb")

        # build a tmp 3-cell notebook (mix str and list[str] sources)
        nb = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {
                "kernelspec": {"display_name": "Python 3",
                               "language": "python",
                               "name": "python3"},
                "custom": {"keep": "me"},
            },
            "cells": [
                {"cell_type": "code", "execution_count": 3,
                 "metadata": {"tags": ["init"]},
                 "outputs": [{"output_type": "stream", "text": ["hi\n"]}],
                 "source": ["x = 1\n", "print(x)\n"]},
                {"cell_type": "markdown", "metadata": {},
                 "source": "# Title\nSome text\n"},
                {"cell_type": "raw", "metadata": {}, "source": ["raw data"]},
            ],
        }
        Path(nb_path).write_text(json.dumps(nb), encoding="utf-8")

        # --- read ---
        out = read_notebook(nb_path)
        check("read formats 3 cells",
              "Cell 0 [code]:" in out and "Cell 1 [markdown]:" in out
              and "Cell 2 [raw]:" in out and "---" in out
              and "x = 1" in out and "# Title" in out)

        # --- edit ---
        msg = edit_cell(nb_path, 0, "x = 42\nprint(x * 2)\n")
        check("edit returns confirmation", msg == "Edited cell 0")
        check("edit changes source on read",
              "x = 42" in read_notebook(nb_path))

        # --- insert ---
        msg = insert_cell(nb_path, 1, "markdown", "## Inserted\n")
        check("insert confirmation",
              msg == "Inserted markdown cell at index 1")
        cells = json.loads(Path(nb_path).read_text())["cells"]
        check("insert position and type",
              len(cells) == 4 and cells[1]["cell_type"] == "markdown"
              and cells[1]["source"] == ["## Inserted\n"])

        # --- delete ---
        msg = delete_cell(nb_path, 1)
        check("delete confirmation", msg == "Deleted cell 1")
        cells = json.loads(Path(nb_path).read_text())["cells"]
        check("delete removes the cell", len(cells) == 3
              and cells[1]["cell_type"] == "markdown"
              and cells[1]["source"] == "# Title\nSome text\n".splitlines(
                  keepends=True))

        # --- append ---
        msg = add_cell(nb_path, "code", ["y = 2\n"])
        check("append confirmation",
              msg == "Appended code cell at index 3")
        cells = json.loads(Path(nb_path).read_text())["cells"]
        check("append grows the list",
              len(cells) == 4 and cells[3]["cell_type"] == "code"
              and cells[3]["outputs"] == []
              and cells[3]["execution_count"] is None)

        # --- JSON validity + metadata preservation ---
        raw = json.loads(Path(nb_path).read_text(encoding="utf-8"))
        check("JSON valid", isinstance(raw, dict) and "cells" in raw)
        check("top-level metadata preserved",
              raw["nbformat"] == 4 and raw["nbformat_minor"] == 5
              and raw["metadata"]["kernelspec"]["name"] == "python3"
              and raw["metadata"]["custom"]["keep"] == "me")
        check("cell metadata preserved",
              cells[0]["metadata"].get("tags") == ["init"])
        check("written sources are line lists",
              all(isinstance(c["source"], list) for c in raw["cells"]))

        # --- validation errors ---
        for label, fn in [
            ("missing file", lambda: read_notebook(str(Path(d) / "no.ipynb"))),
            ("wrong extension", lambda: read_notebook(str(Path(d) / "x.txt"))),
            ("index out of range", lambda: edit_cell(nb_path, 99, "z")),
            ("bad cell_type",
             lambda: add_cell(nb_path, "html", "x")),
        ]:
            try:
                fn()
                raised = False
            except (FileNotFoundError, ValueError, IndexError, TypeError):
                raised = True
            check(f"raises on {label}", raised)

        # --- register() wires 4 tools ---
        class FakeAgent:
            def __init__(self):
                self.tools = {}

        register(FakeAgent())  # type: ignore[arg-type]
        # exercised via a real agent object instead
        fa = FakeAgent()
        register(fa)
        check("register exposes 4 tools",
              {"NotebookRead", "NotebookEdit", "NotebookInsert",
               "NotebookDelete"} <= set(fa.tools))
        check("tool handlers work end-to-end",
              fa.tools["NotebookRead"].handler(path=nb_path).startswith(
                  "Cell 0 [code]:")
              and fa.tools["NotebookEdit"].handler(
                  path=nb_path, cell_index=0, new_source="final = True\n")
              == "Edited cell 0"
              and fa.tools["NotebookInsert"].handler(
                  path=nb_path, index=0, cell_type="raw", source="hdr")
              == "Inserted raw cell at index 0"
              and fa.tools["NotebookDelete"].handler(
                  path=nb_path, cell_index=0) == "Deleted cell 0"
              and fa.tools["NotebookRead"].handler(
                  path="/nope/missing.ipynb").startswith("ERROR:"))

    print("ALL PASS")
