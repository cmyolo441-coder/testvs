"""Database tools: real SQLite inspection and querying via stdlib sqlite3.

Three tools:

* ``DbTables`` — list every table (and view) in a database with row counts.
* ``DbSchema`` — show the real ``CREATE TABLE`` statement plus per-column
  info (type, nullability, defaults, primary keys).
* ``DbQuery`` — run a SQL statement. ``SELECT`` (and other read-only
  statements) run in read-only URI mode (``mode=ro``) so they can never
  write. Any non-SELECT statement requires ``confirm=True``, otherwise it
  is refused outright.

``db_path`` must exist and be a file; anything else is an error. Results
are capped at 100 rows.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from urllib.parse import quote

from .tools import (
    RISK_CONFIRM,
    RISK_SAFE,
    Tool,
    _checked,
)

_READ_ONLY_STARTERS = ("SELECT", "WITH", "EXPLAIN", "VALUES", "PRAGMA")


def _validate_db(db_path: str) -> tuple[Path, str | None]:
    """Validate db_path: must resolve to an existing regular file."""
    p, err = _checked(db_path)
    if err:
        return p, err
    if not p.exists():
        return p, f"ERROR: database file not found: {p}"
    if not p.is_file():
        return p, f"ERROR: not a file: {p}"
    return p, None


def _connect_readonly(p: Path) -> sqlite3.Connection:
    """Open the database read-only via URI mode=ro.

    mode=ro refuses to create the file (missing file is a checked error
    above anyway) and refuses any write statement — a guard so SELECT
    tools can never accidentally modify data.
    """
    uri = "file:" + quote(p.as_posix(), safe="/") + "?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=5.0)


def _format_table(columns: list[str], rows: list[tuple]) -> str:
    """Render rows as a plain-text table with column headers."""
    strs = [[str(v) if v is not None else "NULL" for v in r]
            for r in rows]
    widths = [len(c) for c in columns]
    for row in strs:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))
    lines = []
    lines.append(" | ".join(c.ljust(widths[i])
                            for i, c in enumerate(columns)))
    lines.append("-+-".join("-" * w for w in widths))
    for row in strs:
        cells = list(row) + [""] * (len(columns) - len(row))
        lines.append(" | ".join(c.ljust(widths[i])
                                for i, c in enumerate(cells)))
    return "\n".join(lines)


def db_tables(db_path: str) -> str:
    """Return the real list of tables/views in the DB with row counts."""
    p, err = _validate_db(db_path)
    if err:
        return err
    try:
        conn = _connect_readonly(p)
    except sqlite3.Error as e:
        return f"ERROR: could not open database {p}: {e}"
    try:
        try:
            objs = conn.execute(
                "SELECT type, name FROM sqlite_master "
                "WHERE type IN ('table', 'view') "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        except sqlite3.DatabaseError as e:
            return f"ERROR: {p} is not a valid SQLite database: {e}"
        if not objs:
            return f"{p}: no tables or views found"
        rows = []
        for kind, name in objs:
            if kind == "table":
                try:
                    count = conn.execute(
                        f'SELECT COUNT(*) FROM "{name.replace(chr(34), chr(34)*2)}"'
                    ).fetchone()[0]
                except sqlite3.Error:
                    count = "?"
            else:
                count = "-"
            rows.append((name, kind, count))
        out = f"Database: {p}\n"
        out += _format_table(["name", "type", "row_count"],
                             [tuple(map(str, r)) for r in rows])
        return out
    finally:
        conn.close()


def db_schema(db_path: str, table: str) -> str:
    """Return the real CREATE TABLE statement plus column details."""
    p, err = _validate_db(db_path)
    if err:
        return err
    if not isinstance(table, str) or not table.strip():
        return "ERROR: table must be a non-empty string"
    name = table.strip()
    try:
        conn = _connect_readonly(p)
    except sqlite3.Error as e:
        return f"ERROR: could not open database {p}: {e}"
    try:
        try:
            row = conn.execute(
                "SELECT sql FROM sqlite_master "
                "WHERE type = 'table' AND name = ?", (name,)
            ).fetchone()
        except sqlite3.DatabaseError as e:
            return f"ERROR: {p} is not a valid SQLite database: {e}"
        if row is None or row[0] is None:
            exists = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE name = ? AND type IN ('table', 'view')", (name,)
            ).fetchone()
            if exists:
                return (f"ERROR: {name!r} is a view, not a table "
                        "(no CREATE TABLE statement)")
            return f"ERROR: table {name!r} not found in {p}"
        ddl = row[0]
        cols = conn.execute(f'PRAGMA table_info("{name.replace(chr(34), chr(34)*2)}")').fetchall()
        col_rows = []
        for _cid, col, ctype, notnull, dflt, pk in cols:
            col_rows.append((
                col,
                ctype or "",
                "NO" if notnull else "YES",
                "YES" if pk else "NO",
                "NULL" if dflt is None else str(dflt),
            ))
        out = f"Table: {name} (in {p})\n\n{ddl};\n\nColumns:\n"
        out += _format_table(["column", "type", "nullable", "pk",
                              "default"], col_rows)
        fks = conn.execute(f'PRAGMA foreign_key_list("{name.replace(chr(34), chr(34)*2)}")').fetchall()
        if fks:
            fk_rows = [(str(r[2]), str(r[3]), str(r[4])) for r in fks]
            out += "\n\nForeign keys:\n"
            out += _format_table(["ref_table", "from", "to"], fk_rows)
        return out
    finally:
        conn.close()


def db_query(db_path: str, sql: str, confirm: bool = False) -> str:
    """Run SQL. SELECT-only by default; writes need confirm=True.

    Read-only statements always execute in ``mode=ro``. Anything else is
    refused unless ``confirm=True`` is passed explicitly, in which case it
    runs in a normal connection and commits.
    """
    p, err = _validate_db(db_path)
    if err:
        return err
    if not isinstance(sql, str) or not sql.strip():
        return "ERROR: sql must be a non-empty string"
    text = sql.strip()
    if isinstance(confirm, str):
        confirm = confirm.strip().lower() in ("1", "true", "yes", "y")
    first = text.lstrip("(").split(None, 1)
    keyword = (first[0].upper() if first else "")
    read_only = keyword in _READ_ONLY_STARTERS
    if not read_only and not confirm:
        return ("ERROR: refusing to run non-SELECT statement "
                f"({keyword or 'unknown'}) without confirm=True — "
                "pass confirm=True to allow writes/DDL explicitly")
    try:
        if read_only:
            conn = _connect_readonly(p)
        else:
            conn = sqlite3.connect(os.fspath(p), timeout=10.0)
    except sqlite3.Error as e:
        return f"ERROR: could not open database {p}: {e}"
    try:
        try:
            cur = conn.execute(text)
        except sqlite3.Error as e:
            return f"ERROR: SQL failed: {e}"
        if cur.description is not None:
            columns = [d[0] for d in cur.description]
            rows = cur.fetchmany(100)
            out = f"Database: {p}\n\n{_format_table(columns, rows)}"
            extra = cur.fetchone()
            if extra is not None:
                out += "\n\n… 100 rows shown (limit reached)"
            return out
        affected = cur.rowcount
        if not read_only:
            conn.commit()
        return ("OK: statement executed"
                + (f", {affected} row(s) affected" if affected >= 0
                   else ""))
    finally:
        conn.close()


_DB_TABLES_DESC = (
    "List every table and view in a SQLite database with its row count. "
    "db_path must be an existing file. Read-only — never modifies the "
    "database."
)

_DB_SCHEMA_DESC = (
    "Show a SQLite table's real CREATE TABLE statement plus per-column "
    "info (type, nullability, primary key, default) and foreign keys. "
    "db_path must be an existing file. Read-only."
)

_DB_QUERY_DESC = (
    "Run a SQL statement against a SQLite database. SELECT/WITH/EXPLAIN/"
    "VALUES/PRAGMA queries run in read-only mode and return up to 100 "
    "rows as a readable table with column headers. Any non-SELECT "
    "statement (INSERT/UPDATE/DELETE/DDL/…) is REFUSED unless "
    "confirm=True is passed explicitly. db_path must be an existing file."
)

_DB_TABLES_PARAMS = {
    "type": "object",
    "properties": {
        "db_path": {
            "type": "string",
            "description": "Path to the SQLite database file.",
        },
    },
    "required": ["db_path"],
}

_DB_SCHEMA_PARAMS = {
    "type": "object",
    "properties": {
        "db_path": {
            "type": "string",
            "description": "Path to the SQLite database file.",
        },
        "table": {
            "type": "string",
            "description": "Name of the table to describe.",
        },
    },
    "required": ["db_path", "table"],
}

_DB_QUERY_PARAMS = {
    "type": "object",
    "properties": {
        "db_path": {
            "type": "string",
            "description": "Path to the SQLite database file.",
        },
        "sql": {
            "type": "string",
            "description": "The SQL statement to execute.",
        },
        "confirm": {
            "type": "boolean",
            "description": "Required True for non-SELECT statements "
                           "(INSERT/UPDATE/DELETE/DDL). Defaults False.",
        },
    },
    "required": ["db_path", "sql"],
}


def register(agent) -> None:
    """Register the DbTables / DbSchema / DbQuery tools on an agent."""
    agent.tools["DbTables"] = Tool("DbTables", _DB_TABLES_DESC,
                                   _DB_TABLES_PARAMS, db_tables,
                                   risk=RISK_SAFE)
    agent.tools["DbSchema"] = Tool("DbSchema", _DB_SCHEMA_DESC,
                                   _DB_SCHEMA_PARAMS, db_schema,
                                   risk=RISK_SAFE)
    agent.tools["DbQuery"] = Tool("DbQuery", _DB_QUERY_DESC,
                                  _DB_QUERY_PARAMS, db_query,
                                  risk=RISK_CONFIRM)


if __name__ == "__main__":
    import tempfile

    def check(name: str, cond: bool) -> None:
        print(("PASS" if cond else "FAIL"), "-", name)
        if not cond:
            raise SystemExit(f"self-test failed: {name}")

    with tempfile.TemporaryDirectory() as d:
        db = str(Path(d) / "shop.db")
        # Build a REAL database with real tables and rows.
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE users ("
                     "id INTEGER PRIMARY KEY, name TEXT NOT NULL, "
                     "email TEXT DEFAULT 'none')")
        conn.execute("CREATE TABLE orders ("
                     "id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL "
                     "REFERENCES users(id), total REAL)")
        conn.executemany(
            "INSERT INTO users (name, email) VALUES (?, ?)",
            [("ada", "ada@x.io"), ("bob", "bob@x.io"),
             ("cat", "cat@x.io")])
        conn.executemany(
            "INSERT INTO orders (user_id, total) VALUES (?, ?)",
            [(1, 9.99), (2, 42.5)])
        conn.commit()
        conn.close()

        # 1. DbTables lists real tables with real row counts
        out = db_tables(db)
        print(out)
        check("DbTables lists users", "users" in out and "table" in out)
        check("DbTables lists orders", "orders" in out)
        check("DbTables shows count 3 for users",
              any("users" in ln and "3" in ln for ln in out.splitlines()))
        check("DbTables shows count 2 for orders",
              any("orders" in ln and "2" in ln for ln in out.splitlines()))

        # 2. DbSchema returns the real DDL + column info
        out = db_schema(db, "users")
        print(out)
        check("DbSchema returns CREATE TABLE", "CREATE TABLE users" in out)
        check("DbSchema lists id column", "id" in out)
        check("DbSchema marks id as pk",
              any("id" in ln and "YES" in ln for ln in
                  out.split("Columns:")[1].splitlines()))
        check("DbSchema shows default 'none'",
              "none" in out)
        out = db_schema(db, "nope")
        check("DbSchema missing table errors", out.startswith("ERROR"))
        out = db_schema(db, "orders")
        check("DbSchema shows foreign key to users", "users" in out)

        # 3. DbQuery SELECT returns real rows with headers
        out = db_query(db, "SELECT name, email FROM users ORDER BY id")
        print(out)
        check("DbQuery headers", "name" in out and "email" in out)
        check("DbQuery returns ada row", "ada" in out)
        check("DbQuery returns bob row", "bob" in out)

        # 4. non-SELECT without confirm is refused, WITH confirm it works
        out = db_query(db, "DELETE FROM orders WHERE id = 1")
        check("DELETE refused without confirm",
              out.startswith("ERROR") and "confirm=True" in out)
        cnt = sqlite3.connect(db).execute(
            "SELECT COUNT(*) FROM orders").fetchone()[0]
        check("refused DELETE did not delete", cnt == 2)
        out = db_query(db, "DELETE FROM orders WHERE id = 1",
                       confirm=True)
        check("DELETE with confirm succeeds", out.startswith("OK"))
        cnt = sqlite3.connect(db).execute(
            "SELECT COUNT(*) FROM orders").fetchone()[0]
        check("confirmed DELETE really deleted", cnt == 1)

        # 5. missing file / bad inputs error cleanly
        out = db_tables(str(Path(d) / "missing.db"))
        check("missing file errors", out.startswith("ERROR"))
        out = db_query(str(Path(d) / "missing.db"), "SELECT 1")
        check("missing file errors on query", out.startswith("ERROR"))
        out = db_query(db, "")
        check("empty SQL errors", out.startswith("ERROR"))
        out = db_query(db, "SELEC 1")
        check("bad SQL errors cleanly", out.startswith("ERROR"))

        # 6. read-only mode: SELECT cannot write via a trick
        out = db_query(
            db, "SELECT 1; INSERT INTO users (name) VALUES ('hack')")
        # sqlite3.execute runs only one statement → second is rejected
        # by the driver; assert the row never appears either way.
        cnt = sqlite3.connect(db).execute(
            "SELECT COUNT(*) FROM users").fetchone()[0]
        check("SELECT-injection attempt left users at 3", cnt == 3)

    # 7. register() wires all three tools
    class FakeAgent:
        def __init__(self):
            self.tools = {}

    a = FakeAgent()
    register(a)
    for tname in ("DbTables", "DbSchema", "DbQuery"):
        check(f"register sets agent.tools['{tname}']",
              tname in a.tools)
    check("DbQuery risk is confirm",
          a.tools["DbQuery"].risk == "confirm")
    check("DbTables risk is safe",
          a.tools["DbTables"].risk == "safe")

    print("ALL SELF-TESTS PASSED")
