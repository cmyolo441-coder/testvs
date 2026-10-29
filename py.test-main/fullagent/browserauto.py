"""Browser automation: real Playwright-driven browser tools, gracefully degraded.

Main entry point::

    register(agent) -> None

Registers four tools on the agent:

- ``BrowserOpen(url)``      — load a URL in a real headless Chromium page;
                              returns the page title and final URL.
- ``BrowserSnapshot()``     — real accessibility-tree snapshot of the
                              current page (truncated), so the model can see
                              what is on screen.
- ``BrowserClick(selector)`` — click a CSS selector on the current page.
- ``BrowserClose()``        — shut the browser down.

One persistent browser context is kept per agent: it is created lazily on
the first use and torn down by ``BrowserClose``. Every Playwright operation
uses a 30-second timeout.

Graceful degradation: the module only ever imports ``playwright`` inside
function calls. If the package is not installed, every tool returns the
clear message ``"playwright not installed (pip install playwright)"`` —
nothing is faked or simulated. Never imports ``.agent``.
"""

from __future__ import annotations

import os
import shutil
import threading

__all__ = ["register", "browser_available", "NOT_INSTALLED_MSG",
           "DEFAULT_TIMEOUT_MS", "_BrowserSession"]

DEFAULT_TIMEOUT_MS = 30_000  # every Playwright operation: 30s
SNAPSHOT_CAP = 8_000         # max chars of the accessibility snapshot

NOT_INSTALLED_MSG = ("playwright not installed (pip install playwright) "
                     "— browser tools are unavailable until it is installed.")
BROWSER_MISSING_MSG = (
    "Chromium browser not found. Run 'python -m playwright install chromium' "
    "or install Chromium and put it on PATH.")


# ---------------------------------------------------------------------------
# Optional playwright import (never raises)
# ---------------------------------------------------------------------------

def _import_playwright():
    """Return playwright's ``sync_api`` module, or None if unavailable."""
    try:
        from playwright import sync_api
        return sync_api
    except Exception:
        return None


def _resolve_chromium_executable(chromium=None) -> str | None:
    """Prefer an explicit, system, or Playwright-managed Chromium binary."""
    configured = os.environ.get("FULLAGENT_CHROMIUM_EXECUTABLE", "").strip()
    if configured:
        if os.path.isfile(configured) and os.access(configured, os.X_OK):
            return configured
        return None
    if chromium is not None:
        try:
            bundled = chromium.executable_path
            if bundled and os.path.isfile(bundled):
                return bundled
        except Exception:
            pass
    for name in ("chromium", "chromium-browser", "google-chrome",
                 "google-chrome-stable", "chrome"):
        found = shutil.which(name)
        if found:
            return found
    return None


def browser_available() -> bool:
    """True when Playwright and a usable Chromium executable are available."""
    sync_api = _import_playwright()
    if sync_api is None:
        return False
    pw = None
    try:
        pw = sync_api.sync_playwright().start()
        return _resolve_chromium_executable(pw.chromium) is not None
    except Exception:
        return False
    finally:
        if pw is not None:
            try:
                pw.stop()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Persistent browser session (one per agent, lazy init)
# ---------------------------------------------------------------------------

class _BrowserSession:
    """Lazy headless-Chromium session: playwright -> browser -> page.

    Thread-safe enough for the agent's serialized tool calls. Never raises
    from public methods; failures come back as ``"ERROR: ..."`` strings.
    """

    def __init__(self, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> None:
        self._timeout = timeout_ms
        self._lock = threading.RLock()
        self._pw = None       # playwright instance
        self._browser = None  # Browser
        self._page = None     # Page

    # -- lifecycle ------------------------------------------------------
    def _launch(self) -> str:
        """Start playwright/browser/page. Returns '' on success, else error."""
        sync_api = _import_playwright()
        if sync_api is None:
            return NOT_INSTALLED_MSG
        pw = browser = None
        try:
            pw = sync_api.sync_playwright().start()
            executable = _resolve_chromium_executable(pw.chromium)
            if executable is None:
                pw.stop()
                return BROWSER_MISSING_MSG
            browser = pw.chromium.launch(
                headless=True,
                executable_path=executable,
                args=["--no-sandbox", "--disable-dev-shm-usage"])
            page = browser.new_page()
            page.set_default_timeout(self._timeout)
        except Exception as e:  # noqa: BLE001 — real launch failure
            if browser is not None:
                try:
                    browser.close()
                except Exception:
                    pass
            if pw is not None:
                try:
                    pw.stop()
                except Exception:
                    pass
            return (f"ERROR: could not launch chromium: "
                    f"{type(e).__name__}: {e}")
        self._pw, self._browser, self._page = pw, browser, page
        return ""

    def _ensure(self):
        """Return (error, page); launches lazily. Never raises."""
        with self._lock:
            if self._page is not None:
                return "", self._page
            err = self._launch()
            if err:
                return err, None
            return "", self._page

    def close(self) -> str:
        """Shut down browser + playwright. Safe to call when not open."""
        with self._lock:
            if self._page is None and self._browser is None:
                if _import_playwright() is None:
                    return NOT_INSTALLED_MSG
                return "browser already closed"
            errs: list[str] = []
            if self._browser is not None:
                try:
                    self._browser.close()
                except Exception as e:  # noqa: BLE001
                    errs.append(f"{type(e).__name__}: {e}")
            if self._pw is not None:
                try:
                    self._pw.stop()
                except Exception as e:  # noqa: BLE001
                    errs.append(f"{type(e).__name__}: {e}")
            self._page = self._browser = self._pw = None
            if errs:
                return "ERROR: " + "; ".join(errs)
            return "browser closed"

    # -- operations ------------------------------------------------------
    def open_url(self, url: str) -> str:
        """Real page load. Returns title + final url."""
        if not isinstance(url, str) or not url.strip():
            return "ERROR: url must be a non-empty string"
        url = url.strip()
        if not url.lower().startswith(("http://", "https://")):
            return "ERROR: url must start with http:// or https://"
        err, page = self._ensure()
        if err:
            return err
        try:
            page.goto(url, timeout=self._timeout,
                      wait_until="domcontentloaded")
            return f"title: {page.title()}\nurl: {page.url}"
        except Exception as e:  # noqa: BLE001 — real nav failure
            return f"ERROR: {type(e).__name__}: {e}"

    def snapshot(self) -> str:
        """Real accessibility-tree snapshot of the current page."""
        err, page = self._ensure()
        if err:
            return err
        tree = None
        try:
            accessibility = getattr(page, "accessibility", None)
            if accessibility is not None:
                tree = accessibility.snapshot()
        except Exception:
            # Newer Playwright releases removed page.accessibility; fall
            # back to Locator.aria_snapshot() or visible page text below.
            tree = None
        if tree is None:
            try:
                tree = page.locator("body").aria_snapshot()
            except Exception:
                tree = None
        if isinstance(tree, str):
            out = tree.strip()
            if not out:
                try:
                    out = page.locator("body").inner_text().strip()
                except Exception:
                    out = ""
            if len(out) > SNAPSHOT_CAP:
                kept = len(out)
                out = out[:SNAPSHOT_CAP].rsplit("\n", 1)[0]
                out += f"\n… [{kept - len(out)} chars truncated]"
            return out or "(empty page)"
        if not tree:
            # Empty AX tree (e.g. canvas-only page): fall back to raw text.
            try:
                text = (page.locator("body").inner_text()
                        if page.locator("body").count() else "")
            except Exception:
                text = ""
            text = " ".join(text.split())
            return text[:SNAPSHOT_CAP] or "(empty page)"
        lines: list[str] = []
        stack: list[tuple[dict, int]] = [(tree, 0)]
        try:
            while stack:
                node, depth = stack.pop()
                if not isinstance(node, dict):
                    continue
                role = str(node.get("role", "?"))
                name = str(node.get("name", "") or "").strip()
                value = str(node.get("value", "") or "").strip()
                detail = " ".join(p for p in (name, value) if p)
                line = f"{'  ' * depth}[{role}]" + (f" {detail}" if detail else "")
                lines.append(line)
                children = node.get("children") or []
                for child in reversed(children):
                    if isinstance(child, dict):
                        stack.append((child, depth + 1))
        except Exception:
            pass
        out = "\n".join(lines).strip()
        if len(out) > SNAPSHOT_CAP:
            kept = len(out)
            out = out[:SNAPSHOT_CAP].rsplit("\n", 1)[0]
            out += f"\n… [{kept - len(out)} chars truncated]"
        return out or "(empty accessibility tree)"

    def click(self, selector: str) -> str:
        """Real click on a CSS selector on the current page."""
        if not isinstance(selector, str) or not selector.strip():
            return "ERROR: selector must be a non-empty string"
        err, page = self._ensure()
        if err:
            return err
        try:
            page.click(selector.strip(), timeout=self._timeout)
            return f"clicked: {selector.strip()}"
        except Exception as e:  # noqa: BLE001 — real click failure
            return f"ERROR: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Agent integration
# ---------------------------------------------------------------------------

def register(agent) -> None:
    """Register BrowserOpen/BrowserSnapshot/BrowserClick/BrowserClose tools.

    The browser session is stored on ``agent.browserauto_session`` so one
    persistent context lives per agent; it is created lazily and torn down
    by ``BrowserClose``. Click needs user approval (RISK_CONFIRM); the rest
    are read-only-ish reads (RISK_SAFE). Duck-typed: never imports .agent.
    """
    from .tools import Tool, RISK_SAFE, RISK_CONFIRM  # local: no cycles

    session = getattr(agent, "browserauto_session", None)
    if not isinstance(session, _BrowserSession):
        session = _BrowserSession()
        agent.browserauto_session = session

    agent.tools["BrowserOpen"] = Tool(
        "BrowserOpen",
        "Open a URL in a real headless Chromium browser and return the "
        "page title and final URL. Use this for pages that need JavaScript.",
        {"type": "object",
         "properties": {"url": {"type": "string",
                               "description": "http(s) URL to load"}},
         "required": ["url"]},
        session.open_url, risk=RISK_SAFE)

    agent.tools["BrowserSnapshot"] = Tool(
        "BrowserSnapshot",
        "Snapshot the current browser page as an accessibility tree "
        "(roles + visible names/values) so you can see what is on screen.",
        {"type": "object", "properties": {}},
        lambda: session.snapshot(), risk=RISK_SAFE)

    agent.tools["BrowserClick"] = Tool(
        "BrowserClick",
        "Click an element on the current browser page by CSS selector "
        "(e.g. 'button.submit', '#login', 'a[href=\"/docs\"]').",
        {"type": "object",
         "properties": {"selector": {"type": "string",
                                    "description": "CSS selector to click"}},
         "required": ["selector"]},
        session.click, risk=RISK_CONFIRM)

    agent.tools["BrowserClose"] = Tool(
        "BrowserClose",
        "Close the headless Chromium browser and free its resources.",
        {"type": "object", "properties": {}},
        session.close, risk=RISK_SAFE)


# ---------------------------------------------------------------------------
# Self-test: python3 -m fullagent.browserauto  →  PASS
#
# With Playwright and Chromium: opens a local data URL, snapshots and clicks.
# Without either dependency: tools return an actionable message, not a crash.
# ---------------------------------------------------------------------------

def _selftest() -> None:
    fails = 0

    def check(name: str, cond: bool) -> None:
        nonlocal fails
        print(("PASS" if cond else "FAIL"), "-", name)
        if not cond:
            fails += 1

    have_browser = browser_available()
    print("browser available:", have_browser)

    if not have_browser:
        # --- graceful degradation when Playwright or Chromium is absent ---
        expected = (NOT_INSTALLED_MSG if _import_playwright() is None
                    else BROWSER_MISSING_MSG)
        session = _BrowserSession()
        for name, call in (
            ("BrowserOpen", lambda: session.open_url("https://example.com")),
            ("BrowserSnapshot", session.snapshot),
            ("BrowserClick", lambda: session.click("button")),
        ):
            try:
                out = call()
            except Exception as e:  # noqa: BLE001
                check(f"{name} does not raise when playwright missing",
                      False)
                print("   raised:", e)
                continue
            check(f"{name} returns actionable unavailable message",
                  out == expected)
        close_expected = (NOT_INSTALLED_MSG
                          if _import_playwright() is None
                          else "browser already closed")
        check("BrowserClose is safe when unavailable",
              session.close() == close_expected)

        # register() works on a duck-typed agent and wires 4 tools
        class _FakeAgent:
            def __init__(self):
                self.tools = {}

        agent = _FakeAgent()
        try:
            register(agent)
            check("register() does not raise without playwright", True)
        except Exception as e:  # noqa: BLE001
            check("register() does not raise without playwright", False)
            print("   raised:", e)
        check("4 tools registered",
              {"BrowserOpen", "BrowserSnapshot", "BrowserClick",
               "BrowserClose"} <= set(agent.tools))
        tool_outs = []
        for t in ("BrowserOpen", "BrowserSnapshot", "BrowserClick"):
            h = agent.tools[t].handler
            try:
                if t == "BrowserSnapshot":
                    tool_outs.append(h())
                elif t == "BrowserOpen":
                    tool_outs.append(h("https://example.com"))
                else:
                    tool_outs.append(h("button"))
            except Exception as e:  # noqa: BLE001
                tool_outs.append(f"RAISED {e}")
        check("tool handlers return actionable unavailable message",
              all(o == expected for o in tool_outs))
        try:
            close_out = agent.tools["BrowserClose"].handler()
        except Exception as e:  # noqa: BLE001
            close_out = f"RAISED {e}"
        check("BrowserClose is safe when unavailable",
              close_out == close_expected)
    else:
        # --- local data URL: deterministic, no internet required -----------
        class _FakeAgent:
            def __init__(self):
                self.tools = {}

        agent = _FakeAgent()
        register(agent)
        check("4 tools registered",
              {"BrowserOpen", "BrowserSnapshot", "BrowserClick",
               "BrowserClose"} <= set(agent.tools))

        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        _page = ("<html><head><title>Example</title></head><body>"
                 "<main><h1>Example page</h1>"
                 "<a href='#more'>More information</a>"
                 "<p id='more'>Browser automation works.</p></main></body></html>")

        class _PageHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                body = _page.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), _PageHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        page_url = f"http://127.0.0.1:{server.server_port}/"
        out = agent.tools["BrowserOpen"].handler(page_url)
        check("BrowserOpen returns output (no exception)", bool(out))
        print("   open:", out.replace("\n", " | ")[:120])
        check("title contains 'Example'", "Example" in out)

        snap = agent.tools["BrowserSnapshot"].handler()
        check("snapshot returns text", bool(snap and len(snap) > 20))
        print("   snapshot head:", snap.splitlines()[0][:100]
              if snap else "")
        check("snapshot contains page text",
              "Example" in snap or "example" in snap)

        # real click on the local test page
        click_out = agent.tools["BrowserClick"].handler("a")
        check("BrowserClick returns (no exception)", bool(click_out))
        print("   click:", click_out[:100])

        close_out = agent.tools["BrowserClose"].handler()
        check("BrowserClose reports closed",
              close_out == "browser closed")
        print("   close:", close_out)

        # after close, lazy re-init works (session still functional)
        out2 = agent.tools["BrowserOpen"].handler(page_url)
        check("re-open after close works", "Example" in out2)
        final = agent.tools["BrowserClose"].handler()
        check("final close clean", final == "browser closed")
        server.shutdown()
        server.server_close()

        # invalid inputs return errors, never raise
        check("empty url -> error",
              agent.tools["BrowserOpen"].handler("").startswith("ERROR"))
        check("empty selector -> error",
              agent.tools["BrowserClick"].handler("").startswith("ERROR"))

    print("PASS" if fails == 0 else "FAIL", f"- {fails} failure(s)")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    _selftest()
