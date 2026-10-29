"""Upgraded web fetch: better extraction, caching, redirects, JS-heavy detection.

Drop-in improvement over ``tools.web_fetch``. Stdlib only (urllib, html.parser,
hashlib, json). Never imports ``.agent``.

Main entry point::

    fetch_text(url, timeout=20) -> dict

Returns one of::

    {"url": ..., "title": ..., "text": ..., "truncated": bool, "note": ...}
    {"error": "..."}

Errors are always returned as a dict, never raised.

Cache lives at ``~/.fullagent/cache/webfetch/`` (keyed by sha256(url), 5-minute
TTL); ``clear_cache()`` wipes it. Pass ``cache_dir=`` in tests to redirect it.
"""

from __future__ import annotations

import hashlib
import html
import ipaddress
import json
import os
import threading
import re
import socket
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser

__all__ = ["fetch_text", "clear_cache", "register"]

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT = 20
TEXT_CAP = 8_000          # max chars of extracted text kept
MAX_HTML_BYTES = 2_000_000  # download ceiling: never pull more than 2 MB
CACHE_TTL = 300            # seconds (5 minutes)
CACHE_SUBDIR = ("cache", "webfetch")
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) FullAgent/1.0"

# Tags whose content is never part of the visible text.
_SKIP_TAGS = frozenset({"script", "style", "nav", "footer", "template",
                        "svg", "noscript"})

# Markers that strongly suggest the page needs a JS-capable browser.
_JS_MARKERS = (
    "enable javascript",
    "javascript is required",
    "javascript required",
    "please enable js",
    "turn on javascript",
    "browser does not support javascript",
    "requires javascript",
    "this site requires javascript",
)


# ---------------------------------------------------------------------------
# HTML -> text extraction
# ---------------------------------------------------------------------------

class _TextExtractor(HTMLParser):
    """HTMLParser that captures the title and visible text, skipping boilerplate.

    ``script``/``style``/``nav``/``footer``/``template``/``svg``/``noscript``
    subtrees are dropped. Everything else is emitted with block-level tags
    producing line breaks so the result stays readable.
    """

    _BLOCK_TAGS = frozenset({
        "p", "div", "section", "article", "header", "main", "h1", "h2", "h3",
        "h4", "h5", "h6", "li", "ul", "ol", "table", "tr", "br", "hr",
        "blockquote", "pre", "figure", "figcaption", "aside", "dd", "dt",
    })

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._in_title = False
        self._title_parts: list[str] = []
        self._chunks: list[str] = []

    # -- title -------------------------------------------------------------
    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = True
        elif tag in self._BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = False
        elif tag in self._BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self._title_parts.append(data)
        else:
            self._chunks.append(data)

    @property
    def title(self) -> str:
        return _collapse("".join(self._title_parts)).strip()

    @property
    def text(self) -> str:
        return _collapse("".join(self._chunks)).strip()


def _collapse(s: str) -> str:
    """Collapse horizontal whitespace; keep paragraph breaks."""
    s = html.unescape(s)
    s = re.sub(r"[ \t\r\f\v\u00a0]+", " ", s)
    s = re.sub(r"\n[ \t]*\n+", "\n\n", s)
    s = re.sub(r"[ \t]*\n[ \t]*", "\n", s)
    return s


def _extract(html_text: str) -> tuple[str, str]:
    """Return (title, text) for an HTML document. Never raises."""
    parser = _TextExtractor()
    try:
        parser.feed(html_text)
        parser.close()
    except Exception:
        pass  # malformed HTML: return whatever we got
    return parser.title, parser.text


# ---------------------------------------------------------------------------
# Charset + content-type handling
# ---------------------------------------------------------------------------

def _detect_charset(raw: bytes, content_type: str) -> str:
    """Best-effort charset: header param -> <meta> tag -> utf-8 fallback."""
    m = re.search(r"charset\s*=\s*['\"]?([\w\-\.]+)", content_type, re.I)
    if m:
        return m.group(1)
    head = raw[:4096].decode("latin-1", errors="replace")
    m = re.search(r'<meta[^>]+charset\s*=\s*["\']?([\w\-\.]+)', head, re.I)
    if m:
        return m.group(1)
    m = re.search(
        r'<meta[^>]+http-equiv\s*=\s*["\']?content-type["\']?[^>]*'
        r'content\s*=\s*["\'][^"\']*charset\s*=\s*([\w\-\.]+)',
        head, re.I)
    if m:
        return m.group(1)
    return "utf-8"


def _is_html_content_type(content_type: str) -> bool:
    ct = (content_type or "").lower()
    return ("html" in ct or "xhtml" in ct or "text/" in ct
            or ct.endswith("+xml"))


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _cache_dir(cache_dir: str | None) -> str:
    if cache_dir:
        return cache_dir
    base = os.environ.get("FULLAGENT_HOME") or os.path.join(os.path.expanduser("~"), ".fullagent")
    return os.path.join(base, *CACHE_SUBDIR)


def _cache_path(url: str, cache_dir: str | None) -> str:
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return os.path.join(_cache_dir(cache_dir), key + ".json")


def _cache_get(url: str, cache_dir: str | None):
    """Return cached result dict on fresh hit, else None. Never raises."""
    try:
        path = _cache_path(url, cache_dir)
        with open(path, "r", encoding="utf-8") as f:
            entry = json.load(f)
        result = entry.get("result")
        if (isinstance(result, dict)
                and time.time() - float(entry.get("fetched_at", 0))
                <= CACHE_TTL):
            return result
    except Exception:
        pass
    return None


def _cache_put(url: str, result: dict, cache_dir: str | None) -> None:
    """Store a successful result. Never raises."""
    try:
        d = _cache_dir(cache_dir)
        os.makedirs(d, exist_ok=True)
        path = _cache_path(url, cache_dir)
        # unique tmp name: two threads fetching the same URL must not
        # interleave writes into one shared tmp file (torn cache entry)
        tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"fetched_at": time.time(), "url": url,
                       "result": result}, f)
        os.replace(tmp, path)
    except Exception:
        pass


def clear_cache(cache_dir: str | None = None) -> int:
    """Delete cached webfetch entries. Returns number of files removed."""
    d = _cache_dir(cache_dir)
    removed = 0
    try:
        for name in os.listdir(d):
            if name.endswith(".json"):
                try:
                    os.unlink(os.path.join(d, name))
                    removed += 1
                except OSError:
                    pass
    except OSError:
        pass
    return removed


# ---------------------------------------------------------------------------
# JS-heavy detection
# ---------------------------------------------------------------------------

def _js_heavy(text: str, html_size: int, raw_html: str) -> bool:
    """Heuristic: tiny extracted text from a large doc, or explicit markers."""
    if len(text) < 500 and html_size > 20_000:
        return True
    lowered = raw_html.lower()
    return any(marker in lowered for marker in _JS_MARKERS)


_JS_NOTE = ("note: page may require JavaScript; content may be incomplete")


# ---------------------------------------------------------------------------
# Main fetch
# ---------------------------------------------------------------------------

_METADATA_HOSTS = frozenset({
    "metadata.google.internal",
    "metadata.goog",
    "instance-data",
    "instance-data.ec2.internal",
})


def _host_error(host: str | None) -> str:
    """Return a reason if *host* is not a public internet address."""
    if not host:
        return "URL has no host"
    h = host.strip().strip("[]").rstrip(".").lower()
    if not h:
        return "URL has no host"
    if h in _METADATA_HOSTS:
        return f"blocked cloud-metadata host: {host}"
    try:
        infos = socket.getaddrinfo(h, None, proto=socket.IPPROTO_TCP)
    except OSError:
        return ""  # unresolvable: let urlopen report the network error
    for info in infos:
        try:
            ip = ipaddress.ip_address(str(info[4][0]).split("%", 1)[0])
        except ValueError:
            continue
        if not ip.is_global:
            # loopback / private / link-local (169.254.169.254) / CGNAT
            return f"blocked non-public address for {host}: {ip}"
    return ""


def _validate_url(url: object) -> str:
    """Return an error message, or '' if the URL is fetchable."""
    if not isinstance(url, str) or not url.strip():
        return "url must be a non-empty string"
    try:
        parsed = urllib.parse.urlparse(url.strip())
        scheme = parsed.scheme.lower()
        host = parsed.hostname
    except ValueError:
        return f"malformed URL: {url[:80]}"
    if scheme not in ("http", "https"):
        # SSRF guard: no file://, gopher://, or cloud-metadata endpoints
        # reached through exotic schemes.
        return (f"only http(s) URLs are allowed, "
                f"got {scheme or 'no'} scheme")
    blocked = _host_error(host)
    if blocked:
        return f"blocked URL: {blocked}"
    return ""


def fetch_text(url: str, timeout: int = DEFAULT_TIMEOUT,
               cache_dir: str | None = None) -> dict:
    """Fetch *url* and return extracted text.

    Success: ``{"url", "title", "text", "truncated", "note"}`` where ``note``
    is "" unless the page looks JS-heavy or the text was truncated.
    Failure: ``{"error": "..."}``. Never raises.
    """
    err = _validate_url(url)
    if err:
        return {"error": err}
    url = url.strip()

    cached = _cache_get(url, cache_dir)
    if cached is not None:
        return dict(cached)

    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT,
                 "Accept": "text/html,application/xhtml+xml,"
                           "application/xml;q=0.9,text/plain;q=0.8,"
                           "*/*;q=0.5"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = getattr(resp, "status", 200)
            if status >= 400:
                return {"error": f"HTTP {status}"}
            final_url = resp.geturl()
            # urllib follows redirects on its own: validate the final hop
            # before reading any body, so an attacker page cannot redirect
            # us to 169.254.169.254 / localhost and leak the response.
            final_err = _validate_url(final_url)
            if final_err:
                return {"error": f"redirect blocked: {final_err}"}
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read(MAX_HTML_BYTES)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    if not _is_html_content_type(content_type):
        return {"error": f"unsupported content type: "
                         f"{content_type or 'unknown'}"}

    charset = _detect_charset(raw, content_type)
    try:
        html_text = raw.decode(charset, errors="replace")
    except (LookupError, UnicodeError):
        html_text = raw.decode("utf-8", errors="replace")

    title, text = _extract(html_text)

    truncated = False
    note = ""
    if len(text) > TEXT_CAP:
        text = text[:TEXT_CAP].rsplit(" ", 1)[0]
        truncated = True
        note = f"note: text truncated at {TEXT_CAP} chars; ask for a " \
               f"specific section to see more"

    if _js_heavy(text, len(html_text), html_text):
        note = (note + "; " if note else "") + _JS_NOTE

    result = {"url": final_url, "title": title, "text": text,
              "truncated": truncated, "note": note}
    _cache_put(url, result, cache_dir)
    return result


# ---------------------------------------------------------------------------
# Agent integration
# ---------------------------------------------------------------------------

def _web_fetch_str(url: str) -> str:
    """String adapter matching tools.web_fetch's return contract."""
    res = fetch_text(url)
    if "error" in res:
        return f"ERROR: {res['error']}"
    head = f"# {res['title']}\n\n" if res.get("title") else ""
    note = f"\n\n({res['note']})" if res.get("note") else ""
    return f"{head}URL: {res['url']}\n\n{res['text']}{note}"


def register(agent) -> None:
    """Swap the agent's ``web_fetch`` handler for the improved implementation.

    Keeps the tool name, description, schema and risk exactly as registered,
    so the LLM-visible contract is unchanged; the handler now extracts better
    (title, no script/style/nav/footer), caches for 5 minutes, follows
    redirects and flags JS-heavy pages. If no ``web_fetch`` tool exists yet,
    registers one under that name.
    """
    from .tools import Tool  # local import: avoids any import-time cycles

    tools = getattr(agent, "tools", None)
    if not isinstance(tools, dict):
        tools = {}
        setattr(agent, "tools", tools)
    existing = tools.get("web_fetch")
    if existing is not None:
        agent.tools["web_fetch"] = Tool(
            existing.name, existing.description, existing.parameters,
            _web_fetch_str, risk=existing.risk)
    else:
        agent.tools["web_fetch"] = Tool(
            "web_fetch", "Fetch a URL and return its text content.",
            {"type": "object", "properties": {"url": {"type": "string"}},
             "required": ["url"]},
            _web_fetch_str)


# ---------------------------------------------------------------------------
# Self-test (no network): python3 -m fullagent.webfetch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    _fails = 0

    def check(name: str, cond: bool) -> None:
        global _fails
        print(("PASS" if cond else "FAIL"), "-", name)
        if not cond:
            _fails += 1

    # --- 1. extraction on crafted HTML -------------------------------------
    html_doc = """<!DOCTYPE html><html><head>
    <title>  Test &amp; Page </title>
    <style>body { color: red; }</style>
    <script>var x = 1; document.write('secret-script-text');</script>
    </head><body>
    <nav>Home | About | Contact</nav>
    <main><h1>Hello</h1><p>Visible   paragraph  one.</p>
    <p>Visible paragraph two.</p></main>
    <footer>Copyright 2026 - footer junk</footer>
    </body></html>"""
    title, text = _extract(html_doc)
    check("title captured", title == "Test & Page")
    check("script stripped", "secret-script-text" not in text)
    check("style stripped", "color: red" not in text)
    check("nav stripped", "Home | About" not in text)
    check("footer stripped", "Copyright" not in text)
    check("visible text kept",
          "Hello" in text and "Visible paragraph one." in text
          and "Visible paragraph two." in text)
    check("whitespace collapsed", "paragraph  one" not in text)

    # --- 2. fetch_text end-to-end with mocked urlopen (no network) ---------
    class _FakeHeaders(dict):
        def get(self, k, default=None):
            return super().get(k, default)

    class _FakeResp:
        def __init__(self, body: bytes, final_url: str):
            self._body = body
            self.status = 200
            self.headers = _FakeHeaders(
                {"Content-Type": "text/html; charset=utf-8"})
            self._final = final_url

        def geturl(self):
            return self._final

        def read(self, n=-1):
            return self._body[:n] if n != -1 else self._body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    calls = {"n": 0}
    payload = html_doc.encode("utf-8")

    def _fake_urlopen(req, timeout=None):
        calls["n"] += 1
        full = req.full_url if hasattr(req, "full_url") else str(req)
        return _FakeResp(payload, full + "/final")

    real_urlopen = urllib.request.urlopen
    urllib.request.urlopen = _fake_urlopen
    try:
        with tempfile.TemporaryDirectory() as d:
            url = "http://example.test/page"
            r1 = fetch_text(url, cache_dir=d)
            check("fetch_text no error", "error" not in r1)
            check("final url from redirect chain",
                  r1.get("url") == url + "/final")
            check("title in result", r1.get("title") == "Test & Page")
            check("text in result", "Visible paragraph one." in r1["text"])
            check("truncated False on short page",
                  r1.get("truncated") is False)
            check("first call hit network", calls["n"] == 1)

            r2 = fetch_text(url, cache_dir=d)
            check("cache hit on second call (no second urlopen)",
                  calls["n"] == 1 and r2 == r1)

            # cache files exist on disk
            files = list(Path(d).glob("*.json"))
            check("cache file written", len(files) == 1)

            # expired entries are treated as miss
            p = files[0]
            stale = json.loads(p.read_text(encoding="utf-8"))
            stale["fetched_at"] = time.time() - CACHE_TTL - 10
            p.write_text(json.dumps(stale), encoding="utf-8")
            r3 = fetch_text(url, cache_dir=d)
            check("stale entry refetched", calls["n"] == 2
                  and r3.get("title") == "Test & Page")

            check("clear_cache removes entries",
                  clear_cache(cache_dir=d) == 1
                  and list(Path(d).glob("*.json")) == [])
    finally:
        urllib.request.urlopen = real_urlopen

    # --- 3. JS-heavy detection ---------------------------------------------
    js_html = ("<html><head><title>App</title></head><body>"
               "<div id='root'></div><script src='/bundle.js'></script>"
               "<p>Please enable JavaScript to view this site.</p>"
               "</body></html>")
    t3, x3 = _extract(js_html)
    check("low-text extractor output", len(x3) < 500)
    check("js-heavy via marker",
          _js_heavy(x3, len(js_html), js_html))
    big_html = ("<html><head><title>Big</title></head><body>"
                + "<script>" + "x" * 30000 + "</script>"
                + "<p>Tiny.</p></body></html>")
    tb, xb = _extract(big_html)
    check("js-heavy via low-text+large-html",
          _js_heavy(xb, len(big_html), big_html))
    normal = ("<html><head><title>N</title></head><body>"
              + "<p>" + "word " * 200 + "</p></body></html>")
    tn, xn = _extract(normal)
    check("normal page not flagged", not _js_heavy(xn, len(normal), normal))

    # --- 4. truncation ------------------------------------------------------
    long_html = ("<html><head><title>Long</title></head><body><p>"
                 + "word " * 5000 + "</p></body></html>")
    tl, xl = _extract(long_html)
    check("long text extracted", len(xl) > TEXT_CAP)

    # --- 5. error contract (never raises) -----------------------------------
    e1 = fetch_text("")
    check("empty url -> error dict, no raise",
          e1.get("error") == "url must be a non-empty string")
    e2 = fetch_text("file:///etc/passwd")
    check("file scheme rejected",
          e2.get("error", "").startswith("only http(s) URLs are allowed"))
    urllib.request.urlopen = lambda *a, **k: (_ for _ in ()).throw(
        OSError("boom"))
    try:
        e3 = fetch_text("http://example.test/", cache_dir=None)
        check("network failure -> error dict, no raise",
              "error" in e3 and "OSError" in e3["error"])
    finally:
        urllib.request.urlopen = real_urlopen

    print("PASS" if _fails == 0 else "FAIL",
          f"- {_fails} failure(s)")
    raise SystemExit(1 if _fails else 0)
