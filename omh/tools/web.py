"""``web_fetch``: download a page and reduce it to readable text."""

from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Any, Dict, List

from ..errors import ToolError
from .base import ToolContext, ToolResult, ToolSpec

MAX_BYTES = 2_000_000
SKIP_TAGS = {"script", "style", "noscript", "svg", "head", "iframe", "template"}
BLOCK_TAGS = {
    "p", "div", "br", "li", "tr", "section", "article", "header", "footer",
    "h1", "h2", "h3", "h4", "h5", "h6", "pre", "blockquote", "table",
}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in SKIP_TAGS:
            self._skip_depth += 1
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts).replace("\r", "")
        lines = [" ".join(line.split()) for line in raw.splitlines()]
        cleaned: List[str] = []
        blank = False
        for line in lines:
            if not line:
                if not blank and cleaned:
                    cleaned.append("")
                blank = True
            else:
                cleaned.append(line)
                blank = False
        return "\n".join(cleaned).strip()


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # malformed HTML should still yield something
        return html
    return parser.text()


def _web_fetch(arguments: Dict[str, Any], context: ToolContext) -> ToolResult:
    url = str(arguments["url"]).strip()
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        raise ToolError("url must start with http:// or https://")
    if not parsed.netloc:
        raise ToolError(f"invalid url: {url}")
    limit = min(100_000, max(1, int(arguments.get("max_chars", 20_000) or 20_000)))
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; omh/1.0; +https://github.com/rlaope/oh-my-hermes)",
            "Accept": "text/html,application/xhtml+xml,application/json,text/plain;q=0.9,*/*;q=0.5",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            status = getattr(response, "status", 200)
            content_type = (response.headers.get("Content-Type") or "").lower()
            raw = response.read(MAX_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise ToolError(f"{url} returned HTTP {exc.code}") from exc
    except Exception as exc:
        raise ToolError(f"cannot fetch {url}: {exc}") from exc

    truncated = len(raw) > MAX_BYTES
    raw = raw[:MAX_BYTES]
    charset = "utf-8"
    if "charset=" in content_type:
        charset = content_type.split("charset=", 1)[1].split(";")[0].strip() or "utf-8"
    try:
        body = raw.decode(charset, "replace")
    except LookupError:
        body = raw.decode("utf-8", "replace")

    if "html" in content_type or body.lstrip().startswith("<"):
        text = html_to_text(body)
    elif "json" in content_type or "text" in content_type or not content_type:
        text = body
    else:
        raise ToolError(f"unsupported content type: {content_type or 'unknown'}")

    text = text.strip() or "(empty body)"
    if len(text) > limit:
        text = text[:limit] + f"\n… [truncated at {limit:,} characters]"
    header = f"GET {url} → HTTP {status}, {len(raw):,} bytes"
    if truncated:
        header += " (download capped at 2 MB)"
    return ToolResult.success(f"{header}\n\n{text}", url=url, status=status)


def web_tools() -> List[ToolSpec]:
    return [
        ToolSpec(
            name="web_fetch",
            description=(
                "Fetch an http(s) URL and return readable text (HTML is stripped). "
                "Use it for documentation, issues or changelogs."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "max_chars": {"type": "integer"},
                },
                "required": ["url"],
            },
            run=_web_fetch,
            preview=lambda arguments: str(arguments.get("url") or ""),
        )
    ]
