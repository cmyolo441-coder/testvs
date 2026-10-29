"""Vision support: read image files and attach them to model requests.

Public API:
    - :func:`register` -- register the ``ImageRead`` tool on an agent
      (duck-typed) and attach ``agent.image_to_part``.
    - :func:`image_to_message_part` -- build an OpenAI-style content part
      for an image path; never raises.
    - :func:`describe_image` -- compact text descriptor (dimensions, size,
      base64 data URI) used by the ``ImageRead`` tool.
    - :func:`detect_mime` / :func:`image_dimensions` -- low-level helpers
      (magic-byte sniffing + ``struct`` header parsing, no PIL).

TUI convention (coordinator wires this): when user input contains
``@image <path>``, the input preprocessor expands the token by calling
:func:`image_to_message_part` and appending the returned part to the
outgoing message's ``content`` list, e.g.::

    @image /tmp/shot.png  what do you see here?

becomes a user message whose content is::

    [{"type": "text", "text": "what do you see here?"},
     {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]

Files larger than 10 MiB are refused; unsupported or unreadable files
produce a ``[image unreadable: reason]`` text part instead of raising.
"""

from __future__ import annotations

import base64
import os
import struct
from pathlib import Path
from typing import Any, Optional, Tuple

from .tools import Tool
from ._foundation import validate_path, ValidationError

MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MiB

SUPPORTED_MIMES = ("image/png", "image/jpeg", "image/gif", "image/webp")

_PNG_SIG = b"\x89PNG\r\n\x1a\n"
_JPEG_SIG = b"\xff\xd8\xff"
_WEBP_RIFF = b"RIFF"
_WEBP_FTYP = b"WEBP"


# ---------------------------------------------------------------------------
# Magic-byte detection (not the file extension)
# ---------------------------------------------------------------------------

def detect_mime(data: bytes) -> Optional[str]:
    """Return the MIME type from magic bytes, or None if not a supported image."""
    if data.startswith(_PNG_SIG):
        return "image/png"
    if data.startswith(_JPEG_SIG):
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if (data[:4] == _WEBP_RIFF and data[8:12] == _WEBP_FTYP):
        return "image/webp"
    return None


# ---------------------------------------------------------------------------
# Dimension parsing via struct (no PIL / Pillow dependency)
# ---------------------------------------------------------------------------

def _png_dims(data: bytes) -> Optional[Tuple[int, int]]:
    # IHDR: width/height are big-endian u32 at offsets 16/20.
    if len(data) < 24 or data[12:16] != b"IHDR":
        return None
    w, h = struct.unpack(">II", data[16:24])
    return (w, h) if w > 0 and h > 0 else None


def _jpeg_dims(data: bytes) -> Optional[Tuple[int, int]]:
    # Walk markers to the first SOFn (excluding DHT/JPG/arithmetic-codings).
    i = 2
    n = len(data)
    sof = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
           0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    while i + 4 <= n:
        if data[i] != 0xFF:
            return None
        # skip padding bytes
        while i < n and data[i] == 0xFF:
            i += 1
        if i >= n:
            return None
        marker = data[i]
        i += 1
        if marker == 0xD9:            # EOI
            return None
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:
            continue                 # standalone markers, no length field
        if i + 2 > n:
            return None
        length = struct.unpack(">H", data[i:i + 2])[0]
        if length < 2 or i + length > n:
            return None
        if marker in sof:
            # SOF payload: precision(1) height(2) width(2)
            if i + 7 > n:
                return None
            h, w = struct.unpack(">HH", data[i + 3:i + 7])
            return (w, h) if w > 0 and h > 0 else None
        if marker == 0xDA:           # SOS: scan data follows, no dims after
            return None
        i += length
    return None


def _gif_dims(data: bytes) -> Optional[Tuple[int, int]]:
    # Logical Screen Descriptor: little-endian u16 width/height at 6/8.
    if len(data) < 10:
        return None
    w, h = struct.unpack("<HH", data[6:10])
    return (w, h) if w > 0 and h > 0 else None


def _webp_dims(data: bytes) -> Optional[Tuple[int, int]]:
    # RIFF....WEBP then chunks: VP8X (canvas), VP8 (lossy), VP8L (lossless).
    if len(data) < 30:
        return None
    chunk = data[12:16]
    if chunk == b"VP8X":
        w = int.from_bytes(data[24:27], "little") + 1
        h = int.from_bytes(data[27:30], "little") + 1
        return (w, h) if w > 0 and h > 0 else None
    if chunk == b"VP8 " and len(data) >= 30 and data[23:26] == b"\x9d\x01\x2a":
        w = int.from_bytes(data[26:28], "little") & 0x3FFF
        h = int.from_bytes(data[28:30], "little") & 0x3FFF
        return (w, h) if w > 0 and h > 0 else None
    if chunk == b"VP8L" and len(data) >= 25 and data[20] == 0x2F:
        # VP8L: 14-bit width-1, then 14-bit height-1 (LSB-first).
        word = int.from_bytes(data[21:25], "little")
        w = (word & 0x3FFF) + 1
        h = ((word >> 14) & 0x3FFF) + 1
        return (w, h) if w > 0 and h > 0 else None
    return None


def image_dimensions(mime: str, data: bytes) -> Optional[Tuple[int, int]]:
    """Parse (width, height) from image headers. None when unparseable."""
    try:
        if mime == "image/png":
            return _png_dims(data)
        if mime == "image/jpeg":
            return _jpeg_dims(data)
        if mime == "image/gif":
            return _gif_dims(data)
        if mime == "image/webp":
            return _webp_dims(data)
    except (struct.error, IndexError, ValueError):
        pass
    return None


# ---------------------------------------------------------------------------
# Loading / validation
# ---------------------------------------------------------------------------

def _resolve(path: Any) -> Path:
    """Resolve a user-supplied path against the process cwd (traversal-safe)."""
    p = validate_path(path)  # raises ValidationError on bad input
    if not p.is_absolute():
        p = Path.cwd() / p
    return p


def load_image(path: Any) -> Tuple[Optional[str], Optional[bytes],
                                   Optional[str]]:
    """Validate and read an image file.

    Returns ``(mime, data, error)``: on success ``error`` is None; on any
    failure ``mime``/``data`` are None and ``error`` explains why. Files
    over :data:`MAX_IMAGE_BYTES` are refused.
    """
    try:
        p = _resolve(path)
    except (ValidationError, ValueError, TypeError) as e:
        return None, None, f"invalid path: {e}"
    if not p.exists():
        return None, None, f"file not found: {p}"
    if not p.is_file():
        return None, None, f"not a file: {p}"
    try:
        size = p.stat().st_size
    except OSError as e:
        return None, None, f"cannot stat file: {e}"
    if size > MAX_IMAGE_BYTES:
        return (None, None,
                f"file is {size:,} bytes, over the "
                f"{MAX_IMAGE_BYTES:,} byte (10 MiB) limit")
    if size == 0:
        return None, None, "file is empty"
    try:
        data = p.read_bytes()
    except OSError as e:
        return None, None, f"cannot read file: {e}"
    mime = detect_mime(data)
    if mime is None:
        return (None, None,
                "not a supported image (magic bytes do not match "
                "png/jpeg/gif/webp)")
    return mime, data, None


def data_uri(mime: str, data: bytes) -> str:
    """Build a ``data:<mime>;base64,...`` URI for image bytes."""
    return "data:{};base64,{}".format(
        mime, base64.b64encode(data).decode("ascii"))


def describe_image(path: Any) -> str:
    """Compact descriptor for the ImageRead tool: dims, size, data URI."""
    mime, data, error = load_image(path)
    if error is not None:
        return f"Error: {error}"
    dims = image_dimensions(mime, data)
    dim_str = f"{dims[0]}x{dims[1]}" if dims else "unknown"
    lines = [
        f"Image: {Path(str(path)).name}",
        f"Format: {mime}",
        f"Dimensions: {dim_str}",
        f"Size: {len(data):,} bytes",
        f"Data URI: {data_uri(mime, data)}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Message part for providers (never raises)
# ---------------------------------------------------------------------------

def image_to_message_part(path: Any) -> dict:
    """Return an OpenAI-style content part for an image.

    Success: ``{"type": "image_url", "image_url": {"url": "data:...;base64,..."}}``.
    Any failure: ``{"type": "text", "text": "[image unreadable: reason]"}``.
    Never raises.
    """
    try:
        mime, data, error = load_image(path)
    except Exception as e:  # defensive: this function must never raise
        return {"type": "text",
                "text": f"[image unreadable: unexpected error: {e}]"}
    if error is not None:
        return {"type": "text", "text": f"[image unreadable: {error}]"}
    return {"type": "image_url",
            "image_url": {"url": data_uri(mime, data)}}


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

def _handle_image_read(path: Any = None, **kwargs: Any) -> str:
    if path is None or (isinstance(path, str) and not path.strip()):
        return "Error: 'path' is required."
    return describe_image(path)


def make_image_read_tool() -> Tool:
    return Tool(
        name="ImageRead",
        description=(
            "Read an image file (png/jpeg/gif/webp, max 10 MiB) and return "
            "a compact descriptor: dimensions, byte size, and a base64 data "
            "URI you can pass back to the user message as an image_url "
            "content part. Use to inspect screenshots, diagrams, or photos "
            "the user references."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the image file.",
                },
            },
            "required": ["path"],
        },
        handler=_handle_image_read,
    )


def register(agent: Any) -> None:
    """Wire the ImageRead tool into an agent (duck-typed, no imports).

    Registers ``agent.tools["ImageRead"]`` and attaches
    ``agent.image_to_part = image_to_message_part`` so the TUI input
    preprocessor can expand ``@image <path>`` tokens.
    """
    agent.tools["ImageRead"] = make_image_read_tool()
    agent.image_to_part = image_to_message_part


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    import zlib

    def _make_1x1_png(path: str) -> None:
        # Minimal valid 1x1 RGBA PNG, built by hand (no PIL).
        def chunk(ctype: bytes, payload: bytes) -> bytes:
            body = ctype + payload
            return (struct.pack(">I", len(payload)) + body +
                    struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))
        ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
        raw = b"\x00\xff\x00\x00\xff"  # 1 red pixel + filter byte
        png = (_PNG_SIG + chunk(b"IHDR", ihdr) +
               chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
        with open(path, "wb") as fh:
            fh.write(png)

    tmp = tempfile.mkdtemp(prefix="images_selftest_")
    png_path = os.path.join(tmp, "one.png")
    _make_1x1_png(png_path)

    class FakeAgent:
        def __init__(self):
            self.session_id = "selftest-session"
            self.tools = {}

    a = FakeAgent()
    register(a)
    assert "ImageRead" in a.tools, "tool not registered"
    assert callable(getattr(a, "image_to_part", None)), \
        "agent.image_to_part missing"

    # 1. validation + descriptor
    out = a.tools["ImageRead"].handler(path=png_path)
    assert "Dimensions: 1x1" in out, out
    assert "Format: image/png" in out, out
    assert "Data URI: data:image/png;base64," in out, out
    size_line = [ln for ln in out.splitlines()
                 if ln.startswith("Size:")][0]
    assert int(size_line.split()[1].replace(",", "")) > 0

    # 2. message part shape
    part = image_to_message_part(png_path)
    assert part["type"] == "image_url", part
    url = part["image_url"]["url"]
    assert url.startswith("data:image/png;base64,"), url[:40]
    assert base64.b64decode(url.split(",", 1)[1])[:8] == _PNG_SIG

    # 3. JPEG + GIF + WebP dimension parsing (hand-built headers)
    # minimal JPEG with SOF0 2x3
    jpeg = (b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01"
            b"\x00\x00\xff\xc0\x00\x11\x08\x00\x03\x00\x02\x03"
            b"\x01\x11\x00\x02\x11\x01\x03\x11\x01")
    assert image_dimensions("image/jpeg", jpeg) == (2, 3), \
        image_dimensions("image/jpeg", jpeg)
    gif = b"GIF89a" + struct.pack("<HH", 4, 5) + b"\x00" * 10
    assert detect_mime(gif) == "image/gif"
    assert image_dimensions("image/gif", gif) == (4, 5)
    # WebP VP8X canvas 6x7
    webp = (b"RIFF" + b"\x00" * 4 + b"WEBP" + b"VP8X" +
            struct.pack("<I", 10) + b"\x00" * 4 +
            (5).to_bytes(3, "little") + (6).to_bytes(3, "little"))
    assert detect_mime(webp) == "image/webp"
    assert image_dimensions("image/webp", webp) == (6, 7)

    # 4. graceful failures (never raise)
    missing = os.path.join(tmp, "nope.png")
    assert image_to_message_part(missing)["type"] == "text"
    assert "not found" in image_to_message_part(missing)["text"]
    assert a.tools["ImageRead"].handler(
        path=missing).startswith("Error:")
    corrupt = os.path.join(tmp, "junk.bin")
    with open(corrupt, "wb") as fh:
        fh.write(b"this is definitely not an image")
    part = image_to_message_part(corrupt)
    assert part["type"] == "text" and "not a supported image" in part["text"]
    assert a.tools["ImageRead"].handler(
        path=corrupt).startswith("Error:")
    assert a.tools["ImageRead"].handler().startswith("Error:")  # no path

    # 5. oversize refusal (sparse file so it doesn't eat disk)
    big = os.path.join(tmp, "big.png")
    with open(big, "wb") as fh:
        fh.truncate(MAX_IMAGE_BYTES + 1)
    part = image_to_message_part(big)
    assert part["type"] == "text" and "over the" in part["text"], part
    assert a.tools["ImageRead"].handler(
        path=big).startswith("Error: file is")

    print("images self-test PASSED")
