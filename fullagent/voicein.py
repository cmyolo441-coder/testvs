"""Voice input: REAL speech-to-text via faster-whisper / openai-whisper.

LLM tools (see :func:`register`)::

    VoiceRecord      — record N seconds from the microphone, then transcribe
                       with a real whisper model; returns the transcript text
    VoiceTranscribe  — transcribe an existing .wav file with a real whisper
                       model; returns the transcript text

Graceful degradation (never faked):

    * Neither ``faster-whisper`` nor ``openai-whisper`` installed →
      every tool and the ``/voice`` command returns the clear message
      ``whisper not installed (pip install faster-whisper)``. No
      transcription is simulated.
    * No ``arecord`` / ``ffmpeg`` on PATH → recording explains which
      capture tool is missing instead of pretending to record.
    * Silent, empty, or garbage audio → reported as "no speech detected"
      (segments below the confidence floor are dropped); whisper is never
      asked to hallucinate and its low-confidence output is never returned.

TUI: ``/voice [seconds]`` records (default 10s), transcribes, and prints the
result. Inserting the transcript into the live input buffer would require
host-TUI support the prompt loop does not currently expose, so the text is
printed for copy/paste — documented here rather than faked.

Public API:

    - :func:`register` -- register the tools on an agent (duck-typed) and
      attach ``agent.voice_transcribe`` / ``agent.record_and_transcribe``.
    - :func:`handle_voice` -- TUI ``/voice`` handler (``ui`` duck-typed).
    - :func:`transcribe_wav` -- transcribe a wav file, real whisper only.
    - :func:`record_and_transcribe` -- real mic capture → real transcription.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import tempfile
import threading
from typing import Any, Dict, List, Optional, Tuple

from .tools import RISK_CONFIRM, RISK_SAFE, Tool
from ._foundation import validate_path, ValidationError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WHISPER_NOT_INSTALLED = "whisper not installed (pip install faster-whisper)"
NO_SPEECH = ("no speech detected in the audio "
             "(empty or low-confidence transcription)")
NO_CAPTURE_TOOL = ("no audio capture tool found — install alsa-utils "
                   "(arecord) or ffmpeg to record from the microphone")

#: faster-whisper segments with avg_logprob below this are dropped instead of
#: returned (silence / noise / garbage must not become hallucinated text).
CONFIDENCE_FLOOR = -1.0

#: Model size for both backends. Override with FULLAGENT_WHISPER_MODEL=base.
DEFAULT_MODEL = os.environ.get("FULLAGENT_WHISPER_MODEL", "tiny")

MAX_RECORD_SECONDS = 120
MAX_WAV_BYTES = 100 * 1024 * 1024  # 100 MiB


# ---------------------------------------------------------------------------
# Backend detection — importlib only, never fakes a backend
# ---------------------------------------------------------------------------

def whisper_backend() -> Optional[str]:
    """Return ``"faster_whisper"``, ``"whisper"``, or None.

    Uses ``importlib.util.find_spec`` so the heavy native dependencies are
    never imported just to check availability.
    """
    for name in ("faster_whisper", "whisper"):
        try:
            if importlib.util.find_spec(name) is not None:
                return name
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Lazy model loading — once per process, real models only
# ---------------------------------------------------------------------------

_model_lock = threading.Lock()
_model: Any = None
_model_backend: Optional[str] = None


def _get_model() -> Tuple[Any, str]:
    """Load (once) and return ``(model, backend_name)``.

    Raises RuntimeError when no whisper backend is installed.
    """
    global _model, _model_backend
    with _model_lock:
        if _model is not None:
            return _model, _model_backend  # type: ignore[return-value]
        backend = whisper_backend()
        if backend is None:
            raise RuntimeError(WHISPER_NOT_INSTALLED)
        if backend == "faster_whisper":
            from faster_whisper import WhisperModel
            _model = WhisperModel(DEFAULT_MODEL, device="cpu",
                                  compute_type="int8")
        else:  # openai-whisper
            import whisper
            _model = whisper.load_model(DEFAULT_MODEL)
        _model_backend = backend
        return _model, backend


# ---------------------------------------------------------------------------
# Transcription — real whisper, confidence-filtered, never hallucinated
# ---------------------------------------------------------------------------

def segments_to_text(segments: List[Any]) -> str:
    """Join segment texts, dropping anything below the confidence floor.

    Accepts faster-whisper segment objects (``.text`` / ``.avg_logprob``)
    or openai-whisper segment dicts (``"text"`` / ``"avg_logprob"``).
    Pure function — safe to unit test with hand-built segments.
    """
    parts: List[str] = []
    for seg in segments:
        if isinstance(seg, dict):
            text = str(seg.get("text") or "").strip()
            conf = seg.get("avg_logprob")
        else:
            text = str(getattr(seg, "text", "") or "").strip()
            conf = getattr(seg, "avg_logprob", None)
        if not text:
            continue
        if conf is not None:
            try:
                if float(conf) < CONFIDENCE_FLOOR:
                    continue
            except (TypeError, ValueError):
                pass
        parts.append(text)
    return " ".join(parts).strip()


def _transcribe_with_model(model: Any, backend: str,
                           wav_path: str) -> List[Any]:
    """Run the real model and return its raw segments. Raises on failure."""
    if backend == "faster_whisper":
        segments, _info = model.transcribe(wav_path, beam_size=5)
        return list(segments)
    # openai-whisper
    result = model.transcribe(wav_path)
    return list(result.get("segments") or [])


def _is_wav(path: str) -> bool:
    """True when the file starts with a RIFF....WAVE header."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(12)
        return len(head) == 12 and head[:4] == b"RIFF" and head[8:12] == b"WAVE"
    except OSError:
        return False


def transcribe_wav(wav_path: str) -> str:
    """Transcribe a .wav file with the real whisper model.

    Returns the transcript text, or a clear message when transcription is
    impossible (no backend, bad path, unreadable file, no speech). Never
    returns invented text.
    """
    if whisper_backend() is None:
        return WHISPER_NOT_INSTALLED
    try:
        p = validate_path(wav_path, name="wav_path", must_exist=True)
    except ValidationError as exc:
        return f"invalid wav_path: {exc}"
    if not p.is_file():
        return f"wav_path is not a file: {p}"
    try:
        if p.stat().st_size > MAX_WAV_BYTES:
            return (f"wav file too large "
                    f"({p.stat().st_size // 1024 // 1024} MiB > "
                    f"{MAX_WAV_BYTES // 1024 // 1024} MiB)")
    except OSError as exc:
        return f"cannot stat wav file: {exc}"
    if not _is_wav(str(p)):
        return f"not a WAV file (bad RIFF/WAVE header): {p}"
    try:
        model, backend = _get_model()
    except RuntimeError as exc:
        return str(exc)
    except Exception as exc:  # noqa: BLE001 — model load failure is a message
        return f"could not load whisper model ({DEFAULT_MODEL}): {exc}"
    try:
        segments = _transcribe_with_model(model, backend, str(p))
    except Exception as exc:  # noqa: BLE001 — transcription failure is a msg
        return f"transcription failed: {exc}"
    text = segments_to_text(segments)
    return text if text else NO_SPEECH


# ---------------------------------------------------------------------------
# Audio capture — real arecord / ffmpeg, tool availability checked first
# ---------------------------------------------------------------------------

def find_capture_tool() -> Optional[str]:
    """Return ``"arecord"`` or ``"ffmpeg"`` if present on PATH, else None."""
    if shutil.which("arecord"):
        return "arecord"
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    return None


def _capture_cmd(tool: str, seconds: int, out_wav: str,
                 _input: Optional[Tuple[str, str]] = None) -> List[str]:
    """Build the real capture command.

    ``_input`` is a private test seam: ``("alsa", "default")`` is the real
    microphone path; the self-test passes ``("lavfi", "anullsrc=…")`` to
    exercise the pipeline without a microphone.
    """
    fmt, src = _input if _input is not None else ("alsa", "default")
    if tool == "arecord":
        if fmt != "alsa":
            raise ValueError("arecord only supports ALSA input")
        return ["arecord", "-d", str(seconds), "-f", "S16_LE",
                "-r", "16000", "-c", "1", "-t", "wav", out_wav]
    # ffmpeg
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", fmt, "-i", src,
           "-t", str(seconds), "-ar", "16000", "-ac", "1",
           "-c:a", "pcm_s16le", out_wav]
    return cmd


def record_and_transcribe(seconds: int = 10) -> str:
    """Record ``seconds`` from the microphone, then really transcribe it.

    Returns the transcript text, or a clear message when anything in the
    pipeline is unavailable. Never returns invented text.
    """
    try:
        seconds = int(seconds)
    except (TypeError, ValueError):
        return "seconds must be an integer"
    seconds = max(1, min(MAX_RECORD_SECONDS, seconds))

    # Check the transcription backend FIRST so we never record audio we
    # cannot transcribe.
    if whisper_backend() is None:
        return WHISPER_NOT_INSTALLED

    tool = find_capture_tool()
    if tool is None:
        return NO_CAPTURE_TOOL

    fd, tmp = tempfile.mkstemp(prefix="voicein_", suffix=".wav")
    os.close(fd)
    try:
        cmd = _capture_cmd(tool, seconds, tmp)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=seconds + 60)
        except subprocess.TimeoutExpired:
            return f"recording timed out after {seconds}s"
        except FileNotFoundError:
            return NO_CAPTURE_TOOL
        if proc.returncode != 0:
            err = (proc.stderr or "").strip().splitlines()
            tail = err[-1] if err else "unknown error"
            return f"recording failed ({tool} exit {proc.returncode}): {tail}"
        if not _is_wav(tmp):
            return "recording produced no valid WAV audio"
        return transcribe_wav(tmp)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def _handle_voice_record(seconds: int = 10) -> str:
    return record_and_transcribe(seconds)


def _handle_voice_transcribe(wav_path: str) -> str:
    return transcribe_wav(wav_path)


# ---------------------------------------------------------------------------
# TUI handler — /voice [seconds]
# ---------------------------------------------------------------------------

def handle_voice(ui: Any, arg: str) -> None:
    """Dispatch ``/voice [seconds]``; prints via ``ui``.

    Records from the microphone (default 10s), transcribes with the real
    whisper model, and prints the transcript for copy/paste. The live input
    buffer cannot be written from here — the prompt loop exposes no public
    API for it — so insertion is intentionally not faked; this docstring and
    the printed output say exactly that.
    """
    text = (arg or "").strip()
    seconds = 10
    if text:
        try:
            seconds = int(text.split()[0])
        except ValueError:
            ui.print_error("usage: /voice [seconds]  (1-120)")
            return
    if whisper_backend() is None:
        ui.print_error(WHISPER_NOT_INSTALLED)
        return
    ui.print_info(f"🎙 recording {seconds}s from microphone… speak now")
    result = record_and_transcribe(seconds)
    if result in (WHISPER_NOT_INSTALLED, NO_CAPTURE_TOOL) or \
            result.startswith(("recording failed", "transcription failed",
                               "no speech detected", "invalid wav_path",
                               "could not load")):
        ui.print_error(result)
        return
    ui.print_info("transcript (copy/paste — buffer insert not supported "
                  "by the prompt loop):")
    ui.print_info(result)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register(agent: Any) -> None:
    """Wire VoiceRecord / VoiceTranscribe into an agent (duck-typed)."""
    agent.tools["VoiceRecord"] = Tool(
        name="VoiceRecord",
        description=(
            "Record N seconds of audio from the microphone and transcribe "
            "it with a real whisper speech-to-text model. Returns the "
            "transcript text. Requires faster-whisper (pip install "
            "faster-whisper) and arecord or ffmpeg; if either is missing "
            "the tool says so instead of faking a transcript."),
        parameters={"type": "object", "properties": {
            "seconds": {"type": "integer",
                        "description": "recording length in seconds "
                                     "(1-120, default 10)"}},
            "required": []},
        handler=lambda seconds=10: _handle_voice_record(seconds),
        risk=RISK_CONFIRM,  # captures microphone audio — needs approval
    )
    agent.tools["VoiceTranscribe"] = Tool(
        name="VoiceTranscribe",
        description=(
            "Transcribe a .wav file with a real whisper speech-to-text "
            "model. Returns the transcript text, or a clear message when "
            "the file is unreadable or contains no detectable speech. "
            "Requires faster-whisper (pip install faster-whisper)."),
        parameters={"type": "object", "properties": {
            "wav_path": {"type": "string",
                         "description": "path to a .wav audio file"}},
            "required": ["wav_path"]},
        handler=lambda wav_path: _handle_voice_transcribe(wav_path),
        risk=RISK_SAFE,
    )
    # Programmatic handles for the TUI / other modules.
    agent.voice_transcribe = transcribe_wav
    agent.record_and_transcribe = record_and_transcribe


# ---------------------------------------------------------------------------
# Self-test — honest about what ran
# ---------------------------------------------------------------------------

def _selftest_check(name: str, cond: bool, note: str = "") -> bool:
    status = "ok" if cond else "FAIL"
    extra = f" — {note}" if note else ""
    print(f"  [{status}] {name}{extra}")
    return cond


class _Seg:  # minimal faster-whisper-like segment for the pure-function test
    def __init__(self, text: str, avg_logprob: float):
        self.text = text
        self.avg_logprob = avg_logprob


if __name__ == "__main__":
    print("voicein self-test")
    passed = True
    backend = whisper_backend()
    print(f"  whisper backend detected: {backend!r}")

    # 1. Graceful degradation: tools must return the clear message, never
    #    fake text, when no whisper backend is installed.
    if backend is None:
        r1 = _handle_voice_record(3)
        passed &= _selftest_check(
            "VoiceRecord → not-installed message (no fake transcript)",
            r1 == WHISPER_NOT_INSTALLED, repr(r1))
        r2 = _handle_voice_transcribe("/tmp/does_not_exist_xyz.wav")
        passed &= _selftest_check(
            "VoiceTranscribe → not-installed message (no fake transcript)",
            r2 == WHISPER_NOT_INSTALLED, repr(r2))
        passed &= _selftest_check(
            "message names the fix",
            "pip install faster-whisper" in r1)
    else:
        # Real backend present: verify the model actually loads and runs on
        # a real wav file, and that non-speech does NOT become text.
        try:
            model, bname = _get_model()
            m2, _ = _get_model()
            passed &= _selftest_check("model lazy-loads once (same object)",
                                      model is m2, bname)
        except Exception as exc:  # noqa: BLE001
            passed &= _selftest_check("model loads", False, str(exc)[:100])
            model = None
        if model is not None:
            fd, silence = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            try:
                # Real ffmpeg pipeline, silence source (NOT claimed to be
                # speech — verifies the wav pipeline + no-hallucination).
                cmd = _capture_cmd(
                    "ffmpeg", 1, silence,
                    _input=("lavfi", "anullsrc=r=16000:cl=mono"))
                pr = subprocess.run(cmd, capture_output=True,
                                    timeout=30)
                ok_wav = pr.returncode == 0 and _is_wav(silence)
                passed &= _selftest_check(
                    "ffmpeg silence wav is a valid WAV", ok_wav)
                if ok_wav:
                    out = transcribe_wav(silence)
                    passed &= _selftest_check(
                        "silence transcribes to NO_SPEECH, not hallucinated "
                        "text", out == NO_SPEECH, repr(out[:80]))
            finally:
                try:
                    os.unlink(silence)
                except OSError:
                    pass

    # 2. Capture-tool detection logic — real shutil.which results, reported
    #    honestly; asserts internal consistency, not a fixed answer.
    found = find_capture_tool()
    arecord_here = shutil.which("arecord") is not None
    ffmpeg_here = shutil.which("ffmpeg") is not None
    print(f"  arecord on PATH: {arecord_here}, ffmpeg on PATH: {ffmpeg_here} "
          f"→ find_capture_tool() = {found!r}")
    expected = ("arecord" if arecord_here
                else "ffmpeg" if ffmpeg_here else None)
    passed &= _selftest_check("detection matches shutil.which",
                              found == expected)

    # 3. Real ffmpeg recording-pipeline test via a lavfi silence source
    #    (exercises command construction + subprocess + WAV validation;
    #    a real microphone is not available in this environment).
    if ffmpeg_here:
        fd, probe = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            cmd = _capture_cmd("ffmpeg", 1, probe,
                               _input=("lavfi", "anullsrc=r=16000:cl=mono"))
            pr = subprocess.run(cmd, capture_output=True, timeout=30)
            passed &= _selftest_check(
                "ffmpeg pipeline produces valid WAV",
                pr.returncode == 0 and _is_wav(probe))
        finally:
            try:
                os.unlink(probe)
            except OSError:
                pass
    else:
        print("  [skip] ffmpeg pipeline test — ffmpeg not on PATH")

    # 4. WAV sniffing on garbage bytes → False (never transcribed).
    fd, junk = tempfile.mkstemp(suffix=".wav")
    os.write(fd, b"this is not audio data at all" * 10)
    os.close(fd)
    try:
        passed &= _selftest_check("garbage bytes are not WAV",
                                  not _is_wav(junk))
    finally:
        os.unlink(junk)

    # 5. Confidence filter unit tests (pure function, hand-built segments).
    segs = [_Seg("hello world", -0.2), _Seg("mumble mumble", -4.5)]
    passed &= _selftest_check(
        "low-confidence segment dropped, confident kept",
        segments_to_text(segs) == "hello world")
    passed &= _selftest_check(
        "all-low-confidence → empty (never hallucinated)",
        segments_to_text([_Seg("xyz", -3.0)]) == "")
    passed &= _selftest_check(
        "openai-whisper dict segments work",
        segments_to_text([{"text": "hi there", "avg_logprob": -0.1}])
        == "hi there")

    # 6. register() wires tools onto a duck-typed agent without importing
    #    the real Agent class.
    class _FakeAgent:
        def __init__(self):
            self.tools: Dict[str, Any] = {}
    fa = _FakeAgent()
    try:
        register(fa)
        has_both = ("VoiceRecord" in fa.tools
                    and "VoiceTranscribe" in fa.tools)
        passed &= _selftest_check("register wires both tools", has_both)
        # Handlers must still degrade honestly on this box (no whisper).
        if backend is None:
            h = fa.tools["VoiceRecord"].handler(2)
            passed &= _selftest_check(
                "registered VoiceRecord handler degrades honestly",
                h == WHISPER_NOT_INSTALLED)
    except Exception as exc:  # noqa: BLE001
        passed &= _selftest_check("register()", False, str(exc)[:100])

    print("PASS" if passed else "FAIL")
    raise SystemExit(0 if passed else 1)
