"""Closed local probes and a killable, bounded caller-injected evaluation seam.

This is not a security sandbox for hostile Python. Audit only trusted source
trees and trusted adapters. No shell, provider client, or discovered commands.
"""

from __future__ import annotations

import ast
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
import multiprocessing
from multiprocessing.connection import Connection
import pickle
from pathlib import Path
import re
import sys
import time
from typing import TypedDict

from ..catalogs.documentation_claims import DocumentationClaim
from .documentation_claims_model import ModelAdapter, ModelRequest, ModelRun, model_provenance, model_result


SOURCE_BYTE_CAP = 512 * 1024
OUTPUT_BYTE_CAP = 64 * 1024
MODEL_INPUT_BYTE_CAP = 16 * 1024
PROBE_MODES = {
    "release-checklist-observed": "cli_probe",
    "reported-rate-schema": "schema_assertion",
    "release-checklist-symbol": "symbol_check",
    "empty-reported-rate": "fixture_behavior",
    "roles-equality": "render_equality",
    "evidence-language": "model_assisted",
}


class ProbeResult(TypedDict, total=False):
    fact: bool | str
    error: str
    model_run: ModelRun
    model_started: bool
    started: ModelRun


def bounded_read(path: Path, cap: int = SOURCE_BYTE_CAP) -> str:
    with path.open("rb") as stream:
        data = stream.read(cap + 1)
    if len(data) > cap:
        raise ValueError("input_limit")
    return data.decode("utf-8")


def checked_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("path_outside_root")
    return path


def _load_module(root: Path, relative: str, name: str):
    path = checked_path(root, relative)
    source = bounded_read(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None:
        raise ImportError(name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    # Compile current bytes, not a possibly same-size/same-second .pyc fixture.
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


class _BoundedOutput(io.StringIO):
    def __init__(self):
        super().__init__()
        self.bytes_written: int = 0

    def write(self, text: str) -> int:
        self.bytes_written += len(text.encode("utf-8"))
        if self.bytes_written > OUTPUT_BYTE_CAP:
            raise ValueError("output_limit")
        return super().write(text)


def _probe(root: Path, claim: DocumentationClaim) -> bool | str:
    if claim.probe in {"release-checklist-observed", "release-checklist-symbol"}:
        module = _load_module(root, "src/maintenance/release.py", "omh.maintenance.release")
        if claim.probe == "release-checklist-symbol":
            return callable(getattr(module, "release_readiness_checklist"))
        _ = _load_module(root, "src/commands/release.py", "omh.commands.release")
        # Load after the selected source modules, so the real parser binds to
        # this fixture's handler rather than a parent-process cached import.
        from ..commands.main import main
        output = _BoundedOutput()
        with redirect_stdout(output):
            code = main(["release", "checklist", "--json"])
        if code != 0:
            raise RuntimeError("probe_exit")
        return json.loads(output.getvalue())["observed"]
    if claim.probe in {"reported-rate-schema", "empty-reported-rate"}:
        module = _load_module(root, "src/quality/reported_rate.py", "omh.quality.reported_rate")
        if claim.probe == "reported-rate-schema":
            return module.reported_rate(numerator=0, denominator=0, numerator_of=("supported",),
                                        denominator_of="observed claims").to_payload()["schema_version"]
        rate = module.reported_rate(numerator=0, denominator=0, numerator_of=("supported",),
                                    denominator_of="observed claims").to_payload()
        return rate["percent"] is None and rate["basis"] == "no_observations"
    if claim.probe == "roles-equality":
        module = _load_module(root, "src/catalogs/roles.py", "omh.catalogs.roles")
        # Match the public generated check for CRLF checkouts, without changing
        # the byte cap, other whitespace, or source/model evidence reads.
        current = bounded_read(checked_path(root, "docs/ROLES.md")).replace("\r\n", "\n")
        return current == module.roles_reference_markdown()
    raise ValueError("unsupported_probe")


def _worker(connection: Connection, root: Path, claim: DocumentationClaim, adapter: ModelAdapter | None) -> None:
    # Keep unexpected bootstrap tracebacks out of the caller's stderr too.
    # The parent classifies abnormal exits; no raw child diagnostics escape.
    sys.stdout = _BoundedOutput()
    sys.stderr = _BoundedOutput()
    with connection:
        try:
            if PROBE_MODES.get(claim.probe) != claim.mode:
                raise ValueError("unsupported_probe")
            for page in claim.pages:
                _ = bounded_read(checked_path(root, page))
            for anchor in claim.anchors:
                tree = ast.parse(bounded_read(checked_path(root, anchor.path)))
                names = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
                names.update(target.id for node in tree.body if isinstance(node, ast.Assign)
                             for target in node.targets if isinstance(target, ast.Name))
                if anchor.symbol not in names:
                    raise LookupError("anchor_unavailable")
            if claim.mode == "model_assisted":
                if adapter is None:
                    raise ValueError("model_unavailable")
                evidence = bounded_read(checked_path(root, claim.pages[0]), MODEL_INPUT_BYTE_CAP)
                binding = json.dumps([claim.claim_id, claim.question, claim.invariant, claim.expected_fact, evidence],
                                     ensure_ascii=True, separators=(",", ":"))
                request = ModelRequest(claim.claim_id, claim.question, claim.invariant, claim.expected_fact,
                                       evidence, hashlib.sha256(binding.encode()).hexdigest())
                provenance = model_provenance(adapter, request)
                connection.send_bytes(json.dumps({"started": provenance}, sort_keys=True).encode())
                result = {"model_run": model_result(adapter, request)}
            else:
                fact = _probe(root, claim)
                if type(fact) is not type(claim.expected_fact):
                    raise ValueError("fact_shape")
                # Unexpected strings never become a raw output channel.
                if isinstance(fact, str) and (len(fact) > 96 or not re.fullmatch(r"[a-z_]+/v[0-9]+", fact)):
                    raise ValueError("fact_shape")
                result = {"fact": fact}
            connection.send_bytes(json.dumps(result, sort_keys=True).encode())
        except (OSError, ValueError, TypeError, LookupError, ImportError, AttributeError, RuntimeError, SyntaxError):
            # Raw exception text can contain secrets. Unknown crashes likewise
            # close the pipe and are classified by the parent, never supported.
            connection.send_bytes(b'{"error":"probe_unavailable"}')


def run_bounded_probe(root: Path, claim: DocumentationClaim, timeout: float, adapter: ModelAdapter | None = None) -> ProbeResult:
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(sender, root, claim, adapter), daemon=True)
    deadline = time.monotonic() + timeout
    started: ProbeResult = {}
    try:
        process.start()
        sender.close()
        if not receiver.poll(max(0, deadline - time.monotonic())):
            return {"error": "timeout"}
        result: ProbeResult = json.loads(receiver.recv_bytes(OUTPUT_BYTE_CAP))
        if "started" in result:
            started = {"model_run": result["started"], "model_started": True}
            if not receiver.poll(max(0, deadline - time.monotonic())):
                return {**started, "error": "timeout"}
            result = json.loads(receiver.recv_bytes(OUTPUT_BYTE_CAP))
        process.join(max(0, deadline - time.monotonic()))
        if process.is_alive():
            return {**started, "error": "timeout"}
        if process.exitcode != 0:
            return {**started, "error": "probe_unavailable"}
        combined: ProbeResult = {**started, **result}
        return combined
    except (OSError, EOFError, ValueError, TypeError, RuntimeError, AttributeError, pickle.PicklingError):
        return {**started, "error": "probe_unavailable"}
    finally:
        sender.close()
        receiver.close()
        if process.pid is not None:
            if process.is_alive():
                process.kill()
            process.join()
            process.close()
