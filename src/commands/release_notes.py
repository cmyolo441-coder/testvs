"""Maintainer-only local changelog preparation and body verification commands."""
from __future__ import annotations

import argparse
from pathlib import Path

from ..maintenance.changelog import ChangelogError
from ..maintenance.release_notes import prepare_notes, verify_body
from .common import _print_json, _wants_json


def cmd_release_notes(args: argparse.Namespace) -> int:
    try:
        metadata = prepare_notes(Path(args.repo_root), args.version, Path(args.notes_file), stamp=args.stamp)
    except (ChangelogError, OSError) as exc:
        _print_json({'status': 'invalid_input', 'error': exc.code if isinstance(exc, ChangelogError) else 'file_error'})
        return 2
    payload = {'status': 'prepared', 'release_notes': metadata}
    if _wants_json(args):
        _print_json(payload)
    else:
        print(f"Release notes: {metadata['path']} ({metadata['sha256']})")
    return 0


def cmd_release_notes_verify(args: argparse.Namespace) -> int:
    try:
        matching = verify_body(Path(args.notes_file), Path(args.body_json))
    except (ChangelogError, OSError) as exc:
        _print_json({'status': 'invalid_input', 'error': exc.code if isinstance(exc, ChangelogError) else 'file_error'})
        return 2
    verdict = 'matching' if matching else 'mismatched'
    if _wants_json(args):
        _print_json({'verification': verdict})
    else:
        print(f'Release body: {verdict}')
    return 0 if matching else 1


def add_release_notes_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    for name, stamp in (('prepare', True), ('notes', False)):
        command = sub.add_parser(name, help='Prepare local curated notes; no tag, build or publication.')
        command.add_argument('--version', required=True)
        command.add_argument('--repo-root', required=True)
        command.add_argument('--notes-file', required=True)
        command.add_argument('--json', action='store_true')
        command.set_defaults(func=cmd_release_notes, stamp=stamp)
    verify = sub.add_parser('notes-verify', help='Compare a decoded release body without remote mutation.')
    verify.add_argument('--notes-file', required=True)
    verify.add_argument('--body-json', required=True)
    verify.add_argument('--json', action='store_true')
    verify.set_defaults(func=cmd_release_notes_verify)
