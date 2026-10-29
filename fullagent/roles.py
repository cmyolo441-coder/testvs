"""Shared role capability catalog for planning and delegated subagents.

The catalog defines tool capabilities and specialist intent only; lifecycle,
concurrency, path scopes, and approval policy are enforced by
``fullagent.subagents`` and the host agent rather than by role labels or
prompt text.
"""
from __future__ import annotations

ROLES: dict[str, dict] = {
    "researcher": {
        "tools": ("read_file", "list_dir", "file_info", "search_files",
                  "glob_files", "web_search", "web_fetch"),
        "writes": False,
    },
    "coder": {
        "tools": ("read_file", "list_dir", "file_info", "search_files",
                  "glob_files", "write_file", "edit_file", "create_directory"),
        "writes": True,
    },
    "tester": {
        "tools": ("read_file", "list_dir", "file_info", "search_files",
                  "glob_files"),
        "writes": False,
    },
    "reviewer": {
        "tools": ("read_file", "list_dir", "file_info", "search_files",
                  "glob_files"),
        "writes": False,
    },
    "analyst": {
        "tools": ("read_file", "list_dir", "file_info", "search_files",
                  "glob_files", "web_search", "web_fetch"),
        "writes": False,
    },
    "architect": {
        "tools": ("read_file", "list_dir", "file_info", "search_files",
                  "glob_files", "write_file", "create_directory"),
        "writes": True,
    },
    "debugger": {
        "tools": ("read_file", "list_dir", "file_info", "search_files",
                  "glob_files"),
        "writes": False,
    },
    "optimizer": {
        "tools": ("read_file", "list_dir", "file_info", "search_files",
                  "glob_files"),
        "writes": False,
    },
    "refactorer": {
        "tools": ("read_file", "list_dir", "file_info", "search_files",
                  "glob_files", "write_file", "edit_file", "create_directory"),
        "writes": True,
    },
    "documenter": {
        "tools": ("read_file", "list_dir", "file_info", "search_files",
                  "glob_files", "write_file", "edit_file", "create_directory"),
        "writes": True,
    },
    "devops": {
        "tools": ("read_file", "list_dir", "file_info", "search_files",
                  "glob_files", "write_file", "edit_file", "create_directory"),
        "writes": True,
    },
    "integrator": {
        "tools": ("read_file", "list_dir", "file_info", "search_files",
                  "glob_files", "write_file", "edit_file", "create_directory"),
        "writes": True,
    },
    "planner": {
        "tools": ("read_file", "list_dir", "file_info", "search_files",
                  "glob_files"),
        "writes": False,
    },
    "security": {
        "tools": ("read_file", "list_dir", "file_info", "search_files",
                  "glob_files"),
        "writes": False,
    },
}

ROLE_LABELS: dict[str, str] = {
    "researcher": "Codebase Researcher",
    "coder": "Scoped Coder",
    "tester": "Test Analyst",
    "reviewer": "Code Reviewer",
    "analyst": "Evidence Analyst",
    "architect": "Software Architect",
    "debugger": "Root-Cause Debugger",
    "optimizer": "Performance Analyst",
    "refactorer": "Refactoring Specialist",
    "documenter": "Technical Writer",
    "devops": "Build & DevOps",
    "integrator": "Integration Analyst",
    "planner": "Task Planner",
    "security": "Security Reviewer",
}

ROLE_GUIDANCE: dict[str, str] = {
    "researcher": "Trace the relevant implementation and report exact files, symbols, and verified behavior.",
    "coder": "Implement the assigned change minimally inside the approved paths; inspect conventions and verify the result with available read-only evidence.",
    "tester": "Map behavior to existing tests, edge cases, and missing coverage. You cannot execute shell commands; never claim tests ran.",
    "reviewer": "Look for concrete correctness, regression, compatibility, and security defects; prioritize actionable findings with file and line evidence.",
    "analyst": "Answer the assigned question by separating direct evidence, inference, and unresolved uncertainty.",
    "architect": "Design a focused, maintainable change; compare meaningful trade-offs and identify migration or compatibility risks.",
    "debugger": "Trace a failure from observable symptoms to a specific, evidence-backed root cause and smallest repair.",
    "optimizer": "Find measurable performance opportunities; explain the bottleneck evidence and preserve behavior.",
    "refactorer": "Improve structure only within the approved paths; preserve public behavior and call out any unverified assumptions.",
    "documenter": "Write concise, accurate documentation grounded in the inspected implementation and existing project voice.",
    "devops": "Inspect build, packaging, and deployment configuration; change only approved paths and identify operational impact.",
    "integrator": "Trace interactions across components and report interface mismatches, duplicate work, and integration risks.",
    "planner": "Return an ordered, dependency-aware plan with parallelizable work separated from sequential decisions; do not claim execution.",
    "security": "Audit trust boundaries, authorization, secret handling, injection, path handling, and error exposure; report only evidence-backed risks and practical mitigations.",
}

DEFAULT_ROLE = "coder"


def role_label(role: str) -> str:
    """Return the human-readable specialist title for a known role."""
    return ROLE_LABELS.get(role, role.replace("_", " ").title())


def role_guidance(role: str) -> str:
    """Return focused instructions without changing the role's capabilities."""
    return ROLE_GUIDANCE.get(role, ROLE_GUIDANCE[DEFAULT_ROLE])
