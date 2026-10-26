"""OMH command line: one-shot mode, REPL and slash commands."""

from __future__ import annotations

import argparse
import difflib
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import TAGLINE, __version__
from .agent import Agent
from .config import Config, load_config, omh_home
from .errors import ConfigError, OmhError
from .providers import build_providers
from .router import ModelRouter
from .session import Session, SessionStore, store_for
from .skills import Skill, load_skills
from .theme import (
    PALETTES,
    Theme,
    banner,
    render_markdown,
    shorten_path,
    table,
    terminal_width,
)
from .tools import default_registry
from .tools.base import ToolRegistry
from .ui import Console, Prompt, Spinner, TurnRenderer, ask_approval

SLASH_COMMANDS = {
    "/help": "this help",
    "/model": "switch model, or `auto` to load balance",
    "/models": "show the model pool",
    "/provider": "restrict to one provider, or `auto`",
    "/strategy": "balanced | round_robin | priority",
    "/balance": "live routing stats",
    "/tools": "tools and their approval",
    "/skill": "load a skill, or `off`",
    "/theme": "hermes | amber | mono",
    "/approval": "ask | auto | deny",
    "/new": "start a fresh session",
    "/sessions": "list saved sessions",
    "/resume": "resume a session by id",
    "/clear": "forget this session's messages",
    "/cost": "session tokens and spend",
    "/doctor": "configuration and provider health",
    "/init": "write a starter config file",
    "/save": "save the session now",
    "/version": "print the version",
    "/exit": "leave omh",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omh",
        description="OMH — a fast terminal AI agent with a load-balanced model pool.",
        epilog=(
            "examples:\n"
            "  omh                       start the REPL\n"
            "  omh -p 'fix the failing test'\n"
            "  omh --offline             try the harness without keys\n"
            "  omh --model auto --strategy balanced\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("prompt", nargs="*", help="prompt to run once, then exit")
    parser.add_argument("-p", "--prompt-text", metavar="TEXT", help="prompt to run once")
    parser.add_argument("--offline", action="store_true", help="use the built-in mock provider")
    parser.add_argument("-y", "--yes", action="store_true", help="approve tool writes and shell commands")
    parser.add_argument("--no-tools", action="store_true", help="disable tools for this run")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI colour")
    parser.add_argument("-q", "--quiet", action="store_true", help="do not print the banner")
    parser.add_argument("--config", help="path to a config JSON file")
    parser.add_argument("--model", help="model id, alias, or `auto` for the whole pool")
    parser.add_argument("--provider", help="only use this provider")
    parser.add_argument("--strategy", choices=["balanced", "round_robin", "priority"])
    parser.add_argument("--approval", choices=["ask", "auto", "deny"])
    parser.add_argument("--theme", choices=sorted(PALETTES))
    parser.add_argument("--max-steps", type=int, help="maximum model/tool round trips")
    parser.add_argument("--session", help="resume a session id")
    parser.add_argument("--resume", action="store_true", help="resume the latest session")
    parser.add_argument("--list-models", action="store_true", help="print the model pool and exit")
    parser.add_argument("--list-sessions", action="store_true", help="print saved sessions and exit")
    parser.add_argument("--doctor", action="store_true", help="print health and exit")
    parser.add_argument("--init", action="store_true", help="write a starter config and exit")
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    return parser


def _suggest(config: Config, name: str) -> str:
    names = [profile.name for profile in config.models] + list(config.providers)
    matches = difflib.get_close_matches(name, names, n=3, cutoff=0.4)
    return ("did you mean: " + ", ".join(matches)) if matches else "run `/models` to see the pool"


def _apply_args(config: Config, args: argparse.Namespace) -> Optional[str]:
    """Apply CLI overrides; return an error message when a value is unknown."""
    if args.model:
        if args.model.lower() in {"auto", "*"}:
            config.model = ""
        elif args.model in config.providers and config.resolve_model_name(args.model) is None:
            config.provider = args.model
            config.model = ""
        elif config.resolve_model_name(args.model) is None:
            return f"unknown model {args.model!r}; " + _suggest(config, args.model)
        else:
            config.model = args.model
    if args.provider:
        if args.provider not in config.providers:
            return f"unknown provider {args.provider!r}; " + _suggest(config, args.provider)
        config.provider = args.provider
    if args.strategy:
        config.router.strategy = args.strategy
    if args.approval:
        config.approval_mode = args.approval
    if args.theme:
        config.theme = args.theme
    if args.no_tools:
        config.tools_enabled = False
    if args.max_steps:
        config.max_steps = max(1, args.max_steps)
    if args.yes:
        config.approval_mode = "auto"
    return None


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        print(f"omh {__version__} · {TAGLINE}")
        return 0

    try:
        config = load_config(args.config, offline=args.offline)
    except ConfigError as exc:
        print(f"omh: {exc.message}", file=sys.stderr)
        if exc.hint:
            print(f"     {exc.hint}", file=sys.stderr)
        return 2

    error = _apply_args(config, args)
    if error:
        print(f"omh: {error}", file=sys.stderr)
        return 2
    if args.offline:
        # The demo flag wins over model/provider overrides.
        config.model = "mock-1"
        config.provider = "mock"

    theme = Theme(config.theme, enabled=False if args.no_color else None)

    if args.init:
        target = omh_home() / "config.json"
        if target.exists():
            print(theme.warn(f"{target} already exists"))
            return 1
        config.save(target)
        print(theme.ok(f"wrote {target}"))
        return 0
    if args.list_models:
        print(_models_table(config, theme))
        return 0
    if args.list_sessions:
        print(_sessions_table(store_for(config), theme))
        return 0

    interactive = sys.stdin.isatty()
    try:
        app = App(
            config,
            interactive=interactive,
            no_color=args.no_color,
            session_id=args.session,
            resume=args.resume,
        )
    except OmhError as exc:
        print(theme.err("omh: " + exc.message), file=sys.stderr)
        if exc.hint:
            print(theme.muted("     " + exc.hint), file=sys.stderr)
        return 1

    if args.doctor:
        return app.doctor()
    if not args.quiet and interactive:
        app.welcome()

    prompt_text = (args.prompt_text or " ".join(args.prompt)).strip()
    if prompt_text:
        return app.run_once(prompt_text)
    if not interactive:
        data = sys.stdin.read().strip()
        if not data:
            parser.print_help()
            return 0
        return app.run_once(data)
    return app.repl()


class App:
    def __init__(
        self,
        config: Config,
        *,
        interactive: bool,
        no_color: bool = False,
        session_id: Optional[str] = None,
        resume: bool = False,
    ) -> None:
        self.config = config
        self.interactive = interactive
        self.theme = Theme(config.theme, enabled=False if no_color else None)
        self.console = Console(self.theme)
        self.spinner = Spinner(self.theme, enabled=interactive)
        self.prompt = Prompt(self.theme, config.history_file) if interactive else None
        self.store = store_for(config)
        self.session = self._open_session(session_id, resume)
        self.skills: Dict[str, Skill] = load_skills(config.workspace)
        self.skill: Optional[Skill] = (
            self.skills.get(self.session.skill) if self.session.skill else None
        )
        self.router = ModelRouter(config, build_providers(config))
        self.registry = self._build_registry()
        self._refresh_completion()
        self._session_notice = bool(session_id) or (
            resume and bool(self.session.messages)
        )

    # -- setup -----------------------------------------------------------
    def _open_session(self, session_id: Optional[str], resume: bool) -> Session:
        if session_id:
            return self.store.load(session_id)
        if resume:
            latest = self.store.latest()
            if latest is not None:
                return latest
        return self.store.new(model=self.config.model)

    def _build_registry(self) -> ToolRegistry:
        return default_registry(
            self.config.workspace,
            max_output=self.config.max_tool_output,
            approval_mode=self.config.approval_mode,
        )

    def _refresh_completion(self) -> None:
        if self.prompt is None:
            return
        words = list(SLASH_COMMANDS)
        words += [profile.name for profile in self.config.models]
        words += [profile.id for profile in self.config.models]
        words += list(self.config.providers)
        words += ["auto", "off", *PALETTES]
        self.prompt.set_words(words)

    def _model_label(self) -> str:
        if self.config.model.strip().lower() in {"", "auto", "*"}:
            lanes = len(self.config.enabled_profiles())
            return f"auto · {self.config.router.strategy} · {lanes} lane{'s' if lanes != 1 else ''}"
        return self.config.model

    def welcome(self) -> None:
        enabled = self.config.enabled_profiles()
        self.console.line(
            banner(
                self.theme,
                version=__version__,
                model_label=self._model_label(),
                workspace=self.config.workspace,
                strategy=self.config.router.strategy,
                profile_count=len(enabled),
                provider_count=len({profile.provider for profile in enabled}),
            )
        )
        self.console.line()
        if self._session_notice:
            title = self.session.title or "(new conversation)"
            self.console.line(
                self.theme.muted(f"  resumed {self.session.id} · {title}")
            )
        else:
            self.console.line(
                self.theme.muted(
                    f"  session {self.session.id} · saved in {shorten_path(self.config.session_dir, 40)}"
                )
            )
        if not enabled:
            self.console.line()
            self.console.line(self.theme.warn("  no provider credentials found"))
            for hint in self.config.missing_credentials()[:3]:
                self.console.line(self.theme.muted("    · " + hint))
            if any(
                not provider.requires_key and provider.kind != "mock"
                for provider in self.config.providers.values()
            ):
                self.console.line(
                    self.theme.muted("    · local server? `/provider ollama`")
                )
            self.console.line(
                self.theme.muted("    · run `omh --offline` for a zero-key demo")
            )
        self.console.line()

    # -- one turn --------------------------------------------------------
    def _make_agent(self) -> Agent:
        return Agent(
            self.config,
            self.router,
            self.registry,
            self.session,
            skill=self.skill,
            approver=self._approve if self.interactive else None,
        )

    def _approve(self, spec: Any, arguments: Dict[str, Any], summary: str) -> bool:
        self.spinner.stop()
        return ask_approval(self.console, spec, arguments, summary)

    def _dispatch(self, renderer: TurnRenderer, event: Any) -> Optional[str]:
        kind = event.kind
        if kind == "text":
            renderer.text(event.text, str(event.get("model") or ""))
        elif kind == "info":
            renderer.info(str(event.get("message") or ""))
        elif kind == "tool_start":
            call = event.get("call")
            renderer.tool_start(
                getattr(call, "name", "tool"),
                str(event.get("summary") or ""),
                str(event.get("model") or ""),
            )
        elif kind == "tool_result":
            call = event.get("call")
            renderer.tool_result(getattr(call, "name", "tool"), event.get("result"))
        elif kind == "usage":
            renderer.usage(event.data)
        elif kind == "error":
            renderer.error(event.get("error"))
        elif kind == "done":
            return str(event.get("reason") or "complete")
        return None


    def run_once(self, text: str) -> int:
        agent = self._make_agent()
        renderer = TurnRenderer(self.console, self.spinner, show_usage=True)
        renderer.begin()
        outcome = "error"
        payload: Dict[str, Any] = {}
        try:
            for event in agent.run(text):
                reason = self._dispatch(renderer, event)
                if reason:
                    outcome = reason
                    payload = event.data
        except KeyboardInterrupt:
            agent.cancel()
            outcome = "cancelled"
            payload = {"reason": "cancelled"}
        except OmhError as exc:
            renderer.error(exc)
            outcome = "error"
        except Exception as exc:
            if os.environ.get("OMH_DEBUG"):
                raise
            renderer.error(OmhError(f"{type(exc).__name__}: {exc}"))
            outcome = "error"
        finally:
            self.spinner.stop()
            renderer.done(payload or {"reason": outcome})
            self._save()
        return 0 if outcome in {"complete", "cancelled", "max_steps"} else 1

    def _save(self) -> None:
        try:
            self.store.save(self.session)
        except OSError as exc:
            self.console.error(self.theme.err(f"  could not save session: {exc}"))

    # -- repl ------------------------------------------------------------
    def repl(self) -> int:
        while True:
            try:
                line = self.prompt.ask() if self.prompt else None
            except KeyboardInterrupt:
                self.console.line(self.theme.muted("  (ctrl-c) type /exit to leave"))
                continue
            if line is None:
                self.console.line()
                break
            text = line.strip()
            if not text:
                continue
            if text.startswith("!"):
                self._shell(text[1:].strip())
                continue
            if text.startswith("/"):
                if not self._slash(text):
                    break
                continue
            self.run_once(text)
        self.console.line(self.theme.muted("  bye"))
        return 0

    def _shell(self, command: str) -> None:
        if not command:
            return
        try:
            completed = subprocess.run(command, shell=True, cwd=self.config.workspace)
        except OSError as exc:
            self.console.line(self.theme.err(f"  {exc}"))
            return
        if completed.returncode:
            self.console.line(self.theme.muted(f"  exit {completed.returncode}"))


    # -- slash commands --------------------------------------------------
    def _slash(self, text: str) -> bool:
        parts = text.split()
        command = parts[0].lower()
        args = parts[1:]
        if command in {"/exit", "/quit", "/q"}:
            return False
        handlers = {
            "/help": lambda: self._help(),
            "/model": lambda: self._model(args),
            "/models": lambda: self.console.line(_models_table(self.config, self.theme)),
            "/provider": lambda: self._provider(args),
            "/strategy": lambda: self._strategy(args),
            "/balance": lambda: self._balance(),
            "/tools": lambda: self._tools(),
            "/skill": lambda: self._skill_command(args),
            "/theme": lambda: self._theme_command(args),
            "/approval": lambda: self._approval(args),
            "/new": lambda: self._new(),
            "/sessions": lambda: self.console.line(_sessions_table(self.store, self.theme)),
            "/resume": lambda: self._resume(args),
            "/clear": lambda: self._clear(),
            "/cost": lambda: self._cost(),
            "/doctor": lambda: self.doctor(),
            "/init": lambda: self._init(),
            "/save": lambda: self._save_command(),
            "/version": lambda: self.console.line(
                self.theme.text(f"omh {__version__} · {TAGLINE}")
            ),
        }
        handler = handlers.get(command)
        if handler is None:
            matches = difflib.get_close_matches(command, list(handlers), n=1)
            hint = f"did you mean {matches[0]}?" if matches else "try /help"
            self.console.line(
                self.theme.err(f"  unknown command {command}")
                + self.theme.muted(f"  {hint}")
            )
            return True
        handler()
        return True

    def _rebuild(self) -> None:
        self.router = ModelRouter(self.config, build_providers(self.config))
        self.registry = self._build_registry()
        self._refresh_completion()

    def _help(self) -> None:
        t = self.theme
        lines = ["", t.title("  Commands")]
        for command, description in SLASH_COMMANDS.items():
            lines.append(t.accent(f"  {command:<11}") + t.muted(description))
        lines += [
            "",
            t.title("  Keys"),
            t.muted("  !<cmd>      run a shell command in the workspace"),
            t.muted("  ctrl-c      cancel the current turn / clear the input"),
            t.muted("  ctrl-d      leave omh"),
            "",
            t.title("  Examples"),
            t.muted("  omh --offline"),
            t.muted("  omh -p 'summarise this repo' --yes"),
            t.muted("  /model sonnet-4   /theme amber   /balance"),
        ]
        self.console.line("\n".join(lines))

    def _model(self, args: List[str]) -> None:
        t = self.theme
        if not args or args[0].lower() == "status":
            self.console.line(
                t.muted("  model  ") + t.accent(self._model_label())
                + t.muted("   provider  ") + t.accent(self.config.provider or "auto")
            )
            self.console.line(t.muted("  use /model <name>, /model auto, or /models"))
            return
        name = args[0]
        if name.lower() in {"list", "ls"}:
            self.console.line(_models_table(self.config, self.theme))
            return
        if name.lower() in {"auto", "*"}:
            self.config.model = ""
            self.config.provider = ""
            self._rebuild()
            self.console.line(t.ok("  model → auto") + t.muted(f"  {self._model_label()}"))
            return
        profile = self.config.resolve_model_name(name)
        if profile is not None:
            self.config.model = profile.id
            self.config.provider = ""
            self._rebuild()
            self.console.line(
                t.ok(f"  model → {profile.name}") + t.muted(f"  ({profile.provider})")
            )
            return
        if name in self.config.providers:
            self.config.provider = name
            self.config.model = ""
            self._rebuild()
            self.console.line(t.ok(f"  provider → {name}"))
            return
        self.console.line(
            t.err(f"  unknown model {name!r}")
            + t.muted(f"  {_suggest(self.config, name)}")
        )


    def _provider(self, args: List[str]) -> None:
        t = self.theme
        if not args:
            providers = ", ".join(self.config.providers)
            self.console.line(t.muted("  providers  ") + t.text(providers))
            self.console.line(t.muted(f"  current    {self.config.provider or 'auto'}"))
            return
        name = args[0].lower()
        if name in {"auto", "*", "off"}:
            self.config.provider = ""
        elif name in self.config.providers:
            self.config.provider = name
        else:
            self.console.line(t.err(f"  unknown provider {name!r}"))
            return
        self._rebuild()
        self.console.line(t.ok(f"  provider → {self.config.provider or 'auto'}"))

    def _strategy(self, args: List[str]) -> None:
        t = self.theme
        if not args:
            self.console.line(
                t.muted("  strategy  ") + t.accent(self.config.router.strategy)
                + t.muted("   hedge  ")
                + t.text(f"{self.config.router.hedge_delay_s:g}s")
            )
            self.console.line(t.muted("  balanced | round_robin | priority"))
            return
        strategy = args[0].lower()
        if strategy not in {"balanced", "round_robin", "priority"}:
            self.console.line(
                t.err("  strategy must be balanced, round_robin or priority")
            )
            return
        self.config.router.strategy = strategy
        self._rebuild()
        self.console.line(t.ok(f"  strategy → {strategy}"))

    def _balance(self) -> None:
        t = self.theme
        stats = self.router.stats()
        self.console.line(
            t.muted("  strategy  ") + t.accent(self.config.router.strategy)
            + t.muted("   attempts  ") + t.text(str(self.config.router.max_attempts))
            + t.muted("   hedge  ")
            + t.text(f"{self.config.router.hedge_delay_s:g}s")
        )
        if not stats:
            self.console.line(t.warn("  no routable models right now"))
            for hint in self.config.missing_credentials()[:3]:
                self.console.line(t.muted("    · " + hint))
            return
        rows = [
            [
                row["model"],
                row["provider"],
                f"{row['weight']:g}",
                str(row["requests"]),
                str(row["failures"]),
                f"{row['latency_ms']}ms",
                f"{row['tokens']:,}",
                row["health"],
            ]
            for row in stats
        ]
        self.console.line()
        self.console.line(
            table(
                t,
                ["model", "provider", "wt", "req", "fail", "ttft", "tokens", "health"],
                rows,
                max_width=terminal_width() - 4,
            )
        )
        for row in stats:
            if row["last_error"] and row["health"] == "cooling":
                self.console.line(
                    t.muted(f"  {row['model']}: {row['last_error'][:100]}")
                )

    def _tools(self) -> None:
        t = self.theme
        rows = []
        for name in self.registry.names():
            spec = self.registry.specs[name]
            rows.append(
                [
                    name,
                    "ask" if spec.requires_approval else "auto",
                    spec.description.split(". ")[0],
                ]
            )
        self.console.line()
        self.console.line(
            table(
                t,
                ["tool", "approval", "what it does"],
                rows,
                max_width=terminal_width() - 4,
            )
        )


    def _skill_command(self, args: List[str]) -> None:
        t = self.theme
        if not args:
            if not self.skills:
                self.console.line(t.warn("  no skills found"))
                self.console.line(
                    t.muted("  add markdown files to ./skills or ~/.omh/skills")
                )
                return
            rows = [
                [
                    name,
                    "on" if self.skill and self.skill.name == name else "",
                    skill.description,
                ]
                for name, skill in sorted(self.skills.items())
            ]
            self.console.line()
            self.console.line(
                table(
                    t,
                    ["skill", "active", "description"],
                    rows,
                    max_width=terminal_width() - 4,
                )
            )
            return
        name = args[0].lower()
        if name in {"off", "none", "clear"}:
            self.skill = None
            self.session.skill = ""
            self.console.line(t.ok("  skill cleared"))
            return
        skill = self.skills.get(name)
        if skill is None:
            matches = difflib.get_close_matches(name, list(self.skills), n=2)
            hint = (
                ("did you mean " + ", ".join(matches) + "?")
                if matches
                else "run /skill to list"
            )
            self.console.line(t.err(f"  unknown skill {name!r}") + t.muted(f"  {hint}"))
            return
        self.skill = skill
        self.session.skill = skill.name
        self.console.line(
            t.ok(f"  skill → {skill.name}")
            + t.muted(f"  {skill.description or shorten_path(skill.path, 50)}")
        )

    def _theme_command(self, args: List[str]) -> None:
        t = self.theme
        if not args:
            self.console.line(
                t.muted("  theme  ") + t.accent(self.theme.palette.name)
                + t.muted("   available  ") + t.text(", ".join(PALETTES))
            )
            return
        name = args[0].lower()
        if name not in PALETTES:
            self.console.line(
                t.err(f"  unknown theme {name!r}; choose {', '.join(PALETTES)}")
            )
            return
        self.config.theme = name
        self.theme = Theme(name, enabled=self.theme.enabled)
        self.console.theme = self.theme
        self.spinner.theme = self.theme
        if self.prompt is not None:
            self.prompt.theme = self.theme
        self.console.line(
            render_markdown(f"**{name}** · `accent` · plain text · **bold**", self.theme)
        )

    def _approval(self, args: List[str]) -> None:
        t = self.theme
        if not args:
            self.console.line(
                t.muted("  approval  ") + t.accent(self.config.approval_mode)
            )
            self.console.line(t.muted("  ask (default) · auto · deny"))
            return
        mode = args[0].lower()
        if mode not in {"ask", "auto", "deny"}:
            self.console.line(t.err("  approval must be ask, auto or deny"))
            return
        self.config.approval_mode = mode
        self.registry.context.approval_mode = mode
        self.console.line(t.ok(f"  approval → {mode}"))

    def _new(self) -> None:
        self._save()
        self.session = self.store.new(model=self.config.model)
        self.skill = None
        self.console.line(
            self.theme.ok(f"  new session {self.session.id}")
            + self.theme.muted("  (/resume to go back)")
        )

    def _resume(self, args: List[str]) -> None:
        t = self.theme
        wanted = args[0] if args else "last"
        try:
            session = (
                self.store.latest()
                if wanted in {"last", "-"}
                else self.store.load(wanted)
            )
        except OmhError as exc:
            self.console.line(t.err("  " + exc.message))
            return
        if session is None:
            self.console.line(t.warn("  no sessions saved yet"))
            return
        self._save()
        self.session = session
        self.skills = load_skills(self.config.workspace)
        self.skill = self.skills.get(session.skill) if session.skill else None
        self.console.line(
            t.ok(f"  resumed {session.id}")
            + t.muted(f"  {session.title or '(empty)'} · {len(session.messages)} messages")
        )


    def _clear(self) -> None:
        self.session.clear()
        self._save()
        self.console.line(self.theme.ok("  session cleared"))

    def _cost(self) -> None:
        t = self.theme
        usage = self.session.usage
        self.console.line(
            t.muted("  tokens  ")
            + t.text(
                f"{usage.prompt_tokens:,} in · {usage.completion_tokens:,} out · {usage.total:,} total"
            )
        )
        self.console.line(
            t.muted("  spend   ")
            + t.text(f"${self.session.cost:.4f}")
            + t.muted("   model  ")
            + t.text(self.session.model or "auto")
        )
        self.console.line(
            t.muted("  session ")
            + t.text(self.session.id)
            + t.muted(f"   messages  {len(self.session.messages)}")
        )

    def _init(self) -> None:
        target = omh_home() / "config.json"
        if target.exists():
            self.console.line(self.theme.warn(f"  {target} already exists"))
            return
        self.config.save(target)
        self.console.line(self.theme.ok(f"  wrote {target}"))

    def _save_command(self) -> None:
        path = self.store.save(self.session)
        self.console.line(self.theme.muted(f"  saved {path}"))

    def doctor(self) -> int:
        t = self.theme
        providers = build_providers(self.config)
        config_paths = [
            path
            for path in (omh_home() / "config.json", Path.cwd() / ".omh.json")
            if path.is_file()
        ]
        self.console.line()
        self.console.line(
            t.title("  omh " + __version__)
            + t.muted(f"  · python {sys.version.split()[0]} · {sys.platform}")
        )
        self.console.line(t.muted("  workspace  ") + t.text(self.config.workspace))
        self.console.line(
            t.muted("  config     ")
            + t.text(", ".join(str(path) for path in config_paths) or "(defaults)")
        )
        self.console.line(t.muted("  sessions   ") + t.text(self.config.session_dir))
        self.console.line(
            t.muted("  router     ")
            + t.text(
                f"{self.config.router.strategy} · attempts {self.config.router.max_attempts}"
                f" · cooldown {self.config.router.cooldown_s:g}s"
                f" · hedge {self.config.router.hedge_delay_s:g}s"
            )
        )
        self.console.line(
            t.muted("  approval   ")
            + t.text(self.config.approval_mode)
            + t.muted("   theme  ")
            + t.text(self.config.theme)
            + t.muted("   tools  ")
            + t.text(str(len(self.registry.specs)))
        )
        self.console.line()
        rows = [
            [
                provider.name,
                provider.kind,
                shorten_path(provider.base_url or "-", 40),
                provider.api_key_env or "-",
                (
                    "demo"
                    if provider.kind == "mock"
                    else "local"
                    if not provider.requires_key
                    else "ready"
                    if provider.available()
                    else "missing key"
                ),
            ]
            for provider in self.config.providers.values()
        ]
        self.console.line(
            table(
                t,
                ["provider", "kind", "base_url", "key env", "status"],
                rows,
                max_width=terminal_width() - 4,
            )
        )
        enabled = self.config.enabled_profiles()
        self.console.line()
        self.console.line(
            t.muted("  routable  ")
            + t.text(f"{len(enabled)} profile(s) via {len(providers)} provider(s)")
        )
        if not enabled:
            for hint in self.config.missing_credentials()[:4]:
                self.console.line(t.muted("    · " + hint))
            if any(
                not provider.requires_key and provider.kind != "mock"
                for provider in self.config.providers.values()
            ):
                self.console.line(
                    t.muted("    · local server? `/provider ollama` (or set a model)")
                )
            self.console.line(t.muted("    · run `omh --offline` for a demo"))
            return 1
        return 0


def _models_table(config: Config, theme: Theme) -> str:
    auto = config.model.strip().lower() in {"", "auto", "*"}
    mode = "auto · load balanced" if auto else config.model
    lines = [
        "",
        theme.muted("  mode  ")
        + theme.accent(mode)
        + theme.muted("   strategy  ")
        + theme.text(config.router.strategy),
        "",
    ]
    rows = []
    for profile in config.models:
        provider = config.providers[profile.provider]
        if provider.kind == "mock":
            status = "offline demo"
        elif not provider.requires_key:
            status = "local"
        elif provider.available():
            status = "ready"
        else:
            status = "no key"
        marker = "→ " if not auto and profile.id == config.model else "  "
        rows.append(
            [
                marker + profile.name,
                profile.provider,
                f"{profile.weight:g}",
                f"{profile.context // 1000}k",
                f"${profile.input_cost:.2f}/${profile.output_cost:.2f}",
                status,
            ]
        )
    lines.append(
        table(
            theme,
            ["model", "provider", "wt", "ctx", "in/out $/M", "status"],
            rows,
            max_width=terminal_width() - 4,
        )
    )
    enabled = config.enabled_profiles()
    if not enabled:
        lines.append("")
        lines.extend(theme.muted("  · " + hint) for hint in config.missing_credentials()[:4])
        if any(
            not provider.requires_key and provider.kind != "mock"
            for provider in config.providers.values()
        ):
            lines.append(
                theme.muted("  · local server? select it with `/provider ollama`")
            )
        lines.append(theme.muted("  · or run `omh --offline` for a demo"))
    else:
        lines.append("")
        lines.append(
            theme.muted(f"  auto rotates {len(enabled)} ready lane(s); `/model <name>` pins one.")
        )
    return "\n".join(lines)


def _sessions_table(store: SessionStore, theme: Theme) -> str:
    rows = store.summaries(limit=15)
    if not rows:
        return theme.muted("  no sessions saved yet")
    rendered = []
    for row in rows:
        when = time.strftime("%m-%d %H:%M", time.localtime(row["updated"]))
        rendered.append(
            [
                row["id"],
                when,
                str(row["messages"]),
                row["model"] or "auto",
                row["title"],
            ]
        )
    return "\n".join(
        [
            "",
            table(
                theme,
                ["session", "updated", "msgs", "model", "title"],
                rendered,
                max_width=terminal_width() - 4,
            ),
        ]
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())








