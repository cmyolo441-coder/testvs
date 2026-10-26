from __future__ import annotations

import base64
import os
import platform
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

from ..command_path import inspect_omh_command_path
from ..local_store import atomic_write_text
from ..install.uninstall_safety import configured_link_reason, unsafe_removal_reason
from ..paths import OmhPaths
from ..runtime.artifacts import update_state


MENUBAR_APP_SCHEMA_VERSION = "menubar_app/v1"
MENUBAR_LABEL = "com.rlaope.omh.menubar"
DEFAULT_REFRESH_INTERVAL_SECONDS = 8
# Idle backoff ladder. One status reading costs about 0.7s wall and 0.55s
# user — roughly 0.46s of interpreter start plus importing the CLI, and
# 0.24s of the `ps` scan — and at a flat 8s that is about 9% of a core and
# 10,800 spawns a day, paid whether or not anything is running.
#
# Each consecutive reading that renders the same menu steps one rung, and
# any change drops straight back to rung 0. Rungs multiply the ACTIVE
# interval, so `--interval` still sets the live cadence.
IDLE_BACKOFF_MULTIPLIERS = (1, 2, 4, 8)
# What the top rung costs: once the helper has reached it, a change that no
# watched file announces — the `ps` scan finding a Hermes process that has
# not touched its session store — is up to this many seconds late. Session
# activity does move a watched file, so it is bounded by the tick below
# instead. The ceiling never pulls the cadence below the active interval: an
# operator who asked for something slower than this keeps it.
IDLE_BACKOFF_CEILING_SECONDS = 60
# The timer's own cadence. Most ticks only stat the payload's `watch_paths`,
# which is microseconds against the 0.7s of a spawn, so this is what bounds
# how late a starting session shows up — at any rung, not just rung 0.
WATCH_TICK_SECONDS = 2
_LAUNCH_AGENT_SYSTEM_PATHS = (
    "/usr/local/bin",
    "/opt/homebrew/bin",
    "/usr/bin",
    "/bin",
    "/usr/sbin",
    "/sbin",
)
MENUBAR_ICON_BASE64 = "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"


def menubar_idle_interval(active_interval: float, idle_steps: int) -> float:
    """Seconds until the next status reading after `idle_steps` unchanged ones.

    The Swift helper runs the same arithmetic over the same constants, which
    are substituted into its source below rather than written there, so the
    two cannot drift apart by editing one of them.
    """
    index = max(0, min(int(idle_steps), len(IDLE_BACKOFF_MULTIPLIERS) - 1))
    ceiling = max(float(active_interval), float(IDLE_BACKOFF_CEILING_SECONDS))
    return min(float(active_interval) * IDLE_BACKOFF_MULTIPLIERS[index], ceiling)


def menubar_should_read(
    seconds_since_last_read: float,
    active_interval: float,
    idle_steps: int,
    watch_stamp_moved: bool,
) -> bool:
    """Whether a tick should pay for a status reading.

    Two ways to say yes, and the second one has a floor. A moved watch stamp
    shortens the wait but never below the active interval, because Hermes
    writes its session store every few seconds while a session is live.
    Measured against a home with one live session writing once a second, an
    unfloored watch path took 169 readings in 15 minutes where a flat 8s
    timer took 102: two thirds more cost, spent at exactly the moment the
    machine is busiest. Floored, the same run took 93.

    The Swift helper calls the same rule over the same constants, mirrored
    into its source below.
    """
    if seconds_since_last_read >= menubar_idle_interval(active_interval, idle_steps):
        return True
    return bool(watch_stamp_moved) and seconds_since_last_read >= float(active_interval)


def setup_menubar_app(
    paths: OmhPaths,
    *,
    dry_run: bool = False,
    start: bool = True,
    force: bool = False,
    command_path: str | None = None,
    platform_name: str | None = None,
) -> dict[str, Any]:
    system = platform_name or platform.system()
    app_paths = menubar_app_paths(paths)
    base_payload: dict[str, Any] = {
        "schema_version": MENUBAR_APP_SCHEMA_VERSION,
        "status": "skipped",
        "supported": False,
        "platform": system,
        "dry_run": bool(dry_run),
        "started": False,
        "installed": False,
        "label": MENUBAR_LABEL,
        **{key: str(value) for key, value in app_paths.items()},
    }
    if system != "Darwin":
        return {**base_payload, "reason": "macOS menu bar helper is only supported on Darwin."}

    swiftc = shutil.which("swiftc")
    if not swiftc:
        return {**base_payload, "reason": "swiftc is not available, so the native menu bar helper cannot be built."}

    resolved_command = command_path or _resolved_omh_command()
    if not resolved_command:
        return {**base_payload, "reason": "`omh` command path is not available yet; rerun setup after the command is on PATH."}

    payload = {
        **base_payload,
        "supported": True,
        "status": "dry_run" if dry_run else "installed",
        "reason": "",
        "swiftc": swiftc,
        "omh_command": resolved_command,
    }
    if dry_run:
        return payload

    app_paths["app_dir"].mkdir(parents=True, exist_ok=True)
    app_paths["icon"].write_bytes(base64.b64decode(MENUBAR_ICON_BASE64, validate=True))
    atomic_write_text(app_paths["source"], _SWIFT_SOURCE)
    _compile_swift_helper(swiftc, app_paths["source"], app_paths["executable"])
    _write_launch_agent(app_paths["launch_agent"], app_paths["executable"], resolved_command, paths)
    payload["installed"] = True

    if start:
        start_result = start_menubar_app(paths, platform_name=system)
        payload["started"] = bool(start_result.get("started", False))
        payload["start_status"] = start_result.get("status", "unknown")
        payload["start_message"] = start_result.get("message", "")
        if payload["started"]:
            payload["status"] = "running"
        else:
            payload["status"] = "installed_start_failed"
    update_state(paths, {"last_menubar_app": payload})
    return payload


def start_menubar_app(paths: OmhPaths, *, platform_name: str | None = None) -> dict[str, Any]:
    system = platform_name or platform.system()
    app_paths = menubar_app_paths(paths)
    payload = {
        "schema_version": MENUBAR_APP_SCHEMA_VERSION,
        "operation": "start",
        "platform": system,
        "label": MENUBAR_LABEL,
        "launch_agent": str(app_paths["launch_agent"]),
        "started": False,
    }
    if system != "Darwin":
        return {**payload, "status": "skipped", "message": "macOS menu bar helper is only supported on Darwin."}
    if not app_paths["launch_agent"].exists():
        return {**payload, "status": "missing", "message": "LaunchAgent is not installed; run `omh menubar install`."}

    domain = f"gui/{os.getuid()}"
    _run_launchctl(["bootout", domain, str(app_paths["launch_agent"])], check=False)
    bootstrap = _run_launchctl(["bootstrap", domain, str(app_paths["launch_agent"])], check=False)
    if bootstrap.returncode != 0 and "already bootstrapped" not in bootstrap.stderr:
        return {
            **payload,
            "status": "failed",
            "message": (bootstrap.stderr or bootstrap.stdout or "launchctl bootstrap failed").strip(),
        }
    kickstart = _run_launchctl(["kickstart", "-k", f"{domain}/{MENUBAR_LABEL}"], check=False)
    if kickstart.returncode != 0:
        return {
            **payload,
            "status": "failed",
            "message": (kickstart.stderr or kickstart.stdout or "launchctl kickstart failed").strip(),
        }
    return {**payload, "status": "running", "started": True, "message": "OMH menu bar helper started."}


def stop_menubar_app(paths: OmhPaths, *, platform_name: str | None = None) -> dict[str, Any]:
    system = platform_name or platform.system()
    app_paths = menubar_app_paths(paths)
    payload = {
        "schema_version": MENUBAR_APP_SCHEMA_VERSION,
        "operation": "stop",
        "platform": system,
        "label": MENUBAR_LABEL,
        "launch_agent": str(app_paths["launch_agent"]),
        "stopped": False,
    }
    if system != "Darwin":
        return {**payload, "status": "skipped", "message": "macOS menu bar helper is only supported on Darwin."}
    domain = f"gui/{os.getuid()}"
    result = _run_launchctl(["bootout", domain, str(app_paths["launch_agent"])], check=False)
    if result.returncode != 0 and "No such process" not in result.stderr and "No such file" not in result.stderr:
        return {**payload, "status": "failed", "message": (result.stderr or result.stdout).strip()}
    return {**payload, "status": "stopped", "stopped": True, "message": "OMH menu bar helper stopped."}


def uninstall_menubar_app(paths: OmhPaths, *, dry_run: bool = False, platform_name: str | None = None) -> dict[str, Any]:
    system = platform_name or platform.system()
    app_paths = menubar_app_paths(paths)
    candidates = [app_paths["launch_agent"], app_paths["app_dir"]]
    existing = [path for path in candidates if path.exists() or path.is_symlink()]
    reason = (
        unsafe_removal_reason(paths.omh_home, hermes_home=paths.hermes_home)
        or configured_link_reason(paths.omh_home, "OMH_HOME")
        or unsafe_removal_reason(app_paths["app_dir"], hermes_home=paths.hermes_home)
    )
    if not reason and not is_managed_menubar_install(paths):
        reason = "menu bar ownership is unproven: no matching OMH LaunchAgent"
    removable = [] if reason else existing
    payload = {
        "schema_version": MENUBAR_APP_SCHEMA_VERSION,
        "operation": "uninstall",
        "platform": system,
        "label": MENUBAR_LABEL,
        "dry_run": bool(dry_run),
        "removed": [],
        "kept_paths": [{"path": str(path), "reason": reason} for path in existing] if reason else [],
        "would_remove": [str(path) for path in removable] if dry_run else [],
    }
    if dry_run:
        return {**payload, "status": "dry_run"}
    if reason:
        return {**payload, "status": "kept" if existing else "absent"}

    stop_result = stop_menubar_app(paths, platform_name=system)
    removed: list[str] = []
    for path in removable:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(str(path))
    payload["removed"] = removed
    payload["stop_status"] = stop_result.get("status", "unknown")
    payload["status"] = "removed" if removed else "absent"
    update_state(paths, {"last_menubar_app": payload})
    return payload


def menubar_app_paths(paths: OmhPaths) -> dict[str, Path]:
    app_dir = paths.omh_home / "menubar"
    return {
        "app_dir": app_dir,
        "source": app_dir / "OMHMenuBar.swift",
        "executable": app_dir / "omh-menubar",
        "icon": app_dir / "omh-character-mask.png",
        "launch_agent": Path.home() / "Library" / "LaunchAgents" / f"{MENUBAR_LABEL}.plist",
    }


def is_managed_menubar_install(paths: OmhPaths) -> bool:
    app_paths = menubar_app_paths(paths)
    launch_agent = app_paths["launch_agent"]
    if _path_contains_symlink(launch_agent, Path.home()):
        return False
    for name in ("app_dir", "source", "executable", "icon"):
        if _path_contains_symlink(app_paths[name], paths.omh_home):
            return False
    if not launch_agent.is_file():
        return False
    try:
        payload = plistlib.loads(launch_agent.read_bytes())
    except (OSError, plistlib.InvalidFileException):
        return False
    if not isinstance(payload, dict) or payload.get("Label") != MENUBAR_LABEL:
        return False
    raw_arguments = payload.get("ProgramArguments")
    if not isinstance(raw_arguments, list) or any(not isinstance(value, str) for value in raw_arguments):
        return False
    arguments = [str(value) for value in raw_arguments]
    if not arguments or arguments[0] != str(app_paths["executable"]):
        return False
    return (
        _launch_agent_argument(arguments, "--omh-home") == str(paths.omh_home)
        and _launch_agent_argument(arguments, "--hermes-home") == str(paths.hermes_home)
    )


def _path_contains_symlink(path: Path, boundary: Path) -> bool:
    try:
        path.relative_to(boundary)
    except ValueError:
        return True
    current = path
    while True:
        if current.is_symlink():
            return True
        if current == boundary:
            return False
        current = current.parent


def _launch_agent_argument(arguments: list[str], name: str) -> str:
    try:
        index = arguments.index(name)
    except ValueError:
        return ""
    value_index = index + 1
    return arguments[value_index] if value_index < len(arguments) else ""


def _resolved_omh_command() -> str:
    command = inspect_omh_command_path()
    if command.get("found") and command.get("path"):
        return str(command["path"])
    which = shutil.which("omh")
    return which or ""


def _compile_swift_helper(swiftc: str, source: Path, executable: Path) -> None:
    result = subprocess.run(
        [swiftc, "-framework", "AppKit", str(source), "-o", str(executable)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "swiftc failed").strip())


def _write_launch_agent(plist_path: Path, executable: Path, omh_command: str, paths: OmhPaths) -> None:
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "Label": MENUBAR_LABEL,
        "ProgramArguments": [
            str(executable),
            "--omh-command",
            omh_command,
            "--omh-home",
            str(paths.omh_home),
            "--hermes-home",
            str(paths.hermes_home),
            "--icon",
            str(menubar_app_paths(paths)["icon"]),
            "--interval",
            str(DEFAULT_REFRESH_INTERVAL_SECONDS),
        ],
        "RunAtLoad": True,
        "KeepAlive": True,
        "EnvironmentVariables": {"PATH": _launch_agent_path(omh_command)},
        "StandardOutPath": str(paths.runtime_dir / "menubar.out.log"),
        "StandardErrorPath": str(paths.runtime_dir / "menubar.err.log"),
    }
    paths.runtime_dir.mkdir(parents=True, exist_ok=True)
    plist_path.write_bytes(plistlib.dumps(payload))


def _launch_agent_path(omh_command: str) -> str:
    node_command = shutil.which("node")
    candidates = [
        _launch_agent_executable_parent(node_command or ""),
        _launch_agent_executable_parent(sys.executable),
        _launch_agent_executable_parent(omh_command),
        *_LAUNCH_AGENT_SYSTEM_PATHS,
    ]
    path_entries: list[str] = []
    canonical_entries: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue
        canonical = str(Path(candidate).resolve()) if platform.system() == "Darwin" else candidate
        if canonical not in canonical_entries:
            path_entries.append(candidate)
            canonical_entries.add(canonical)
    return ":".join(path_entries)


def _launch_agent_executable_parent(command: str) -> str:
    if not command:
        return ""
    expanded = os.path.expanduser(command)
    if platform.system() == "Darwin":
        command_path = Path(expanded)
        return str(command_path.resolve().parent) if command_path.parent != Path(".") else ""
    command_path = PurePosixPath(expanded)
    return str(command_path.parent) if command_path.parent != PurePosixPath(".") else ""


def _run_launchctl(args: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["launchctl", *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check)


_SWIFT_TEMPLATE = r'''
import AppKit
import Foundation

// Seconds until the next status reading after `idleSteps` consecutive
// readings that rendered the same menu. Mirrors menubar_idle_interval() in
// src/surfaces/menubar_app.py, which substitutes the three constants below
// into this source: there is one place to change them and the mirror cannot
// drift by editing the other copy.
func omhIdleInterval(_ activeInterval: Double, _ idleSteps: Int) -> Double {
    let multipliers: [Double] = __OMH_IDLE_BACKOFF_MULTIPLIERS__
    let ceiling: Double = __OMH_IDLE_BACKOFF_CEILING_SECONDS__
    let index = max(0, min(idleSteps, multipliers.count - 1))
    return min(activeInterval * multipliers[index], max(activeInterval, ceiling))
}

// Whether a tick should pay for a status reading. Mirrors
// menubar_should_read() in src/surfaces/menubar_app.py.
//
// The floor on the second branch is the whole point of it. Hermes writes
// its session store every few seconds while a session is live, so a watch
// path that reads whenever the stamp moved reads far more often than the
// rung allows. Measured over 15 minutes against one live session writing
// once a second: unfloored 169 readings, a flat 8s timer 102, floored 93.
// So the floor is what makes a busy machine cost no more than it did
// before, while an idle one still backs off.
func omhShouldRead(
    _ secondsSinceLastRead: Double,
    _ activeInterval: Double,
    _ idleSteps: Int,
    _ watchStampMoved: Bool
) -> Bool {
    if secondsSinceLastRead >= omhIdleInterval(activeInterval, idleSteps) {
        return true
    }
    return watchStampMoved && secondsSinceLastRead >= activeInterval
}

// How often the timer fires. Most ticks only stat the payload's
// `watch_paths`, so this is cheap enough to keep short at every rung.
let omhWatchTickSeconds: Double = __OMH_WATCH_TICK_SECONDS__

// One NSMenu for the life of the process. AppKit updates an open menu in
// place when its items change, so a refresh that lands while the user is
// looking at the menu is visible immediately; replacing `statusItem.menu`
// (the previous design) only took effect on the NEXT open, which read as
// "the menu never updates". The status subprocess runs off the main thread:
// a ~0.5s synchronous wait every 8s froze the menu bar for that long.
final class OMHMenuBarDelegate: NSObject, NSApplicationDelegate, NSMenuDelegate {
    private let statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
    private let menu = NSMenu()
    private let refreshQueue = DispatchQueue(label: "omh.menubar.refresh", qos: .utility)
    private let clock: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss"
        return formatter
    }()
    private var timer: Timer?
    private var refreshInFlight = false
    private var lastPayload: [String: Any]?
    private var lastRefreshedAt: Date?
    private var lastRefreshStartedAt: Date?
    private var consecutiveFailures = 0
    private var idleSteps = 0
    private var lastRenderFingerprint: String?
    private var watchPaths: [String] = []
    private var lastWatchStamp = ""
    private var wokeOnWatch = false
    private var omhCommand = "omh"
    private var omhHome = ""
    private var hermesHome = ""
    private var iconPath = ""
    private var activeInterval: TimeInterval = 8

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        parseArguments()
        configureStatusButton()
        // Rows are information, not commands: keep them enabled so they render
        // in the normal text color instead of the disabled gray that reads as
        // "still loading". Nothing has an action, so clicking does nothing.
        menu.autoenablesItems = false
        menu.delegate = self
        statusItem.menu = menu
        updateStatusDescription(headline: "OMH", summary: "Loading status")
        renderMenu(headline: "OMH", summary: "Loading status", cards: [], footer: "Loading…")
        refresh()
        // `.common` so the timer keeps firing while the menu is open (menu
        // tracking runs the event-tracking mode, where a default-mode timer
        // is silent for as long as the menu stays down).
        let timer = Timer(timeInterval: omhWatchTickSeconds, repeats: true) { [weak self] _ in
            self?.tick()
        }
        RunLoop.main.add(timer, forMode: .common)
        self.timer = timer
    }

    func menuWillOpen(_ menu: NSMenu) {
        // Opening the menu is the moment freshness matters most.
        refresh()
    }

    private func parseArguments() {
        let args = CommandLine.arguments
        var index = 1
        while index < args.count {
            let key = args[index]
            let value = index + 1 < args.count ? args[index + 1] : ""
            switch key {
            case "--omh-command":
                omhCommand = value
                index += 2
            case "--omh-home":
                omhHome = value
                index += 2
            case "--hermes-home":
                hermesHome = value
                index += 2
            case "--icon":
                iconPath = value
                index += 2
            case "--interval":
                activeInterval = TimeInterval(value) ?? 8
                index += 2
            default:
                index += 1
            }
        }
    }

    private func configureStatusButton() {
        guard
            let button = statusItem.button,
            let image = NSImage(contentsOfFile: iconPath)
        else {
            return
        }
        image.size = NSSize(width: 18, height: 18)
        image.isTemplate = true
        button.image = image
        button.imagePosition = .imageLeading
    }

    private func updateStatusDescription(headline: String, summary: String) {
        guard let button = statusItem.button else {
            return
        }
        let description = "OMH — \(headline) — \(summary)"
        button.toolTip = description
        button.setAccessibilityLabel(description)
    }

    @objc private func refreshClicked(_ sender: Any?) {
        refresh()
    }

    @objc private func quitClicked(_ sender: Any?) {
        NSApp.terminate(nil)
    }

    // Fires every `omhWatchTickSeconds`. A reading costs about 0.7s of CPU
    // and almost always says nothing changed, so the tick decides whether
    // to pay for one; the stats below cost microseconds.
    private func tick() {
        guard let startedAt = lastRefreshStartedAt else {
            refresh()
            return
        }
        // A watched file that moved means Hermes recorded session activity,
        // so a backed-off rung would be showing a stale menu. Waking on it
        // is what bounds how late a session shows up, and the floor inside
        // omhShouldRead is what stops that waking from costing more than
        // the flat timer once a session is actually live.
        //
        // The new stamp is recorded by the reading, not by this tick, so a
        // tick that decides against reading -- below the floor, or with a
        // refresh already in flight -- leaves the change pending for a
        // later tick instead of marking it seen.
        //
        // One race is left and is left knowingly: a reading still running
        // when the write lands records the new stamp on completion even
        // though it may have queried the store first. That costs one
        // ladder period of staleness, needs a reading slower than the
        // floor to happen at all, and the rung's own deadline still fires.
        let elapsed = Date().timeIntervalSince(startedAt)
        let stampMoved = watchStamp() != lastWatchStamp
        if omhShouldRead(elapsed, activeInterval, idleSteps, stampMoved) {
            // Hermes wrote something, so the machine is not idle however
            // identical the rendered menu looks. Restart at the active rung
            // instead of letting the ladder climb through a live session.
            wokeOnWatch = stampMoved
            refresh()
        }
    }

    // Stats only. Reading these files is what the subprocess is for.
    private func watchStamp() -> String {
        if watchPaths.isEmpty {
            return ""
        }
        let manager = FileManager.default
        var parts: [String] = []
        for path in watchPaths {
            guard let attributes = try? manager.attributesOfItem(atPath: path) else {
                parts.append("-")
                continue
            }
            let modified = (attributes[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
            let size = (attributes[.size] as? NSNumber)?.int64Value ?? 0
            parts.append("\(modified)/\(size)")
        }
        return parts.joined(separator: "|")
    }

    // What the menu renders, and nothing else. `hermes_processes.observed_at`
    // and `process_overlay.age_seconds` are stamped on every reading, so two
    // whole payloads are never equal and a payload-level comparison would
    // leave the ladder pinned to rung 0 forever.
    private func renderFingerprint(_ display: [String: Any]?) -> String {
        guard
            let display,
            let data = try? JSONSerialization.data(withJSONObject: display, options: [.sortedKeys])
        else {
            return ""
        }
        return String(decoding: data, as: UTF8.self)
    }

    private func refresh() {
        if refreshInFlight {
            return
        }
        refreshInFlight = true
        lastRefreshStartedAt = Date()
        refreshQueue.async { [weak self] in
            guard let self else {
                return
            }
            let payload = self.readStatusPayload()
            DispatchQueue.main.async {
                self.refreshInFlight = false
                self.apply(payload)
            }
        }
    }

    private func apply(_ payload: [String: Any]?) {
        // Before the failure branch: a reading that could not be taken
        // still consumed the change that asked for it, and leaving the old
        // stamp here would make every tick spawn again until it succeeds.
        lastWatchStamp = watchStamp()
        guard let payload else {
            consecutiveFailures += 1
            // A command that cannot be run is no reason to run it 450 times
            // an hour, so a failure steps the ladder like an unchanged
            // reading does. The mark below still arrives on the third one.
            // The wake flag is consumed here too: left set, it would hold
            // the ladder at rung 0 for every later reading.
            wokeOnWatch = false
            idleSteps += 1
            // One failed read (a busy state.db, a slow disk) must not flash
            // the attention mark and blank the cards; keep the last good
            // reading until the failure persists.
            if lastPayload != nil && consecutiveFailures < 3 {
                renderFooter()
                return
            }
            statusItem.button?.title = "!"
            updateStatusDescription(headline: "OMH needs attention", summary: "Status unavailable")
            renderMenu(
                headline: "OMH needs attention",
                summary: "Status unavailable",
                cards: [
                    [
                        "title": "Recovery",
                        "rows": [
                            ["label": "Next", "value": "Run omh doctor"],
                            ["label": "Then", "value": "Run omh setup if registration needs repair"]
                        ]
                    ]
                ],
                footer: footerText()
            )
            return
        }
        consecutiveFailures = 0
        lastPayload = payload
        lastRefreshedAt = Date()
        watchPaths = (payload["watch_paths"] as? [String]) ?? []
        // Again, deliberately: the stamp taken above was over the previous
        // path list, and this reading may have replaced it.
        lastWatchStamp = watchStamp()
        let display = payload["display"] as? [String: Any]
        // Compare what the menu renders, not the payload.
        let fingerprint = renderFingerprint(display)
        if let previous = lastRenderFingerprint, previous == fingerprint, !wokeOnWatch {
            idleSteps += 1
        } else {
            idleSteps = 0
        }
        wokeOnWatch = false
        lastRenderFingerprint = fingerprint
        let headline = (display?["headline"] as? String) ?? "OMH ready"
        let summary = (display?["summary_line"] as? String) ?? "OMH ready"
        let menuBarTitle = (display?["menu_bar_title"] as? String) ?? ""
        statusItem.button?.title = menuBarTitle
        updateStatusDescription(headline: headline, summary: summary)
        renderMenu(headline: headline, summary: summary, cards: menuCards(from: payload), footer: footerText())
    }

    private func footerText() -> String {
        guard let refreshedAt = lastRefreshedAt else {
            return "Not updated yet"
        }
        let stamp = "Updated \(clock.string(from: refreshedAt))"
        if consecutiveFailures > 0 {
            return "\(stamp) · last read failed, retrying"
        }
        return stamp
    }

    private func renderFooter() {
        guard let item = menu.items.first(where: { $0.representedObject as? String == "footer" }) else {
            return
        }
        item.title = " \(footerText())"
    }

    private func renderMenu(headline: String, summary: String, cards: [[String: Any]], footer: String) {
        menu.removeAllItems()
        let headlineItem = infoItem(" \(headline)")
        let font = NSFont.menuBarFont(ofSize: 0)
        headlineItem.attributedTitle = NSAttributedString(
            string: " \(headline)",
            attributes: [.font: NSFont.boldSystemFont(ofSize: font.pointSize)]
        )
        menu.addItem(headlineItem)
        menu.addItem(infoItem(" \(summary)"))

        for card in cards {
            menu.addItem(NSMenuItem.separator())
            if let title = card["title"] as? String, !title.isEmpty {
                let item = infoItem(" \(title)")
                item.attributedTitle = NSAttributedString(
                    string: " \(title)",
                    attributes: [.font: NSFont.boldSystemFont(ofSize: font.pointSize)]
                )
                menu.addItem(item)
            }
            if let columns = card["columns"] as? [String], !columns.isEmpty {
                let line = tableHeaderTitle(columns)
                let item = infoItem("   \(line)")
                item.toolTip = line
                item.attributedTitle = NSAttributedString(
                    string: "   \(line)",
                    attributes: [
                        .font: NSFont.monospacedSystemFont(ofSize: font.pointSize, weight: .medium),
                        .foregroundColor: NSColor.secondaryLabelColor
                    ]
                )
                menu.addItem(item)
            }
            if let rows = card["rows"] as? [[String: Any]] {
                for row in rows.prefix(6) {
                    let line = rowTitle(row)
                    let item = infoItem("   \(line)")
                    item.toolTip = rowToolTip(row)
                    if (row["kind"] as? String) == "table_row" {
                        item.attributedTitle = NSAttributedString(
                            string: "   \(line)",
                            attributes: [.font: NSFont.monospacedSystemFont(ofSize: font.pointSize, weight: .regular)]
                        )
                    }
                    menu.addItem(item)
                }
            }
            if let footer = card["footer"] as? String, !footer.isEmpty {
                let item = infoItem("   \(footer)")
                item.toolTip = footer
                item.attributedTitle = NSAttributedString(
                    string: "   \(footer)",
                    attributes: [
                        .font: NSFont.systemFont(ofSize: NSFont.smallSystemFontSize),
                        .foregroundColor: NSColor.secondaryLabelColor
                    ]
                )
                menu.addItem(item)
            }
        }
        menu.addItem(NSMenuItem.separator())
        let footerItem = infoItem(" \(footer)")
        footerItem.representedObject = "footer"
        menu.addItem(footerItem)
        menu.addItem(NSMenuItem(title: "Refresh", action: #selector(refreshClicked(_:)), keyEquivalent: "r"))
        menu.addItem(NSMenuItem(title: "Quit OMH Menu Bar", action: #selector(quitClicked(_:)), keyEquivalent: "q"))
    }

    private func infoItem(_ title: String) -> NSMenuItem {
        let item = NSMenuItem(title: title, action: nil, keyEquivalent: "")
        item.isEnabled = true
        return item
    }

    private func tableHeaderTitle(_ columns: [String]) -> String {
        let padded = columns.prefix(3).enumerated().map { index, value in
            if index == 0 {
                return fixedWidth(value, 18)
            }
            if index == 1 {
                return fixedWidth(value, 24)
            }
            return value
        }
        return padded.joined()
    }

    private func fixedWidth(_ value: String, _ length: Int) -> String {
        if value.count > length {
            let endIndex = value.index(value.startIndex, offsetBy: max(0, length - 1))
            return String(value[..<endIndex]) + "…"
        }
        return value.padding(toLength: length, withPad: " ", startingAt: 0)
    }

    private func rowTitle(_ row: [String: Any]) -> String {
        if (row["kind"] as? String) == "table_row" {
            let left = fixedWidth((row["left"] as? String) ?? "", 18)
            let right = fixedWidth((row["right"] as? String) ?? "", 24)
            return "\(left)\(right)"
        }
        let label = (row["label"] as? String) ?? ""
        let value = (row["value"] as? String) ?? ""
        let detail = (row["detail"] as? String) ?? ""
        var pieces: [String] = []
        if !label.isEmpty {
            pieces.append(label)
        }
        if !value.isEmpty {
            pieces.append(value)
        }
        var title = pieces.joined(separator: ": ")
        if !detail.isEmpty {
            title += " — \(detail)"
        }
        return title.isEmpty ? "Unavailable" : title
    }

    private func rowToolTip(_ row: [String: Any]) -> String {
        if (row["kind"] as? String) == "table_row" {
            let left = (row["left"] as? String) ?? ""
            let right = (row["right"] as? String) ?? ""
            return "\(left): \(right)"
        }
        return rowTitle(row)
    }

    private func menuCards(from payload: [String: Any]) -> [[String: Any]] {
        if
            let display = payload["display"] as? [String: Any],
            let cards = display["menu_cards"] as? [[String: Any]],
            !cards.isEmpty
        {
            return cards
        }
        return fallbackCards(from: payload)
    }

    private func fallbackCards(from payload: [String: Any]) -> [[String: Any]] {
        let settings = payload["settings"] as? [String: Any]
        var rows: [[String: String]] = []
        for key in ["omh_connection", "hermes_targets", "coding_handoff", "send_mode"] {
            if
                let row = settings?[key] as? [String: Any],
                let label = row["label"] as? String
            {
                rows.append(["label": label, "value": ""])
            }
        }
        return [["title": "Overview", "rows": rows]]
    }

    private func readStatusPayload() -> [String: Any]? {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: omhCommand)
        var args: [String] = []
        if !omhHome.isEmpty {
            args.append(contentsOf: ["--omh-home", omhHome])
        }
        if !hermesHome.isEmpty {
            args.append(contentsOf: ["--hermes-home", hermesHome])
        }
        args.append(contentsOf: ["menubar", "status", "--observe-local-processes", "--json"])
        process.arguments = args
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = Pipe()
        do {
            try process.run()
        } catch {
            return nil
        }
        // Read before waiting: a payload larger than the pipe buffer would
        // otherwise block the child on write while we block on exit.
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            return nil
        }
        guard
            let object = try? JSONSerialization.jsonObject(with: data, options: []),
            let payload = object as? [String: Any]
        else {
            return nil
        }
        return payload
    }
}

let app = NSApplication.shared
let delegate = OMHMenuBarDelegate()
app.delegate = delegate
app.run()
'''


def _render_swift_source() -> str:
    """Substitute the ladder constants into the helper source.

    Keeps `menubar_idle_interval` and its Swift mirror reading the same three
    numbers. A test compiles the mirrored function out of this rendered
    source and compares it against the Python one over a grid.
    """
    multipliers = ", ".join(str(float(value)) for value in IDLE_BACKOFF_MULTIPLIERS)
    replacements = {
        "__OMH_IDLE_BACKOFF_MULTIPLIERS__": f"[{multipliers}]",
        "__OMH_IDLE_BACKOFF_CEILING_SECONDS__": str(float(IDLE_BACKOFF_CEILING_SECONDS)),
        "__OMH_WATCH_TICK_SECONDS__": str(float(WATCH_TICK_SECONDS)),
    }
    rendered = _SWIFT_TEMPLATE
    for placeholder, value in replacements.items():
        if placeholder not in rendered:
            raise RuntimeError(f"menu bar helper source lost its {placeholder} placeholder")
        rendered = rendered.replace(placeholder, value)
    return rendered


_SWIFT_SOURCE = _render_swift_source()
