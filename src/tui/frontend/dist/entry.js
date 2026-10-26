// OMH TUI frontend shim.
//
// Hermes' launcher runs `node <HERMES_TUI_DIR>/dist/entry.js` in place of its
// own Ink app when bare `omh` sets HERMES_TUI_DIR. This file only hands the
// terminal to the Python OMH TUI, with the interpreter `omh` named in
// OMH_TUI_PYTHON, and returns its exit code to the launcher.
//
// The launcher's PYTHONPATH points at Hermes' source root and Hermes' own
// runtime packages. The OMH interpreter must not import those, so it starts
// with -E (PYTHONPATH and PYTHONHOME ignored) and -P (no working directory on
// the import path), and the value moves to OMH_TUI_GATEWAY_PYTHONPATH, which
// the TUI hands back to the gateway child it spawns. Not -I: that also drops
// the user's site-packages, where a `pip install --user` OMH lives.
//
// The TUI has to import the `omh` that `omh` itself runs, and that package is
// not always on this interpreter's own path: the npm and Bun launcher puts its
// extracted cache on the path in-process (python_bridge.py). So `omh` names, in
// OMH_TUI_SITE, the directory its `omh` package was imported from, and the
// bootstrap puts it first on the path unless this interpreter already imports
// `omh` from there. Only the environment carries it, never this source. With
// it on the path no bytecode is written: the npm launcher checks its cache
// against a digest on every run. An OMH that still cannot import itself exits
// with FALLBACK_EXIT, so `omh` opens Hermes' own terminal instead of showing a
// traceback.
//
// Exit 75 alone is not a fallback request, because Hermes exits 75 on paths of
// its own: `omh` also needs the file it named in OMH_TUI_FALLBACK_SENTINEL to
// exist. So every fallback exit here, the bootstrap's included, creates that
// file first, as `omh.tui.launch.request_fallback` does for the TUI.
const { spawn } = require('node:child_process')
const fs = require('node:fs')
const { constants } = require('node:os')

const FALLBACK_EXIT = 75
const FALLBACK_SENTINEL_ENV = 'OMH_TUI_FALLBACK_SENTINEL'
const SITE_ENV = 'OMH_TUI_SITE'

const BOOTSTRAP = `
import os, sys
try:
    site = os.environ.get("${SITE_ENV}", "")
    if site:
        import importlib.util
        spec = importlib.util.find_spec("omh")
        own = os.path.dirname(os.path.dirname(spec.origin)) if spec is not None and spec.origin else ""
        if not own or os.path.realpath(own) != os.path.realpath(site):
            sys.dont_write_bytecode = True
            sys.path.insert(0, site)
    from omh.tui.app import main
except ImportError as error:
    sys.stderr.write(f"omh: the OMH TUI could not import itself: {error}\\n")
    sentinel = os.environ.get("${FALLBACK_SENTINEL_ENV}", "")
    if sentinel:
        try:
            os.close(os.open(sentinel, os.O_WRONLY | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600))
        except OSError:
            pass
    sys.exit(${FALLBACK_EXIT})
sys.exit(main(os.environ))
`

function requestFallback () {
  const sentinel = process.env[FALLBACK_SENTINEL_ENV]
  if (sentinel) {
    try {
      fs.closeSync(fs.openSync(sentinel, fs.constants.O_WRONLY | fs.constants.O_CREAT | (fs.constants.O_NOFOLLOW ?? 0), 0o600))
    } catch {}
  }
  process.exit(FALLBACK_EXIT)
}

const python = process.env.OMH_TUI_PYTHON
if (!python) {
  process.stderr.write('omh: OMH_TUI_PYTHON is not set; the OMH TUI is opened through `omh`.\n')
  requestFallback()
}

const env = { ...process.env }
env.OMH_TUI_GATEWAY_PYTHONPATH = env.PYTHONPATH ?? ''
delete env.PYTHONPATH
delete env.NODE_OPTIONS

const child = spawn(python, ['-E', '-P', '-c', BOOTSTRAP], { stdio: 'inherit', env })

// The TUI reads Ctrl+C as a key in raw mode. A SIGINT that still reaches this
// process (sent to the whole group) must not end the shim and orphan the TUI.
process.on('SIGINT', () => {})
for (const signal of ['SIGTERM', 'SIGHUP']) {
  process.on(signal, () => child.kill(signal))
}

// 'error' also fires when a kill fails after the child started; only a child
// that never started asks for the fallback, or a live session's end would open
// a second terminal.
let started = false
child.on('spawn', () => { started = true })
child.on('error', () => { if (!started) requestFallback() })
child.on('exit', (code, signal) => {
  if (code !== null) process.exit(code)
  process.exit(signal ? 128 + (constants.signals[signal] ?? 0) : 1)
})
