#!/usr/bin/env bash
# FullAgent installer — detects architecture, installs the right way.
set -euo pipefail

REPO="cmyolo441-coder/testvs"
NAME="fullagent"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"

# Add INSTALL_DIR to PATH via the user's shell rc files if it isn't there.
ensure_path() {
  case ":$PATH:" in
    *":$INSTALL_DIR:"*) return 0 ;;
  esac
  local line="export PATH=\"$INSTALL_DIR:\$PATH\""
  local handled=0
  for rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
    if [ -f "$rc" ]; then
      if grep -qF "$INSTALL_DIR" "$rc" 2>/dev/null; then
        handled=1   # rc already covers it — just needs a shell restart
      else
        printf '\n# Added by FullAgent installer\n%s\n' "$line" >> "$rc"
        echo ">> Added $INSTALL_DIR to PATH in $rc"
        handled=1
      fi
    fi
  done
  if [ "$handled" -eq 0 ]; then
    echo "!! $INSTALL_DIR is not in your PATH — add this to your shell rc:"
    echo "   $line"
  else
    echo ">> Restart your shell (or run: source ~/.bashrc) for the PATH change to take effect"
  fi
}

ARCH="$(uname -m)"
OS="$(uname -s)"

echo ">> FullAgent installer ($OS/$ARCH)"

# ---------------------------------------------------------------------------
# x86_64 Linux: prefer a fresh SOURCE install (always current); only fall
# back to the prebuilt release binary when a venv is impossible — the
# release artifact can lag behind main (stale version).
# ---------------------------------------------------------------------------
if [ "$OS" = "Linux" ] && { [ "$ARCH" = "x86_64" ] || [ "$ARCH" = "amd64" ]; }; then
  if ! command -v python3 >/dev/null 2>&1; then
    echo ">> python3 not found; trying the prebuilt binary..."
    if command -v curl >/dev/null 2>&1; then
      mkdir -p "$INSTALL_DIR"
      TMP="$(mktemp -d)"
      trap 'rm -rf "$TMP"' EXIT
      if curl -fsSL "https://github.com/${REPO}/releases/latest/download/fullagent-linux-x64" -o "$TMP/fullagent"; then
        chmod +x "$TMP/fullagent"
        mv "$TMP/fullagent" "$INSTALL_DIR/$NAME"
        echo ">> Installed: $INSTALL_DIR/$NAME"
        ensure_path
        echo ">> Run: $NAME"
        exit 0
      fi
    fi
    echo "!! python3 is required for the source install" >&2
    exit 1
  fi
  echo ">> Source install (always the freshest code)..."
fi

# ---------------------------------------------------------------------------
# Everything else (ARM64 Linux, macOS, etc.): install from source via pip.
# ---------------------------------------------------------------------------
echo ">> Source install for $ARCH..."

if ! command -v python3 >/dev/null 2>&1; then
  echo "!! python3 is required" >&2; exit 1
fi

if ! python3 -c "import venv, ensurepip" >/dev/null 2>&1; then
  echo "!! python3 venv support is missing (Debian/Ubuntu: sudo apt install python3-venv)" >&2
  exit 1
fi

VENV_DIR="${FULLAGENT_VENV:-$HOME/.fullagent-venv}"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo ">> Creating virtualenv..."
  rm -rf "$VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

echo ">> Installing FullAgent..."
"$VENV_DIR/bin/pip" install -q --upgrade pip
# A git URL without a ref is a DIRECT reference: pip compares it to what is
# already installed and can report "Requirement already satisfied", keeping a
# stale checkout (that is how a fresh install kept booting v4.0.0 while main
# was v4.1.2). Force the rebuild and bypass the wheel cache.
"$VENV_DIR/bin/pip" install -q --upgrade --force-reinstall --no-cache-dir \
  "git+https://github.com/${REPO}.git"

mkdir -p "$INSTALL_DIR"
# Launch through a tiny script instead of `python -m fullagent`. With -m,
# Python puts the CURRENT DIRECTORY first on sys.path — so running the command
# from inside a checkout of this repo imported THAT copy (e.g. an old v4.0.0
# clone) instead of the freshly installed package, and the app kept showing
# the old version and the old "no API key" error. A script's sys.path[0] is
# its own folder ($VENV_DIR/bin), which never contains `fullagent`.
cat > "$VENV_DIR/bin/_fullagent_shim.py" <<'PY'
import sys

from fullagent.__main__ import main

sys.exit(main())
PY
cat > "$INSTALL_DIR/$NAME" <<EOF
#!/bin/sh
exec "$VENV_DIR/bin/python" "$VENV_DIR/bin/_fullagent_shim.py" "\$@"
EOF
chmod +x "$INSTALL_DIR/$NAME"

if [ -d "$PWD/fullagent" ]; then
  echo ">> note: $PWD contains a fullagent/ checkout — the launcher ignores it"
  echo "   (the installed package is used; a local clone only matters for python main.py)"
fi

# A stale `fullagent` EARLIER in PATH (typical: an old single-file binary in
# /usr/local/bin, or the previous release left in ~/.local/bin before pip had
# a venv) shadows this install — the user then keeps running the old build
# and sees the old version/errors. Move it aside when we can, else say so.
shadowed=""
IFS=':' read -r -a _path_dirs <<< "$PATH"
for _d in "${_path_dirs[@]}"; do
  [ -z "$_d" ] && continue
  [ "$_d" = "$INSTALL_DIR" ] && break
  if [ -f "$_d/$NAME" ] && [ -x "$_d/$NAME" ]; then
    shadowed="$_d/$NAME"
    break
  fi
done
if [ -n "$shadowed" ]; then
  if [ -w "$shadowed" ]; then
    mv "$shadowed" "$shadowed.bak-$(date +%Y%m%d%H%M%S)"
    echo ">> Moved stale $shadowed aside (it shadowed the new install)"
  else
    echo "!! $shadowed shadows the new install and is not writable."
    echo "   Remove it: sudo rm -f $shadowed"
  fi
fi

# Probe through the installed command itself (not `python -m`, which a local
# checkout in the current directory would shadow) so the reported version is
# exactly what the user gets when they type `fullagent`.
INSTALLED_VERSION="$("$INSTALL_DIR/$NAME" --version 2>/dev/null || true)"
echo ">> Installed: $INSTALL_DIR/$NAME (${INSTALLED_VERSION:-version unknown})"
ensure_path
echo ">> Run: $NAME   (open a new shell, or: hash -r; source ~/.bashrc)"
