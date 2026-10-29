#!/usr/bin/env bash
# FullAgent installer — detects architecture, installs the right way.
set -euo pipefail

REPO="cmyolo441-coder/py.test"
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
"$VENV_DIR/bin/pip" install -q --upgrade "git+https://github.com/${REPO}.git"

mkdir -p "$INSTALL_DIR"
cat > "$INSTALL_DIR/$NAME" <<EOF
#!/bin/sh
exec "$VENV_DIR/bin/python" -m fullagent "\$@"
EOF
chmod +x "$INSTALL_DIR/$NAME"

echo ">> Installed: $INSTALL_DIR/$NAME"
ensure_path
echo ">> Run: $NAME"
