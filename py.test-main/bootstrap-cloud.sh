#!/usr/bin/env bash
# ============================================================================
# FullAgent Cloud — one-command VPS deploy.
#
# Installs FullAgent + the cloud server on this machine, exposes it through
# a Cloudflare quick tunnel, and prints the URL + token your phone needs.
#
# Run (after every VPS reset, too — it's idempotent):
#   curl -fsSL https://raw.githubusercontent.com/cmyolo441-coder/py.test/main/bootstrap-cloud.sh | bash
#
# Your phone (Termux) then needs only the tiny `fac` client:
#   curl -fsSL https://raw.githubusercontent.com/cmyolo441-coder/py.test/main/fac -o ~/.local/bin/fac
#   chmod +x ~/.local/bin/fac
#   fac --set-url <URL below> --token <token below>
#   fac "hello"
# ============================================================================
set -euo pipefail

REPO="cmyolo441-coder/py.test"
RAW="https://raw.githubusercontent.com/${REPO}/main"
PORT="${FULLAGENT_CLOUD_PORT:-18791}"
DIR="$HOME/.fullagent-cloud"
VENV="$HOME/.fullagent-venv"
WHEEL_URL="https://github.com/${REPO}/releases/download/v1.0.0/fullagent-3.1.0-py3-none-any.whl"
# NOTE: the x86_64 standalone tarball on the release is STALE — always use the wheel.

mkdir -p "$DIR" "$HOME/.local/bin"
export PATH="$HOME/.local/bin:$PATH"

echo ">> [1/6] Stopping any previous cloud server/tunnel..."
pkill -f "cloudserve\.py" 2>/dev/null || true
# Kill ONLY our own previous tunnel (matched by its exact --url). Never use a
# bare "cloudflared tunnel" pattern here: on 2Z2-style sandbox VMs the remote
# desktop itself is served by another cloudflared tunnel, and killing it
# drops the user's desktop access mid-deploy.
pkill -f "cloudflared tunnel --url http://127.0.0.1:${PORT}" 2>/dev/null || true
sleep 1

echo ">> [2/6] Python environment..."
command -v python3 >/dev/null 2>&1 || { echo "!! python3 not found" >&2; exit 1; }
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV" || { echo "!! venv failed — try: sudo apt install python3-venv" >&2; exit 1; }
fi
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q "$WHEEL_URL"
echo "    fullagent installed."

echo ">> [3/6] Downloading cloud server..."
curl -fsSL "$RAW/cloudserve.py" -o "$DIR/cloudserve.py"

echo ">> [4/6] cloudflared..."
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64|amd64) CFARCH="amd64" ;;
  aarch64|arm64) CFARCH="arm64" ;;
  *) echo "!! unsupported arch: $ARCH" >&2; exit 1 ;;
esac
if ! command -v cloudflared >/dev/null 2>&1; then
  curl -fsSL "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${CFARCH}" \
    -o "$HOME/.local/bin/cloudflared"
  chmod +x "$HOME/.local/bin/cloudflared"
fi
echo "    cloudflared ready."

echo ">> [5/6] Starting server (port $PORT)..."
: > "$DIR/server.log"
nohup "$VENV/bin/python" "$DIR/cloudserve.py" --port "$PORT" \
  >"$DIR/server.log" 2>&1 < /dev/null &
for _i in $(seq 1 30); do
  grep -q "agent ready" "$DIR/server.log" 2>/dev/null && break
  sleep 1
done
if ! grep -q "agent ready" "$DIR/server.log" 2>/dev/null; then
  echo "!! server failed to start — tail of server.log:" >&2
  tail -20 "$DIR/server.log" >&2 || true
  exit 1
fi
TOKEN="$(cat "$DIR/token")"
echo "    server running."

echo ">> [6/6] Opening Cloudflare tunnel..."
nohup cloudflared tunnel --url "http://127.0.0.1:${PORT}" \
  >"$DIR/tunnel.log" 2>&1 < /dev/null &
URL=""
for i in $(seq 1 40); do
  URL="$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$DIR/tunnel.log" 2>/dev/null | head -1 || true)"
  [ -n "$URL" ] && break
  sleep 3
done
if [ -z "$URL" ]; then
  echo "!! tunnel URL not found — tail of tunnel.log:" >&2
  tail -20 "$DIR/tunnel.log" >&2 || true
  exit 1
fi

cat <<EOF

================================================================
  FullAgent Cloud is LIVE  🚀
================================================================
  Server URL : $URL
  Token      : $TOKEN

  On your phone (Termux), run:
    curl -fsSL $RAW/fac -o ~/.local/bin/fac
    chmod +x ~/.local/bin/fac
    fac --set-url $URL --token $TOKEN
    fac "hello, what can you do?"

  Logs: $DIR/server.log  $DIR/tunnel.log
  NOTE: this tunnel URL is temporary — if the VPS resets or the
  tunnel drops, just re-run the bootstrap command for a fresh URL.
================================================================
EOF
