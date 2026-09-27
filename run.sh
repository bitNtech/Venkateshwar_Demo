#!/usr/bin/env bash
#
# Start the VAD + ASR pipeline. One command, from anywhere:
#
#     ./run.sh
#
# Finds Python 3.10/3.11, builds .venv if missing, installs requirements only
# when they are actually missing, installs the AI4Bharat NeMo fork, seeds .env,
# starts the API, waits for it, reports whether ASR loaded, and prints the
# console URL. Ctrl+C stops it cleanly.
#
#   BACKEND_PORT=9000 ./run.sh     serve on another port
#   BACKEND_HOST=0.0.0.0 ./run.sh  bind elsewhere (read the warning it prints)
#   REINSTALL=1 ./run.sh           force a dependency reinstall
#   RELOAD=1 ./run.sh              uvicorn --reload (development)
#
set -Eeuo pipefail

# Works no matter where it is invoked from, including through a symlink.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_DIR="$ROOT_DIR/.venv"
LOG_DIR="$ROOT_DIR/logs"

# ------------------------------------------------------------------ .env ----
# Seeded and loaded HERE, before anything below reads a variable out of it.
# backend/__init__.py loads .env for the PYTHON process, but this shell needs
# it too: BACKEND_PORT and BACKEND_HOST are read by this script.
#
# Parsed line by line rather than sourced: a .env is data, and sourcing it
# executes it. An unquoted value with a space in it - a comma-separated
# CORS_ALLOW_ORIGINS is the realistic one - would run its second word as a
# command. A variable already exported into this shell still wins over the
# file, so `BACKEND_PORT=9000 ./run.sh` keeps working, matching dotenv.
if [[ ! -f "$ROOT_DIR/.env" && -f "$ROOT_DIR/.env.example" ]]; then
  cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
  printf '\033[36m==>\033[0m %s\n' "Created .env from .env.example"
  printf '\033[33m !\033[0m %s\n' "Set HF_TOKEN in .env for the microphone (the ASR model is gated)." >&2
  printf '\033[33m !\033[0m %s\n' "  (Not needed if the model is already in the Hugging Face cache.)" >&2
fi
if [[ -f "$ROOT_DIR/.env" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"                                    # CRLF-safe
    [[ "$line" =~ ^[[:space:]]*(#|$) ]] && continue
    [[ "$line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    env_key="${BASH_REMATCH[2]}"
    env_value="${BASH_REMATCH[3]}"
    env_value="${env_value%"${env_value##*[![:space:]]}"}"  # trailing space
    case "$env_value" in
      \"*\"|\'*\') env_value="${env_value:1:${#env_value}-2}" ;;
    esac
    [[ -n "${!env_key+set}" ]] || export "$env_key=$env_value"
  done < "$ROOT_DIR/.env"
  unset env_key env_value line
fi

BACKEND_PORT="${BACKEND_PORT:-8000}"
# logs/server.log for the default port, because that path is what HANDOFF.md
# and every debugging note refer to. A second instance on another port gets its
# own file rather than overwriting the first one's - two servers clobbering one
# log is how you end up reading the wrong process's output.
if [[ "$BACKEND_PORT" == "8000" ]]; then
  LOG_FILE="$LOG_DIR/server.log"
else
  LOG_FILE="$LOG_DIR/server-$BACKEND_PORT.log"
fi
# 127.0.0.1, never "localhost": on Windows "localhost" can try ::1 first and
# wait out a timeout before falling back.
BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
mkdir -p "$LOG_DIR"

say()  { printf '\033[36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33m !\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31m x\033[0m %s\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------- python ----
# IndicConformer's NeMo build has no wheels above 3.11 - numba and editdistance
# try to compile from source and fail - so the range is pinned rather than
# "3.10 or newer".
SUPPORTED='import sys; raise SystemExit(not ((3,10) <= sys.version_info[:2] <= (3,11)))'
PYTHON_CMD=()

find_python() {
  local candidate version
  for candidate in python3.11 python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c "$SUPPORTED" >/dev/null 2>&1; then
      PYTHON_CMD=("$candidate"); return
    fi
  done
  for version in 3.11 3.10; do
    if command -v py >/dev/null 2>&1 && py "-$version" -c "$SUPPORTED" >/dev/null 2>&1; then
      PYTHON_CMD=(py "-$version"); return
    fi
  done
  die "Python 3.10 or 3.11 is required (NeMo does not support 3.12+).
     Install from https://www.python.org/downloads/ with 'Add Python to PATH'."
}

if [[ ! -d "$VENV_DIR" ]]; then
  find_python
  say "Creating .venv with ${PYTHON_CMD[*]}"
  "${PYTHON_CMD[@]}" -m venv "$VENV_DIR"
fi

# Windows venvs put the interpreter in Scripts/, which is what Git Bash sees.
if   [[ -x "$VENV_DIR/bin/python"        ]]; then PY="$VENV_DIR/bin/python"
elif [[ -x "$VENV_DIR/Scripts/python.exe" ]]; then PY="$VENV_DIR/Scripts/python.exe"
else die ".venv is incomplete. Delete it and run again."
fi

"$PY" -c "$SUPPORTED" >/dev/null 2>&1 \
  || die ".venv was built with an unsupported Python (NeMo needs 3.10 or 3.11).
     Delete .venv and run again."

# ----------------------------------------------------------- dependencies ----
# Checked by import, not reinstalled every run: a full pip pass over this
# requirements file takes minutes and the point of this script is that it can
# be run casually.
deps_present() {
  "$PY" - <<'EOF' >/dev/null 2>&1
import importlib.util as u
raise SystemExit(any(u.find_spec(m) is None for m in ("fastapi", "uvicorn", "soundfile", "ten_vad", "websockets")))
EOF
}

if [[ -n "${REINSTALL:-}" ]] || ! deps_present; then
  say "Installing requirements (first run takes a while)"
  "$PY" -m pip install --upgrade pip --quiet
  "$PY" -m pip install -r "$ROOT_DIR/requirements.txt"
else
  say "Dependencies present"
fi

# IndicConformer needs the AI4Bharat NeMo fork; stock nemo-toolkit cannot load
# its multilingual tokenizer. --no-deps skips Windows-incompatible extras.
if ! "$PY" -c "import nemo.collections.asr" >/dev/null 2>&1; then
  if [[ -d "$ROOT_DIR/NeMo_ai4bharat" ]]; then
    say "Installing the AI4Bharat NeMo fork"
    "$PY" -m pip install -e "$ROOT_DIR/NeMo_ai4bharat" --no-deps
  else
    warn "NeMo_ai4bharat/ is missing - VAD will run but there will be no transcripts."
    warn "  git clone --depth 1 https://github.com/AI4Bharat/NeMo.git NeMo_ai4bharat"
  fi
fi

# ----------------------------------------------------------------- serve ----
# A SERVER ALREADY ON THIS PORT IS A TRAP, not an inconvenience. It caches the
# prompts and the code it started with, so a stale one answers every request
# and every check passes against code that is no longer on disk. That has cost
# a debugging session; uvicorn's own bind error scrolls past in a log file.
if curl -fsS -m 3 "http://127.0.0.1:$BACKEND_PORT/api/health" >/dev/null 2>&1; then
  die "Something is already serving on port $BACKEND_PORT, and it is running the
     code it STARTED with, not the code on disk. Stop it first:
       Windows:  Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" |
                   Where-Object { \$_.CommandLine -like '*uvicorn*' } |
                   ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }
       Linux:    pkill -f 'uvicorn backend.main:app'
     Or serve elsewhere:  BACKEND_PORT=9000 ./run.sh"
fi

if [[ "$BACKEND_HOST" != "127.0.0.1" && "$BACKEND_HOST" != "localhost" ]]; then
  # Measured: binding 0.0.0.0 does not get you a working console from another
  # machine. Browsers only allow getUserMedia on localhost or HTTPS, so the
  # microphone dies on a plain-HTTP LAN origin - port-forward instead:
  #   ssh -L $BACKEND_PORT:127.0.0.1:$BACKEND_PORT user@host
  warn "Binding $BACKEND_HOST: the console's microphone will NOT work over plain"
  warn "  HTTP from another machine (getUserMedia needs localhost or HTTPS)."
  warn "  Prefer: ssh -L $BACKEND_PORT:127.0.0.1:$BACKEND_PORT user@host"
  [[ -z "${AUDIO_WS_AUTH_TOKEN:-}" ]] && \
    warn "  AUDIO_WS_AUTH_TOKEN is unset, so /ws/audio has NO auth on that interface."
fi

# stdout AND stderr to a file: several real bugs in this project were only ever
# visible in the server log, and a backgrounded process has no console.
RELOAD_FLAG=()
[[ -n "${RELOAD:-}" ]] && RELOAD_FLAG=(--reload)

say "Starting API on $BACKEND_HOST:$BACKEND_PORT (log: ${LOG_FILE#$ROOT_DIR/})"
"$PY" -m uvicorn backend.main:app --host "$BACKEND_HOST" --port "$BACKEND_PORT" "${RELOAD_FLAG[@]}" \
  > "$LOG_FILE" 2>&1 &
SERVER_PID=$!

cleanup() {
  printf '\n'
  say "Stopping"
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ASR model loading takes a while on a cold start; poll rather
# than guess, and fail loudly if the process dies during it.
say "Loading models..."
for _ in $(seq 1 60); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    warn "The server exited during startup. Last lines of the log:"
    tail -20 "$LOG_FILE" >&2
    exit 1
  fi
  HEALTH="$(curl -fsS -m 3 "http://127.0.0.1:$BACKEND_PORT/api/health" 2>/dev/null || true)"
  [[ -n "$HEALTH" ]] && break
  sleep 2
done

if [[ -z "${HEALTH:-}" ]]; then
  warn "The server did not become healthy in time. Last lines of the log:"
  tail -20 "$LOG_FILE" >&2
  exit 1
fi

# A server that answers requests is not evidence that ASR loaded - say so.
printf '\n'
"$PY" - "$HEALTH" <<'EOF'
import json, sys
h = json.loads(sys.argv[1])
ok = bool(h.get("asr_ready"))
print(f"    {'OK  ' if ok else 'DOWN'}  speech-to-text ({h.get('asr_language')}, {h.get('asr_decoding')})"
      + ("" if ok else "   (see logs/server.log)"))
EOF

cat <<EOF

    Console:  http://localhost:$BACKEND_PORT/console

    Ctrl+C to stop.

EOF

wait "$SERVER_PID"
