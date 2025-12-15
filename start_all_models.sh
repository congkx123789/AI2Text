#!/usr/bin/env bash
# Start/stop all backend services and frontend for AI2Text
# Services:
#   - AI2Text Transformer (8000)
#   - CTC ASR (8001)
#   - Whisper CT2 finetune (8002)
#   - Whisper HF base (8003)
#   - Frontend static (8080)

set -euo pipefail

ROOT="/home/alida/Documents/Cursor/AI2Text"
LOG_DIR="$ROOT/logs/run"
mkdir -p "$LOG_DIR"

AI2TEXT_CMD=(uvicorn api.app:app --host 0.0.0.0 --port 8000)
CTC_CMD=("$ROOT/ai-llm/.venv/bin/python" -m uvicorn src.asr.api:app --host 0.0.0.0 --port 8001)
WHISPER_CT2_CMD=(bash -lc "cd \"$ROOT/ai-llm\" && source .venv/bin/activate && ASR_MODEL=\"$ROOT/ai-llm/models/final/whisper-vi-en-ct2\" ASR_DEVICE=cpu ASR_COMPUTE=int8 uvicorn src.api.server:app --host 0.0.0.0 --port 8002")
WHISPER_BASE_CMD=(bash -lc "cd \"$ROOT/ai-llm\" && source .venv/bin/activate && ASR_MODEL=\"$ROOT/ai-llm/models/base/whisper-small\" ASR_DEVICE=cpu ASR_COMPUTE=int8 uvicorn src.api.server:app --host 0.0.0.0 --port 8003")
FRONTEND_CMD=(python3 -m http.server 8080)

pidfile() { echo "$LOG_DIR/$1.pid"; }
logfile() { echo "$LOG_DIR/$1.log"; }

is_running() {
  local pid_file="$1"
  [[ -f "$pid_file" ]] || return 1
  local pid
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

stop_one() {
  local name="$1"
  local pf
  pf="$(pidfile "$name")"
  if is_running "$pf"; then
    local pid
    pid="$(cat "$pf")"
    echo "Stopping $name (pid $pid)..."
    kill "$pid" 2>/dev/null || true
    sleep 1
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  fi
  rm -f "$pf"
}

start_one() {
  local name="$1"; shift
  local workdir="$1"; shift
  local -a cmd=("$@")
  local pf lf
  pf="$(pidfile "$name")"
  lf="$(logfile "$name")"
  stop_one "$name"
  echo "Starting $name ..."
  (
    cd "$workdir"
    nohup "${cmd[@]}" >"$lf" 2>&1 &
    echo $! >"$pf"
  )
}

status_one() {
  local name="$1"
  local pf
  pf="$(pidfile "$name")"
  if is_running "$pf"; then
    echo "$name: running (pid $(cat "$pf"))"
  else
    echo "$name: stopped"
  fi
}

case "${1:-}" in
  start)
    start_one ai2text        "$ROOT/AI2Text"   "${AI2TEXT_CMD[@]}"
    start_one ctc            "$ROOT/ai-llm-ss" "${CTC_CMD[@]}"
    start_one whisper_ct2    "$ROOT/ai-llm"    "${WHISPER_CT2_CMD[@]}"
    start_one whisper_base   "$ROOT/ai-llm"    "${WHISPER_BASE_CMD[@]}"
    start_one frontend       "$ROOT/frontend"  "${FRONTEND_CMD[@]}"
    ;;
  stop)
    stop_one frontend
    stop_one whisper_base
    stop_one whisper_ct2
    stop_one ctc
    stop_one ai2text
    ;;
  restart)
    "$0" stop
    "$0" start
    ;;
  status)
    status_one ai2text
    status_one ctc
    status_one whisper_ct2
    status_one whisper_base
    status_one frontend
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status}"
    exit 1
    ;;
esac

