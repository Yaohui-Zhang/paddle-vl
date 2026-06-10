#!/bin/bash
# General entry point: ensure the vLLM server is up, then run run.py.
# Args are passed straight through to run.py (same interface):
#   run.sh [pdfs/dirs ...] [--workers N] [--vl-conc N] [--server URL] [-o OUT]
# Example:
#   ./run.sh /workspace/papers --workers 4 --vl-conc 64
set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=python3
PADDLEOCR=paddleocr
MODEL=/workspace/.cache/paddle/official_models/PaddleOCR-VL-1.6
PORT=8118
LOG=/workspace/vllm_server.log
export PADDLE_PDX_CACHE_HOME=/workspace/.cache/paddle
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

if [ "$#" -eq 0 ]; then
  echo "usage: $0 [pdfs/dirs ...] [run.py options]" >&2
  exit 2
fi

# Always cold-restart: kill any running vLLM, wait for the GPU to free, then start fresh.
echo "===== restarting vLLM server on :$PORT ====="
pkill -9 -f genai_server 2>/dev/null || true; pkill -9 -f EngineCore 2>/dev/null || true
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | grep -oE '^[0-9]+')" -ge 5000 ]; do sleep 2; done

# First run: if the model dir is missing, OMIT --model_dir so paddlex auto-downloads it
# (to $MODEL); later runs find the cache and use --model_dir directly.
MODEL_ARGS=(--model_dir "$MODEL")
TIMEOUT=300
if [ ! -d "$MODEL" ]; then
  echo "model not found locally -> will auto-download to $MODEL (first run, allow extra time)"
  MODEL_ARGS=()
  TIMEOUT=1800
fi
: > "$LOG"
nohup "$PADDLEOCR" genai_server \
  --model_name PaddleOCR-VL-1.6-0.9B "${MODEL_ARGS[@]}" \
  --backend vllm --host 0.0.0.0 --port "$PORT" --backend_config "$HERE/backend_config.yaml" \
  > "$LOG" 2>&1 &
disown
s=0
until curl -s --max-time 3 "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q PaddleOCR; do
  sleep 6; s=$((s+6)); [ $s -ge $TIMEOUT ] && { echo "SERVER TIMEOUT (see $LOG)"; exit 1; }
done
echo "server ready (${s}s)"

echo "===== running run.py ====="
exec "$PY" "$HERE/run.py" "$@"
