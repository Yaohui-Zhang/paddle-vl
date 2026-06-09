#!/bin/bash
# Validate 4-worker throughput on 100 papers (offsets 0/50/100/150 over the 200-paper set), 4 batches x 25.
# Cold start: restart vLLM first to clear prefix cache; papers are never-tested too.
set +e
PY=python3
PADDLEOCR=paddleocr
MODEL=/workspace/.cache/paddle/official_models/PaddleOCR-VL-1.6
export PADDLE_PDX_CACHE_HOME=/workspace/.cache/paddle
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

echo "===== [1/2] COLD RESTART vLLM (clears prefix cache) ====="
pkill -9 -f genai_server 2>/dev/null; pkill -9 -f EngineCore 2>/dev/null
# wait GPU free
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | grep -oE '^[0-9]+')" -ge 5000 ]; do sleep 2; done
: > /workspace/vllm_server.log
# First run: if the model dir is missing, OMIT --model_dir so paddlex auto-downloads it.
# The auto-download target ($PADDLE_PDX_CACHE_HOME/official_models/PaddleOCR-VL-1.6) is
# exactly $MODEL, so later runs find the cache and use --model_dir directly.
MODEL_ARGS=(--model_dir "$MODEL")
TIMEOUT=300
if [ ! -d "$MODEL" ]; then
  echo "model not found locally -> will auto-download to $MODEL (first run, allow extra time)"
  MODEL_ARGS=()
  TIMEOUT=1800
fi
nohup "$PADDLEOCR" genai_server \
  --model_name PaddleOCR-VL-1.6-0.9B "${MODEL_ARGS[@]}" \
  --backend vllm --host 0.0.0.0 --port 8118 --backend_config /workspace/backend_config.yaml \
  > /workspace/vllm_server.log 2>&1 &
disown
# wait ready
s=0
until curl -s --max-time 3 http://localhost:8118/v1/models 2>/dev/null | grep -q PaddleOCR; do
  sleep 6; s=$((s+6)); [ $s -ge $TIMEOUT ] && { echo "SERVER TIMEOUT"; exit 1; }
done
echo "server ready (${s}s), api servers: $(grep -ac 'Started server process' /workspace/vllm_server.log)"
echo "prefix cache hit at start should be ~0% (cold)"

echo "===== [2/2] 4 batches x 25 papers, 4 workers, conc 64 ====="
for off in 0 50 100 150; do
  echo "######## batch offset=$off (25 papers) ########"
  "$PY" /workspace/run.py 25 4 64 "$off" 2>&1 \
    | grep -aE '^\[cfg\]|====|^\[err\]' | grep -vaE 'ccache'
  echo
done
echo "==== BATCH TEST DONE ===="