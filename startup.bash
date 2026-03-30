#!/usr/bin/env bash

set -euo pipefail

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export REDIS_URL="${REDIS_URL:-redis://redis:6379}"

mkdir -p logs data/models/pretrained

if [ -x "scripts/download_models.sh" ]; then
    echo "Checking pretrained model files..."
    ./scripts/download_models.sh
fi

# Keep child PIDs so we can stop everything cleanly on container shutdown.
PIDS=()

cleanup() {
    echo "Stopping services..."
    for pid in "${PIDS[@]:-}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    wait || true
    echo "All services stopped."
}

trap cleanup SIGINT SIGTERM

uvicorn api.app:app --host 0.0.0.0 --port 8080 >> logs/api.log 2>&1 &
API_PID=$!
PIDS+=("$API_PID")
echo "FastAPI started on :8080 (pid=$API_PID)"

GPU_COUNT=0
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT="$(nvidia-smi -L | wc -l | tr -d ' ')"
fi
if [ -n "${GPU_WORKERS:-}" ] && [ "${GPU_WORKERS}" -lt "${GPU_COUNT}" ]; then
    GPU_COUNT="${GPU_WORKERS}"
fi
echo "Detected ${GPU_COUNT} GPU worker(s)"

if [ "${GPU_COUNT}" -gt 0 ]; then
    for gpu_id in $(seq 0 $((GPU_COUNT - 1))); do
        python -m ml_engine.jobs.worker --redis-url "${REDIS_URL}" --gpu "$gpu_id" >> "logs/worker_${gpu_id}.log" 2>&1 &
        worker_pid=$!
        PIDS+=("$worker_pid")
        echo "Worker started on GPU ${gpu_id} (pid=${worker_pid})"
    done
else
    echo "No GPU detected; worker processes are not started."
fi

echo ""
echo "All services running."
echo "  API:  http://localhost:8080/docs"
echo "  Redis URL: ${REDIS_URL}"
echo "  Workers: ${GPU_COUNT}"
echo "  Logs: tail -f logs/*.log"

wait -n
cleanup
