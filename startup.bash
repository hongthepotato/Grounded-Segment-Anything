#!/bin/bash

export TOKENIZERS_PARALLELISM=false

# 启动 Redis (如果没运行)
if ! pgrep -x "redis-server" > /dev/null; then
    redis-server --daemonize yes
    echo "✅ Redis started"
fi

# 启动 FastAPI (后台)
uvicorn api.app:app --host 0.0.0.0 --port 8080 &
echo "✅ FastAPI started on :8000"

# 自动检测GPU数量并启动Workers
GPU_COUNT=$(nvidia-smi -L | wc -l)
echo "🔍 Detected $GPU_COUNT GPUs"

for gpu_id in $(seq 0 $((GPU_COUNT - 1))); do
    python -m ml_engine.jobs.worker --gpu $gpu_id &
    echo "✅ Worker started on GPU $gpu_id"
done

echo ""
echo "🚀 All services running!"
echo "   API:  http://localhost:8080/docs"
echo "   Workers: $GPU_COUNT (GPU 0-$((GPU_COUNT - 1)))"
echo "   Logs: tail -f logs/*.log"
