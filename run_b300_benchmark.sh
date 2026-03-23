#!/usr/bin/env bash
#
# Run InferenceX Kimi-K2.5 FP4 benchmark on a p6-b300.48xlarge EC2 instance.
#
# Prerequisites:
#   - Docker with NVIDIA runtime
#   - Model downloaded to $MODEL_DIR (default: /opt/dlami/nvme/models/Kimi-K2.5-NVFP4)
#
# Usage:
#   bash run_b300_benchmark.sh [--model-dir /path/to/model] [--results-dir /path/to/results]
#
set -euo pipefail

# --- Configuration ---
IMAGE="${IMAGE:-vllm/vllm-openai:v0.18.0-cu130}"
MODEL_HF="nvidia/Kimi-K2.5-NVFP4"
MODEL_DIR="${MODEL_DIR:-/opt/dlami/nvme/models/Kimi-K2.5-NVFP4}"
RESULTS_HOST="${RESULTS_DIR:-/opt/dlami/nvme/inferencex_results}"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

# Benchmark parameters (matching InferenceX B200 config)
TP=4
MAX_MODEL_LEN=4096
RANDOM_RANGE_RATIO=0.8
ISL=1024
OSL=1024
CONC_LIST="${CONC_LIST:-4 8 16 32 64}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-dir)    MODEL_DIR="$2"; shift 2 ;;
        --results-dir)  RESULTS_HOST="$2"; shift 2 ;;
        --image)        IMAGE="$2"; shift 2 ;;
        --conc)         CONC_LIST="$2"; shift 2 ;;
        *)              echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$RESULTS_HOST"

echo "============================================"
echo " InferenceX Kimi-K2.5 FP4 Benchmark (B300)"
echo "============================================"
echo "Image:      $IMAGE"
echo "Model:      $MODEL_HF"
echo "Model dir:  $MODEL_DIR"
echo "TP:         $TP"
echo "ISL/OSL:    ${ISL}/${OSL}"
echo "Conc list:  $CONC_LIST"
echo "Results:    $RESULTS_HOST"
echo "============================================"

# Verify model exists
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
    echo "Model not found at $MODEL_DIR. Downloading..."
    pip install -q huggingface_hub 2>/dev/null || true
    huggingface-cli download "$MODEL_HF" --local-dir "$MODEL_DIR"
fi

# Verify Docker image
if ! docker image inspect "$IMAGE" &>/dev/null; then
    echo "Pulling Docker image $IMAGE..."
    docker pull "$IMAGE"
fi

# Run benchmark for each concurrency level
for CONC in $CONC_LIST; do
    RESULT_FILENAME="kimik2.5_fp4_vllm_tp${TP}_ep1_dpa_false_conc${CONC}_b300"

    echo ""
    echo ">>> Benchmark: concurrency=$CONC, result=$RESULT_FILENAME"
    echo ""

    docker run --rm \
        --gpus all \
        --ipc=host \
        --network=host \
        -v "$MODEL_DIR":/workspace/model:ro \
        -v "$REPO_DIR":/workspace/inferencex:ro \
        -v "$RESULTS_HOST":/workspace/results \
        -e MODEL=/workspace/model \
        -e TP="$TP" \
        -e CONC="$CONC" \
        -e ISL="$ISL" \
        -e OSL="$OSL" \
        -e MAX_MODEL_LEN="$MAX_MODEL_LEN" \
        -e RANDOM_RANGE_RATIO="$RANDOM_RANGE_RATIO" \
        -e RESULT_FILENAME="$RESULT_FILENAME" \
        -e PYTHONNOUSERSITE=1 \
        -e TORCH_CUDA_ARCH_LIST="10.0 10.3" \
        -w /workspace/inferencex \
        --entrypoint bash \
        "$IMAGE" \
        -c '
            # The "hf download" in the script expects huggingface-cli;
            # model is already downloaded, so stub it out
            hf() { echo "Model pre-downloaded at $MODEL, skipping hf download"; }
            export -f hf

            # Override result dir to mounted volume
            export RESULT_DIR_OVERRIDE=/workspace/results

            source benchmarks/benchmark_lib.sh

            nvidia-smi

            SERVER_LOG=/workspace/server.log
            PORT=8888

            start_gpu_monitor

            set -x
            vllm serve $MODEL --host 0.0.0.0 --port $PORT \
                --tensor-parallel-size=$TP \
                --gpu-memory-utilization 0.90 \
                --max-model-len $MAX_MODEL_LEN \
                --max-num-seqs $CONC \
                --reasoning-parser kimi_k2 \
                --tool-call-parser kimi_k2 \
                --compilation_config.pass_config.fuse_allreduce_rms true \
                --trust-remote-code > $SERVER_LOG 2>&1 &

            SERVER_PID=$!

            wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

            pip install -q datasets pandas 2>/dev/null

            run_benchmark_serving \
                --model "$MODEL" \
                --port "$PORT" \
                --backend vllm \
                --input-len "$ISL" \
                --output-len "$OSL" \
                --random-range-ratio "$RANDOM_RANGE_RATIO" \
                --num-prompts $(( CONC * 10 )) \
                --max-concurrency "$CONC" \
                --result-filename "$RESULT_FILENAME" \
                --result-dir /workspace/results \
                --trust-remote-code

            stop_gpu_monitor

            # Copy GPU metrics to results
            cp -f /workspace/gpu_metrics.csv /workspace/results/gpu_metrics_conc${CONC}.csv 2>/dev/null || true
            set +x
        '

    echo ">>> Completed concurrency=$CONC"

    # Check if result file was created
    if [[ -f "$RESULTS_HOST/${RESULT_FILENAME}.json" ]]; then
        echo "    Result: $RESULTS_HOST/${RESULT_FILENAME}.json"
    else
        echo "    WARNING: Result file not found!"
    fi
done

echo ""
echo "============================================"
echo " Benchmark Summary"
echo "============================================"
echo ""
echo "| Concurrency | Output tok/s | tok/s/gpu (4 GPUs) | Mean TPOT (ms) | Mean TTFT (ms) |"
echo "|:-----------:|:------------:|:------------------:|:--------------:|:--------------:|"

for CONC in $CONC_LIST; do
    RESULT_FILENAME="kimik2.5_fp4_vllm_tp${TP}_ep1_dpa_false_conc${CONC}_b300"
    RESULT_FILE="$RESULTS_HOST/${RESULT_FILENAME}.json"
    if [[ -f "$RESULT_FILE" ]]; then
        python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
tput = d.get('output_throughput', 0)
ttft = d.get('mean_ttft_ms', 0)
tpot = d.get('mean_tpot_ms', 0)
print(f'| {sys.argv[2]:^11} | {tput:>12.1f} | {tput/4:>18.1f} | {tpot:>14.2f} | {ttft:>14.2f} |')
" "$RESULT_FILE" "$CONC" 2>/dev/null || echo "| $CONC | (parse error) | | | |"
    else
        echo "| $CONC | (no data) | | | |"
    fi
done

echo ""
echo "Results saved to: $RESULTS_HOST/"
echo "============================================"
