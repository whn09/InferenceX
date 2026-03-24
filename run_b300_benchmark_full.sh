#!/usr/bin/env bash
#
# Run InferenceX Kimi-K2.5 full benchmark matrix on AWS p6-b300.48xlarge.
#
# Reproduces the B200 InferenceX test matrix:
#   - Precision: fp4 (TP=4 EP=4, TP=8 EP=1) and int4 (TP=8 EP=1)
#   - ISL/OSL:   1024/1024, 1024/8192, 8192/1024
#   - Concurrency: 4, 8, 16, 32, 64
#
# Prerequisites:
#   - Docker with NVIDIA runtime
#   - FP4 model at $FP4_MODEL_DIR (default: /opt/dlami/nvme/models/Kimi-K2.5-NVFP4)
#   - INT4 model at $INT4_MODEL_DIR (default: /opt/dlami/nvme/models/Kimi-K2.5-NVINT4)
#
# Usage:
#   bash run_b300_benchmark_full.sh [--fp4-model-dir /path] [--int4-model-dir /path]
#                                   [--results-dir /path] [--image IMAGE]
#                                   [--filter FILTER]
#
# Filter examples:
#   --filter "fp4_tp4_ep4"           Only run fp4 TP=4 EP=4 configs
#   --filter "int4"                  Only run int4 configs
#   --filter "1024_1024"             Only run ISL=1024/OSL=1024
#   --filter "conc64"                Only run concurrency=64
#   --filter "fp4_tp4_ep4_1024_1024_conc64"  Single specific run
#
set -euo pipefail

# --- Configuration ---
IMAGE="${IMAGE:-vllm/vllm-openai:v0.18.0-cu130}"
FP4_MODEL_HF="nvidia/Kimi-K2.5-NVFP4"
INT4_MODEL_HF="nvidia/Kimi-K2.5-NVINT4"
FP4_MODEL_DIR="${FP4_MODEL_DIR:-/opt/dlami/nvme/models/Kimi-K2.5-NVFP4}"
INT4_MODEL_DIR="${INT4_MODEL_DIR:-/opt/dlami/nvme/models/Kimi-K2.5-NVINT4}"
RESULTS_HOST="${RESULTS_DIR:-/opt/dlami/nvme/inferencex_results}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")" && pwd)}"
FILTER=""
RANDOM_RANGE_RATIO=0.8

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --fp4-model-dir)   FP4_MODEL_DIR="$2"; shift 2 ;;
        --int4-model-dir)  INT4_MODEL_DIR="$2"; shift 2 ;;
        --results-dir)     RESULTS_HOST="$2"; shift 2 ;;
        --image)           IMAGE="$2"; shift 2 ;;
        --filter)          FILTER="$2"; shift 2 ;;
        *)                 echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$RESULTS_HOST"

# --- Test Matrix ---
# Format: "PRECISION TP EP ISL OSL MAX_MODEL_LEN GPU_MEM_UTIL CONC_LIST"
#
# From InferenceX B200 configs (nvidia-master.yaml):
#   fp4 TP=4 EP=4: 1K/1K (c=4-64), 1K/8K (c=4-64), 8K/1K (c=4-64)
#   fp4 TP=8 EP=1: 1K/1K (c=4), 8K/1K (c=4)
#   int4 TP=8 EP=1: 1K/1K (c=4-64), 1K/8K (c=4-64), 8K/1K (c=4-64)

CONFIGS=(
    # --- fp4, TP=4, EP=4 ---
    "fp4 4 4 1024 1024 4096  0.90 4 8 16 32 64"
    "fp4 4 4 1024 8192 16384 0.90 4 8 16 32 64"
    "fp4 4 4 8192 1024 16384 0.90 4 8 16 32 64"
    # --- fp4, TP=8, EP=1 ---
    "fp4 8 1 1024 1024 4096  0.90 4"
    "fp4 8 1 8192 1024 16384 0.90 4"
    # --- int4, TP=8, EP=1 ---
    "int4 8 1 1024 1024 4096  0.95 4 8 16 32 64"
    "int4 8 1 1024 8192 16384 0.95 4 8 16 32 64"
    "int4 8 1 8192 1024 16384 0.95 4 8 16 32 64"
)

echo "============================================"
echo " InferenceX Kimi-K2.5 Full Benchmark (B300)"
echo "============================================"
echo "Image:       $IMAGE"
echo "FP4 model:   $FP4_MODEL_DIR"
echo "INT4 model:  $INT4_MODEL_DIR"
echo "Results:     $RESULTS_HOST"
echo "Filter:      ${FILTER:-<none>}"
echo "Configs:     ${#CONFIGS[@]} groups"
echo "============================================"

run_single_benchmark() {
    local precision=$1 tp=$2 ep=$3 isl=$4 osl=$5 max_model_len=$6 gpu_mem=$7 conc=$8

    # Build filter tag for matching
    local tag="${precision}_tp${tp}_ep${ep}_${isl}_${osl}_conc${conc}"
    if [[ -n "$FILTER" && "$tag" != *"$FILTER"* ]]; then
        return 0
    fi

    # Select model dir
    local model_dir
    if [[ "$precision" == "fp4" ]]; then
        model_dir="$FP4_MODEL_DIR"
        if [[ ! -f "$model_dir/config.json" ]]; then
            echo "SKIP: FP4 model not found at $model_dir"
            return 0
        fi
    else
        model_dir="$INT4_MODEL_DIR"
        if [[ ! -f "$model_dir/config.json" ]]; then
            echo "SKIP: INT4 model not found at $model_dir"
            return 0
        fi
    fi

    # EP flag
    local ep_flag=""
    if [[ "$ep" -gt 1 ]]; then
        ep_flag="--enable-expert-parallel"
    fi

    # Extra flags per precision
    local extra_flags=""
    if [[ "$precision" == "int4" ]]; then
        extra_flags="--disable-log-requests"
    fi

    local result_filename="kimik2.5_${precision}_vllm_tp${tp}_ep${ep}_dpa_false_conc${conc}_b300"

    echo ""
    echo ">>> Benchmark: ${precision} TP=${tp} EP=${ep} ISL=${isl} OSL=${osl} conc=${conc}"
    echo "    Result: ${result_filename}"
    echo ""

    docker run --rm \
        --gpus all \
        --ipc=host \
        --network=host \
        -v "$model_dir":/workspace/model:ro \
        -v "$REPO_DIR":/workspace/inferencex:ro \
        -v "$RESULTS_HOST":/workspace/results \
        -e TORCH_CUDA_ARCH_LIST="10.0 10.3" \
        -e PYTHONNOUSERSITE=1 \
        -w /workspace/inferencex \
        --entrypoint bash \
        "$IMAGE" \
        -c "
            source benchmarks/benchmark_lib.sh

            SERVER_LOG=/tmp/server.log
            PORT=8888

            start_gpu_monitor

            set -x
            vllm serve /workspace/model --host 0.0.0.0 --port \$PORT \
                --tensor-parallel-size=${tp} \
                ${ep_flag} \
                --gpu-memory-utilization ${gpu_mem} \
                --max-model-len ${max_model_len} \
                --max-num-seqs ${conc} \
                --reasoning-parser kimi_k2 \
                --tool-call-parser kimi_k2 \
                --compilation_config.pass_config.fuse_allreduce_rms true \
                --trust-remote-code \
                ${extra_flags} > \$SERVER_LOG 2>&1 &

            SERVER_PID=\$!

            wait_for_server_ready --port \$PORT --server-log \$SERVER_LOG --server-pid \$SERVER_PID

            pip install -q datasets pandas 2>/dev/null

            run_benchmark_serving \
                --model /workspace/model \
                --port \$PORT \
                --backend vllm \
                --input-len ${isl} \
                --output-len ${osl} \
                --random-range-ratio ${RANDOM_RANGE_RATIO} \
                --num-prompts \$(( ${conc} * 10 )) \
                --max-concurrency ${conc} \
                --result-filename ${result_filename} \
                --result-dir /workspace/results \
                --trust-remote-code

            stop_gpu_monitor
            cp -f /workspace/gpu_metrics.csv /workspace/results/gpu_metrics_${result_filename}.csv 2>/dev/null || true
            set +x
        "

    if [[ -f "$RESULTS_HOST/${result_filename}.json" ]]; then
        echo "    OK: $RESULTS_HOST/${result_filename}.json"
    else
        echo "    WARNING: Result file not found!"
    fi
}

# --- Run all configs ---
TOTAL=0
DONE=0

for config in "${CONFIGS[@]}"; do
    read -r precision tp ep isl osl max_model_len gpu_mem conc_rest <<< "$config"
    for conc in $conc_rest; do
        TOTAL=$((TOTAL + 1))
    done
done

echo ""
echo "Total benchmark runs (before filter): $TOTAL"
echo ""

for config in "${CONFIGS[@]}"; do
    read -r precision tp ep isl osl max_model_len gpu_mem conc_rest <<< "$config"
    for conc in $conc_rest; do
        run_single_benchmark "$precision" "$tp" "$ep" "$isl" "$osl" "$max_model_len" "$gpu_mem" "$conc"
        DONE=$((DONE + 1))
        echo "    Progress: $DONE / $TOTAL"
    done
done

# --- Summary ---
echo ""
echo "============================================"
echo " Full Benchmark Summary"
echo "============================================"
echo ""

for config in "${CONFIGS[@]}"; do
    read -r precision tp ep isl osl max_model_len gpu_mem conc_rest <<< "$config"

    local_gpu_count=$tp
    echo ""
    echo "### ${precision} TP=${tp} EP=${ep} ISL=${isl}/OSL=${osl} (${local_gpu_count} GPUs)"
    echo "| Conc | Output tok/s | tok/s/gpu | Mean TPOT (ms) | Mean TTFT (ms) |"
    echo "|:----:|:------------:|:---------:|:--------------:|:--------------:|"

    for conc in $conc_rest; do
        result_filename="kimik2.5_${precision}_vllm_tp${tp}_ep${ep}_dpa_false_conc${conc}_b300"
        result_file="$RESULTS_HOST/${result_filename}.json"
        if [[ -f "$result_file" ]]; then
            python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
tput = d.get('output_throughput', 0)
ttft = d.get('mean_ttft_ms', 0)
tpot = d.get('mean_tpot_ms', 0)
gpus = int(sys.argv[2])
print(f'| {sys.argv[3]:^4} | {tput:>12.1f} | {tput/gpus:>9.1f} | {tpot:>14.2f} | {ttft:>14.2f} |')
" "$result_file" "$local_gpu_count" "$conc" 2>/dev/null || echo "| $conc | (parse error) | | | |"
        else
            echo "| $conc | (no data) | | | |"
        fi
    done
done

echo ""
echo "Results saved to: $RESULTS_HOST/"
echo "============================================"
