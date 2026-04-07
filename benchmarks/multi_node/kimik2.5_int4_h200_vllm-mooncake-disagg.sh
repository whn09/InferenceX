#!/usr/bin/env bash
#
# Kimi-K2.5 INT4 PD Disaggregated Serving Benchmark (Mooncake EFA)
# Runs on 2x AWS p5en.48xlarge instances.
#
# Prerequisites:
#   - P5EN-1 (PREFILL_HOST): vLLM kv_producer running on PREFILL_PORT
#   - P5EN-2 (DECODE_HOST):  vLLM kv_consumer running on DECODE_PORT
#   - Mooncake connector proxy running on PROXY_PORT
#   - InferenceX benchmark_serving.py available
#
# Usage:
#   # Start vLLM instances first (see start_prefill.sh / start_decode.sh)
#   # Start proxy (see start_proxy.sh)
#   # Then run this benchmark:
#   bash kimik2.5_int4_h200_vllm-mooncake-disagg.sh
#
set -euo pipefail

source "$(dirname "$0")/../benchmark_lib.sh"

# --- Configuration ---
MODEL="${MODEL:-moonshotai/Kimi-K2.5}"
PROXY_URL="${PROXY_URL:-http://127.0.0.1:8000}"
PROXY_PORT="${PROXY_PORT:-8000}"
ISL="${ISL:-1024}"
OSL="${OSL:-1024}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.8}"
CONC_LIST="${CONC_LIST:-4 8 16 32 64 128 256 512}"
RESULT_DIR="${RESULT_DIR:-/tmp/bench_pd_results}"
BENCH_SCRIPT="${BENCH_SCRIPT:-$(dirname "$0")/../../utils/bench_serving/benchmark_serving.py}"

mkdir -p "$RESULT_DIR"

echo "============================================"
echo " PD Disagg Benchmark (Mooncake EFA, H200)"
echo "============================================"
echo "Proxy:      $PROXY_URL"
echo "Model:      $MODEL"
echo "ISL/OSL:    ${ISL}/${OSL}"
echo "Conc list:  $CONC_LIST"
echo "Results:    $RESULT_DIR"
echo "============================================"

# Verify proxy is healthy
if ! curl -s --fail "${PROXY_URL}/v1/chat/completions" -X POST \
    -H "Content-Type: application/json" \
    -d '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"hi"}],"max_tokens":1}' > /dev/null 2>&1; then
    echo "ERROR: Proxy not responding at $PROXY_URL"
    exit 1
fi
echo "Proxy health check passed."

start_gpu_monitor

for CONC in $CONC_LIST; do
    NUM_PROMPTS=$(( CONC * 10 ))
    NUM_WARMUPS=$(( CONC * 2 ))
    RESULT_FILENAME="kimik2.5_int4_pd_disagg_mooncake_efa_tp8_isl${ISL}_osl${OSL}_conc${CONC}_h200"

    echo ""
    echo ">>> Benchmark: concurrency=$CONC, num_prompts=$NUM_PROMPTS, warmups=$NUM_WARMUPS"
    echo ""

    run_benchmark_serving \
        --model "$MODEL" \
        --port "$PROXY_PORT" \
        --backend vllm \
        --input-len "$ISL" \
        --output-len "$OSL" \
        --random-range-ratio "$RANDOM_RANGE_RATIO" \
        --num-prompts "$NUM_PROMPTS" \
        --max-concurrency "$CONC" \
        --result-filename "$RESULT_FILENAME" \
        --result-dir "$RESULT_DIR" \
        --trust-remote-code

    echo ">>> Completed concurrency=$CONC"
    sleep 2
done

stop_gpu_monitor

# --- Summary ---
echo ""
echo "============================================"
echo " PD Disagg Benchmark Summary (Mooncake EFA)"
echo " ISL=${ISL} / OSL=${OSL}, TP=8, H200"
echo "============================================"
echo ""
echo "| Conc | Output tok/s | tok/s/gpu (8 GPUs) | Mean TPOT (ms) | Mean TTFT (ms) |"
echo "|:----:|:------------:|:------------------:|:--------------:|:--------------:|"

for CONC in $CONC_LIST; do
    RESULT_FILENAME="kimik2.5_int4_pd_disagg_mooncake_efa_tp8_isl${ISL}_osl${OSL}_conc${CONC}_h200"
    RESULT_FILE="$RESULT_DIR/${RESULT_FILENAME}.json"
    if [[ -f "$RESULT_FILE" ]]; then
        python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
tput = d.get('output_throughput', 0)
ttft = d.get('mean_ttft_ms', 0)
tpot = d.get('mean_tpot_ms', 0)
print(f'| {sys.argv[2]:^4} | {tput:>12.1f} | {tput/8:>18.1f} | {tpot:>14.2f} | {ttft:>14.2f} |')
" "$RESULT_FILE" "$CONC" 2>/dev/null || echo "| $CONC | (parse error) | | | |"
    else
        echo "| $CONC | (no data) | | | |"
    fi
done

echo ""
echo "Results saved to: $RESULT_DIR/"
echo "============================================"
