#!/usr/bin/env bash
set -euo pipefail

MODEL="nvidia/Kimi-K2.5-NVFP4"
PROXY_URL="http://127.0.0.1:8000"
PROXY_PORT=8000
ISL=8192
OSL=1024
RANDOM_RANGE_RATIO=0.8
CONC_LIST="${CONC_LIST:-4 8 16 32 64}"
RESULT_DIR="/tmp/bench_pd_results"
BENCH_SCRIPT="/tmp/bench_serving/benchmark_serving.py"

mkdir -p "$RESULT_DIR"
pip install -q datasets pandas 2>/dev/null || true

echo "============================================"
echo " PD Disagg NIXL Benchmark (EP=4, EFA)"
echo "============================================"
echo "ISL/OSL:    ${ISL}/${OSL}"
echo "Conc list:  $CONC_LIST"
echo "============================================"

# Verify proxy
if ! curl -s --fail "http://127.0.0.1:${PROXY_PORT}/v1/chat/completions" -X POST \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" > /dev/null 2>&1; then
    echo "ERROR: Proxy not responding"
    exit 1
fi
echo "Proxy health check passed."

for CONC in $CONC_LIST; do
    NUM_PROMPTS=$(( CONC * 10 ))
    NUM_WARMUPS=$(( CONC * 2 ))
    RESULT_FILENAME="kimik2.5_fp4_pd_disagg_nixl_tp4_ep4_isl${ISL}_osl${OSL}_conc${CONC}_b300"

    echo ""
    echo ">>> Benchmark: concurrency=$CONC, num_prompts=$NUM_PROMPTS (NIXL)"
    echo ""

    python3 "$BENCH_SCRIPT" \
        --model "$MODEL" \
        --backend vllm \
        --base-url "$PROXY_URL" \
        --dataset-name random \
        --random-input-len "$ISL" \
        --random-output-len "$OSL" \
        --random-range-ratio "$RANDOM_RANGE_RATIO" \
        --num-prompts "$NUM_PROMPTS" \
        --max-concurrency "$CONC" \
        --request-rate inf \
        --ignore-eos \
        --num-warmups "$NUM_WARMUPS" \
        --percentile-metrics 'ttft,tpot,itl,e2el' \
        --save-result \
        --result-dir "$RESULT_DIR" \
        --result-filename "${RESULT_FILENAME}.json" \
        --trust-remote-code \
        2>&1 | tee "$RESULT_DIR/${RESULT_FILENAME}.log"

    echo ">>> Completed concurrency=$CONC"
    sleep 2
done

echo ""
echo "============================================"
echo " NIXL Benchmark Summary (ISL=${ISL})"
echo "============================================"
echo ""
echo "| Conc | Output tok/s | tok/s/gpu | Mean TPOT (ms) | Mean TTFT (ms) |"
echo "|:----:|:------------:|:---------:|:--------------:|:--------------:|"

for CONC in $CONC_LIST; do
    RESULT_FILENAME="kimik2.5_fp4_pd_disagg_nixl_tp4_ep4_isl${ISL}_osl${OSL}_conc${CONC}_b300"
    RESULT_FILE="$RESULT_DIR/${RESULT_FILENAME}.json"
    if [ -f "$RESULT_FILE" ]; then
        python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
tput = d.get('output_throughput', 0)
ttft = d.get('mean_ttft_ms', 0)
tpot = d.get('mean_tpot_ms', 0)
print(f'| {sys.argv[2]:^4} | {tput:>12.1f} | {tput/4:>9.1f} | {tpot:>14.2f} | {ttft:>14.2f} |')
" "$RESULT_FILE" "$CONC" 2>/dev/null || echo "| $CONC | (parse error) | | | |"
    else
        echo "| $CONC | (no data) | | | |"
    fi
done
