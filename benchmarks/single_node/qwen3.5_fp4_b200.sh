#!/usr/bin/env bash

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME \
    MAX_MODEL_LEN \
    EP_SIZE

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

nvidia-smi

hf download "$MODEL"

export NCCL_NVLS_ENABLE=1
export SGL_ENABLE_JIT_DEEPGEMM=false
export SGLANG_ENABLE_FLASHINFER_GEMM=true
export PYTHONUNBUFFERED=1

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}
MEM_FRAC_STATIC=0.85

echo "Config: ISL=$ISL, OSL=$OSL, CONC=$CONC, EP=$EP_SIZE, MEM=$MEM_FRAC_STATIC, CUDA_BS=$CUDA_GRAPH_MAX_BS, MAX_RR=$MAX_RUNNING_REQUESTS"

# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

set -x
PYTHONNOUSERSITE=1 python3 -m sglang.launch_server --model-path=$MODEL --host=0.0.0.0 --port=$PORT \
--trust-remote-code \
--tensor-parallel-size $TP \
--ep-size $EP_SIZE \
--cuda-graph-max-bs $CONC \
--max-running-requests $CONC \
--mem-fraction-static $MEM_FRAC_STATIC \
--chunked-prefill-size 32768 \
--max-prefill-tokens 32768 \
--context-length $MAX_MODEL_LEN \
--attention-backend trtllm_mha \
--moe-runner-backend flashinfer_trtllm \
--fp4-gemm-backend flashinfer_cutlass \
--quantization modelopt_fp4 \
--kv-cache-dtype fp8_e4m3 \
--mamba-ssm-dtype bfloat16 \
--disable-radix-cache \
--scheduler-recv-interval 30 --stream-interval 30 > $SERVER_LOG 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

pip install -q datasets pandas

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$((CONC * 10))" \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT" --concurrent-requests $CONC
    append_lm_eval_summary
fi

# Stop GPU monitoring
stop_gpu_monitor
set +x
