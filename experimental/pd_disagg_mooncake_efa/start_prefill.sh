#!/bin/bash
# Start vLLM Prefill (kv_producer) on P6-1
#
# Usage: PREFILL_PORT=8010 BOOTSTRAP_PORT=8998 bash start_prefill.sh

IMAGE="${IMAGE:-vllm-b300-sm103a:latest}"
MODEL="${MODEL:-nvidia/Kimi-K2.5-NVFP4}"
HF_CACHE="${HF_CACHE:-/opt/dlami/nvme/hf_cache}"
PREFILL_PORT="${PREFILL_PORT:-8010}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
TP="${TP:-4}"

docker run --rm --gpus all --ipc=host --net=host \
    --device=/dev/infiniband \
    -v "$HF_CACHE":/root/.cache/huggingface \
    -e FI_PROVIDER=efa \
    -e FI_EFA_USE_DEVICE_RDMA=1 \
    -e MC_SLICE_SIZE=262144 \
    -e HF_HOME=/root/.cache/huggingface \
    -e VLLM_MOONCAKE_BOOTSTRAP_PORT="$BOOTSTRAP_PORT" \
    "$IMAGE" \
    "$MODEL" \
    --host 0.0.0.0 \
    --port "$PREFILL_PORT" \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096 \
    --trust-remote-code \
    --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer","kv_connector_extra_config":{"mooncake_protocol":"efa"}}'
