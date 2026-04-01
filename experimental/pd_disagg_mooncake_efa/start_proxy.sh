#!/bin/bash
# Start Mooncake Connector Proxy on P6-2
#
# Usage:
#   PREFILL_HOST=172.31.53.87 PREFILL_PORT=8010 BOOTSTRAP_PORT=8998 \
#   DECODE_PORT=8020 PROXY_PORT=8000 \
#   bash start_proxy.sh

PREFILL_HOST="${PREFILL_HOST:?Set PREFILL_HOST to prefill node IP}"
PREFILL_PORT="${PREFILL_PORT:-8010}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
DECODE_PORT="${DECODE_PORT:-8020}"
PROXY_PORT="${PROXY_PORT:-8000}"
PYTHON="${PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Use upstream vLLM proxy if available, otherwise use local copy
PROXY_SCRIPT="${PROXY_SCRIPT:-${SCRIPT_DIR}/mooncake_connector_proxy.py}"

echo "Starting Mooncake Connector Proxy..."
echo "  Prefill: http://${PREFILL_HOST}:${PREFILL_PORT} (bootstrap: ${BOOTSTRAP_PORT})"
echo "  Decode:  http://localhost:${DECODE_PORT}"
echo "  Proxy:   http://0.0.0.0:${PROXY_PORT}"

exec "$PYTHON" "$PROXY_SCRIPT" \
    --prefill "http://${PREFILL_HOST}:${PREFILL_PORT}" "$BOOTSTRAP_PORT" \
    --decode "http://localhost:${DECODE_PORT}" \
    --host 0.0.0.0 \
    --port "$PROXY_PORT"
