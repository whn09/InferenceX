# PD Disaggregated Serving with Mooncake EFA on AWS B300

This directory contains scripts for running Prefill-Decode (PD) disaggregated
inference using vLLM + Mooncake Connector over AWS EFA.

## Architecture

```
                    ┌─────────────┐
                    │   Client    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │    Proxy    │  (mooncake_connector_proxy.py)
                    │  P6-2:8000  │
                    └──┬──────┬───┘
                       │      │
          ┌────────────▼┐   ┌─▼────────────┐
          │   Prefill   │   │    Decode     │
          │  P6-1:8010  │──▶│  P6-2:8020   │
          │ kv_producer  │EFA│ kv_consumer   │
          │    TP=4     │   │    TP=4       │
          └─────────────┘   └──────────────┘
```

## Prerequisites

- 2x AWS p6-b300.48xlarge instances with EFA connectivity
- Docker image with:
  - vLLM v0.18.0 (rebuilt with native sm_103a)
  - Mooncake Transfer Engine (built with `-DUSE_EFA=ON`)
  - AWS EFA 1.47.0 + GDRCopy 2.5.1
- Model: nvidia/Kimi-K2.5-NVFP4

## Quick Start

```bash
# 1. On P6-1 (prefill node):
bash start_prefill.sh

# 2. On P6-2 (decode node):
bash start_decode.sh

# 3. On P6-2 (proxy):
bash start_proxy.sh

# 4. On P6-2 (benchmark):
bash ../../benchmarks/multi_node/kimik2.5_fp4_b300_vllm-mooncake-disagg.sh
```

## Docker Image Build

See `build_efa_image.sh` for the full Docker image build script that adds
EFA + GDRCopy + Mooncake (USE_EFA=ON) to the base vLLM image.

## Key Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FI_PROVIDER` | `efa` | libfabric provider |
| `FI_EFA_USE_DEVICE_RDMA` | `1` | Enable EFA device RDMA |
| `MC_SLICE_SIZE` | `262144` | Mooncake transfer slice size |
| `VLLM_MOONCAKE_BOOTSTRAP_PORT` | `8998` | Bootstrap server port (prefill only) |
