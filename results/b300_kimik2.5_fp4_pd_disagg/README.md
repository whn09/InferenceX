# Kimi-K2.5 FP4 PD Disaggregated Serving on NVIDIA B300 (AWS p6-b300.48xlarge)

> **Unofficial** — This benchmark is run on AWS p6-b300.48xlarge instances, not on the official InferenceX infrastructure.

## Architecture

Prefill-Decode (PD) disaggregation splits inference into two stages running on separate GPU nodes, connected via Mooncake KV cache transfer over AWS EFA (Elastic Fabric Adapter).

```
Client → Proxy (P6-2:8000)
  ├─→ Prefill (P6-1:8010, kv_producer, TP=4) → Mooncake EFA KV transfer
  └─→ Decode  (P6-2:8020, kv_consumer, TP=4) → Stream response back
```

## Setup

| Item | Detail |
|------|--------|
| **Instances** | 2x AWS p6-b300.48xlarge (8x NVIDIA B300 SXM6 AC, 275GB HBM3e each) |
| **Model** | nvidia/Kimi-K2.5-NVFP4 (551GB, FP4 quantized) |
| **Framework** | vLLM v0.18.0 (cu130), rebuilt with native sm_103a + FlashInfer 0.6.6 |
| **KV Transfer** | Mooncake Connector (EFA protocol), built from source with `-DUSE_EFA=ON` |
| **Network** | AWS EFA v1.47.0 with GDRCopy 2.5.1 (32 NICs per node, 3200 Gbps aggregate) |
| **Prefill Config** | TP=4, kv_role=kv_producer, gpu-memory-utilization=0.90 |
| **Decode Config** | TP=4, kv_role=kv_consumer, gpu-memory-utilization=0.90 |
| **Proxy** | vLLM official mooncake_connector_proxy.py (round-robin) |
| **Benchmark** | InferenceX benchmark_serving.py (random-range-ratio=0.8, ignore-eos) |
| **Date** | 2026-04-01 |

### Docker Image Build

The Docker image was built on top of `vllm/vllm-openai:v0.18.0-cu130` with the following additions:
- **GDRCopy 2.5.1** — GPU Direct RDMA support
- **AWS EFA 1.47.0** — Elastic Fabric Adapter with GDR enabled
- **yalantinglibs** — Async networking library (Mooncake dependency)
- **Mooncake Transfer Engine** — Built from source with `-DUSE_EFA=ON -DUSE_CUDA=ON -DWITH_TE=ON -DWITH_STORE=ON`

Environment variables:
```
FI_PROVIDER=efa
FI_EFA_USE_DEVICE_RDMA=1
MC_SLICE_SIZE=262144
```

## Results — ISL=1024, OSL=1024 (max-model-len=4096)

| Concurrency | Output tok/s | tok/s/gpu (4 GPUs) | Mean TPOT (ms) | Mean TTFT (ms) |
|:-----------:|:------------:|:------------------:|:--------------:|:--------------:|
| 4           | 403.1        | 100.8              | 8.15           | 1,397.14       |
| 8           | 761.9        | 190.5              | 10.08          | 204.35         |
| 16          | 1,230.8      | 307.7              | 12.46          | 246.58         |
| 32          | 1,976.2      | 494.1              | 15.49          | 297.24         |
| 64          | 2,914.8      | 728.7              | 21.02          | 386.68         |
| 128         | 4,513.7      | 1,128.4            | 27.01          | 559.50         |
| 256         | 7,060.1      | 1,765.0            | 34.33          | 940.06         |
| 512         | 10,634.2     | 2,658.6            | 45.04          | 1,738.00       |

## Comparison with Single-Node B300 (sm_103a, ISL=1024, OSL=1024)

| Concurrency | Single-Node tok/s/gpu | PD Disagg tok/s/gpu | Speedup | Single TTFT | PD TTFT | TTFT Improvement |
|:-----------:|:---------------------:|:-------------------:|:-------:|:-----------:|:-------:|:----------------:|
| 4           | 103.0                 | 100.8               | 0.98x   | 634ms       | 1,397ms | -1.2x            |
| 8           | 171.8                 | 190.5               | 1.11x   | 796ms       | 204ms   | **3.9x**         |
| 16          | 245.6                 | 307.7               | 1.25x   | 1,234ms     | 247ms   | **5.0x**         |
| 32          | 389.2                 | 494.1               | 1.27x   | 840ms       | 297ms   | **2.8x**         |
| 64          | 550.9                 | 728.7               | 1.32x   | 1,171ms     | 387ms   | **3.0x**         |
| 128         | 795.7                 | 1,128.4             | 1.42x   | 1,329ms     | 560ms   | **2.4x**         |
| 256         | 1,163.3               | 1,765.0             | 1.52x   | 1,574ms     | 940ms   | **1.7x**         |
| 512         | 1,692.1               | 2,658.6             | 1.57x   | 1,516ms     | 1,738ms | -0.9x            |

### Key Findings

1. **Throughput**: PD disagg consistently outperforms single-node from c=8 onward, reaching **1.57x at c=512**. The decode GPU is fully dedicated to token generation without prefill contention. Higher concurrency shows greater benefit.

2. **TTFT**: Dramatic improvement at c=8-256 (up to **5.0x lower** at c=16). Since prefill runs on a separate node, decode requests are not blocked by long prefill computations. At c=512, TTFT advantage diminishes as the proxy becomes a bottleneck.

3. **TPOT**: PD disagg has lower TPOT across all concurrency levels, as the decode GPU memory and compute are not shared with prefill.

4. **c=4 anomaly**: The high TTFT at c=4 is due to proxy warmup and the sequential prefill→decode handshake being more visible at low concurrency.

5. **EFA bandwidth**: With ISL=1024, KV cache per request is small (~30MB due to MLA architecture), so EFA NICs are underutilized (<2 Gbps per NIC). Larger ISL values would better saturate the 3200 Gbps aggregate EFA bandwidth.

6. **Why not 2x speedup?** With ISL=OSL=1024, decode accounts for >90% of GPU compute time (1024 sequential forward passes vs 1 prefill pass). Offloading prefill frees only ~5-10% of decode GPU capacity. The throughput gain mainly comes from reduced scheduling contention, not freed compute. Higher ISL (e.g., 8192) would shift the balance and yield larger disaggregation benefits.

### Resource Consideration

PD disagg uses **2 nodes × 4 GPUs = 8 GPUs total** vs single-node **1 node × 4 GPUs**. The throughput per total GPU:

| Concurrency | Single-Node tok/s/total-GPU | PD Disagg tok/s/total-GPU | Efficiency |
|:-----------:|:---------------------------:|:-------------------------:|:----------:|
| 64          | 550.9 (4 GPUs)              | 364.4 (8 GPUs)            | 66%        |
| 128         | 795.7 (4 GPUs)              | 564.2 (8 GPUs)            | 71%        |
| 256         | 1,163.3 (4 GPUs)            | 882.5 (8 GPUs)            | 76%        |
| 512         | 1,692.1 (4 GPUs)            | 1,329.3 (8 GPUs)          | 79%        |

While per-GPU efficiency is lower (expected for disaggregation), PD disagg enables **higher absolute throughput** and **dramatically lower TTFT** at moderate concurrency, which is critical for latency-sensitive applications. Efficiency improves with higher concurrency as the decode pipeline stays busier.

## Results — ISL=8192, OSL=1024 (max-model-len=16384)

> **Note**: These results use TP=4 without --enable-expert-parallel. The single-node baseline uses TP=4 EP=4 (--enable-expert-parallel). EP-enabled PD disagg results will be added separately for fair comparison.

| Concurrency | Output tok/s | tok/s/gpu (4 GPUs) | Mean TPOT (ms) | Mean TTFT (ms) |
|:-----------:|:------------:|:------------------:|:--------------:|:--------------:|
| 4           | 390.4        | 97.6               | 8.70           | 1,126.53       |
| 8           | 748.6        | 187.1              | 10.12          | 344.93         |
| 16          | 1,200.9      | 300.2              | 12.41          | 459.91         |
| 32          | 1,852.7      | 463.2              | 15.92          | 766.98         |
| 64          | 2,691.5      | 672.9              | 21.91          | 1,152.81       |

### ISL=8192 vs ISL=1024 (PD Disagg, both without EP)

| Concurrency | ISL=1024 tok/s/gpu | ISL=8192 tok/s/gpu | ISL=1024 TTFT | ISL=8192 TTFT |
|:-----------:|:------------------:|:------------------:|:-------------:|:-------------:|
| 4           | 100.8              | 97.6               | 1,397ms       | 1,127ms       |
| 8           | 190.5              | 187.1              | 204ms         | 345ms         |
| 16          | 307.7              | 300.2              | 247ms         | 460ms         |
| 32          | 494.1              | 463.2              | 297ms         | 767ms         |
| 64          | 728.7              | 672.9              | 387ms         | 1,153ms       |

### ISL=8192 Observations

1. **Throughput**: ISL=8192 tok/s/gpu is ~5-8% lower than ISL=1024, which is expected since the decode GPU now handles longer KV caches (8x more KV cache per request → more memory bandwidth consumed during attention).

2. **TTFT**: Significantly higher than ISL=1024 (e.g., 460ms vs 247ms at c=16). This is expected — 8x longer input means ~8x more prefill compute, plus larger KV cache transfer over EFA (~240MB vs ~30MB per request due to MLA architecture).

3. **EFA bandwidth**: With ISL=8192, per-NIC bandwidth reaches ~7 Gbps (vs <2 Gbps at ISL=1024). Still well below the 100 Gbps per-NIC capacity. MLA's compressed KV cache (~240MB per request at ISL=8192) limits EFA utilization — non-MLA architectures (e.g., Llama) would have 10x larger KV transfers.

4. **TPOT**: Nearly identical to ISL=1024 results, confirming that per-token decode latency is dominated by model forward pass, not KV cache size.
