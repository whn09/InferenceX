# Kimi-K2.5 INT4 PD Disaggregated Serving on H200 (AWS p5en.48xlarge)

> **Unofficial** — This benchmark is run on AWS p5en.48xlarge instances, not on the official InferenceX infrastructure.

## Architecture

Prefill-Decode (PD) disaggregation splits inference into two stages running on separate GPU nodes, connected via Mooncake KV cache transfer over AWS EFA (Elastic Fabric Adapter).

```
Client → Proxy (P5EN-2:8000)
  ├─→ Prefill (P5EN-1:8010, kv_producer, TP=8) → Mooncake EFA KV transfer
  └─→ Decode  (P5EN-2:8020, kv_consumer, TP=8) → Stream response back
```

## Setup

| Item | Detail |
|------|--------|
| **Instances** | 2x AWS p5en.48xlarge (8x NVIDIA H200 SXM, 141GB HBM3e each) |
| **Model** | moonshotai/Kimi-K2.5 (INT4 quantized) |
| **Framework** | vLLM (pip install, cu130 wheel) |
| **KV Transfer** | Mooncake Connector (EFA protocol), built from source with `-DUSE_EFA=ON -DUSE_CUDA=ON` |
| **Network** | AWS EFA (16 NICs per node, 3200 Gbps aggregate) |
| **Prefill Config** | TP=8, kv_role=kv_producer, gpu-memory-utilization=0.90 |
| **Decode Config** | TP=8, kv_role=kv_consumer, gpu-memory-utilization=0.90 |
| **Proxy** | vLLM mooncake_connector_proxy.py (round-robin) |
| **Benchmark** | InferenceX benchmark_serving.py (random-range-ratio=0.5, ignore-eos) |

### Mooncake Tuning

```
MC_SLICE_SIZE=262144        # 256KB, optimal for EFA throughput
MC_MAX_WR=4096              # WR depth per endpoint
MC_MAX_CQE_PER_CTX=32768   # CQ capacity per device
MC_NUM_CQ_PER_CTX=4         # Round-robin CQ assignment (131072 total CQE)
```

## Results — ISL=1024, OSL=1024 (max-model-len=4096)

### PD Disagg (Mooncake EFA)

| Concurrency | Output tok/s | tok/s/gpu (8 GPUs) | Mean TPOT (ms) | Mean TTFT (ms) |
|:-----------:|:------------:|:------------------:|:--------------:|:--------------:|
| 4           | 402.9        | 50.4               | 9.38           | 247            |
| 8           | 683.4        | 85.4               | 11.18          | 275            |
| 16          | 1,035.3      | 129.4              | 14.76          | 341            |
| 32          | 1,614.8      | 201.8              | 18.91          | 444            |
| 64          | 2,353.4      | 294.2              | 25.86          | 650            |

### Single-Node Baseline

| Concurrency | Output tok/s | tok/s/gpu (8 GPUs) | Mean TPOT (ms) | Mean TTFT (ms) |
|:-----------:|:------------:|:------------------:|:--------------:|:--------------:|
| 4           | 404.6        | 50.6               | 9.50           | 136            |
| 8           | 423.0        | 52.9               | 9.18           | 6,770          |
| 16          | 434.5        | 54.3               | 9.04           | 20,276         |
| 32          | 435.1        | 54.4               | 9.02           | 46,986         |
| 64          | 436.2        | 54.5               | 9.03           | 99,806         |

### Comparison

| Concurrency | Single-Node tok/s/gpu | PD Disagg tok/s/gpu | Throughput Speedup | Single TTFT | PD TTFT | TTFT Improvement |
|:-----------:|:---------------------:|:-------------------:|:------------------:|:-----------:|:-------:|:----------------:|
| 4           | 50.6                  | 50.4                | 1.00x              | 136ms       | 247ms   | -1.8x            |
| 8           | 52.9                  | 85.4                | **1.62x**          | 6,770ms     | 275ms   | **24.6x**        |
| 16          | 54.3                  | 129.4               | **2.38x**          | 20,276ms    | 341ms   | **59.5x**        |
| 32          | 54.4                  | 201.8               | **3.71x**          | 46,986ms    | 444ms   | **105.9x**       |
| 64          | 54.5                  | 294.2               | **5.39x**          | 99,806ms    | 650ms   | **153.6x**       |

### Key Findings (ISL=1024)

1. **Throughput**: PD disagg scales nearly linearly (50→294 tok/s/gpu from c=4 to c=64), while single-node saturates at ~55 tok/s/gpu by c=8. At c=64, PD disagg achieves **5.39x** throughput.

2. **TTFT**: Single-node TTFT explodes from 136ms (c=4) to 100 seconds (c=64) as prefill requests queue on shared GPUs. PD disagg keeps TTFT under 650ms even at c=64 — a **153.6x improvement**. This is because prefill runs on a dedicated node and never blocks decode.

3. **TPOT**: Single-node maintains ~9ms constant TPOT (low decode concurrency). PD disagg TPOT rises to 26ms at c=64 as more decode requests compete — the expected cost of higher throughput.

4. **c=4**: PD disagg has slightly higher TTFT (247ms vs 136ms) due to KV transfer overhead (~30MB per request with MLA at ISL=1024). At low concurrency this overhead is visible; by c=8 the single-node queuing delay (6.8s) far exceeds it.

## Results — ISL=8192, OSL=1024 (max-model-len=16384)

### PD Disagg (Mooncake EFA)

| Concurrency | Output tok/s | tok/s/gpu (8 GPUs) | Mean TPOT (ms) | Mean TTFT (ms) |
|:-----------:|:------------:|:------------------:|:--------------:|:--------------:|
| 4           | 354.5        | 44.3               | 9.95           | 958            |
| 8           | 574.0        | 71.7               | 12.16          | 1,355          |
| 16          | 848.7        | 106.1              | 16.37          | 1,747          |
| 32          | 1,256.5      | 157.1              | 22.66          | 1,916          |
| 64          | 1,681.0      | 210.1              | 33.89          | 3,015          |

### Single-Node Baseline

| Concurrency | Output tok/s | tok/s/gpu (8 GPUs) | Mean TPOT (ms) | Mean TTFT (ms) |
|:-----------:|:------------:|:------------------:|:--------------:|:--------------:|
| 4           | 337.9        | 42.2               | 10.90          | 453            |
| 8           | 354.6        | 44.3               | 10.61          | 8,199          |
| 16          | 360.0        | 45.0               | 10.55          | 24,572         |
| 32          | 363.6        | 45.4               | 10.53          | 56,590         |
| 64          | 362.7        | 45.3               | 10.55          | 120,282        |

### Comparison

| Concurrency | Single-Node tok/s/gpu | PD Disagg tok/s/gpu | Throughput Speedup | Single TTFT | PD TTFT | TTFT Improvement |
|:-----------:|:---------------------:|:-------------------:|:------------------:|:-----------:|:-------:|:----------------:|
| 4           | 42.2                  | 44.3                | 1.05x              | 453ms       | 958ms   | -2.1x            |
| 8           | 44.3                  | 71.7                | **1.62x**          | 8,199ms     | 1,355ms | **6.0x**         |
| 16          | 45.0                  | 106.1               | **2.36x**          | 24,572ms    | 1,747ms | **14.1x**        |
| 32          | 45.4                  | 157.1               | **3.46x**          | 56,590ms    | 1,916ms | **29.5x**        |
| 64          | 45.3                  | 210.1               | **4.63x**          | 120,282ms   | 3,015ms | **39.9x**        |

### Key Findings (ISL=8192)

1. **Throughput**: PD disagg scales from 44→210 tok/s/gpu (c=4 to c=64), while single-node saturates at ~45 tok/s/gpu. At c=64, PD disagg achieves **4.63x** throughput.

2. **TTFT**: Single-node TTFT reaches 120 seconds at c=64. PD disagg keeps it under 3 seconds (**39.9x improvement**).

3. **c=4 anomaly**: PD disagg has higher TTFT at c=4 (958ms vs 453ms) due to larger KV transfer overhead (~240MB per request with MLA at ISL=8192). At higher concurrency the queuing delay in single-node far exceeds the transfer time.

4. **TPOT**: Similar to ISL=1024 — single-node stays at ~10.5ms, PD disagg rises to 34ms at c=64.

## ISL=1024 vs ISL=8192 (PD Disagg)

| Concurrency | ISL=1024 tok/s/gpu | ISL=8192 tok/s/gpu | ISL=1024 TTFT | ISL=8192 TTFT |
|:-----------:|:------------------:|:------------------:|:-------------:|:-------------:|
| 4           | 50.4               | 44.3               | 247ms         | 958ms         |
| 8           | 85.4               | 71.7               | 275ms         | 1,355ms       |
| 16          | 129.4              | 106.1              | 341ms         | 1,747ms       |
| 32          | 201.8              | 157.1              | 444ms         | 1,916ms       |
| 64          | 294.2              | 210.1              | 650ms         | 3,015ms       |

ISL=8192 shows ~15-30% lower throughput and ~4x higher TTFT compared to ISL=1024. This is expected: 8x longer input means more prefill compute and larger KV cache transfers (~240MB vs ~30MB per request due to MLA).

Notably, ISL=1024 achieves higher throughput speedup (5.39x) than ISL=8192 (4.63x) at c=64. This is because ISL=1024's prefill is very fast, allowing the prefill node to keep up with higher decode demand. At ISL=8192, the prefill node starts becoming a bottleneck at high concurrency.

## Resource Consideration

PD disagg uses **2 nodes × 8 GPUs = 16 GPUs total** vs single-node **1 node × 8 GPUs**. The throughput per total GPU:

### ISL=1024

| Concurrency | Single-Node tok/s/total-GPU | PD Disagg tok/s/total-GPU | Efficiency |
|:-----------:|:---------------------------:|:-------------------------:|:----------:|
| 8           | 52.9 (8 GPUs)              | 42.7 (16 GPUs)            | 81%        |
| 16          | 54.3 (8 GPUs)              | 64.7 (16 GPUs)            | 119%       |
| 32          | 54.4 (8 GPUs)              | 100.9 (16 GPUs)           | 186%       |
| 64          | 54.5 (8 GPUs)              | 147.1 (16 GPUs)           | 270%       |

### ISL=8192

| Concurrency | Single-Node tok/s/total-GPU | PD Disagg tok/s/total-GPU | Efficiency |
|:-----------:|:---------------------------:|:-------------------------:|:----------:|
| 8           | 44.3 (8 GPUs)              | 35.9 (16 GPUs)            | 81%        |
| 16          | 45.0 (8 GPUs)              | 53.0 (16 GPUs)            | 118%       |
| 32          | 45.4 (8 GPUs)              | 78.5 (16 GPUs)            | 173%       |
| 64          | 45.3 (8 GPUs)              | 105.1 (16 GPUs)           | 232%       |

From c=16 onward, PD disagg achieves **higher per-GPU efficiency** than single-node for both ISL values. The single-node baseline is so bottlenecked by prefill/decode contention that adding a second node more than doubles effective throughput. At c=64 with ISL=1024, efficiency reaches **270%** — each GPU in the 2-node setup produces 2.7x the output of a GPU in the single-node setup.

## Results — ISL=8192, OSL=1024: NIXL LIBFABRIC vs Mooncake EFA

vLLM + NIXL LIBFABRIC backend on the same H200/p5en setup. NIXL uses `--enforce-eager` per AWS recommendation.

Ref: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start-nixl.html

### PD Disagg (NIXL LIBFABRIC)

| Concurrency | Output tok/s | tok/s/gpu (8 GPUs) | Mean TPOT (ms) | Mean TTFT (ms) |
|:-----------:|:------------:|:------------------:|:--------------:|:--------------:|
| 4           | 370.6        | 46.3               | 9.71           | 511            |
| 8           | 611.9        | 76.5               | 11.84          | 677            |
| 16          | 875.6        | 109.5              | 16.37          | 889            |
| 32          | 1,265.2      | 158.1              | 22.85          | 1,239          |
| 64          | 1,652.1      | 206.5              | 34.68          | 2,245          |

### NIXL vs Mooncake Comparison (ISL=8192)

| Concurrency | Mooncake tok/s/gpu | NIXL tok/s/gpu | Throughput Diff | Mooncake TTFT | NIXL TTFT | TTFT Diff |
|:-----------:|:------------------:|:--------------:|:---------------:|:-------------:|:---------:|:---------:|
| 4           | 44.3               | 46.3           | +4.5%           | 958ms         | 511ms     | **1.9x better** |
| 8           | 71.7               | 76.5           | +6.7%           | 1,355ms       | 677ms     | **2.0x better** |
| 16          | 106.1              | 109.5          | +3.2%           | 1,747ms       | 889ms     | **2.0x better** |
| 32          | 157.1              | 158.1          | +0.6%           | 1,916ms       | 1,239ms   | **1.5x better** |
| 64          | 210.1              | 206.5          | -1.7%           | 3,015ms       | 2,245ms   | **1.3x better** |

### Key Findings (NIXL vs Mooncake)

1. **Throughput**: Nearly identical at all concurrency levels (within ±5%). Both frameworks saturate the same EFA bandwidth, so the end-to-end serving throughput is comparable.

2. **TTFT**: NIXL consistently achieves **1.3-2.0x lower TTFT** across all concurrency levels. This is likely due to:
   - NIXL LIBFABRIC's auto multi-rail striping (lower per-transfer latency)
   - Less software overhead in the KV transfer path (NIXL is a thin wrapper over libfabric vs Mooncake's multi-layer TransferEngine)

3. **TPOT**: Nearly identical (both ~10ms at c=4, ~35ms at c=64), as TPOT is dominated by decode compute, not KV transfer.

4. **Framework difference**: NIXL uses vLLM, Mooncake also uses vLLM — same inference engine, different KV transfer backends. The throughput parity confirms both backends deliver similar aggregate bandwidth; the TTFT gap reflects per-request transfer latency differences.

5. **nixlbench confirms**: NIXL LIBFABRIC achieves 384 GB/s (8 GPU) vs Mooncake's 337-347 GB/s on the same hardware. The ~10% raw bandwidth advantage translates to lower per-request KV transfer time, hence better TTFT.

---

## Why H200 PD Disagg Shows Much Larger Gains Than B300

B300 PD disagg peaked at 1.57x throughput (ISL=1024, TP=4). H200 shows up to 5.39x because:

1. **TP=8 vs TP=4**: H200 uses all 8 GPUs for a single model instance. In single-node mode, all 8 GPUs are shared between prefill and decode, making contention worse. B300 uses TP=4, leaving more scheduling flexibility.

2. **Single-node saturation**: H200 single-node throughput saturates by c=8 (~55 tok/s/gpu for ISL=1024), while B300 single-node continues scaling to c=512. This means H200 has much more headroom for disaggregation to exploit.

3. **Amplified contention**: With TP=8, a single prefill pass blocks all 8 GPUs simultaneously, preventing any decode progress. With TP=4, the other 4 GPUs can still serve decode requests.
