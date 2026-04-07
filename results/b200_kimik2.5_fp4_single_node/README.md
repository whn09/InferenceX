# Kimi-K2.5 FP4 Single-Node Serving on NVIDIA B200 (AWS p6-b200.48xlarge)

## Setup

| Item | Detail |
|------|--------|
| **Instance** | 1x AWS p6-b200.48xlarge (8x NVIDIA B200, 183GB HBM3e each) |
| **Model** | nvidia/Kimi-K2.5-NVFP4 (551GB, FP4 quantized via modelopt) |
| **Framework** | vLLM v0.19.0, FlashInfer 0.6.6 (FLASHINFER_MLA, HND layout) |
| **Config** | TP=4 (CUDA_VISIBLE_DEVICES=0,1,2,3), max_model_len=4096, prefix_caching=on, expert_parallel=on, gpu_mem_util=0.90 |
| **Quantization** | modelopt_fp4 |
| **Workload** | random ISL=1024, OSL=1024, num_prompts=conc*10 |
| **Mooncake** | fix/efa-read-and-endpoint-eviction branch (a9f6790) — single-node, no KV transfer |
| **Date** | 2026-04-04 |

## Results — ISL=1024, OSL=1024

| Concurrency | Output tok/s | tok/s/gpu (4 GPUs) | Mean TPOT (ms) | Mean TTFT (ms) | P99 TTFT (ms) |
|:-----------:|:------------:|:------------------:|:--------------:|:--------------:|:-------------:|
| 4           | 426.2        | 106.6              | 8.47           | 939.36         | 7,564.94      |
| 8           | 670.1        | 167.5              | 10.00          | 1,995.30       | 10,737.96     |
| 16          | 1,223.8      | 305.9              | 12.86          | 229.25         | 480.11        |
| 32          | 1,948.7      | 487.2              | 16.13          | 302.93         | 683.97        |
| 64          | 2,942.8      | 735.7              | 21.23          | 545.43         | 1,706.57      |
| 128         | 4,478.1      | 1,119.5            | 27.90          | 702.87         | 3,011.89      |
| 256         | 6,247.4      | 1,561.9            | 38.67          | 1,185.84       | 5,717.08      |
| 512         | 5,496.1      | 1,374.0            | 66.27          | 24,424.43      | 59,963.61     |

## Comparison with B300 Single-Node (TP=4, sm_103a, same ISL/OSL)

| Concurrency | B200 tok/s/gpu | B300 tok/s/gpu | B200 vs B300 |
|:-----------:|:--------------:|:--------------:|:------------:|
| 4           | 106.6          | 103.0          | 1.03x        |
| 8           | 167.5          | 171.8          | 0.98x        |
| 16          | 305.9          | 245.6          | 1.25x        |
| 32          | 487.2          | 389.2          | 1.25x        |
| 64          | 735.7          | 550.9          | 1.34x        |
| 128         | 1,119.5        | 795.7          | 1.41x        |
| 256         | 1,561.9        | 1,163.3        | 1.34x        |
| 512         | 1,374.0        | 1,692.1        | 0.81x        |

### Key Findings

1. **Mid-concurrency advantage**: B200 outperforms B300 by 25-41% at c=16-256, likely due to faster sm_100 compute (higher clock speeds or better scheduling than sm_103a for this workload).

2. **High concurrency regression (c=512)**: B200 drops to 0.81x of B300, with TTFT spiking to 24s. B200 has less HBM per GPU (183GB vs 275GB), reducing KV cache capacity and causing more eviction/recomputation at high concurrency.

3. **TTFT warmup (c=4,8)**: High TTFT at low concurrency is a warmup artifact (first-time compilation, cache population). B300 shows similar behavior.

4. **Peak throughput**: B200 peaks at c=256 (6,247 tok/s total, 1,562/gpu), while B300 peaks at c=512 (6,768 tok/s total, 1,692/gpu).
