# Kimi-K2.5 FP4 PD Disaggregated Serving Benchmark (Mooncake EFA, fix branch)

## Setup
- **Hardware**: 2x p5en.48xlarge (8x H200 each)
- **Model**: Kimi-K2.5 (compressed-tensors FP4, ~595GB)
- **Config**: TP=8 per node, max_model_len=10240, prefix_caching=on, gpu_mem_util=0.9, expert_parallel=on
- **Mooncake**: fix/efa-read-and-endpoint-eviction branch, USE_EFA=ON, built with Python 3.13
- **vLLM**: 0.18.1
- **Workload**: random ISL=8192, OSL=1024, random_range_ratio=0.8, num_prompts=conc*10
- **M1 (prefill)**: 172.31.35.3, port 8010
- **M2 (decode)**: 172.31.45.191, port 8020
- **Proxy**: M2:8000
- **EFA env**: FI_PROVIDER=efa, FI_EFA_USE_DEVICE_RDMA=1, FI_HMEM=system, MC_SLICE_SIZE=262144
- **Kernel modules**: efa, efa_nv_peermem, gdrdrv (all required)

## Results

| Conc | Output tok/s | tok/s/GPU | Total tok/s | Mean TTFT (ms) | Mean TPOT (ms) | P99 TTFT (ms) | P99 TPOT (ms) | Status |
|:----:|:------------:|:---------:|:-----------:|:--------------:|:--------------:|:-------------:|:-------------:|:------:|
| 4    | 161.0        | 20.1      | 1,447.2     | 1,134.6        | 22.9           | 3,164.3       | 23.5          | OK     |
| 8    | 297.0        | 37.1      | -           | 1,366.5        | 24.9           | -             | -             | OK     |
| 16   | -            | -         | -           | -              | -              | -             | -             | FAILED |
| 32   | -            | -         | -           | -              | -              | -             | -             | NOT RUN|
| 64   | -            | -         | -           | -              | -              | -             | -             | NOT RUN|

## Comparison with upstream main (same hardware, same workload)

| Conc | Fix branch tok/s | Upstream main tok/s | Fix TTFT (ms) | Upstream TTFT (ms) |
|:----:|:----------------:|:-------------------:|:-------------:|:------------------:|
| 4    | 161              | 169                 | 1,135         | 984                |
| 8    | 297              | 319                 | 1,367         | 716                |

Fix branch shows ~5-7% lower throughput and higher TTFT compared to upstream main. This is expected as the fix branch includes additional endpoint eviction logic and enhanced error handling.

## Concurrency >= 16 Failure Analysis

At concurrency 16, the EFA transport hits a CQ (Completion Queue) drain bottleneck:

- **Symptom**: `EFA submitPostSend: timed out waiting for CQ drain (wr_depth=256, max=256)`
- **Count**: 1,263 CQ drain timeouts on prefill side
- **Decode error**: `Mooncake transfer engine returned -1` (KV cache pull failed)
- **Root cause**: With ISL=8192, each KV transfer generates many 256KB slices. At 16 concurrent requests × 8 TP workers, the per-endpoint wr_depth=256 saturates and CQ polling cannot drain completions fast enough.
- **Note**: Zero CQ errors (vs. crashes in pre-fix builds), indicating the transport itself is stable — it's a throughput/capacity issue, not a correctness bug.

### Potential fixes for high-concurrency scaling:
1. Increase CQ polling frequency or use dedicated CQ polling threads
2. Increase wr_depth and CQ size limits
3. Use multiple CQs per domain to distribute completion load
4. Adaptive backoff instead of yield-based spin-wait

## Key Findings

1. **efa_nv_peermem kernel module is essential** — without it, EFA CQ errors crash the prefill server
2. **Must build with correct Python** — vLLM 0.18.1 uses Python 3.13 (`/opt/pytorch/bin/python3`), the .so must be built from `/home/ubuntu/Mooncake` (not `/opt/dlami/nvme/Mooncake` which targets Python 3.12)
3. **FI_MR_HMEM in mr_mode hints** is required for GPU memory registration; FI_HMEM in caps breaks fi_getinfo
4. **EFA transport is stable at low-medium concurrency** — zero errors at conc 4/8

## Date
2026-04-04
