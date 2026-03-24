# Kimi-K2.5 FP4 Benchmark on NVIDIA B300 (AWS p6-b300.48xlarge)

> **Unofficial** — This benchmark is run on an AWS p6-b300.48xlarge instance, not on the official InferenceX infrastructure.

## Setup

| Item | Detail |
|------|--------|
| **Instance** | AWS p6-b300.48xlarge (8x NVIDIA B300 SXM6 AC, 275GB HBM3e each) |
| **Model** | nvidia/Kimi-K2.5-NVFP4 (551GB, FP4 quantized) |
| **Framework** | vLLM 0.18.1rc1 nightly (cu130), rebuilt with native sm_103a |
| **Config** | TP=4, EP=4, max-model-len=4096, gpu-memory-utilization=0.90 |
| **Benchmark** | InferenceX benchmark_serving.py (ISL=1024, OSL=1024, random-range-ratio=0.8, ignore-eos) |
| **CUDA Arch** | Native sm_100a + sm_103a (built from source) |
| **Date** | 2026-03-23 |

## Results — Native sm_103a (B300)

| Concurrency | Output tok/s | tok/s/gpu | Mean TPOT (ms) | Mean TTFT (ms) |
|:-----------:|:------------:|:---------:|:--------------:|:--------------:|
| 4           | 412.1        | 103.0     | 8.78           | 634.21         |
| 8           | 687.1        | 171.8     | 10.56          | 796.39         |
| 16          | 982.3        | 245.6     | 14.67          | 1,233.70       |
| 32          | 1,556.8      | 389.2     | 19.24          | 840.09         |
| 64          | 2,203.5      | 550.9     | 27.23          | 1,170.87       |
| **128**     | **3,182.8**  | **795.7** | 38.02          | 1,328.91       |
| **256**     | **4,653.3**  | **1,163.3** | 52.34        | 1,573.83       |
| **512**     | **6,768.5**  | **1,692.1** | 72.79        | 1,516.03       |

## Comparison with B200 (InferenceX Official, FP4 TP=4 EP=4)

| Concurrency | B200 tok/s/gpu | B300 (PTX JIT) | B300 (sm_103a) | sm_103a vs B200 |
|:-----------:|:--------------:|:--------------:|:--------------:|:---------------:|
| 4           | 112.7          | 94.5           | 103.0          | -8.6%           |
| 8           | 181.9          | 174.8          | 171.8          | -5.6%           |
| 16          | 276.8          | 252.9          | 245.6          | -11.3%          |
| 32          | 392.6          | 376.9          | 389.2          | -0.9%           |
| 64          | 571.2          | 529.4          | 550.9          | -3.6%           |
| 128         | N/A            | —              | 795.7          | —               |
| 256         | N/A            | —              | 1,163.3        | —               |
| 512         | N/A            | —              | 1,692.1        | —               |

B200 reference data from [InferenceX](https://inferencex.com/) (ISL=1024, OSL=1024, FP4, TP=4, EP=4).

## Key Findings

### 1. Native sm_103a vs PTX JIT
Building vLLM from source with `TORCH_CUDA_ARCH_LIST="10.0a;10.3a"` yields **~4% improvement** at high concurrency (550.9 vs 529.4 tok/s/gpu at c=64). The PTX forward-compatibility overhead on B300 is modest for decode-heavy workloads.

### 2. B300 Memory Advantage at High Concurrency
B300 has **275GB HBM3e** per GPU vs B200's **192GB HBM3e** — 43% more memory. This translates directly into higher concurrency support:

- **KV cache capacity**: ~716 concurrent requests (at 4096 tokens each)
- At c=512, B300 achieves **1,692.1 tok/s/gpu** — **2.96x** the throughput of B200's peak at c=64 (571.2)
- Throughput scales near-linearly from c=64 to c=512, with TPOT increasing from 27ms to 73ms

### 3. B300 vs B200 at Equal Concurrency
At c=64 (B200's max tested), B300 native sm_103a is within **3.6%** of B200. The small gap is likely due to software not yet fully optimized for sm_103a (e.g., flash attention lacks native sm_103 kernels).

## Building vLLM with Native sm_103a

The official vLLM Docker images (including cu130-nightly) only include sm_100 kernels. To build with native sm_103a support:

```bash
bash build_sm103a.sh
```

This script:
1. Clones vLLM source matching the nightly version
2. Patches `CMakeLists.txt` to add 10.3 to the supported arch whitelist
3. Fixes missing CUDA dev headers and nvrtc symlinks in the container
4. Builds the wheel with `TORCH_CUDA_ARCH_LIST="10.0a;10.3a"`
5. Creates a new Docker image (`vllm-b300-sm103a:latest`)

Verification: the built image contains **50 sm_100 kernels + 30 sm_103 kernels** in `_C.so`.

## Running Benchmarks

```bash
# Single concurrency test
bash run_b300_benchmark_full.sh \
  --image vllm-b300-sm103a:latest \
  --filter "fp4_tp4_ep4_1024_1024_conc64"

# Full matrix (fp4 + int4, all ISL/OSL ratios)
bash run_b300_benchmark_full.sh \
  --image vllm-b300-sm103a:latest \
  --results-dir /path/to/results
```

## Files

| File | Description |
|------|-------------|
| `kimik2.5_fp4_vllm_tp4_ep4_dpa_false_conc{N}_b300.json` | Benchmark results for concurrency N |
| `gpu_metrics_*.csv` | Per-second GPU utilization metrics during each run |
| `../../build_sm103a.sh` | Script to build vLLM Docker image with native sm_103a |
| `../../run_b300_benchmark_full.sh` | Full benchmark matrix runner |
