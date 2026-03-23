# Kimi-K2.5 FP4 Benchmark on NVIDIA B300 (AWS p6-b300.48xlarge)

## Setup
- **Instance**: AWS p6-b300.48xlarge (8x NVIDIA B300 SXM6 AC, 275GB each)
- **Model**: nvidia/Kimi-K2.5-NVFP4 (551GB)
- **Framework**: vLLM v0.18.0 (vllm/vllm-openai:v0.18.0-cu130)
- **Config**: TP=4, max-model-len=4096, gpu-memory-utilization=0.90, FP8 KV cache
- **Benchmark**: InferenceX benchmark_serving.py (ISL=1024, OSL=1024, random-range-ratio=0.8, ignore-eos)
- **Date**: 2026-03-23

## Results

| Concurrency | Output tok/s | tok/s/gpu (4 GPUs) | Mean TPOT (ms) | Mean TTFT (ms) |
|:-----------:|:------------:|:------------------:|:--------------:|:--------------:|
| 4           | 378.1        | 94.5               | 9.22           | 989.03         |
| 8           | 699.3        | 174.8              | 11.03          | 187.88         |
| 16          | 1,011.6      | 252.9              | 14.59          | 842.89         |
| 32          | 1,507.6      | 376.9              | 19.82          | 951.41         |
| 64          | 2,117.8      | 529.4              | 28.31          | 1,248.40       |

## Comparison with B200 (InferenceX reference)

B200 FP4 TP=4 peak: **1,142.7 tok/s/gpu** at c=64  
B300 FP4 TP=4 peak: **529.4 tok/s/gpu** at c=64 (46% of B200)

### Why B300 is slower than B200

The B300 (sm_103a, compute capability 10.3) runs via **PTX forward compatibility** — the vLLM Docker image is compiled for sm_100 (B200), and B300 JIT-compiles PTX at runtime. This causes a **~50% performance penalty**.

To achieve native B300 performance, vLLM needs to be compiled from source with `TORCH_CUDA_ARCH_LIST="10.3a"`. As of March 2026, no official Docker image with native sm_103a support exists.

### Known vLLM B300 issues
- SymmMemCommunicator: Device capability 10.3 not fully supported ([#30630](https://github.com/vllm-project/vllm/issues/30630))
- AllReduce fusion disabled for world_size=4 on B300
- Various initialization hangs reported ([#34133](https://github.com/vllm-project/vllm/issues/34133))
