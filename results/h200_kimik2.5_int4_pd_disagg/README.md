# Kimi-K2.5 INT4 PD Disaggregated Serving Benchmark (Mooncake EFA, upstream main)

## Setup
- **Hardware**: 2x p5en.48xlarge (8x H200 each)
- **Model**: Kimi-K2.5 (compressed-tensors W4A16, ~595GB)
- **Config**: TP=8 per node, max_model_len=16384, prefix_caching=on, gpu_mem_util=0.9
- **Mooncake**: upstream main (commit e426e35), EFA protocol
- **vLLM**: 0.18.1
- **Workload**: random ISL=8192, OSL=1024, num_prompts=conc*10
- **M1 (prefill)**: 172.31.35.3, port 8010
- **M2 (decode)**: 172.31.45.191, port 8020
- **Proxy**: M2:8000

## Results

| Conc | Output tok/s | Total tok/s | Mean TTFT (ms) | Mean TPOT (ms) | P99 TTFT (ms) | P99 TPOT (ms) |
|:----:|:------------:|:-----------:|:--------------:|:--------------:|:-------------:|:-------------:|
| 4    | 169.2        | 1,522.8     | 983.5          | 22.60          | 2,333.6       | 22.75         |
| 8    | 319.4        | 2,874.6     | 715.7          | 24.20          | 2,773.2       | 24.51         |
| 16   | 563.5        | 5,071.9     | 904.8          | 27.21          | 4,947.8       | 27.64         |
| 32   | 954.3        | 8,588.9     | 1,675.1        | 31.28          | 14,242.7      | 32.01         |
| 64   | 1,590.5      | 14,314.9    | 2,422.8        | 36.60          | 27,476.6      | 37.47         |

## Comparison with Single-Node (no PD disagg, same model, same hardware)

| Conc | PD Output tok/s | Single Output tok/s | PD TTFT (ms) | Single TTFT (ms) |
|:----:|:---------------:|:-------------------:|:------------:|:----------------:|
| 4    | 169             | 351                 | 984          | 1,346            |
| 8    | 319             | 532                 | 716          | 1,972            |
| 16   | 564             | 720                 | 905          | 2,123            |
| 32   | 954             | 906                 | 1,675        | 2,679            |
| 64   | 1,591           | 1,108               | 2,423        | 3,321            |

PD disagg throughput exceeds single-node at conc>=32 and has significantly lower TTFT at low concurrency.

## Date
2026-04-02
