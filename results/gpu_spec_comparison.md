# NVIDIA GPU Spec Comparison: H100 / H200 / B200 / B300

## Hardware Specifications

| Spec | H100 SXM | H200 SXM | B200 SXM | B300 SXM6 AC |
|------|:--------:|:--------:|:--------:|:------------:|
| **Architecture** | Hopper | Hopper | Blackwell | Blackwell Ultra |
| **Compute Capability** | sm_90 | sm_90 | sm_100a | sm_103a |
| **Process** | TSMC 4N | TSMC 4N | TSMC 4NP | TSMC 4NP |
| **SMs** | 132 | 132 | 160 | 160 |
| **GPU Boost Clock** | 1980 MHz | 1980 MHz | ~1800 MHz | ~1800 MHz |
| **TDP** | 700W | 700W | 1000W | 1100W |
| **Transistors** | 80B | 80B | 208B (dual-die) | 208B (dual-die) |
| | | | | |
| **HBM Type** | HBM3 | HBM3e | HBM3e | HBM3e |
| **HBM Capacity** | 80 GB | 141 GB | 192 GB | 288 GB |
| **HBM Bandwidth** | 3.35 TB/s | 4.8 TB/s | 8.0 TB/s | 12.0 TB/s |
| **HBM Stacks** | 5 | 6 | 8 | 12 |
| | | | | |
| **FP64** | 34 TFLOPS | 34 TFLOPS | 40 TFLOPS | 45 TFLOPS |
| **FP32** | 67 TFLOPS | 67 TFLOPS | 80 TFLOPS | 90 TFLOPS |
| **FP16 / BF16 Tensor** | 990 TFLOPS | 990 TFLOPS | 2250 TFLOPS | 2500 TFLOPS |
| **FP8 Tensor** | 1979 TFLOPS | 1979 TFLOPS | 4500 TFLOPS | 5000 TFLOPS |
| **FP4 Tensor** | — | — | 9000 TFLOPS | 10000 TFLOPS |
| | | | | |
| **NVLink Gen** | 4th | 4th | 5th | 5th |
| **NVLink BW (per GPU)** | 900 GB/s | 900 GB/s | 1800 GB/s | 1800 GB/s |
| **PCIe** | Gen5 x16 | Gen5 x16 | Gen6 x16 | Gen6 x16 |

> Note: Some B300 specs are estimated based on public announcements. Actual clock speeds may vary by SKU.

## Derived Metrics (Per GPU)

| Metric | H100 | H200 | B200 | B300 | B300/B200 |
|--------|:-----:|:-----:|:-----:|:-----:|:---------:|
| **Bytes/FLOP (FP4)** | — | — | 0.89 | 1.20 | **1.35x** |
| **Bytes/FLOP (FP8)** | 1.69 | 2.43 | 1.78 | 2.40 | **1.35x** |
| **Bytes/FLOP (BF16)** | 3.38 | 4.85 | 3.56 | 4.80 | **1.35x** |
| **GB per Watt** | 0.114 | 0.201 | 0.192 | 0.262 | 1.36x |
| **TFLOPS/Watt (FP4)** | — | — | 9.0 | 9.1 | 1.01x |

`Bytes/FLOP = HBM Bandwidth / Peak TFLOPS` — higher = more bandwidth relative to compute, better for memory-bound workloads.

## Why B300 Is Slower Than B200 at Low Batch

Our benchmark shows B300 is **8-11% slower** than B200 at c=4~16, but **within 1-4%** at c=32~64, and far ahead at c=128+.

### Root Cause Analysis

LLM decode at low batch is **purely memory-bandwidth bound** — the bottleneck is loading model weights from HBM, not computing. The key question: if B300 has 50% more bandwidth (12 vs 8 TB/s), why is it slower?

#### 1. Software Immaturity for sm_103a (Primary Cause)

| Component | sm_100 (B200) | sm_103 (B300) | Impact |
|-----------|:------------:|:------------:|--------|
| `_C.so` kernels | 50 native | 30 native + 50 sm_100 | Some kernels fall back to sm_100 |
| Flash Attention | Native sm_100 | **0 sm_103 kernels** | All FA runs as sm_100 via compat |
| FlashInfer MLA | Optimized | Not specifically tuned | Decode attention not optimal |
| DeepGEMM | sm_100 tuned | sm_103 untested | MoE GEMM may not be optimal |

**Flash Attention having zero sm_103 kernels is likely the biggest single factor.** At low batch, decode attention is a significant fraction of total time, and it's running sub-optimally on B300.

#### 2. HBM Latency vs Bandwidth Tradeoff

| | B200 | B300 |
|---|:---:|:---:|
| HBM Stacks | 8 | 12 |
| Capacity per Stack | 24 GB | 24 GB |
| Total Bandwidth | 8 TB/s | 12 TB/s |

More HBM stacks increase aggregate bandwidth but can increase **access latency** due to:
- More complex memory controller routing
- Wider physical interposer distances
- Higher coordination overhead for scattered small reads

At low batch, decode generates **many small, scattered memory accesses** (loading different expert weights in MoE). Latency matters more than throughput for these patterns. At high batch, accesses are amortized and bandwidth dominates → B300 wins.

#### 3. Dual-Die Architecture & NVLink-C2C

Both B200 and B300 use a **dual-die** design connected via NVLink-C2C. However:
- B300 has more memory behind each die (144 GB vs 96 GB per die)
- Cross-die memory accesses may have slightly different latency characteristics
- MoE expert placement across dies may not be optimal

#### 4. Clock Speed

B300 may run at a slightly lower boost clock than B200 to stay within thermal envelope while driving 12 HBM stacks at 1100W. Even a 5% clock reduction directly impacts latency-bound (low-batch) workloads.

### Summary: When Each GPU Wins

| Scenario | Winner | Why |
|----------|--------|-----|
| **Low batch decode (c≤16)** | B200 | Lower latency, mature kernels |
| **Medium batch (c=32~64)** | Tie | B300 bandwidth starts helping |
| **High batch (c=128+)** | **B300** | 50% more memory, 50% more BW |
| **Max throughput** | **B300** | 1692 tok/s/gpu at c=512 vs B200 max ~571 at c=64 |
| **Latency-sensitive SLO** | B200 | Lower TPOT at equal concurrency |
| **Cost per token (high load)** | **B300** | 3x throughput at high concurrency |

### What Would Close the Gap

1. **Flash Attention with native sm_103a kernels** — biggest expected improvement
2. **FlashInfer MLA tuning for B300** — decode attention optimization
3. **vLLM upstream sm_103a support** — proper arch detection, no PTX fallback
4. **DeepGEMM / MoE kernel tuning** — expert parallelism on dual-die B300
