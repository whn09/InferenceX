# Kimi-K2.5 INT4 PD Disagg Benchmark — fix branch + vLLM FINISHED_STOPPED patch

## Setup
- **Hardware**: 2x p5en.48xlarge (8x H200 each)
- **Model**: Kimi-K2.5 (compressed-tensors W4A16, ~595GB)
- **Config**: TP=8 per node, max_model_len=16384, prefix_caching=on, gpu_mem_util=0.9
- **Mooncake**: branch fix/efa-read-and-endpoint-eviction (commit d491918 + stale-slice fix 51371f6)
- **vLLM**: 0.18.1, patched `request_finished()` to handle FINISHED_STOPPED
- **Workload**: random ISL=8192, OSL=1024, num_prompts=conc*10
- **M1 (prefill)**: 172.31.35.3, port 8010
- **M2 (decode)**: 172.31.45.191, port 8020
- **Proxy**: M2:8000

## vLLM Patch Details
The original vLLM `mooncake_connector.py` `request_finished()` only sends KV cache
when `request.status == FINISHED_LENGTH_CAPPED`. The proxy sets `max_tokens=1` for
P-side prefill requests. If the model generates EOS as the first token, the status
is `FINISHED_STOPPED`, and KV cache was never sent — causing D-side "P side ready"
timeout.

**Fix**: Allow both `FINISHED_LENGTH_CAPPED` and `FINISHED_STOPPED` to proceed with
KV cache transfer. Only skip for `FINISHED_ABORTED`.

## Results

| Conc | Successful | Output tok/s | Mean TTFT (ms) | Mean TPOT (ms) | P99 TTFT (ms) | P99 TPOT (ms) |
|:----:|:----------:|:------------:|:--------------:|:--------------:|:-------------:|:-------------:|
| 4    | 40/40      | 158.6        | 898.9          | 22.77          | 1,749.5       | 23.44         |
| 8    | 80/80      | 302.9        | 671.2          | 24.50          | 1,620.4       | 25.51         |
| 16   | 160/160    | 540.8        | 785.0          | 27.28          | 2,094.9       | 27.88         |
| 32   | 320/320    | 931.5        | 1,797.5        | 31.08          | 14,286.5      | 32.09         |
| 64   | 639/640    | 1,548.4      | 2,868.4        | 36.34          | 28,642.3      | 37.23         |

## Comparison with Upstream Main

| Conc | Main tok/s | Fix tok/s | Delta tok/s | Main TTFT (ms) | Fix TTFT (ms) | Delta TTFT |
|:----:|:----------:|:---------:|:-----------:|:--------------:|:-------------:|:----------:|
| 4    | 169.2      | 158.6     | -6.3%       | 983.5          | 898.9         | **-8.6%**  |
| 8    | 319.4      | 302.9     | -5.2%       | 715.7          | 671.2         | **-6.2%**  |
| 16   | 563.5      | 540.8     | -4.0%       | 904.8          | 785.0         | **-13.2%** |
| 32   | 954.3      | 931.5     | -2.4%       | 1,675.1        | 1,797.5       | +7.3%      |
| 64   | 1,590.5    | 1,548.4   | -2.6%       | 2,422.8        | 2,868.4       | +18.4%     |

## Notes
- conc=4 through conc=32: **100% success rate** (was ~1 failure per level before vLLM patch)
- conc=64: 639/640 — the 1 failure is a proxy httpx.ReadError (connection to M1 dropped at high concurrency), not a Mooncake or vLLM issue
- Zero EFA CQ errors — stale-slice bug fix confirmed working
- TTFT at conc=4/8/16 is **better** than upstream (up to 13% improvement), likely due to NIC striping benefits
- TTFT regression at conc=32/64 — striping overhead at high concurrency
- Output throughput is ~2-6% lower than upstream across all concurrency levels — slight overhead from pre-resolved peer info and striping logic
- TPOT is essentially identical to upstream (within noise)

## Date
2026-04-02
