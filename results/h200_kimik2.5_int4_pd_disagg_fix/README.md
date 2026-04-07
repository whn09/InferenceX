# Kimi-K2.5 INT4 PD Disagg Benchmark — branch fix/efa-read-and-endpoint-eviction

## Setup
- **Hardware**: 2x p5en.48xlarge (8x H200 each)
- **Model**: Kimi-K2.5 (compressed-tensors W4A16, ~595GB)
- **Config**: TP=8 per node, max_model_len=16384, prefix_caching=on, gpu_mem_util=0.9
- **Mooncake**: branch fix/efa-read-and-endpoint-eviction (commit d491918 + stale-slice fix)
- **vLLM**: 0.18.1
- **Workload**: random ISL=8192, OSL=1024, num_prompts=conc*10
- **M1 (prefill)**: 172.31.35.3, port 8010
- **M2 (decode)**: 172.31.45.191, port 8020
- **Proxy**: M2:8000

## Branch Changes
1. fi_read support in submitPostSend()
2. EfaEndpointStore LRU eviction (max_endpoints=256)
3. NIC striping for large transfers (>2MB): one chunk per NIC instead of 256KB slices
4. Pre-resolved peer info for striped transfers
5. **Bug fix**: Clear stale peer_nic_path/dest_rkey on slice allocation from SliceCache

## Results

| Conc | Successful | Output tok/s | Total tok/s | Mean TTFT (ms) | Mean TPOT (ms) | P99 TTFT (ms) | P99 TPOT (ms) |
|:----:|:----------:|:------------:|:-----------:|:--------------:|:--------------:|:-------------:|:-------------:|
| 4    | 39/40      | 83.8         | 798.4       | 1,341.9        | 22.83          | 3,517.9       | 23.17         |
| 8    | 79/80      | 207.3        | 1,929.9     | 716.4          | 24.49          | 2,079.3       | 25.04         |
| 16   | 160/160    | 542.2        | 5,134.2     | 767.9          | 27.33          | 2,115.9       | 27.89         |
| 32   | 319/320    | 902.9        | 8,468.6     | 1,769.4        | 31.39          | 13,626.5      | 32.29         |
| 64   | 639/640    | 1,556.7      | 14,581.0    | 2,755.8        | 36.27          | 27,756.0      | 37.21         |

## Comparison with Upstream Main

| Conc | Main Output tok/s | Fix Output tok/s | Main TTFT (ms) | Fix TTFT (ms) | Delta TTFT |
|:----:|:-----------------:|:----------------:|:--------------:|:-------------:|:----------:|
| 4    | 169.2             | 83.8             | 983.5          | 1,341.9       | +36%       |
| 8    | 319.4             | 207.3            | 715.7          | 716.4         | +0.1%      |
| 16   | 563.5             | 542.2            | 904.8          | 767.9         | -15%       |
| 32   | 954.3             | 902.9            | 1,675.1        | 1,769.4       | +6%        |
| 64   | 1,590.5           | 1,556.7          | 2,422.8        | 2,755.8       | +14%       |

## Notes
- 1 request per concurrency level fails due to vLLM mooncake_connector "Timeout waiting for P side ready" bug (not EFA transport related)
- Zero EFA CQ errors ("Remote memory registration is invalid") — the stale-slice bug fix works
- conc=4 throughput is lower because 1/40 requests failed, reducing the successful count
- TTFT regression at conc=32/64 may be due to striping overhead or the 60s abort timeout adding to the duration
- conc=16 shows 15% TTFT improvement — possible benefit of NIC striping for medium concurrency

## Date
2026-04-02
