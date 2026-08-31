# Attention on XDNA2: bounded feasibility study

Answering one question before any integration work:

> For BitNet-2B's **actual** causal-prefill attention shapes on this Strix Halo,
> can XDNA2 run a flash-attention-like kernel fast enough, **after data movement
> is included**, to beat the Zen 5 implementation?

Branch base: `ed97cfcac564be9f85db415faf076695b871e008` (`direct-output-closeout`,
frozen). No llama.cpp integration in this phase.

| tag | meaning |
|---|---|
| **[MEASURED]** | observed on this machine |
| **[DERIVED]** | arithmetic over measured quantities |
| **[DEFERRED]** | not done, with the reason |

---

## A. Real model geometry [MEASURED]

Read from the GGUF's own metadata and from the live runtime, not from
model-family defaults.

| property | value | source |
|---|---|---|
| architecture | `bitnet-b1.58` | GGUF `general.architecture` |
| hidden dimension | 2560 | GGUF `embedding_length` |
| Q heads | **20** | GGUF `attention.head_count` |
| KV heads | **5** | GGUF `attention.head_count_kv` |
| GQA ratio | **4** | derived |
| head dimension | **128** | derived, 2560 / 20 |
| layers | 30 | GGUF `block_count` |
| context length | 4096 | GGUF `context_length` |
| RoPE dim / freq base | 128 / 500000.0 | GGUF `rope.*` |
| RMS norm eps | 1.0e-5 | GGUF |

Runtime facts, from the recorded execution trace and the pinned llama.cpp:

| property | value |
|---|---|
| op | `FLASH_ATTN_EXT`, one node per layer |
| output shape | `ne = [128, 20, T]` = `[head_dim, n_q_heads, tokens]` |
| Q at attention entry | **f32** (`src0_type = 0`) |
| K/V cache precision | **f16** (llama.cpp default `cache_type_k/v`) |
| output precision | f32 |
| sources | `Qcur (view, permuted)`, `cache_k_l<N> (view, permuted)` |
| CPU implementation | `ggml_compute_forward_flash_attn_ext_f16`, dispatching to `_one_chunk` or a split-KV path |

---

## C. CPU oracle [MEASURED]

The baseline any NPU kernel must beat. Measured in situ with the per-node
profiler, 15 threads, medians with the warmup prefill dropped.
Raw: `cpu_oracle.csv`.

| T | prefill | **attention** | share | attn tok/s | logical KV | logical rate | **physical KV** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 414.0 ms | **50.9 ms** | 12.3% | 10059 | 9.38 GiB | 184 GiB/s | **37.5 MiB** |
| 1024 | 876.2 ms | **165.2 ms** | 18.9% | 6198 | 37.50 GiB | 227 GiB/s | **75.0 MiB** |
| 2048 | 2020.9 ms | **602.0 ms** | 29.8% | 3402 | 150.00 GiB | 249 GiB/s | **150.0 MiB** |
| 3968 | 2722.8 ms | **1388.0 ms** | 51.0% | 2859 | 563.09 GiB | 406 GiB/s | **290.6 MiB** |

### The two KV figures are not interchangeable, and the difference decides the experiment

The **logical** rate reaches 406 GiB/s at T=3968, which **exceeds this machine's
DRAM bandwidth**. That is not a measurement error: one layer's KV at T=3968 is
only 10.2 MiB and fits comfortably in the 64 MiB L3, so the CPU is largely
re-reading cache rather than memory.

The **physical** (compulsory) traffic a tiled kernel must move is each layer's KV
once: **37.5 to 290.6 MiB per prefill**. At the NPU's measured weight-stream rate
of 51.2 GiB/s that is **0.7 to 5.5 ms** -- negligible against the CPU's 51 to
1388 ms.

**So this workload is compute-bound, not KV-bandwidth-bound, at these shapes.**
That removes the objection that killed several earlier NPU ideas on this project,
and it means the feasibility question turns almost entirely on whether aie2p can
execute the arithmetic -- above all a numerically stable softmax.

### The arithmetic the NPU would have to beat [DERIVED]

QK^T and PV, causal so ~T/2 keys per query, over 20 Q heads and 30 layers:

| T | attention work | CPU time | **CPU rate** |
|---:|---:|---:|---:|
| 512 | 0.101 TFLOP | 50.9 ms | 1.98 TFLOPS |
| 1024 | 0.403 TFLOP | 165.2 ms | 2.44 TFLOPS |
| 2048 | 1.611 TFLOP | 602.0 ms | 2.68 TFLOPS |
| 3968 | 6.049 TFLOP | 1388.0 ms | 4.36 TFLOPS |

For scale: the same CPU sustains ~10 TFLOPS on the int8 I2_S GEMM, and the NPU
sustains ~11 TOPS there. Attention runs at 2.0-4.4 TFLOPS on the CPU because it
is f32/f16 rather than int8, so the headroom is real **if** aie2p can run this
arithmetic in a format faster than the CPU's f32 path.
