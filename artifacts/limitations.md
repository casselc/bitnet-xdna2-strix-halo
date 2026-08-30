# Known limitations

## Scope

- One checkpoint, one architecture, one quantization: `BitNet-b1.58-2B-4T`, I2_S.
  This is not a generic XDNA ggml backend and does not register through the
  `ggml_backend_reg_t` interface — it is a guarded dispatch inside the CPU
  backend's I2_S `mul_mat` path.
- Three shapes are offloaded (`attn_q`, `attn_output`, `ffn_gate`, `ffn_up`,
  `ffn_down`). `attn_k` and `attn_v` (K=2560, N=640) are deliberately left on the
  CPU: they are 6% of prefill FLOPs but 21% of NPU time, because N=640 cannot
  fill 8 columns.
- Decode is entirely CPU. Fused whole-decoder NPU execution is not attempted.

## Performance

- **The hybrid path is slower than CPU-only at every prompt length** (0.27x at
  128 tokens, 0.66–0.75x at 512–3968). See `e2e/results.md`. The mechanism works;
  the economics currently do not.
- The stock mlir-aie `whole_array` kernel reaches ~9.3 TOPS against a ~50 TOPS
  device peak (~19%). Published hand-tuned XDNA2 int8 GEMM reaches 38–56 TOPS, so
  roughly 4–6x of kernel headroom is untouched. No custom AIE kernel was written.
- Weights are expanded ternary → int8, a **4x inflation**: 1843 MiB resident
  versus 461 MiB of packed I2_S. The native `8b x 4b` mmul (1024 MAC/cycle, half
  the DMA) is the obvious next step and was not attempted.
- One-time ternary→int8 repack and upload costs **3.2–3.5 s**. Acceptable only
  for a long-lived resident controller, not for short-lived processes.

## Correctness caveats

- Bit-exactness is demonstrated for the `2560x2560` shape against real GGUF
  weights. The other two shapes are exercised end-to-end (identical generated
  text) but do not have a dedicated bit-exactness test.
- Output equivalence was checked on one controller-style prompt with greedy
  sampling. No perplexity regression run was performed.

## Contract caveats

- **The KV coordinate is thin by construction.** Because the offload sits inside
  `mul_mat`, KV is produced by unmodified llama.cpp in its canonical host layout;
  there is no NPU-side KV and therefore no conversion to validate. The
  `bitnet_kv_coord` type and its tests exist so a later design that *does* move KV
  cannot silently reuse incompatible state — they are not exercised by a real
  cross-backend handoff here, and this evidence should not be read as if they were.
- The weight-residency cache is keyed on the weight tensor's data pointer, which
  is stable for an mmap'd GGUF but would need revisiting if weights were ever
  relocated or re-quantized in place.

## Build and environment caveats

- `-DGGML_LLAMAFILE=OFF` is **required**, and it changes which CPU kernel runs.
  Both CPU-only and hybrid numbers here use it, so the comparison is fair, but
  these CPU numbers are not comparable to a stock llamafile-enabled build.
- `patches/001-bitnet-xdna.patch` fixes a genuine upstream bug: the fork's
  I2_S path references `src1_cont`, declared only under `#if GGML_USE_LLAMAFILE`.
  That path has evidently never been compiled with llamafile off.
- `llama-completion` crashes in `common_chat_format_example` in this fork, so
  generation comparisons go through `llama-cli`'s REPL with piped input.
- Energy was not measured: `/sys/class/powercap/intel-rapl:0/energy_uj` is
  root-only on this host and the brief does not gate on it.
- `xrt-smi validate` cannot run — Debian's XRT packages strip the prebuilt
  validation xclbins (Xilinx/XRT#8237) — so there is no vendor-supplied GEMM
  reference number to check ours against.

## Measurement caveats

- All figures are single-host, single-session. Prompt-processing throughput
  varied ±10% between `llama-bench` invocations (e.g. pp512 CPU-only measured
  1277 t/s early and 878 t/s later in the same session, on different builds).
  **Comparisons within one table were collected back-to-back; comparisons across
  tables should not be trusted to better than ~15%.**
- Published XDNA2 figures cited for context (0.66 ms dispatch, 2.67 ms context
  switch, 38–56 TOPS) were measured on Strix Point / Krackan, not Strix Halo.
  Where we measured the same quantity here, Strix Halo was substantially better
  (0.197 ms dispatch, 0.101–0.22 ms switch), so those references should be treated
  as weak priors only.
