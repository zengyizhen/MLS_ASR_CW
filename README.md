# GLM-ASR Triton Kernel Optimisation

A GPU kernel optimisation project for a GLM-based speech recognition (ASR) model, targeting end-to-end inference latency on CUDA. Custom Triton kernels progressively replace PyTorch/cuBLAS paths across the encoder–decoder pipeline through a structured ablation study.

---

## Background

The baseline model uses PyTorch's default execution paths (cuBLAS for linear layers, `torch.einsum` for attention). Conditional Triton kernels exist in the codebase but were largely gated-off due to shape constraints:

---

## Full Ablation Table

| # | Configuration | Parent / Reference | E2E (ms/token) | Δ vs Parent | Δ vs Baseline | Component Total (ms) | Encoder (ms) | Decoder 50 steps (ms) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Baseline (cuBLAS, no fusion) | — | 76.02 | — | — | 2766.29 | 209.46 | 2432.55 |
| 2 | + SwiGLU Fused Kernel | #1 | 72.94 | −4.1% | −4.1% | 2699.99 | 209.46 | 2432.55 |
| 3 | + RMSNorm / LayerNorm Triton | #2 | 71.62 | −1.8% | −5.8% | 2549.07 | 204.09 | 2290.79 |
| 4A | + Mixed Precision (FP16 storage) | #3 | 60.55 | −15.4% | −20.3% | 2542.17 | 66.50 | 2420.10 |
| — | Global Triton Linear (regression) | #2 | 76.89 | +5.4% | +1.1% | 3220.67 | ~130 | 3022.25 |
| 4B | + Encoder-only Triton Linear | #3 | 64.39 | −10.1% | −15.3% | 2597.35 | 124.40 | 2415.35 |
| 4C | + KV Cache | 4B | 58.99 | −8.4% | −22.4% | 2430.11 | 123.46 | 2252.97 |
| **5** | **SwiGLU + Norm + Encoder Linear + FP16 + KV Cache** | **4A + 4C** | **58.28** | **−1.2% vs 4C** | **−23.3%** | — | — | — |

> All benchmarks: fixed 3.5s audio input, single CUDA GPU. E2E = end-to-end latency per generated token.

## Bottleneck Analysis and Decision Making

### #1 Baseline: cuBLAS, no fusion
**Bottleneck.** Most Triton kernels existed but were not effectively used. Norm kernels were gated by power-of-two hidden sizes, Conv exceeded Triton shape limits, and encoder attention fell back to `torch.einsum`.

**Decision.** Start from a conservative cuBLAS baseline and optimise only kernels with clear activation paths.

---

### #2 SwiGLU Fused Kernel
**Bottleneck.** Decoder MLP was repeatedly executed during token generation, but gate/up projection and SiLU were still separate operations.

**Decision.** Enable `swiglu_fused_kernel` for decoder MLP.

**Result.** E2E improved from `76.02` to `72.94 ms/token` (`−4.1%`). The gain confirmed fusion was useful, but decoder attention and repeated decode steps remained dominant.

---

### #3 RMSNorm / LayerNorm Triton
**Bottleneck.** RMSNorm and LayerNorm were not activated because hidden sizes such as `1280` and `3584` are not powers of two.

**Decision.** Relax the power-of-two restriction and use masked Triton norm kernels.

**Result.** E2E improved to `71.62 ms/token` (`−1.8% vs #2`). Norm optimisation helped, but the overall gain was limited because decoder generation still dominated total latency.

---

### #4A Mixed Precision: FP16 Storage
**Bottleneck.** Many weights and intermediate buffers were stored or staged as FP32, increasing memory bandwidth and cache pressure.

**Decision.** Use FP16 storage for weights, activations, staging buffers, attention Q/K/V, Conv buffers, and embeddings, while keeping numerically sensitive reductions such as norm, softmax, and attention scores in FP32.

**Result.** E2E improved to `60.55 ms/token` (`−15.4% vs #3`, `−20.3% vs baseline`). This was the largest single-step improvement, showing that memory bandwidth and FP32 expansion were major bottlenecks.

---

### Global Triton Linear Regression
**Bottleneck.** Linear layers were expensive, especially in the encoder and decoder projections.

**Decision.** Test replacing all Linear layers with the custom Triton Linear kernel.

**Result.** Encoder became faster, but decoder became slower; E2E regressed to `76.89 ms/token` (`+5.4% vs #2`). The custom Linear kernel was not competitive with cuBLAS/cuBLASLt for all shapes, especially decoder and large-output paths such as `lm_head`.

**Decision Outcome.** Do not enable Triton Linear globally. Apply it selectively.

---

### #4B Encoder-only Triton Linear
**Bottleneck.** Global Triton Linear hurt decoder performance, but encoder Linear showed clear improvement.

**Decision.** Restrict Triton Linear to encoder-side Linear layers and keep decoder Linear on cuBLAS.

**Result.** E2E improved to `64.39 ms/token` (`−10.1% vs #3`). Encoder latency dropped from `204.09 ms` to `124.40 ms`, confirming that Triton Linear is beneficial for encoder shapes.

---

### #4C KV Cache
**Bottleneck.** Autoregressive decoding repeatedly recomputed previous tokens, making decoder 50-step latency the largest component.

**Decision.** Enable KV cache so each decode step only processes the newly generated token while reusing previous K/V states.

**Result.** E2E improved to `58.99 ms/token` (`−8.4% vs 4B`, `−22.4% vs baseline`). Decoder 50-step latency dropped from `2415.35 ms` to `2252.97 ms`.

---

### #5 Final Combined Configuration
**Bottleneck.** Remaining latency was still dominated by decoder generation, but the best independent optimisations were now identified.

**Decision.** Combine SwiGLU fusion, Triton Norm, encoder-only Triton Linear, FP16 storage, and KV cache.

**Result.** Final E2E latency reached `58.28 ms/token`, a `−23.3%` improvement over baseline.

---

## Key Takeaways

- Kernel fusion helps, but only moderately when applied to isolated MLP paths.
- Mixed precision gives the largest single improvement by reducing FP32 storage and memory bandwidth pressure.
- Triton Linear should be shape-selective; global replacement can regress performance.
- Encoder-only Triton Linear is effective, while decoder Linear remains better served by cuBLAS.
- KV cache is essential because autoregressive decoding dominates end-to-end latency.

## Future Work

Further optimisation should focus on:
- replacing the current attention path with FlashAttention, especially for decoder generation and encoder attention fallback;
- tuning Triton `BLOCK_SIZE`, `BLOCK_M`, `BLOCK_N`, `BLOCK_K`, and tile sizes per operator shape;
- adding autotuning for Linear and fused kernels instead of relying on fixed tile settings.

---

## Tech Stack

Python · PyTorch · [Triton](https://github.com/openai/triton) · CUDA  
Model: GLM-ASR — Encoder hidden 1280, Decoder hidden 3584, Vocab 151,552