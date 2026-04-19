# GLM-ASR Triton FP16 Mixed Precision Scope Notes

## 1. Bottleneck conclusion

Given the current profile:

- Audio Encoder: `204.09 ms` (`8.0%`)
- Multi-modal Projector: `1.66 ms` (`0.1%`)
- Decoder Prefill: `52.54 ms` (`2.1%`)
- Decoder (50 decode steps): `2290.79 ms` (`89.9%`)

The dominant bottleneck is still **decoder decode-stage attention**, not norm or activation.

The key reason is that the real workload does **not** stay on the small-size Triton attention path:

- encoder attention:
  - `head_dim = 64`
  - post-conv `seq_k` is around `1500`
  - `next_power_of_two(seq_k) > 256`
  - result: fallback path
- decoder attention:
  - `head_dim = 128`
  - prefill/decode `seq_k` is already much larger than `256`
  - result: fallback path for the hot decode loop

Before this change, the fallback path was:

1. `torch.einsum("bnqd,bnkd->bnqk", q, k)`
2. explicit softmax
3. `torch.einsum("bnqk,bnkd->bnqd", attn_weights, v)`

This materializes the full attention matrix and keeps the hottest path away from Flash/SDPA-style kernels.

## 2. Scope before the change

### Already effectively FP16

- weights loaded as `torch.float16`
- cuBLAS linear path (`Linear.BACKEND = "cublas"`) consumed FP16 activations
- conv input/weight path mostly followed weight dtype
- embeddings were already stored/output in weight dtype

### Not really using mixed precision end-to-end

- large-sequence attention fallback used explicit `einsum + softmax + einsum`
- Triton small attention path first converted `q/k/v` buffers to `float32`
- Triton RMSNorm / LayerNorm created full FP32 input buffers before launch
- Triton GELU / SiLU created full FP32 flattened buffers before launch
- Triton softmax created full FP32 flattened buffers before launch
- fused decoder SwiGLU path transposed weights to FP32 and expanded activations / padded buffers / intermediate tensors to FP32
- fused encoder `Linear + GELU` path also expanded activations / weights / intermediates to FP32

So the previous implementation was closer to:

- `FP16` weights for many modules
- `FP16` cuBLAS matmul for standard linear layers
- but many Triton-side intermediate buffers still widened to `FP32`
- and the hottest attention fallback did not use a modern mixed-precision attention kernel

## 3. Code changes in this patch

### A. Attention fallback switched to SDPA

File: `attention.py`

For non-Triton attention shapes, the code now prefers:

- `torch.nn.functional.scaled_dot_product_attention(...)`

This matters because on CUDA it can dispatch to:

- Flash Attention backend
- memory-efficient SDPA backend
- or optimized math backend

all without explicitly materializing the full `QK^T` score tensor in Python code.

This is the single highest-impact change for the measured profile, because it targets the `89.9%` decode-stage bottleneck directly.

### B. Small Triton attention path now keeps FP16 storage

File: `attention.py`

The small-size Triton path still uses FP32 score accumulation / softmax output where appropriate, but it no longer widens:

- `q_flat`
- `k_flat`
- `v_flat`
- padded `q/k/v` buffers
- final output buffer

to full FP32 tensors up front.

New behavior:

- storage/buffer dtype follows `q.dtype`
- reductions still happen in higher precision inside the kernels where needed

### C. Norm / activation / softmax stop creating full FP32 staging buffers

File: `layers.py`

Updated:

- `RMSNorm`
- `LayerNorm`
- `gelu`
- `silu`
- `softmax`

New behavior:

- input/output tensors keep the original activation dtype
- kernel internals still cast loaded values to `float32` for numerically sensitive reductions / exponentials

This is the right mixed-precision split for these ops:

- FP16 or BF16 storage / bandwidth
- FP32 internal accumulation or normalization

### D. Fused MLP paths now use mixed-precision buffers

File: `layers.py`

Updated:

- `MLP._forward_fused()` for decoder SwiGLU
- `EncoderMLP._forward_fused()` for fused `Linear + GELU`

New behavior:

- transposed fused weights keep original weight dtype instead of forced FP32
- input/padded/intermediate buffers keep fused weight dtype
- Triton `tl.dot(...)` still accumulates into FP32 accumulators in-kernel

This expands mixed precision to the decoder fused MLP path, which is important because decoder MLP runs inside the hot autoregressive loop.

## 4. Scope after the change

### Now covered by mixed precision more completely

- standard linear layers on cuBLAS: `FP16` input + `FP16` weight path
- large attention fallback: now SDPA-backed instead of manual `einsum`
- small Triton attention path: `FP16` Q/K/V storage, FP32 score math where needed
- RMSNorm / LayerNorm: `FP16` activation storage, FP32 reduction internally
- GELU / SiLU / softmax: `FP16` activation storage, FP32 nonlinear math internally
- fused decoder SwiGLU: `FP16` activations / weights / staging buffers, FP32 accumulation internally
- fused encoder `Linear + GELU`: same mixed-precision pattern

### Still not fully FP16 end-to-end

- Triton attention kernels still keep score/softmax computation in higher precision
- fallback quality/performance still depends on which SDPA backend PyTorch selects on the runtime GPU
- RoPE application is still regular Torch tensor math, not a fused mixed-precision Triton kernel
- conv is still `pad + im2col + matmul`, not a fully fused conv pipeline
- `generate()` still does full-prefix decoding in `model.py`
  - this remains a bigger structural issue than micro-optimizing individual kernels

## 5. Practical expectation

After this patch, the mixed-precision coverage is meaningfully broader in the hot path:

- decoder attention fallback is no longer stuck on manual `einsum`
- decoder fused MLP no longer expands whole buffers to FP32
- norm/activation/softmax kernels no longer pay an avoidable FP32 staging cost

That should improve the real application of FP16 without changing `model.py`.

The biggest remaining optimization opportunity is still:

1. KV-cache incremental generation
2. ensuring decoder attention stays on Flash/SDPA efficiently for the real decode path
3. only then revisiting more operator-level fusion/tile tuning
