# GLM-ASR Triton Example (FP16 Experiment)

This folder records the mixed-precision experiment run against the Triton example implementation.

Important: the notes below describe the manual experiment setup used in the notebook analysis, where `MLP.FUSED=False`. This is the clean baseline used for comparison. Some benchmark entrypoints in the repo temporarily override flags at runtime, so treat this README as the source of truth for the experiment configuration, not for every helper script.

## Experiment Summary

- Baseline E2E: `76.02 ms/token`
- FP16 E2E: `66.76 ms/token`
- Relative improvement: about `12.2%`
- Accuracy on the benchmark sample: unchanged (`100%`)

The measured speedup mainly comes from the audio encoder and projector. Decoder prefill and per-token decode did not improve much, and can even regress slightly, because several decoder-critical paths still run in FP32 and/or fall back to Torch.

## Baseline vs FP16

The FP16 experiment keeps the high-level model structure and most execution choices identical to the baseline. The main change is mixed-precision weight loading plus allowing the Torch/cuBLAS linear path to consume FP16 activations.

### Shared settings

- `Linear.BACKEND = "cublas"` in both baseline and FP16, so the main linear path is still the Torch matmul path backed by CUDA libraries, not the custom Triton GEMM path.
- `MLP.FUSED = False` for the experiment comparison.
- `EncoderMLP.FUSED = True`.
- Attention dispatch policy is unchanged（for 3.5s audio，it‘s still torch matmul path）:
  - use Triton only when `q.is_cuda`
  - `next_power_of_two(seq_k) <= 256`
  - `next_power_of_two(head_dim) <= 256`
- Model architecture, prompt construction, and generation flow are unchanged.

### FP16-specific changes

- Weights are loaded as `torch.float16` instead of `torch.float32` in [weight_loader.py](glm_asr_triton_example_FP16/weight_loader.py:13).
- This applies to:
  - linear weights and bias
  - conv weights and bias
  - LayerNorm / RMSNorm weights
  - embeddings
  - audio positional embeddings
- In the Torch/cuBLAS linear path, activations are cast to `self.weight.dtype` instead of being forced to `torch.float32`, see [layers.py](glm_asr_triton_example_FP16/layers.py:661).

### What did not change

- The Triton linear kernel still computes in FP32 and pads inputs/weights in FP32, see [layers.py](glm_asr_triton_example_FP16/layers.py:680).
- Triton attention still upcasts `q/k/v` to FP32 before computing scores, see [attention.py](glm_asr_triton_example_FP16/attention.py:297).
- Triton RMSNorm / LayerNorm / activations / softmax also use FP32 compute internally, see [layers.py](glm_asr_triton_example_FP16/layers.py:489), [layers.py](glm_asr_triton_example_FP16/layers.py:532), [layers.py](glm_asr_triton_example_FP16/layers.py:568), and [layers.py](glm_asr_triton_example_FP16/layers.py:787).

## Which parts are not really using Triton today

Even though this folder is the "Triton example", a large portion of the end-to-end path still does not run on custom Triton kernels in the benchmarked configuration.

### 1. Linear layers are on cuBLAS, not Triton

- Because `Linear.BACKEND = "cublas"`, all standard `Linear(...)` calls go through `_forward_torch()` instead of `_forward_triton()`, see [__init__.py](glm_asr_triton_example_FP16/__init__.py:19) and [layers.py](glm_asr_triton_example_FP16/layers.py:652).
- This means q/k/v projections, output projections, projector MLP, decoder MLP projections, and LM head are primarily using Torch matmul backed by CUDA libraries.

### 2. Attention is mostly on Torch fallback

- `scaled_dot_product_attention()` only uses Triton for small attention sizes where both padded `seq_k` and padded `head_dim` are at most `256`, see [attention.py](glm_asr_triton_example_FP16/attention.py:291).
- In the benchmarked audio encoder:
  - `head_dim = 1280 / 20 = 64`
  - encoder sequence length after the strided conv is about `1500`
  - padded `seq_k` therefore exceeds `256`
  - result: encoder attention falls back to Torch
- In the benchmarked decoder:
  - `head_dim = 3584 / 28 = 128`
  - prefill sequence length is about `389`
  - decode-step sequence length keeps growing from there
  - padded `seq_k` again exceeds `256`
  - result: decoder attention also falls back to Torch
- The fallback path explicitly materializes the score matrix with `torch.einsum`, softmax, and a second `einsum`, see [attention.py](glm_asr_triton_example_FP16/attention.py:412).

### 3. RoPE application is not a fused Triton kernel

- RoPE frequency generation uses a Triton kernel, but applying rotary embeddings is still done through regular tensor operations in Python/Torch, see [rope.py](glm_asr_triton_example_FP16/rope.py:202).

### 4. Conv1d is only partially Triton

- The audio conv layers can use Triton only when shape constraints are met, see [conv.py](glm_asr_triton_example_FP16/conv.py:193).
- The implementation still performs `pad` and `im2col` outside the Triton kernel, so it is not a fully fused convolution pipeline.

### 5. Decoder generation is not using KV-cache incremental decoding

- `generate()` repeatedly calls `self.decode(inputs_embeds=inputs_embeds)` on the full prefix and then appends one more token embedding, see [model.py](glm_asr_triton_example_FP16/model.py:818).
- The repo contains cache-aware decoder code paths, but the benchmarked generation loop does not use them.
- This keeps attention sequence length large for every token and amplifies decoder overhead.

### 6. Several "Triton" kernels still compute in FP32

- RMSNorm / LayerNorm upcast inputs to FP32.
- GELU / SiLU upcast to FP32.
- Triton softmax upcasts to FP32.
- Triton attention upcasts `q/k/v` to FP32.
- Fused SwiGLU path also prepares FP32 weights and FP32 activations when enabled.

So the current experiment is best described as:

- FP16 weights plus FP16-friendly cuBLAS linear path
- some Triton kernels for selected ops
- many important reductions and attention paths still effectively running in FP32

## Why the current FP16 speedup is limited

The current implementation gets real gains, but they do not fully translate to end-to-end speedup because:

- audio encoder and projector benefit a lot from lower-precision weights and matmuls
- decoder attention still falls back to Torch for the real sequence lengths in this workload
- generation is still full-prefix decoding rather than efficient cached decoding
- several Triton kernels upcast back to FP32, reducing the amount of true mixed-precision execution

This is why the measured E2E improvement is meaningful but still much smaller than a naive "FP16 should make everything much faster" expectation.

## Good next optimization targets

### Highest priority

- Use KV-cache incremental decoding in `generate()`
  - This should reduce decoder per-token work much more than micro-optimizing the current full-prefix loop.
- Replace the current large-sequence attention fallback with a real flash-style attention path or PyTorch SDPA
  - The current fallback materializes the full attention matrix and is memory-bandwidth heavy.

### High priority

- Keep decoder attention in reduced precision where numerically safe
  - Today the Triton attention path upcasts to FP32 before compute.
- Audit decoder MLP for FP32 re-expansion
  - Even when weights are FP16, fused/internal paths still convert activations and cached weights to FP32.
- Re-benchmark the clean comparison with `MLP.FUSED=False` enforced in the benchmark entrypoint
  - This avoids confusion caused by helper scripts that override flags dynamically.

### Medium priority

- Evaluate `bfloat16` as an alternative to `float16`
  - It may offer a better numerical stability / speed trade-off depending on the GPU.
- Fuse more of the conv pipeline
  - `pad + im2col + matmul` still leaves overhead outside the kernel.
- Consider a Triton or SDPA-backed sampling/logits post-processing path only after decoder attention and KV cache are fixed
  - The current top-k sampling path is not the first bottleneck to attack.

## Practical takeaway

This folder is not a "full Triton FP16" implementation yet.

It is a mixed-precision variant of the Triton example where:

- weights are loaded in FP16
- the default linear path can now benefit from FP16/cuBLAS
- many other components still remain on Torch fallback or FP32 compute

That is enough to produce a clear encoder-heavy speedup, but not enough to fully accelerate decoder-dominated end-to-end ASR latency.
