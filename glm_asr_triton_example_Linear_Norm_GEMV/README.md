# Linear + Norm + GEMV Split

## Current Focus

1. `Projector -> Triton`: `MultiModalProjector.linear_1/linear_2` now explicitly use the Triton linear backend. For the common audio prefill shape (`M ~= 375`), this lands on the Triton GEMM path directly.
2. `Decoder GEMM/GEMV split`: decoder attention projections, decoder MLP linears, and `lm_head` now all use the Triton backend with a shape-based dispatch in `Linear.__call__`.

## Linear Dispatch Policy

- `M = 1`: use `linear_gemv_kernel`, targeting autoregressive decode where a tiled GEMM is structurally inefficient.
- `M > 1`: use the existing tiled `linear_kernel_tf32` GEMM path, covering prefill and batched rows.
- `CPU / no CUDA`: explicit Triton layers fall back to the Torch matmul path, which keeps smoke tests usable on non-GPU environments.

## Notes

- The global default backend is still `cublas`, but the projector/decoder call sites in `model.py` now opt into Triton explicitly.
- This step implements the control-plane split first. Weight packing and FP16-specific bandwidth optimization for the GEMV path can be added on top of the new dispatch point.
