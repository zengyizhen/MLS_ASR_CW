# Encoder Triton + Decoder cuBLAS Split

## Current Focus

1. `Encoder / Projector -> Triton`: audio encoder linears and projector linears explicitly use the Triton backend. For the common audio prefill shape (`M ~= 375`), this lands on the Triton GEMM path directly.
2. `Decoder -> cuBLAS with GEMV split`: decoder attention projections, decoder MLP linears, and `lm_head` use the `cublas` backend, and `Linear.__call__` now dispatches `M = 1` to a GEMV path.

## Linear Dispatch Policy

- `backend="triton"`:
  - `M = 1`: use `linear_gemv_kernel`
  - `M > 1`: use the tiled `linear_kernel_tf32` GEMM path
- `backend="cublas"`:
  - `M = 1`: use `torch.mv(...)`, which maps to the CUDA-library GEMV path on GPU
  - `M > 1`: use the regular Torch matmul path backed by CUDA libraries
- `CPU / no CUDA`: explicit Triton layers fall back to the Torch matmul path, which keeps smoke tests usable on non-GPU environments.

## Notes

- The global default backend is still `cublas`, but `model.py` now uses explicit per-module backend selection.
- This keeps the encoder on Triton while preserving a decoder-only `M=1` GEMV split without adding shape checks at every decoder call site.
