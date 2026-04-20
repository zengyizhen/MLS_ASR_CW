## Config：全局默认 cuBLAS，Projector/Decoder 显式走 Triton 分流


```
layers.Linear.BACKEND = "cublas"
layers.MLP.FUSED = False
layers.EncoderMLP.FUSED = False   # 但当前模型没有用

Mixed execution:
- Global Linear default: PyTorch matmul path -> CUDA上通常落到 cuBLAS/cuBLASLt
- Projector overrides:
  - `linear_1`, `linear_2` use `backend="triton"`
  - shape `M≈375` -> Triton GEMM
- Decoder overrides:
  - `q_proj/k_proj/v_proj/o_proj`, decoder MLP linears, `lm_head` use `backend="triton"`
  - shape `M=1` -> Triton GEMV
  - shape `M>1` -> Triton GEMM
- RMSNorm / LayerNorm: Triton when hidden_size is power-of-two and x.is_cuda
- gelu / silu: Triton when x.is_cuda
- Attention Triton path:
  use_triton if q.is_cuda
  and next_power_of_two(seq_k) <= 256
  and next_power_of_two(head_dim) <= 256
  kernels:
    attention_scores_kernel
    softmax_inplace_kernel
    attention_output_kernel
- Attention fallback path:
  scores = torch.einsum("bnqd,bnkd->bnqk", q, k) * scale
  output = torch.einsum("bnqk,bnkd->bnqd", attn_weights, v)
- Conv: Triton only when shape constraints are satisfied

```

