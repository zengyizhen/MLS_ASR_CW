## Config：线性层 cuBLAS + 其他部分按条件混合 Triton


```
layers.Linear.BACKEND = "cublas"
layers.MLP.FUSED = False
layers.EncoderMLP.FUSED = False   # 但当前模型没有用

Mixed execution:
- Linear: PyTorch matmul path -> CUDA上通常落到 cuBLAS/cuBLASLt
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


## 当前attention path(3.5s audio)

**结论：** 在当前 GLM-ASR 配置和 benchmark 输入下，`encoder` 和 `decoder` 的 `scaled_dot_product_attention` 都走的是 **`torch path`**，而非 Triton path。

**Triton path 启用条件：** `q.is_cuda` AND `next_power_of_two(seq_k) <= 256` AND `next_power_of_two(head_dim) <= 256`。

### 具体分析：

1.  **Encoder Attention：**
    *   `head_dim = 1280 / 20 = 64` (满足 Triton 条件：`next_power_of_two(64) = 64 <= 256`)
    *   `seq_k = 1500` (不满足 Triton 条件：`next_power_of_two(1500) = 2048 > 256`)
    *   **结果：** Encoder attention 在当前输入下会回退到 `torch path`。

2.  **Decoder Attention：**
    *   `head_dim = 3584 / 28 = 128` (满足 Triton 条件)
    *   `seq_k` 在 prefill 阶段大约为 `375` (由 `mel_frames // 2 // 4` 决定，例如 `3000 // 2 // 4 = 375`)
    *   `seq_k` 在后续生成阶段约为 `389` 到 `401`。
    *   `next_power_of_two(375)` 或 `next_power_of_two(389)` 都等于 `512` (不满足 Triton 条件：`512 > 256`)
    *   **结果：** Decoder 在 prefill 和后续生成阶段也都会走 `torch path`。