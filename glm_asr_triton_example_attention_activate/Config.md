## Config：线性层 cuBLAS + 其他部分按条件混合 Triton


```
layers.Linear.BACKEND = "cublas"
layers.MLP.FUSED = True
layers.EncoderMLP.FUSED = False   # 但当前模型没有用

Mixed execution:
- Linear: PyTorch matmul path -> CUDA上通常落到 cuBLAS/cuBLASLt
- RMSNorm / LayerNorm: Triton when hidden_size is power-of-two and x.is_cuda
- gelu / silu: Triton when x.is_cuda
- Attention Triton path:
  short seq:
    q.is_cuda
    and next_power_of_two(seq_k) <= 256
    and next_power_of_two(head_dim) <= 256
    kernels:
      attention_scores_kernel
      softmax_inplace_kernel
      attention_output_kernel
  long seq:
    q.is_cuda
    and next_power_of_two(head_dim) <= 256
    and seq_k > 256
    implementation:
      chunk K/V by <= 256 tokens
      attention_scores_kernel per chunk
      attention_output_kernel per chunk
      online softmax accumulation across chunks
- Attention fallback path:
  scores = torch.einsum("bnqd,bnkd->bnqk", q, k) * scale
  output = torch.einsum("bnqk,bnkd->bnqd", attn_weights, v)
- Conv: Triton only when shape constraints are satisfied

```


## 当前 attention path（修改后，3.5s audio）

**结论：** 在 `glm_asr_triton_example_attention_activate` 里，attention 现在分成两类 Triton path：

1. `triton full path`
   - 条件：`q.is_cuda` AND `next_power_of_two(seq_k) <= 256` AND `next_power_of_two(head_dim) <= 256`
   - 适用于短序列，沿用原来的三段式 Triton attention

2. `triton chunked path`
   - 条件：`q.is_cuda` AND `next_power_of_two(head_dim) <= 256` AND `seq_k > 256`
   - 做法：把 K/V 按最多 256 个 token 分块，每个块仍使用 Triton kernel 计算 `QK^T` 和 `P @ V`，块间用 online softmax 在 Python 侧做稳定累积
   - 作用：让长序列 attention 不再因为 `seq_k > 256` 直接回退到 `torch path`

### 对 3.5s 输入的影响

1. **Encoder Attention**
   - `head_dim = 1280 / 20 = 64`
   - `seq_k ≈ 1500`
   - 原来：`next_power_of_two(1500) = 2048 > 256`，只能走 `torch path`
   - 现在：会走 `triton chunked path`

2. **Decoder Attention**
   - `head_dim = 3584 / 28 = 128`
   - prefill 阶段 `seq_k ≈ 375`
   - 后续生成阶段 `seq_k` 继续增长
   - 原来：`next_power_of_two(375) = 512 > 256`，prefill 和后续阶段都回退到 `torch path`
   - 现在：prefill 和后续长序列阶段都可以走 `triton chunked path`

### 结果总结

- 3.5s 音频下，encoder attention 不再卡在 `torch path`
- decoder prefill 也不再卡在 `torch path`
- attention 是否能走 Triton，主要只剩 `head_dim` 是否满足 `<= 256` 这个限制
