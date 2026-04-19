# GLM-ASR Triton FP16 混合精度范围笔记（中文版）

## 1. 瓶颈结论

根据当前 profile：

- Audio Encoder: `204.09 ms`，占比 `8.0%`
- Multi-modal Projector: `1.66 ms`，占比 `0.1%`
- Decoder Prefill: `52.54 ms`，占比 `2.1%`
- Decoder（50 decode steps）: `2290.79 ms`，占比 `89.9%`

当前绝对主瓶颈仍然是 **decoder 的 decode 阶段 attention**，而不是 norm 或 activation。

原因很明确：真实 workload 下，attention 基本没有跑在小尺寸 Triton path 上。

- encoder attention：
  - `head_dim = 64`
  - 卷积降采样后的 `seq_k` 大约在 `1500`
  - `next_power_of_two(seq_k) > 256`
  - 结论：不会走小尺寸 Triton attention，落到 fallback path
- decoder attention：
  - `head_dim = 128`
  - prefill 阶段 `seq_k` 已经明显大于 `256`
  - decode 过程中 `seq_k` 还会继续增长
  - 结论：热路径同样主要走 fallback path

也就是说，当前 profile 中占比最高的部分，本质上不是 Triton attention 在主导，而是大序列 attention fallback 在主导。

## 2. 修改前的混合精度覆盖情况

### 2.1 已经算是 FP16 的部分

- 权重加载时已经使用 `torch.float16`
- 标准 `Linear` 默认走 `Linear.BACKEND = "cublas"`，因此主线性层已经能吃到 FP16 权重和 FP16 激活
- Embedding 权重和输出基本跟随权重 dtype
- Conv 路径大体上也会跟随 weight dtype

### 2.2 仍然没有真正把混合精度用透的部分

- 大序列 attention fallback 仍然是：
  1. `torch.einsum("bnqd,bnkd->bnqk", q, k)`
  2. 显式 softmax
  3. `torch.einsum("bnqk,bnkd->bnqd", attn_weights, v)`
- 小尺寸 Triton attention path 在进入 kernel 之前，会把 `q/k/v` 先整体转成 `float32`
- Triton `RMSNorm` / `LayerNorm` 在 launch 前会生成整块 FP32 输入 buffer
- Triton `gelu` / `silu` 在 launch 前会生成整块 FP32 flattened buffer
- Triton `softmax` 在 wrapper 层会先把输入整体转成 FP32
- fused decoder `SwiGLU` 路径会把转置权重、padding buffer、intermediate tensor 都扩成 FP32
- fused encoder `Linear + GELU` 路径也存在相同问题

所以修改前的状态更接近：

- “权重大多是 FP16”
- “cuBLAS matmul 能用 FP16”
- 但是很多 Triton 侧中间张量仍然在显式扩成 FP32
- 而且最热的大序列 attention fallback 仍然停留在手写 `einsum` 路径

## 3. 这次修改做了什么

### 3.1 大序列 attention fallback 改为 SDPA

文件：[attention.py](/Users/olivia/VScode/MLS_ASR/glm_asr_triton_example_FP16/attention.py:264)

对于不满足小尺寸 Triton 条件的 attention，现在优先走：

- `torch.nn.functional.scaled_dot_product_attention(...)`

这意味着在 CUDA 上，PyTorch 可以根据运行环境选择：

- Flash Attention backend
- memory-efficient attention backend
- 或优化过的 math backend

最大的收益点在于：

- 不再由 Python 侧显式 materialize 完整 `QK^T` score matrix
- 不再固定使用两次 `einsum` + 一次显式 softmax
- 更有机会吃到 CUDA 原生优化 attention 实现

这部分是这次改动里最关键的，因为它直接命中当前 `89.9%` 占比的主瓶颈。

### 3.2 小尺寸 Triton attention path 改为“FP16 存储 + 高精度归约”

文件：[attention.py](/Users/olivia/VScode/MLS_ASR/glm_asr_triton_example_FP16/attention.py:328)

现在的小尺寸 Triton attention path 不再提前把这些张量整体转成 FP32：

- `q_flat`
- `k_flat`
- `v_flat`
- padding 后的 `q/k/v`
- 最终 output buffer

现在的策略是：

- buffer / storage dtype 跟随 `q.dtype`
- 仍然保留 `scores` 为 FP32
- softmax 和 reduction 仍然在更高精度上完成

这更符合混合精度的常见做法：

- 存储和带宽走 FP16
- 对数值敏感的 reduction / normalization 保留 FP32

### 3.3 Norm / Activation / Softmax 改成“外部不扩 FP32，内部再升精度”

文件：[layers.py](/Users/olivia/VScode/MLS_ASR/glm_asr_triton_example_FP16/layers.py:477)

这次收紧了以下算子的 dtype 扩展：

- `RMSNorm`
- `LayerNorm`
- `gelu`
- `silu`
- `softmax`

修改后的行为：

- wrapper 层不再预先把整个输入 tensor 扩成 FP32
- 输入输出保持原始 activation dtype
- kernel 内部在 `tl.load(...)` 之后再转成 `tl.float32` 做归约或指数计算

这样做的好处是：

- 减少整块 FP32 staging buffer
- 降低带宽与显存压力
- 同时保留归一化和非线性计算的稳定性

### 3.4 fused MLP 路径改成真正的混合精度 staging

文件：[layers.py](/Users/olivia/VScode/MLS_ASR/glm_asr_triton_example_FP16/layers.py:817)

这次修改了两条 fused path：

- decoder 侧 `MLP._forward_fused()`，即 fused `SwiGLU`
- encoder 侧 `EncoderMLP._forward_fused()`，即 fused `Linear + GELU`

修改前：

- 转置权重强制转 FP32
- 输入 padding buffer 是 FP32
- intermediate buffer 是 FP32

修改后：

- 转置权重保持原始 weight dtype
- 输入 / padding / intermediate 跟随 fused weight dtype
- `tl.dot(...)` 仍然在 kernel 内使用 FP32 accumulator

这意味着 fused MLP 不再只是“权重是 FP16、但中间几乎全扩回 FP32”，而是更接近真正的混合精度执行。

## 4. 修改后的混合精度覆盖范围

### 4.1 现在已经更完整支持混合精度的部分

- 标准 cuBLAS 线性层：`FP16` 输入 + `FP16` 权重
- 大序列 attention fallback：改为 SDPA-backed path
- 小尺寸 Triton attention：`FP16` Q/K/V storage + FP32 score math
- `RMSNorm` / `LayerNorm`：FP16 activation storage + FP32 reduction
- `GELU` / `SiLU` / `softmax`：FP16 activation storage + FP32 内部计算
- fused decoder `SwiGLU`：FP16 activation / weight / buffer + FP32 accumulation
- fused encoder `Linear + GELU`：同样采用 mixed precision staging

### 4.2 仍然没有完全解决的部分

- Triton attention kernel 本身仍然保留了 FP32 的 score/softmax 计算
- 最终是否走到 Flash / memory-efficient backend，仍依赖 PyTorch 在目标 GPU 上的 SDPA dispatch
- RoPE 的应用仍然是普通 Torch tensor 运算，不是 fused Triton kernel
- Conv 仍然是 `pad + im2col + matmul`，不是 fully fused conv pipeline
- `generate()` 仍然是 full-prefix decode，没有改成 KV-cache 增量解码

其中最关键的一点仍然是最后一条：

- **即使单个 attention / MLP / norm 都继续优化，如果 `generate()` 仍然每步重算整个 prefix，decoder latency 还是会被结构性放大**

## 5. 实际预期

这次改动的现实意义主要有三点：

1. 把 hottest path 的 attention fallback 从手写 `einsum` 换成更现代的 SDPA 路径
2. 把很多“名义上 FP16、实际上 wrapper 里先扩回 FP32”的算子收紧成真正 mixed precision
3. 让 fused decoder MLP 也能更完整地吃到 FP16 的存储与带宽收益

所以这次不是单纯“又调了几个 dtype”，而是把混合精度真正往 decoder 热路径推进了一步。

## 6. 下一步优先级建议

如果继续追端到端延迟，推荐顺序仍然是：

1. 改 `generate()`，真正使用 KV-cache 增量解码
2. 在目标 GPU 上确认 decoder attention 是否已经稳定走到 Flash / memory-efficient SDPA
3. 再考虑继续做更细的 kernel fusion、tile size、block size 调优

原因是：

- 现在最大的收益空间仍然来自“减少重复计算”
- 而不是继续对 full-prefix decode 里的单个算子做边际优化
