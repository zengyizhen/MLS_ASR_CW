# 当前 Attention Path 说明（中文版）

## 1. 总体分流逻辑

当前 `glm_asr_triton_example_FP16` 中的 attention 实现在 [attention.py](/Users/olivia/VScode/MLS_ASR/glm_asr_triton_example_FP16/attention.py:328)。

`scaled_dot_product_attention()` 目前有三层分流：

### Path A：小尺寸 Triton attention

触发条件：

- `q.is_cuda`
- `next_power_of_two(seq_k) <= 256`
- `next_power_of_two(head_dim) <= 256`

满足后会走：

- Triton `attention_scores_kernel`
- Triton `softmax_inplace_kernel`
- Triton `attention_output_kernel`

这条路径适合小序列、小 head_dim 场景。

### Path B：PyTorch SDPA path

当不满足 Path A 时，当前实现会优先走：

- `torch.nn.functional.scaled_dot_product_attention(...)`

这是现在大序列 attention 的主路径。

在 CUDA 上，这条路径理论上可能进一步由 PyTorch 分派到：

- Flash Attention backend
- memory-efficient attention backend
- math backend

这一步由 PyTorch 和当前 GPU/runtime 决定，不是我们在 Python 层硬编码指定的。

### Path C：最后保底的 manual fallback

如果运行环境里没有 `scaled_dot_product_attention`，才会退回到最原始的：

1. `torch.einsum("bnqd,bnkd->bnqk", q, k)`
2. softmax
3. `torch.einsum("bnqk,bnkd->bnqd", attn_weights, v)`

在当前 PyTorch 版本下，这通常只是保底逻辑，不应是主路径。

## 2. 当前 encoder attention path

Audio encoder 的 attention 在 [model.py](/Users/olivia/VScode/MLS_ASR/glm_asr_triton_example_FP16/model.py:85) 到 [model.py](/Users/olivia/VScode/MLS_ASR/glm_asr_triton_example_FP16/model.py:117)。

关键维度：

- `audio_hidden_size = 1280`
- `audio_num_heads = 20`
- 所以 `head_dim = 64`

虽然 `head_dim = 64` 本身满足小尺寸 Triton 条件，但 encoder 真正卡住的是 `seq_k`：

- 音频经过 conv subsample 后，序列长度通常还在千级
- `next_power_of_two(seq_k)` 明显大于 `256`

所以：

- **当前 encoder attention 实际上不会走小尺寸 Triton attention**
- **当前 encoder attention 主路径是 PyTorch SDPA path**

## 3. 当前 decoder attention path

Text decoder 的 attention 在 [model.py](/Users/olivia/VScode/MLS_ASR/glm_asr_triton_example_FP16/model.py:208) 到 [model.py](/Users/olivia/VScode/MLS_ASR/glm_asr_triton_example_FP16/model.py:311)。

关键维度：

- `text_hidden_size = 3584`
- `text_num_heads = 28`
- 所以 `head_dim = 128`
- `text_num_kv_heads = 4`
- 使用 GQA，KV 会通过 `_expand_kv()` 逻辑扩展到 query heads 对齐

`head_dim = 128` 仍然满足小尺寸 Triton 条件，但 decoder 热路径同样卡在 `seq_k`：

- prefill 阶段已经是几百长度
- decode 阶段每生成一个 token，KV 长度都会继续增长
- `next_power_of_two(seq_k)` 很快大于 `256`

因此：

- **当前 decoder attention 主路径也不是小尺寸 Triton**
- **主路径同样是 PyTorch SDPA path**

这也是为什么你的 profile 里 decode 占比极高时，优化 attention fallback 的收益会比优化 norm/activation 更直接。

## 4. 当前 attention path 对性能的意义

### 4.1 为什么以前慢

在改动前，大序列场景会落到手写：

- `einsum(Q, K)`
- softmax
- `einsum(attn, V)`

问题在于：

- 显式 materialize 完整 score matrix
- 显存和带宽压力大
- 无法自动获得 Flash / memory-efficient attention 的优化

### 4.2 为什么现在更合理

改动后，大序列 attention 现在优先走 SDPA path：

- Python 侧逻辑更轻
- 更容易获得 CUDA 优化 attention backend
- 更符合当前 decoder 占主导的 profile 结构

## 5. 当前 attention path 的精度策略

### 小尺寸 Triton attention

- `q/k/v` buffer 保持 activation dtype，通常是 FP16
- `scores` 仍然使用 FP32
- softmax 也在高精度下完成

### 大尺寸 SDPA attention

- 输入 `q/k/v` dtype 跟随当前 activation dtype
- mask 会规范化成 SDPA 可接受的格式
- 精确的内部计算精度由 PyTorch backend 决定

所以当前 attention 的策略不是“全程纯 FP16”，而是：

- **存储和输入尽量保持低精度**
- **对数值敏感的部分仍保留更高精度**

这也是更符合实际部署的 mixed precision 方案。

## 6. 当前 attention path 的限制

虽然现在 path 比之前更好，但仍有几个限制没有解除：

- 小尺寸 Triton attention 只覆盖 `seq_k <= 256` 的场景，覆盖面有限
- 大尺寸 attention 的最终性能高度依赖 PyTorch SDPA dispatch
- decoder 生成过程仍然是 full-prefix decode
- 即使 attention backend 更好了，full-prefix 结构本身仍会持续放大 decode latency

## 7. 结论

当前 attention path 可以概括为：

- 小尺寸场景：走 Triton 三段式 attention kernel
- 大尺寸真实场景：encoder 和 decoder 主要都走 PyTorch SDPA
- 极端保底：才回退到手写 `einsum + softmax + einsum`

结合你现在的 profile，可以直接得出判断：

- **当前 attention path 的真正热点是 decoder 的大序列 SDPA 路径**
- **如果后续继续优化，第一优先级仍然不是再打磨小尺寸 Triton attention，而是改增量解码 / KV-cache**
