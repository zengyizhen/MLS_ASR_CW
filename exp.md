# 明确当前baseline是什么
Torch 模型骨架 + 少量 Triton kernel + baseline 配置，不是全Triton。

glm_asr_triton_example/__init__.py 里直接把 Linear.BACKEND = "cublas"，同时关闭了 MLP.FUSED 和 EncoderMLP.FUSED。这说明 baseline 的线性层默认还是走 Torch 的 @/matmul 路径，底下大多会落到 CUDA 库，比如 cuBLAS/cuBLASLt，而不是强制走自写 Triton GEMM。

attention.py 只在 seq_k/head_dim <= 256 时走三段式 Triton kernel；一旦尺寸大，就退回 torch.einsum 先算 QK^T，再 softmax，再乘 V，而且会显式 materialize 整个 attention matrix。

benchmark.sh 是整体延迟/正确率评测
时间是当前实现的实际运行时间，只是不包含模型加载、读音频、processor 预处理这些开销。
## evaluation
- benchmark.sh
会显式把 example 配置改成 
Linear.BACKEND = 'cublas'
MLP.FUSED = False
EncoderMLP.FUSED = False

- benchmark_detailed.sh 是 benchmark_detailed.py 里的组件级剖析
但 benchmark_detailed.py 没做这一步，直接导入模块后会落到 layers.py 里的默认 Linear.BACKEND = "torch"