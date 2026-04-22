"""
Triton Multi-Head Attention Implementation
End-to-end implementation using Triton kernels
"""

import numpy as np
import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


def get_stream():
    """Get current CUDA stream pointer."""
    if torch.cuda.is_available():
        return torch.cuda.current_stream().cuda_stream
    return None


def _storage_dtype(reference: torch.Tensor, fallback: torch.dtype = torch.float16) -> torch.dtype:
    """Choose the storage dtype for staged GPU buffers."""
    # Q/K/V 缓存按模型 dtype 存储；attention 分数等数值敏感部分在 kernel 内升 FP32。
    if reference.is_floating_point():
        return reference.dtype
    return fallback


# ============================================================================
# Triton Kernels for Attention
# ============================================================================

@triton.jit
def attention_scores_kernel(
    q_ptr,
    k_ptr,
    scores_ptr,
    scale,
    seq_k,
    head_dim,
    stride_q0,
    stride_q1,
    stride_q2,
    stride_k0,
    stride_k1,
    stride_k2,
    stride_s0,
    stride_s1,
    stride_s2,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Compute scaled attention scores for a single query position.
    Grid: (batch_heads, seq_q)
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)

    q = tl.load(
        q_ptr + pid_bh * stride_q0 + pid_q * stride_q1 + offs_d * stride_q2,
        mask=offs_d < head_dim,
        other=0.0,
    ).to(tl.float32)
    k = tl.load(
        k_ptr
        + pid_bh * stride_k0
        + offs_k[:, None] * stride_k1
        + offs_d[None, :] * stride_k2,
        mask=(offs_k[:, None] < seq_k) & (offs_d[None, :] < head_dim),
        other=0.0,
    ).to(tl.float32)
    scores = tl.sum(k * q[None, :], axis=1) * scale
    tl.store(
        scores_ptr
        + pid_bh * stride_s0
        + pid_q * stride_s1
        + offs_k * stride_s2,
        scores,
        mask=offs_k < seq_k,
    )


@triton.jit
def softmax_inplace_kernel(scores_ptr, stride_s, seq_k, BLOCK_SIZE: tl.constexpr):
    """
    Apply softmax along the last dimension (seq_k).
    Grid: (batch_heads * seq_q,)
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < seq_k

    s = tl.load(scores_ptr + row * stride_s + offs, mask=mask, other=-float("inf"))
    s = s - tl.max(s, axis=0)
    exp_s = tl.exp(s)
    denom = tl.sum(exp_s, axis=0)
    out = exp_s / denom

    tl.store(scores_ptr + row * stride_s + offs, out, mask=mask)


@triton.jit
def attention_output_kernel(
    attn_ptr,
    v_ptr,
    output_ptr,
    seq_k,
    head_dim,
    stride_w0,
    stride_w1,
    stride_w2,
    stride_v0,
    stride_v1,
    stride_v2,
    stride_o0,
    stride_o1,
    stride_o2,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Compute attention output: attn_weights @ V
    Grid: (batch_heads, seq_q)
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)

    w = tl.load(
        attn_ptr
        + pid_bh * stride_w0
        + pid_q * stride_w1
        + offs_k * stride_w2,
        mask=offs_k < seq_k,
        other=0.0,
    ).to(tl.float32)
    v = tl.load(
        v_ptr
        + pid_bh * stride_v0
        + offs_k[:, None] * stride_v1
        + offs_d[None, :] * stride_v2,
        mask=(offs_k[:, None] < seq_k) & (offs_d[None, :] < head_dim),
        other=0.0,
    ).to(tl.float32)
    out = tl.sum(v * w[:, None], axis=0)
    tl.store(
        output_ptr
        + pid_bh * stride_o0
        + pid_q * stride_o1
        + offs_d * stride_o2,
        out,
        mask=offs_d < head_dim,
    )


@triton.jit
def causal_mask_kernel(
    scores_ptr,
    seq_k,
    offset,
    stride_s0,
    stride_s1,
    stride_s2,
    BLOCK_K: tl.constexpr,
):
    """
    Apply causal mask to attention scores.
    Grid: (batch_heads, seq_q)
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)
    mask = offs_k < seq_k
    scores = tl.load(
        scores_ptr
        + pid_bh * stride_s0
        + pid_q * stride_s1
        + offs_k * stride_s2,
        mask=mask,
        other=-1e9,
    )
    current_pos = pid_q + offset
    scores = tl.where(offs_k > current_pos, -1e9, scores)
    tl.store(
        scores_ptr
        + pid_bh * stride_s0
        + pid_q * stride_s1
        + offs_k * stride_s2,
        scores,
        mask=mask,
    )


@triton.jit
def flash_attention_v2_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    scale,
    seq_len,
    head_dim,
    num_heads,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_os,
    stride_od,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    FlashAttention-style fused forward kernel for encoder self-attention.
    Grid: (ceil(seq_len / BLOCK_M), batch * heads)
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // num_heads
    pid_h = pid_bh - pid_b * num_heads

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q = tl.load(
        q_ptr
        + pid_b * stride_qb
        + pid_h * stride_qh
        + offs_m[:, None] * stride_qs
        + offs_d[None, :] * stride_qd,
        mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim),
        other=0.0,
    )

    # online softmax 状态：m/l/acc 全程 FP32，避免显式写出完整 attention scores。
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for start_n in tl.range(0, seq_len, BLOCK_N):
        k_offsets = start_n + offs_n
        k = tl.load(
            k_ptr
            + pid_b * stride_kb
            + pid_h * stride_kh
            + k_offsets[:, None] * stride_ks
            + offs_d[None, :] * stride_kd,
            mask=(k_offsets[:, None] < seq_len) & (offs_d[None, :] < head_dim),
            other=0.0,
        )
        scores = tl.dot(q, tl.trans(k)) * scale
        scores = tl.where(k_offsets[None, :] < seq_len, scores, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        p = tl.exp(scores - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v = tl.load(
            v_ptr
            + pid_b * stride_vb
            + pid_h * stride_vh
            + k_offsets[:, None] * stride_vs
            + offs_d[None, :] * stride_vd,
            mask=(k_offsets[:, None] < seq_len) & (offs_d[None, :] < head_dim),
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    out = acc / l_i[:, None]
    tl.store(
        out_ptr
        + pid_b * stride_ob
        + pid_h * stride_oh
        + offs_m[:, None] * stride_os
        + offs_d[None, :] * stride_od,
        out,
        mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim),
    )


@triton.jit
def flash_decode_gqa_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    scale,
    seq_k,
    head_dim,
    queries_per_kv,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_os,
    stride_od,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Flash-Decoding style GQA kernel for q_len=1 decoder steps.
    Grid: (batch, query_heads)
    """
    pid_b = tl.program_id(0)
    pid_qh = tl.program_id(1)
    pid_kvh = pid_qh // queries_per_kv

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q = tl.load(
        q_ptr
        + pid_b * stride_qb
        + pid_qh * stride_qh
        + offs_d * stride_qd,
        mask=offs_d < head_dim,
        other=0.0,
    )

    # q_len=1 的 decoder attention 不写出 scores，直接在线更新 softmax。
    m_i = tl.full((), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((), dtype=tl.float32)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for start_n in tl.range(0, seq_k, BLOCK_N):
        k_offsets = start_n + offs_n
        k = tl.load(
            k_ptr
            + pid_b * stride_kb
            + pid_kvh * stride_kh
            + k_offsets[:, None] * stride_ks
            + offs_d[None, :] * stride_kd,
            mask=(k_offsets[:, None] < seq_k) & (offs_d[None, :] < head_dim),
            other=0.0,
        )
        scores = tl.sum(k.to(tl.float32) * q[None, :].to(tl.float32), axis=1) * scale
        scores = tl.where(k_offsets < seq_k, scores, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        p = tl.exp(scores - m_new)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)

        v = tl.load(
            v_ptr
            + pid_b * stride_vb
            + pid_kvh * stride_vh
            + k_offsets[:, None] * stride_vs
            + offs_d[None, :] * stride_vd,
            mask=(k_offsets[:, None] < seq_k) & (offs_d[None, :] < head_dim),
            other=0.0,
        )
        acc = acc * alpha + tl.sum(p[:, None] * v.to(tl.float32), axis=0)
        m_i = m_new

    out = acc / l_i
    tl.store(
        out_ptr
        + pid_b * stride_ob
        + pid_qh * stride_oh
        + offs_d * stride_od,
        out,
        mask=offs_d < head_dim,
    )


# ============================================================================
# Attention Classes
# ============================================================================

class MultiHeadAttention:
    """Multi-head attention using Triton kernels."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.scale = 1.0 / np.sqrt(self.head_dim)

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Compute multi-head attention.

        Args:
            q: Query (batch, num_heads, seq_q, head_dim)
            k: Key (batch, num_kv_heads, seq_k, head_dim)
            v: Value (batch, num_kv_heads, seq_k, head_dim)
            attention_mask: Optional mask (batch, 1, seq_q, seq_k)
            is_causal: Whether to apply causal masking

        Returns:
            Output (batch, num_heads, seq_q, head_dim)
        """
        batch, num_heads, seq_q, head_dim = q.shape
        _, num_kv_heads, seq_k, _ = k.shape

        if _can_use_flash_decode_gqa(q, k, v, attention_mask):
            # Decoder q_len=1 + GQA：直接用 query head 映射到 KV head，
            # 避免把 4 个 KV heads 物理 expand 成 28 个 query heads。
            return _flash_decode_gqa_triton(q, k, v, self.scale)

        if num_kv_heads != num_heads:
            k = self._expand_kv(k, self.num_queries_per_kv)
            v = self._expand_kv(v, self.num_queries_per_kv)

        return scaled_dot_product_attention(
            q, k, v, attention_mask, is_causal, self.scale
        )

    def _expand_kv(self, x: torch.Tensor, num_repeats: int) -> torch.Tensor:
        """Expand KV heads for GQA using broadcast (zero-copy)."""
        batch, num_kv_heads, seq_len, head_dim = x.shape
        x_expanded = x[:, :, None, :, :].expand(
            batch, num_kv_heads, num_repeats, seq_len, head_dim
        )
        return x_expanded.reshape(batch, num_kv_heads * num_repeats, seq_len, head_dim)


def next_power_of_two(x: int) -> int:
    """Return the smallest power of two >= x."""
    return 1 << (x - 1).bit_length() if x > 0 else 1


MAX_ATTENTION_DIM = 256
FLASH_ATTN_MAX_HEAD_DIM = 128
FLASH_ATTN_BLOCK_M = 16
FLASH_ATTN_BLOCK_N = 64
FLASH_DECODE_BLOCK_N = 64


def _can_use_flash_attention_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    is_causal: bool,
) -> bool:
    """Check the conservative encoder-only FlashAttention v2 Triton path."""
    if not q.is_cuda:
        return False
    if attention_mask is not None or is_causal:
        return False
    if q.dtype not in (torch.float16, torch.bfloat16):
        return False
    if q.dtype != k.dtype or q.dtype != v.dtype:
        return False

    batch, num_heads, seq_q, head_dim = q.shape
    k_batch, k_heads, seq_k, k_head_dim = k.shape
    v_batch, v_heads, seq_v, v_head_dim = v.shape

    # 当前最方便、安全的接入点是 encoder self-attention：
    # q/k/v 头数一致、q_len == kv_len、无 causal mask；decoder KV cache 不会命中。
    return (
        batch == k_batch == v_batch
        and num_heads == k_heads == v_heads
        and seq_q == seq_k == seq_v
        and head_dim == k_head_dim == v_head_dim
        and next_power_of_two(head_dim) <= FLASH_ATTN_MAX_HEAD_DIM
    )


def _flash_attention_v2_triton_encoder(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Run the local Triton FlashAttention v2-style kernel for encoder attention."""
    batch, num_heads, seq_len, head_dim = q.shape
    block_d = next_power_of_two(head_dim)
    q_contig = q.contiguous()
    k_contig = k.contiguous()
    v_contig = v.contiguous()
    output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    grid = (triton.cdiv(seq_len, FLASH_ATTN_BLOCK_M), batch * num_heads)

    flash_attention_v2_fwd_kernel[grid](
        q_contig,
        k_contig,
        v_contig,
        output,
        float(scale),
        seq_len,
        head_dim,
        num_heads,
        q_contig.stride(0),
        q_contig.stride(1),
        q_contig.stride(2),
        q_contig.stride(3),
        k_contig.stride(0),
        k_contig.stride(1),
        k_contig.stride(2),
        k_contig.stride(3),
        v_contig.stride(0),
        v_contig.stride(1),
        v_contig.stride(2),
        v_contig.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        BLOCK_M=FLASH_ATTN_BLOCK_M,
        BLOCK_N=FLASH_ATTN_BLOCK_N,
        BLOCK_D=block_d,
    )
    return output


def _can_use_flash_decode_gqa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> bool:
    """Check whether this is a q_len=1 decoder GQA step."""
    if not q.is_cuda:
        return False
    if attention_mask is not None:
        return False
    if q.dtype not in (torch.float16, torch.bfloat16):
        return False
    if q.dtype != k.dtype or q.dtype != v.dtype:
        return False

    batch, q_heads, seq_q, head_dim = q.shape
    k_batch, kv_heads, seq_k, k_head_dim = k.shape
    v_batch, v_heads, seq_v, v_head_dim = v.shape

    # Decoder decode with KV cache: q_len=1, kv_len grows, num_q_heads > num_kv_heads.
    return (
        batch == k_batch == v_batch
        and seq_q == 1
        and seq_k == seq_v
        and kv_heads == v_heads
        and q_heads % kv_heads == 0
        and head_dim == k_head_dim == v_head_dim
        and next_power_of_two(head_dim) <= FLASH_ATTN_MAX_HEAD_DIM
    )


def _flash_decode_gqa_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Run q_len=1 Flash-Decoding style GQA attention without expanding KV heads."""
    batch, q_heads, _, head_dim = q.shape
    _, kv_heads, seq_k, _ = k.shape
    queries_per_kv = q_heads // kv_heads
    block_d = next_power_of_two(head_dim)

    q_contig = q.contiguous()
    k_contig = k.contiguous()
    v_contig = v.contiguous()
    output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    grid = (batch, q_heads)

    flash_decode_gqa_fwd_kernel[grid](
        q_contig,
        k_contig,
        v_contig,
        output,
        float(scale),
        seq_k,
        head_dim,
        queries_per_kv,
        q_contig.stride(0),
        q_contig.stride(1),
        q_contig.stride(2),
        q_contig.stride(3),
        k_contig.stride(0),
        k_contig.stride(1),
        k_contig.stride(2),
        k_contig.stride(3),
        v_contig.stride(0),
        v_contig.stride(1),
        v_contig.stride(2),
        v_contig.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        BLOCK_N=FLASH_DECODE_BLOCK_N,
        BLOCK_D=block_d,
    )
    return output


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Scaled dot-product attention using Triton kernels.
    """
    batch, num_heads, seq_q, head_dim = q.shape
    _, _, seq_k, _ = k.shape

    if scale is None:
        scale = 1.0 / np.sqrt(head_dim)

    if q.dtype != k.dtype or q.dtype != v.dtype:
        common_dtype = q.dtype
        if common_dtype != k.dtype:
            k = k.to(common_dtype)
        if common_dtype != v.dtype:
            v = v.to(common_dtype)

    if _can_use_flash_attention_v2(q, k, v, attention_mask, is_causal):
        # 最小改动策略：只替换当前最容易收益的 encoder self-attention。
        # 不支持的 mask / causal / decoder KV cache 场景继续走下面原有实现。
        return _flash_attention_v2_triton_encoder(q, k, v, scale)

    seq_k_padded = next_power_of_two(seq_k)
    head_dim_padded = next_power_of_two(head_dim)

    use_triton = (
        q.is_cuda
        and seq_k_padded <= MAX_ATTENTION_DIM
        and head_dim_padded <= MAX_ATTENTION_DIM
    )

    if use_triton:
        # 半精度保存 Q/K/V 和输出缓存，score/softmax 在 FP32 缓冲区中计算。
        storage_dtype = _storage_dtype(q)
        q_flat = q.reshape(batch * num_heads, seq_q, head_dim).contiguous().to(storage_dtype)
        k_flat = k.reshape(batch * num_heads, seq_k, head_dim).contiguous().to(storage_dtype)
        v_flat = v.reshape(batch * num_heads, seq_k, head_dim).contiguous().to(storage_dtype)

        if seq_k_padded != seq_k or head_dim_padded != head_dim:
            k_padded = torch.zeros(
                (batch * num_heads, seq_k_padded, head_dim_padded),
                dtype=storage_dtype,
                device=q.device,
            )
            v_padded = torch.zeros_like(k_padded)
            q_padded = torch.zeros(
                (batch * num_heads, seq_q, head_dim_padded),
                dtype=storage_dtype,
                device=q.device,
            )
            k_padded[:, :seq_k, :head_dim] = k_flat
            v_padded[:, :seq_k, :head_dim] = v_flat
            q_padded[:, :, :head_dim] = q_flat
            k_flat = k_padded
            v_flat = v_padded
            q_flat = q_padded

        scores = torch.empty(
            (batch * num_heads, seq_q, seq_k_padded),
            dtype=torch.float32,
            device=q.device,
        )
        output = torch.empty(
            (batch * num_heads, seq_q, head_dim_padded),
            dtype=storage_dtype,
            device=q.device,
        )

        grid = (batch * num_heads, seq_q)
        attention_scores_kernel[grid](
            q_flat,
            k_flat,
            scores,
            float(scale),
            seq_k_padded,
            head_dim_padded,
            q_flat.stride(0),
            q_flat.stride(1),
            q_flat.stride(2),
            k_flat.stride(0),
            k_flat.stride(1),
            k_flat.stride(2),
            scores.stride(0),
            scores.stride(1),
            scores.stride(2),
            BLOCK_K=seq_k_padded,
            BLOCK_D=head_dim_padded,
        )

        if seq_k_padded != seq_k:
            scores[:, :, seq_k:] = -1e9

        if is_causal:
            mask = torch.triu(
                torch.ones((seq_q, seq_k_padded), dtype=torch.float32, device=q.device),
                diagonal=1,
            ) * -1e9
            scores = scores + mask[None, :, :]

        if attention_mask is not None:
            if attention_mask.ndim == 4:
                attention_mask = attention_mask.reshape(
                    batch * num_heads, seq_q, seq_k
                )
            if seq_k_padded != seq_k:
                mask_padded = torch.zeros(
                    (batch * num_heads, seq_q, seq_k_padded),
                    dtype=torch.float32,
                    device=q.device,
                )
                mask_padded[:, :, :seq_k] = attention_mask
                mask_padded[:, :, seq_k:] = -1e9
                attention_mask = mask_padded
            scores = scores + attention_mask

        scores_2d = scores.reshape(batch * num_heads * seq_q, seq_k_padded)
        block = seq_k_padded
        softmax_inplace_kernel[(scores_2d.shape[0],)](
            scores_2d, scores_2d.stride(0), seq_k_padded, BLOCK_SIZE=block
        )
        scores = scores_2d.reshape(batch * num_heads, seq_q, seq_k_padded)

        attention_output_kernel[grid](
            scores,
            v_flat,
            output,
            seq_k_padded,
            head_dim_padded,
            scores.stride(0),
            scores.stride(1),
            scores.stride(2),
            v_flat.stride(0),
            v_flat.stride(1),
            v_flat.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            BLOCK_K=seq_k_padded,
            BLOCK_D=head_dim_padded,
        )

        if head_dim_padded != head_dim:
            output = output[:, :, :head_dim]

        return output.reshape(batch, num_heads, seq_q, head_dim).to(q.dtype)

    scores = torch.einsum("bnqd,bnkd->bnqk", q, k) * scale

    if is_causal:
        mask = torch.triu(
            torch.ones((seq_q, seq_k), dtype=torch.float32, device=q.device),
            diagonal=1,
        ) * -1e9
        scores = scores + mask[None, None, :, :]

    if attention_mask is not None:
        scores = scores + attention_mask

    scores = scores - torch.max(scores, dim=-1, keepdim=True).values
    attn_weights = torch.exp(scores)
    attn_weights = attn_weights / torch.sum(attn_weights, dim=-1, keepdim=True)
    output = torch.einsum("bnqk,bnkd->bnqd", attn_weights, v)

    return output.to(q.dtype)


if __name__ == "__main__":
    print("Testing Triton Attention...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    num_heads = 4
    seq_len = 16
    head_dim = 64

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    print("\nBasic attention:")
    output = scaled_dot_product_attention(q, k, v)
    print(f"  Output shape: {output.shape}")

    print("\nCausal attention:")
    output_causal = scaled_dot_product_attention(q, k, v, is_causal=True)
    print(f"  Output shape: {output_causal.shape}")

    print("\nWith attention mask:")
    mask = torch.zeros(
        (batch_size, num_heads, seq_len, seq_len), dtype=torch.float32, device=device
    )
    mask[:, :, :, seq_len // 2 :] = -1e9
    output_masked = scaled_dot_product_attention(q, k, v, attention_mask=mask)
    print(f"  Output shape: {output_masked.shape}")

    print("\nGrouped Query Attention (GQA):")
    num_kv_heads = 2
    k_gqa = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device)
    v_gqa = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device)
    attn = MultiHeadAttention(
        hidden_size=num_heads * head_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )
    output_gqa = attn(q, k_gqa, v_gqa)
    print(f"  Output shape: {output_gqa.shape}")

    print("\nOutput statistics:")
    print(f"  Mean: {float(output.mean()):.4f}")
    print(f"  Std:  {float(output.std()):.4f}")
    print(f"  Min:  {float(output.min()):.4f}")
    print(f"  Max:  {float(output.max()):.4f}")

    print("\nTriton Attention working!")
