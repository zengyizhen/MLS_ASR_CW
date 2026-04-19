"""
Triton Multi-Head Attention Implementation
End-to-end implementation using Triton kernels
"""

import numpy as np
import os
import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


def get_stream():
    """Get current CUDA stream pointer."""
    if torch.cuda.is_available():
        return torch.cuda.current_stream().cuda_stream
    return None


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
    )
    k = tl.load(
        k_ptr
        + pid_bh * stride_k0
        + offs_k[:, None] * stride_k1
        + offs_d[None, :] * stride_k2,
        mask=(offs_k[:, None] < seq_k) & (offs_d[None, :] < head_dim),
        other=0.0,
    )
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
    )
    v = tl.load(
        v_ptr
        + pid_bh * stride_v0
        + offs_k[:, None] * stride_v1
        + offs_d[None, :] * stride_v2,
        mask=(offs_k[:, None] < seq_k) & (offs_d[None, :] < head_dim),
        other=0.0,
    )
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


MAX_TRITON_BLOCK = 256
ATTENTION_LOG_MODE = os.environ.get("GLM_ASR_ATTENTION_LOG_MODE", "once").lower()
_ATTENTION_LOGGED_KEYS = set()


def _log_attention_path(
    path_name: str,
    q: torch.Tensor,
    seq_q: int,
    seq_k: int,
    head_dim: int,
    is_causal: bool,
) -> None:
    """Print which attention path is active.

    Logging is deduplicated by default because attention runs many times per
    layer. Set GLM_ASR_ATTENTION_LOG_MODE=always to print every invocation, or
    GLM_ASR_ATTENTION_LOG_MODE=off to disable it.
    """
    if ATTENTION_LOG_MODE == "off":
        return

    log_key = (
        path_name,
        str(q.device),
        str(q.dtype),
        seq_q,
        seq_k,
        head_dim,
        bool(is_causal),
    )
    if ATTENTION_LOG_MODE != "always" and log_key in _ATTENTION_LOGGED_KEYS:
        return

    print(
        "[attention] "
        f"path={path_name} "
        f"device={q.device} "
        f"dtype={q.dtype} "
        f"seq_q={seq_q} "
        f"seq_k={seq_k} "
        f"head_dim={head_dim} "
        f"is_causal={is_causal}"
    )
    _ATTENTION_LOGGED_KEYS.add(log_key)


def _pad_last_dim(x: torch.Tensor, padded_dim: int) -> torch.Tensor:
    """Pad the last dimension with zeros up to padded_dim."""
    if x.shape[-1] == padded_dim:
        return x

    padded_shape = list(x.shape)
    padded_shape[-1] = padded_dim
    padded = torch.zeros(
        padded_shape,
        dtype=torch.float32,
        device=x.device,
    )
    padded[..., : x.shape[-1]] = x.to(torch.float32)
    return padded


def _prepare_attention_mask(
    attention_mask: Optional[torch.Tensor],
    batch: int,
    num_heads: int,
    seq_q: int,
    seq_k: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Normalize attention_mask to (batch * num_heads, seq_q, seq_k)."""
    if attention_mask is None:
        return None

    mask = attention_mask.to(device=device, dtype=torch.float32)

    if mask.ndim == 4:
        if mask.shape[0] != batch or mask.shape[2] != seq_q or mask.shape[3] != seq_k:
            raise ValueError(
                "Expected attention_mask with shape "
                f"(batch, heads, {seq_q}, {seq_k}), got {tuple(mask.shape)}"
            )
        if mask.shape[1] == 1:
            mask = mask.expand(batch, num_heads, seq_q, seq_k)
        elif mask.shape[1] != num_heads:
            raise ValueError(
                "Expected attention_mask head dimension to be 1 or num_heads, got "
                f"{mask.shape[1]} vs {num_heads}"
            )
        return mask.reshape(batch * num_heads, seq_q, seq_k)

    if mask.ndim == 3:
        if mask.shape[0] == batch * num_heads:
            return mask
        if mask.shape[0] == batch:
            mask = mask[:, None, :, :].expand(batch, num_heads, seq_q, seq_k)
            return mask.reshape(batch * num_heads, seq_q, seq_k)

    raise ValueError(
        "attention_mask must have shape (batch, heads, seq_q, seq_k), "
        f"(batch, seq_q, seq_k), or (batch * heads, seq_q, seq_k); got {tuple(mask.shape)}"
    )


def _compute_triton_scores(
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    scale: float,
    seq_q: int,
    seq_k_padded: int,
    head_dim_padded: int,
) -> torch.Tensor:
    """Compute QK^T scores for one Triton attention block."""
    batch_heads = q_flat.shape[0]
    scores = torch.empty(
        (batch_heads, seq_q, seq_k_padded),
        dtype=torch.float32,
        device=q_flat.device,
    )

    grid = (batch_heads, seq_q)
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
    return scores


def _compute_triton_weighted_values(
    attn_weights: torch.Tensor,
    v_flat: torch.Tensor,
    seq_q: int,
    seq_k_padded: int,
    head_dim_padded: int,
) -> torch.Tensor:
    """Compute attention weights @ V for one Triton attention block."""
    batch_heads = attn_weights.shape[0]
    output = torch.empty(
        (batch_heads, seq_q, head_dim_padded),
        dtype=torch.float32,
        device=attn_weights.device,
    )

    grid = (batch_heads, seq_q)
    attention_output_kernel[grid](
        attn_weights,
        v_flat,
        output,
        seq_k_padded,
        head_dim_padded,
        attn_weights.stride(0),
        attn_weights.stride(1),
        attn_weights.stride(2),
        v_flat.stride(0),
        v_flat.stride(1),
        v_flat.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        BLOCK_K=seq_k_padded,
        BLOCK_D=head_dim_padded,
    )
    return output


def _scaled_dot_product_attention_triton_small(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    is_causal: bool,
    scale: float,
    seq_k_padded: int,
    head_dim_padded: int,
) -> torch.Tensor:
    """Original Triton path for sequence lengths that fit in one block."""
    batch, num_heads, seq_q, head_dim = q.shape
    _, _, seq_k, _ = k.shape

    q_flat = q.reshape(batch * num_heads, seq_q, head_dim).to(torch.float32)
    k_flat = k.reshape(batch * num_heads, seq_k, head_dim).to(torch.float32)
    v_flat = v.reshape(batch * num_heads, seq_k, head_dim).to(torch.float32)

    q_flat = _pad_last_dim(q_flat, head_dim_padded)
    k_flat = _pad_last_dim(k_flat, head_dim_padded)
    v_flat = _pad_last_dim(v_flat, head_dim_padded)

    if seq_k_padded != seq_k:
        k_padded = torch.zeros(
            (batch * num_heads, seq_k_padded, head_dim_padded),
            dtype=torch.float32,
            device=q.device,
        )
        v_padded = torch.zeros_like(k_padded)
        k_padded[:, :seq_k, :] = k_flat
        v_padded[:, :seq_k, :] = v_flat
        k_flat = k_padded
        v_flat = v_padded

    scores = _compute_triton_scores(
        q_flat,
        k_flat,
        scale,
        seq_q,
        seq_k_padded,
        head_dim_padded,
    )

    if seq_k_padded != seq_k:
        scores[:, :, seq_k:] = float("-inf")

    if is_causal:
        mask = torch.triu(
            torch.ones((seq_q, seq_k_padded), dtype=torch.bool, device=q.device),
            diagonal=1,
        )
        scores = scores.masked_fill(mask[None, :, :], float("-inf"))

    if attention_mask is not None:
        if seq_k_padded != seq_k:
            padded_mask = torch.zeros(
                (batch * num_heads, seq_q, seq_k_padded),
                dtype=torch.float32,
                device=q.device,
            )
            padded_mask[:, :, :seq_k] = attention_mask
            padded_mask[:, :, seq_k:] = float("-inf")
            attention_mask = padded_mask
        scores = scores + attention_mask

    scores_2d = scores.reshape(batch * num_heads * seq_q, seq_k_padded)
    softmax_inplace_kernel[(scores_2d.shape[0],)](
        scores_2d,
        scores_2d.stride(0),
        seq_k_padded,
        BLOCK_SIZE=seq_k_padded,
    )
    attn_weights = scores_2d.reshape(batch * num_heads, seq_q, seq_k_padded)

    output = _compute_triton_weighted_values(
        attn_weights,
        v_flat,
        seq_q,
        seq_k_padded,
        head_dim_padded,
    )

    if head_dim_padded != head_dim:
        output = output[:, :, :head_dim]

    return output.reshape(batch, num_heads, seq_q, head_dim).to(q.dtype)


def _scaled_dot_product_attention_triton_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    is_causal: bool,
    scale: float,
    head_dim_padded: int,
) -> torch.Tensor:
    """Chunked Triton attention for long seq_k values.

    The block kernels still operate on <= 256 keys each time, while Python-side
    online softmax accumulation keeps the overall result numerically stable.
    """
    batch, num_heads, seq_q, head_dim = q.shape
    _, _, seq_k, _ = k.shape
    batch_heads = batch * num_heads

    q_flat = q.reshape(batch_heads, seq_q, head_dim).to(torch.float32)
    k_flat = k.reshape(batch_heads, seq_k, head_dim).to(torch.float32)
    v_flat = v.reshape(batch_heads, seq_k, head_dim).to(torch.float32)

    q_flat = _pad_last_dim(q_flat, head_dim_padded)

    running_max = torch.full(
        (batch_heads, seq_q),
        float("-inf"),
        dtype=torch.float32,
        device=q.device,
    )
    running_l = torch.zeros(
        (batch_heads, seq_q),
        dtype=torch.float32,
        device=q.device,
    )
    running_acc = torch.zeros(
        (batch_heads, seq_q, head_dim_padded),
        dtype=torch.float32,
        device=q.device,
    )

    query_positions = None
    if is_causal:
        query_positions = torch.arange(seq_q, device=q.device)[None, :, None]

    for chunk_start in range(0, seq_k, MAX_TRITON_BLOCK):
        chunk_end = min(chunk_start + MAX_TRITON_BLOCK, seq_k)
        chunk_len = chunk_end - chunk_start
        chunk_padded = next_power_of_two(chunk_len)

        k_chunk = k_flat[:, chunk_start:chunk_end, :]
        v_chunk = v_flat[:, chunk_start:chunk_end, :]

        k_chunk = _pad_last_dim(k_chunk, head_dim_padded)
        v_chunk = _pad_last_dim(v_chunk, head_dim_padded)

        if chunk_padded != chunk_len:
            padded_shape = (batch_heads, chunk_padded, head_dim_padded)
            k_chunk_padded = torch.zeros(
                padded_shape,
                dtype=torch.float32,
                device=q.device,
            )
            v_chunk_padded = torch.zeros_like(k_chunk_padded)
            k_chunk_padded[:, :chunk_len, :] = k_chunk
            v_chunk_padded[:, :chunk_len, :] = v_chunk
            k_chunk = k_chunk_padded
            v_chunk = v_chunk_padded

        scores = _compute_triton_scores(
            q_flat,
            k_chunk,
            scale,
            seq_q,
            chunk_padded,
            head_dim_padded,
        )

        if chunk_padded != chunk_len:
            scores[:, :, chunk_len:] = float("-inf")

        if is_causal:
            key_positions = torch.arange(
                chunk_start,
                chunk_start + chunk_padded,
                device=q.device,
            )[None, None, :]
            causal_mask = key_positions > query_positions
            scores = scores.masked_fill(causal_mask, float("-inf"))

        if attention_mask is not None:
            mask_chunk = attention_mask[:, :, chunk_start:chunk_end]
            if chunk_padded != chunk_len:
                padded_mask = torch.zeros(
                    (batch_heads, seq_q, chunk_padded),
                    dtype=torch.float32,
                    device=q.device,
                )
                padded_mask[:, :, :chunk_len] = mask_chunk
                padded_mask[:, :, chunk_len:] = float("-inf")
                mask_chunk = padded_mask
            scores = scores + mask_chunk

        chunk_max = scores.max(dim=-1).values
        safe_chunk_max = torch.where(
            torch.isfinite(chunk_max),
            chunk_max,
            torch.zeros_like(chunk_max),
        )
        exp_scores = torch.exp(scores - safe_chunk_max[:, :, None])
        exp_scores = torch.where(
            torch.isfinite(scores),
            exp_scores,
            torch.zeros_like(exp_scores),
        )
        chunk_l = exp_scores.sum(dim=-1)
        chunk_acc = _compute_triton_weighted_values(
            exp_scores,
            v_chunk,
            seq_q,
            chunk_padded,
            head_dim_padded,
        )

        new_max = torch.maximum(running_max, chunk_max)
        alpha = torch.exp(
            torch.where(
                torch.isfinite(running_max),
                running_max - new_max,
                torch.full_like(running_max, float("-inf")),
            )
        )
        beta = torch.exp(
            torch.where(
                torch.isfinite(chunk_max),
                chunk_max - new_max,
                torch.full_like(chunk_max, float("-inf")),
            )
        )

        running_l = alpha * running_l + beta * chunk_l
        running_acc = (
            alpha[:, :, None] * running_acc
            + beta[:, :, None] * chunk_acc
        )
        running_max = new_max

    output = running_acc / torch.clamp(running_l[:, :, None], min=1e-9)

    if head_dim_padded != head_dim:
        output = output[:, :, :head_dim]

    return output.reshape(batch, num_heads, seq_q, head_dim).to(q.dtype)


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

    attention_mask = _prepare_attention_mask(
        attention_mask,
        batch,
        num_heads,
        seq_q,
        seq_k,
        q.device,
    )

    seq_k_padded = next_power_of_two(seq_k)
    head_dim_padded = next_power_of_two(head_dim)

    if q.is_cuda and head_dim_padded <= MAX_TRITON_BLOCK:
        if seq_k_padded <= MAX_TRITON_BLOCK:
            _log_attention_path(
                "triton_full",
                q,
                seq_q,
                seq_k,
                head_dim,
                is_causal,
            )
            return _scaled_dot_product_attention_triton_small(
                q,
                k,
                v,
                attention_mask,
                is_causal,
                scale,
                seq_k_padded,
                head_dim_padded,
            )

        _log_attention_path(
            "triton_chunked",
            q,
            seq_q,
            seq_k,
            head_dim,
            is_causal,
        )
        return _scaled_dot_product_attention_triton_chunked(
            q,
            k,
            v,
            attention_mask,
            is_causal,
            scale,
            head_dim_padded,
        )

    _log_attention_path(
        "torch",
        q,
        seq_q,
        seq_k,
        head_dim,
        is_causal,
    )
    scores = torch.einsum("bnqd,bnkd->bnqk", q, k) * scale

    if is_causal:
        mask = torch.triu(
            torch.ones((seq_q, seq_k), dtype=torch.bool, device=q.device),
            diagonal=1,
        )
        scores = scores.masked_fill(mask[None, None, :, :], float("-inf"))

    if attention_mask is not None:
        scores = scores + attention_mask.reshape(batch, num_heads, seq_q, seq_k)

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
