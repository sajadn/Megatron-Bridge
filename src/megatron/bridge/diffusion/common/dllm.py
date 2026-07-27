# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Diffusion language model utilities: masking, block attention masks, and sampling.

The sampling primitives (``add_gumbel_noise``, ``get_num_transfer_tokens``,
``get_transfer_index``) implement the iterative-denoising step shared by every
block-diffusion / masked-dLLM generation loop in this repo (NemotronLabsDiffusion,
LLaDA1.5, ...). They are model-agnostic: each model keeps its own generation loop
with its own attention semantics (causal-with-KV-cache vs fully bidirectional) but
calls these helpers to score confidence and choose which masked positions to
unmask at each step.
"""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask

from megatron.bridge.diffusion.common.cp_utils import zigzag_local_to_global_idx


# torch's default create_block_mask evaluates mask_mod over the dense
# [B, H, Q_LEN, KV_LEN] grid (see create_mask) before reducing it to 128x128
# tiles, costing ~10 bytes per grid element -- 160 GB at a 128K doubled
# sequence with cp=4, which OOMs. Routing the build through torch.compile
# fuses the predicate into the tile reduction so the dense grid is never
# allocated (28 MiB at that same geometry, bit-identical BlockMask).
#
# dynamic=True is REQUIRED, not cosmetic: Q_LEN/KV_LEN arrive as Python ints
# and static compilation specializes on them by value, needing one compile per
# sequence length (11-60 s each) and hitting Dynamo's default recompile_limit
# of 8 -- after which it silently falls back to eager and the fix disappears.
# With dynamic=True a single kernel serves every shape (two once the grid
# crosses 2**31 and 64-bit indexing kicks in). See
# plans/mask_build_cost_findings.md.
#
# On by default; DIFFU_MASK_COMPILE=off restores the eager path.
_MASK_COMPILE = os.environ.get("DIFFU_MASK_COMPILE", "on").lower() not in (
    "0",
    "off",
    "false",
    "no",
)
_MASK_COMPILE_DYNAMIC = os.environ.get("DIFFU_MASK_COMPILE_DYNAMIC", "on").lower() not in (
    "0",
    "off",
    "false",
    "no",
)
_COMPILED_CREATE_BLOCK_MASK = None


def _build_block_mask(*args, **kwargs):
    """create_block_mask, optionally compiled (one wrapper, reused)."""
    global _COMPILED_CREATE_BLOCK_MASK
    if not _MASK_COMPILE:
        return create_block_mask(*args, **kwargs)
    if _COMPILED_CREATE_BLOCK_MASK is None:
        _COMPILED_CREATE_BLOCK_MASK = torch.compile(
            create_block_mask, dynamic=_MASK_COMPILE_DYNAMIC
        )
        print(
            f"[dllm] BlockMask build compiled (dynamic={_MASK_COMPILE_DYNAMIC})",
            flush=True,
        )
    return _COMPILED_CREATE_BLOCK_MASK(*args, **kwargs)


def forward_process_simple_masking(input_ids, mask_token_id, eps=1e-3, loss_mask=None, generator=None):
    """Uniform random masking for diffusion LM training.

    For each sequence in the batch, sample a masking ratio t ~ U(eps, 1) and
    independently mask each token with probability t.

    Returns:
        noisy_batch: input_ids with masked positions replaced by mask_token_id
        masked_indices: boolean mask of shape (b, l)
        p_mask: per-token masking probability of shape (b, l)
    """
    b, seq_len = input_ids.shape
    device = input_ids.device

    t = torch.rand(b, device=device, generator=generator)

    p_mask = (1 - eps) * t + eps  # shape: (b,)
    p_mask = p_mask[:, None].expand(-1, seq_len)  # shape: (b, l)

    masked_indices = torch.rand((b, seq_len), device=device, generator=generator) < p_mask

    if loss_mask is not None:
        masked_indices[loss_mask == 0] = 0

    noisy_batch = torch.where(masked_indices, mask_token_id, input_ids)

    return noisy_batch, masked_indices, p_mask


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply Gumbel noise to logits for stochastic sampling.

    At ``temperature == 0`` this is a no-op (returns ``logits`` unchanged), so an
    ``argmax`` over the result is plain greedy decoding.

    Args:
        logits: Unnormalized scores of shape ``[..., vocab_size]``.
        temperature: Sampling temperature. ``0`` disables noise (greedy).

    Returns:
        Noised scores (float64 when noise is applied) whose ``argmax`` samples
        from the temperature-scaled distribution.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """Compute how many masked tokens to unmask at each diffusion step.

    Distributes the number of masked positions as evenly as possible across
    ``steps``, giving the earlier steps the remainder.

    Args:
        mask_index: Boolean tensor ``[batch, seq_len]`` (True where masked).
        steps: Number of denoising steps to spread the unmasking over.

    Returns:
        Int64 tensor ``[batch, steps]`` whose rows sum to each sequence's mask
        count.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens


def get_transfer_index(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,
    x: torch.Tensor,
    num_transfer_tokens: torch.Tensor,
    threshold: Optional[float] = None,
    neg_entropy: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select which masked positions to unmask at one diffusion step.

    Samples candidate tokens (``x0``) from ``logits`` and, among currently
    masked positions, transfers the highest-confidence ones from mask to real
    token. Used identically by every block-diffusion generation loop in the repo
    regardless of attention semantics.

    Args:
        logits: Per-position scores ``[batch, seq_len, vocab_size]``.
        temperature: Sampling temperature for Gumbel noise (``0`` = greedy).
        remasking: Confidence source for ranking: ``"low_confidence"`` uses the
            softmax probability of the chosen token; ``"random"`` uses uniform
            noise.
        mask_index: Boolean ``[batch, seq_len]`` marking still-masked positions.
        x: Current token ids ``[batch, seq_len]``; non-masked positions are kept.
        num_transfer_tokens: Per-sequence count of tokens to unmask this step
            (``[batch]`` slice of :func:`get_num_transfer_tokens`). Ignored when
            ``threshold`` is set.
        threshold: If set, transfer every masked position whose confidence
            exceeds this value instead of a fixed count.
        neg_entropy: If True, rank by negative entropy of the distribution
            instead of the chosen token's probability.

    Returns:
        Tuple ``(x0, transfer_index)`` where ``x0`` is the candidate token ids
        (non-masked positions unchanged) and ``transfer_index`` is a boolean mask
        of positions to commit this step.
    """
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    if remasking == "low_confidence":
        p = F.softmax(logits, dim=-1)
        x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
    elif remasking == "random":
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

    if neg_entropy:
        p = F.softmax(logits, dim=-1)
        epsilon = 1e-10
        log_probs = torch.log(p + epsilon)
        confidence_scores = torch.sum(p * log_probs, dim=-1)
    else:
        confidence_scores = x0_p

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, confidence_scores, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index


def compute_block_mask(block_size, max_seq_length):
    """Compute the sbd_block_diff attention mask.

    The semi-block-diffusion mask is composed of three sub-masks over a
    doubled sequence [xt | x0] of length 2*max_seq_length:
      - Block Diagonal (M_BD): self-attention within noised blocks (xt only)
      - Offset Block-Causal (M_OBC): cross-attention from xt to past x0 blocks
      - Fully Causal (M_FC): fully causal attention within x0

    Args:
        block_size: Block size for block-based attention.
        max_seq_length: Length of one half (xt or x0) of the sequence.

    Returns:
        BlockMask for use with ``flex_attention``.
    """
    n = max_seq_length

    def sbd_block_diff_mask(b, h, q_idx, kv_idx):
        x0_flag_q = q_idx >= n
        x0_flag_kv = kv_idx >= n

        block_q = torch.where(x0_flag_q, (q_idx - n) // block_size, q_idx // block_size)
        block_kv = torch.where(x0_flag_kv, (kv_idx - n) // block_size, kv_idx // block_size)

        block_diagonal = (block_q == block_kv) & (~x0_flag_kv) & (~x0_flag_q)
        offset_block_causal = (block_q > block_kv) & x0_flag_kv & (~x0_flag_q)
        fully_causal = (q_idx >= kv_idx) & x0_flag_kv & x0_flag_q

        return block_diagonal | offset_block_causal | fully_causal

    q_len = max_seq_length * 2
    return _build_block_mask(sbd_block_diff_mask, B=None, H=None, Q_LEN=q_len, KV_LEN=q_len)


def compute_asymmetric_semi_ar_mask(
    block_size,
    noisy_length,
    clean_length,
    noisy_response_offset,
    prompt_lengths,
    noisy_valid_lengths,
    clean_lengths,
    cp_rank=0,
    cp_size=1,
):
    """Compute compact asymmetric semi-AR attention mask.

    Layout is ``[noisy_response | clean_prompt_response]``. Noisy
    response queries attend bidirectionally within their current noisy block,
    to the clean prompt, and to clean response tokens from previous blocks.
    Clean queries use ordinary causal attention over the clean side only.

    With ``cp_size > 1`` the mask is built for local-Q context parallelism:
    query rows are this rank's zigzag shard of the sequence (``Q_LEN =
    full_seq_len / cp_size``, indices remapped to global positions) while KV
    columns span the full gathered sequence.
    """
    if (
        prompt_lengths.ndim != 1
        or noisy_valid_lengths.ndim != 1
        or clean_lengths.ndim != 1
    ):
        raise ValueError(
            "prompt_lengths, noisy_valid_lengths, and clean_lengths must be 1D tensors"
        )
    if (
        prompt_lengths.shape != noisy_valid_lengths.shape
        or prompt_lengths.shape != clean_lengths.shape
    ):
        raise ValueError(
            "Asymmetric semi-AR attention metadata tensors must have matching shapes"
        )

    full_seq_len = noisy_length + clean_length

    def asymmetric_semi_ar_mask(b, h, q_idx, kv_idx):
        del h
        prompt_len = prompt_lengths[b]
        noisy_valid_len = noisy_valid_lengths[b]
        clean_len = clean_lengths[b]

        q_is_noisy = q_idx < noisy_length
        kv_is_noisy = kv_idx < noisy_length
        q_noisy_rel = q_idx - noisy_response_offset
        kv_noisy_rel = kv_idx - noisy_response_offset
        q_noisy_valid = q_is_noisy & (q_noisy_rel >= 0) & (q_noisy_rel < noisy_valid_len)
        kv_noisy_valid = kv_is_noisy & (kv_noisy_rel >= 0) & (kv_noisy_rel < noisy_valid_len)

        q_block = torch.div(q_noisy_rel, block_size, rounding_mode="floor")
        kv_block = torch.div(kv_noisy_rel, block_size, rounding_mode="floor")
        noisy_same_block = q_noisy_valid & kv_noisy_valid & (q_block == kv_block)

        q_clean_idx = q_idx - noisy_length
        kv_clean_idx = kv_idx - noisy_length
        q_clean_valid = (~q_is_noisy) & (q_clean_idx >= 0) & (q_clean_idx < clean_len)
        kv_clean_valid = (~kv_is_noisy) & (kv_clean_idx >= 0) & (kv_clean_idx < clean_len)

        clean_prompt = kv_clean_valid & (kv_clean_idx < prompt_len)
        clean_response_rel = kv_clean_idx - prompt_len
        clean_previous_response_blocks = (
            kv_clean_valid
            & (clean_response_rel >= 0)
            & (clean_response_rel < q_block * block_size)
        )
        noisy_query = q_noisy_valid & (
            noisy_same_block | clean_prompt | clean_previous_response_blocks
        )

        clean_causal = q_clean_valid & kv_clean_valid & (kv_clean_idx <= q_clean_idx)

        valid_query = q_noisy_valid | q_clean_valid
        invalid_query_self = (~valid_query) & (q_idx == kv_idx)
        return noisy_query | clean_causal | invalid_query_self

    mask_mod = asymmetric_semi_ar_mask
    q_len = full_seq_len
    if cp_size > 1:
        assert full_seq_len % (2 * cp_size) == 0, (
            f"full_seq_len {full_seq_len} not divisible by 2*cp_size {2 * cp_size}"
        )
        q_len = full_seq_len // cp_size

        def cp_local_q_mask(b, h, q_idx, kv_idx):
            return asymmetric_semi_ar_mask(
                b, h, zigzag_local_to_global_idx(q_idx, cp_rank, cp_size, q_len), kv_idx
            )

        mask_mod = cp_local_q_mask

    return _build_block_mask(
        mask_mod,
        B=prompt_lengths.shape[0],
        H=None,
        Q_LEN=q_len,
        KV_LEN=full_seq_len,
        device=prompt_lengths.device,
    )


def compute_active_block_bidirectional_mask(block_starts, block_ends, q_len, kv_len, offset=0):
    """Build a causal mask with bidirectional attention inside each row's active block."""
    if block_starts.ndim != 1 or block_ends.ndim != 1:
        raise ValueError("block_starts and block_ends must be 1D tensors")
    if block_starts.shape != block_ends.shape:
        raise ValueError(f"block_starts shape {block_starts.shape} must match block_ends shape {block_ends.shape}")

    q_pos = torch.arange(offset, offset + q_len, device=block_starts.device)
    kv_pos = torch.arange(kv_len, device=block_starts.device)
    causal_mask = kv_pos.unsqueeze(0) <= q_pos.unsqueeze(1)

    block_starts = block_starts.view(-1, 1, 1)
    block_ends = block_ends.view(-1, 1, 1)
    q_pos = q_pos.view(1, -1, 1)
    kv_pos = kv_pos.view(1, 1, -1)
    q_in_block = (q_pos >= block_starts) & (q_pos < block_ends)
    block_mask = q_in_block & (kv_pos < block_ends)
    return causal_mask.unsqueeze(0) | block_mask


def compute_block_bias(block_size, max_seq_length, dtype, device):
    """Dense additive ``post_scale_bias`` equivalent of the sbd_block_diff mask.

    flex_attention's ``BlockMask`` is not compatible with context parallelism
    (Transformer Engine disables the unfused backend under CP). The same mask can
    instead be supplied to ``TEDotProductAttention`` as an additive bias
    (``core_attention_bias_type="post_scale_bias"``), which TE *does* support
    under CP and slices per zigzag chunk internally.

    The returned bias is ``0`` where attention is allowed and a large negative
    value where it is disallowed, shaped ``[1, 1, 2L, 2L]`` (broadcastable
    ``11ss``). Every query row has at least its own position allowed (xt within
    its block; x0 via causal self), so no row is fully masked.

    Args:
        block_size: Block size for block-based attention.
        max_seq_length: Length of one half (xt or x0) of the doubled sequence.
        dtype: Bias dtype (match the attention compute dtype).
        device: Device to build the bias on.

    Returns:
        Tensor of shape ``[1, 1, 2*max_seq_length, 2*max_seq_length]``.
    """
    n = max_seq_length
    q_len = 2 * n
    idx = torch.arange(q_len, device=device)
    q_idx = idx[:, None]
    kv_idx = idx[None, :]

    x0_flag_q = q_idx >= n
    x0_flag_kv = kv_idx >= n
    block_q = torch.where(x0_flag_q, (q_idx - n) // block_size, q_idx // block_size)
    block_kv = torch.where(x0_flag_kv, (kv_idx - n) // block_size, kv_idx // block_size)

    block_diagonal = (block_q == block_kv) & (~x0_flag_kv) & (~x0_flag_q)
    offset_block_causal = (block_q > block_kv) & x0_flag_kv & (~x0_flag_q)
    fully_causal = (q_idx >= kv_idx) & x0_flag_kv & x0_flag_q
    allowed = block_diagonal | offset_block_causal | fully_causal

    bias = torch.zeros(q_len, q_len, dtype=dtype, device=device)
    bias.masked_fill_(~allowed, torch.finfo(dtype).min)
    return bias.view(1, 1, q_len, q_len)
