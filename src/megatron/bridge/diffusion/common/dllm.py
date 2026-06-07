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

"""Diffusion language model utilities: masking and block attention masks."""

import torch
from torch.nn.attention.flex_attention import create_block_mask


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
    return create_block_mask(sbd_block_diff_mask, B=None, H=None, Q_LEN=q_len, KV_LEN=q_len)


def compute_asymmetric_semi_ar_mask(
    block_size,
    noisy_length,
    clean_length,
    noisy_response_offset,
    prompt_lengths,
    noisy_valid_lengths,
    clean_lengths,
):
    """Compute compact asymmetric semi-AR attention mask.

    Layout is ``[noisy_response | clean_prompt_response]``. Noisy
    response queries attend bidirectionally within their current noisy block,
    to the clean prompt, and to clean response tokens from previous blocks.
    Clean queries use ordinary causal attention over the clean side only.
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

    return create_block_mask(
        asymmetric_semi_ar_mask,
        B=prompt_lengths.shape[0],
        H=None,
        Q_LEN=full_seq_len,
        KV_LEN=full_seq_len,
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
