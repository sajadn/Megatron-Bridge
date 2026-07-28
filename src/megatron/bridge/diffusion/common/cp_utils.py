# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Context-parallel helpers for the sbd_block_diff diffusion LM.

Megatron context parallelism shards the sequence with a "load-balanced zigzag"
layout: the sequence is cut into ``2 * cp_size`` chunks and rank ``r`` owns
chunks ``[r, 2*cp_size - 1 - r]`` (this is exactly what
``megatron.core.utils.get_batch_on_this_cp_rank`` produces and what
``TEDotProductAttention`` assumes when it slices a ``post_scale_bias``).

This module provides:
  - ``zigzag_slice``: deterministically pick this CP rank's zigzag slice of a
    full-sequence tensor (no communication). Used to shard RoPE cos/sin and the
    Llama-4 scale, which are recomputed identically on every rank.
  - ``all_gather_seq_cp``: autograd all-gather that reconstructs the full
    sequence (undoing the zigzag) so the diffusion loss can split ``[xt | x0]``.
"""

import torch


def zigzag_slice(tensor: torch.Tensor, cp_rank: int, cp_size: int, seq_dim: int) -> torch.Tensor:
    """Pick this CP rank's load-balanced zigzag slice along ``seq_dim``.

    Mirrors ``get_batch_on_this_cp_rank``: split into ``2*cp_size`` chunks and
    take chunks ``cp_rank`` and ``2*cp_size - 1 - cp_rank``.

    Args:
        tensor: Full-sequence tensor.
        cp_rank: This rank's index within the CP group.
        cp_size: CP world size.
        seq_dim: Dimension to slice.

    Returns:
        Local slice of length ``tensor.shape[seq_dim] / cp_size``.
    """
    if cp_size == 1:
        return tensor
    seq_len = tensor.shape[seq_dim]
    assert seq_len % (2 * cp_size) == 0, f"seq_len {seq_len} not divisible by 2*cp_size {2 * cp_size}"
    shard = seq_len // (2 * cp_size)
    idx1, idx2 = cp_rank, 2 * cp_size - 1 - cp_rank
    s1 = [slice(None)] * tensor.dim()
    s1[seq_dim] = slice(idx1 * shard, (idx1 + 1) * shard)
    s2 = [slice(None)] * tensor.dim()
    s2[seq_dim] = slice(idx2 * shard, (idx2 + 1) * shard)
    return torch.cat([tensor[tuple(s1)], tensor[tuple(s2)]], dim=seq_dim).contiguous()


def zigzag_local_to_global_idx(
    local_idx: torch.Tensor, cp_rank: int, cp_size: int, local_len: int
) -> torch.Tensor:
    """Map local zigzag positions to global sequence positions.

    Inverse of ``zigzag_slice`` for index tensors: local positions
    ``[0, local_len)`` on ``cp_rank`` correspond to global chunks ``cp_rank``
    (first half) and ``2*cp_size - 1 - cp_rank`` (second half). Pure tensor op
    (usable inside a flex-attention ``mask_mod``). Identity when cp_size == 1.
    """
    if cp_size == 1:
        return local_idx
    half = local_len // 2
    in_first = local_idx < half
    g_first = cp_rank * half + local_idx
    g_second = (2 * cp_size - 1 - cp_rank) * half + (local_idx - half)
    return torch.where(in_first, g_first, g_second)


class _AllGatherSeqCP(torch.autograd.Function):
    """All-gather a CP-zigzag-sharded tensor to the full sequence on every rank.

    Forward reconstructs the global order; the result is identical on all CP
    ranks (replicated downstream). Backward picks this rank's zigzag chunks from
    the (identical) full gradient -- no all-reduce, which would over-count by
    ``cp_size``.
    """

    @staticmethod
    def forward(ctx, tensor, cp_group, seq_dim):
        cp_size = torch.distributed.get_world_size(cp_group)
        ctx.cp_group = cp_group
        ctx.cp_size = cp_size
        ctx.seq_dim = seq_dim
        if cp_size == 1:
            return tensor.contiguous()

        tensor = tensor.contiguous()
        gathered = [torch.empty_like(tensor) for _ in range(cp_size)]
        torch.distributed.all_gather(gathered, tensor, group=cp_group)

        # Each rank contributed two zigzag chunks; restore global chunk order.
        chunks = []
        for shard in gathered:
            chunks.extend(torch.chunk(shard, chunks=2, dim=seq_dim))
        indices = []
        for r in range(cp_size):
            indices.append(r)
            indices.append(2 * cp_size - 1 - r)
        ordered = [c for _, c in sorted(zip(indices, chunks), key=lambda t: t[0])]
        return torch.cat(ordered, dim=seq_dim).contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        cp_size = ctx.cp_size
        seq_dim = ctx.seq_dim
        if cp_size == 1:
            return grad_output, None, None
        cp_rank = torch.distributed.get_rank(ctx.cp_group)
        chunk = grad_output.shape[seq_dim] // (2 * cp_size)
        lo = cp_rank * chunk
        hi = (2 * cp_size - 1 - cp_rank) * chunk
        grad_input = torch.cat(
            [grad_output.narrow(seq_dim, lo, chunk), grad_output.narrow(seq_dim, hi, chunk)],
            dim=seq_dim,
        )
        return grad_input, None, None


def all_gather_seq_cp(tensor: torch.Tensor, cp_group, seq_dim: int = 1) -> torch.Tensor:
    """Autograd all-gather of a CP-sharded sequence tensor to full length."""
    return _AllGatherSeqCP.apply(tensor, cp_group, seq_dim)


def _reorder_zigzag_chunks(gathered, cp_size, seq_dim):
    """Reassemble global sequence order from per-rank zigzag shards."""
    chunks = []
    for shard in gathered:
        chunks.extend(torch.chunk(shard, chunks=2, dim=seq_dim))
    indices = []
    for r in range(cp_size):
        indices.append(r)
        indices.append(2 * cp_size - 1 - r)
    ordered = [c for _, c in sorted(zip(indices, chunks), key=lambda t: t[0])]
    return torch.cat(ordered, dim=seq_dim).contiguous()


class _ScatterSeqCP(torch.autograd.Function):
    """Inverse of the gather: take the full sequence (identical on every CP rank)
    and keep this rank's zigzag slice. Used to scatter the attention output back
    after a redundant full-sequence flex_attention.

    Forward = zigzag_slice. Backward = all-gather + reorder (each rank contributes
    the grad for the positions it owns; assembled into the full-sequence grad).
    """

    @staticmethod
    def forward(ctx, tensor, cp_group, seq_dim):
        cp_size = torch.distributed.get_world_size(cp_group)
        cp_rank = torch.distributed.get_rank(cp_group)
        ctx.cp_group = cp_group
        ctx.cp_size = cp_size
        ctx.seq_dim = seq_dim
        if cp_size == 1:
            return tensor.contiguous()
        return zigzag_slice(tensor, cp_rank, cp_size, seq_dim)

    @staticmethod
    def backward(ctx, grad_output):
        cp_size = ctx.cp_size
        seq_dim = ctx.seq_dim
        if cp_size == 1:
            return grad_output, None, None
        grad_output = grad_output.contiguous()
        gathered = [torch.empty_like(grad_output) for _ in range(cp_size)]
        torch.distributed.all_gather(gathered, grad_output, group=ctx.cp_group)
        return _reorder_zigzag_chunks(gathered, cp_size, seq_dim), None, None


def scatter_seq_cp(tensor: torch.Tensor, cp_group, seq_dim: int = 1) -> torch.Tensor:
    """Autograd scatter of a full-sequence tensor to this CP rank's zigzag slice."""
    return _ScatterSeqCP.apply(tensor, cp_group, seq_dim)


class _AllGatherKVSeqCP(torch.autograd.Function):
    """All-gather K/V shards to the full sequence for local-Q attention.

    Forward is identical to ``_AllGatherSeqCP`` (every rank reconstructs the
    full global order). Backward differs: with Q kept local, each rank's K/V
    gradient only carries the contributions of its own query rows, so the true
    gradient of the local shard is the SUM over ranks of the full-length grad,
    sliced to this rank's zigzag chunks. ``_AllGatherSeqCP``'s slice-only
    backward assumes the downstream grad is replicated and would silently drop
    the cross-rank terms here.
    """

    @staticmethod
    def forward(ctx, tensor, cp_group, seq_dim):
        cp_size = torch.distributed.get_world_size(cp_group)
        ctx.cp_group = cp_group
        ctx.cp_size = cp_size
        ctx.seq_dim = seq_dim
        if cp_size == 1:
            return tensor.contiguous()
        tensor = tensor.contiguous()
        gathered = [torch.empty_like(tensor) for _ in range(cp_size)]
        torch.distributed.all_gather(gathered, tensor, group=cp_group)
        return _reorder_zigzag_chunks(gathered, cp_size, seq_dim)

    @staticmethod
    def backward(ctx, grad_output):
        cp_size = ctx.cp_size
        if cp_size == 1:
            return grad_output, None, None
        # Clone: all_reduce writes in place, and the incoming grad must not be
        # mutated (matches the no-mutation discipline of the sibling Functions).
        grad_output = grad_output.contiguous().clone()
        torch.distributed.all_reduce(grad_output, group=ctx.cp_group)
        cp_rank = torch.distributed.get_rank(ctx.cp_group)
        return (
            zigzag_slice(grad_output, cp_rank, cp_size, ctx.seq_dim),
            None,
            None,
        )


def all_gather_kv_seq_cp(tensor: torch.Tensor, cp_group, seq_dim: int = 1) -> torch.Tensor:
    """Autograd K/V all-gather for local-Q attention (backward = all-reduce + slice)."""
    return _AllGatherKVSeqCP.apply(tensor, cp_group, seq_dim)


def local_zigzag_mask(seq_len: int, cp_rank: int, cp_size: int, device) -> torch.Tensor:
    """Boolean ``[seq_len]`` mask of the positions this CP rank owns.

    True at the two zigzag chunks (``cp_rank`` and ``2*cp_size-1-cp_rank``). Used
    to restrict the (full, gathered) loss to this rank's share so that summing
    loss/num_tokens across the CP group recovers the global totals exactly --
    i.e. standard Megatron context-parallel loss reduction stays valid. All-True
    when ``cp_size == 1``.
    """
    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    if cp_size == 1:
        mask[:] = True
        return mask
    assert seq_len % (2 * cp_size) == 0, f"seq_len {seq_len} not divisible by 2*cp_size {2 * cp_size}"
    cs = seq_len // (2 * cp_size)
    for idx in (cp_rank, 2 * cp_size - 1 - cp_rank):
        mask[idx * cs : (idx + 1) * cs] = True
    return mask


# ---------------------------------------------------------------------------
# Segment-aware (block-aware) sharding
#
# Stage (b) of plans/cp_kv_sharding_blockaware_ring.md splits the sequence into
# independently-zigzagged SEGMENTS (e.g. [noisy | clean], or [noisy | prompt |
# clean_response]) instead of one global zigzag. Each segment then gets the
# treatment its cost structure wants -- the noisy segment stays rank-local
# because it is only ever consumed block-diagonally, while the clean segment is
# gathered (b) or ringed (c). The helpers below are the layout primitives; the
# mask remap that consumes them lives in ``dllm``.
# ---------------------------------------------------------------------------


def segmented_zigzag_slice(tensor, segment_lengths, cp_rank: int, cp_size: int, seq_dim: int):
    """Zigzag-shard each segment of a concatenated sequence independently.

    The local layout is the concatenation, in segment order, of each segment's
    own zigzag slice: ``[seg0_local | seg1_local | ...]`` with
    ``len(seg_i_local) == segment_lengths[i] / cp_size``.

    Args:
        tensor: Full-sequence tensor; ``tensor.shape[seq_dim]`` must equal
            ``sum(segment_lengths)``.
        segment_lengths: Per-segment full lengths, in layout order.
        cp_rank: This rank's index within the CP group.
        cp_size: CP world size.
        seq_dim: Dimension to slice.

    Returns:
        Local shard of length ``sum(segment_lengths) / cp_size``.
    """
    if cp_size == 1:
        return tensor
    total = sum(segment_lengths)
    assert tensor.shape[seq_dim] == total, (
        f"segmented_zigzag_slice: tensor length {tensor.shape[seq_dim]} along dim {seq_dim} "
        f"does not match sum(segment_lengths)={total}"
    )
    parts = []
    offset = 0
    for seg_len in segment_lengths:
        segment = tensor.narrow(seq_dim, offset, seg_len)
        parts.append(zigzag_slice(segment, cp_rank, cp_size, seq_dim))
        offset += seg_len
    return torch.cat(parts, dim=seq_dim).contiguous()


def segmented_zigzag_local_to_global_idx(
    local_idx: torch.Tensor, segment_lengths, cp_rank: int, cp_size: int
) -> torch.Tensor:
    """Map positions in the segmented local layout to global sequence positions.

    Index-space inverse of :func:`segmented_zigzag_slice`, and the independent
    arithmetic definition of the layout that :func:`segmented_zigzag_slice`
    defines in data space (the parity suite cross-checks the two).

    The loop over segments runs at BUILD time only: consumers that need this
    map inside a compiled kernel go through
    :func:`segmented_zigzag_index_table`, which collapses it to one gather.
    Identity when ``cp_size == 1``.

    Args:
        local_idx: Positions in ``[0, sum(segment_lengths) / cp_size)``.
        segment_lengths: Per-segment full lengths, in layout order.
        cp_rank: This rank's index within the CP group.
        cp_size: CP world size.

    Returns:
        Global positions in ``[0, sum(segment_lengths))``.
    """
    if cp_size == 1:
        return local_idx
    global_idx = torch.zeros_like(local_idx)
    local_offset = 0
    global_offset = 0
    for seg_len in segment_lengths:
        local_len = seg_len // cp_size
        in_segment = (local_idx >= local_offset) & (local_idx < local_offset + local_len)
        mapped = global_offset + zigzag_local_to_global_idx(
            local_idx - local_offset, cp_rank, cp_size, local_len
        )
        global_idx = torch.where(in_segment, mapped, global_idx)
        local_offset += local_len
        global_offset += seg_len
    return global_idx


def reorder_segmented_zigzag_shards(shards, segment_lengths, cp_size: int, seq_dim: int):
    """Reassemble the full sequence from every rank's segmented local shard.

    Data-space inverse of :func:`segmented_zigzag_slice`, and the reordering
    step a segmented all-gather would perform after its collective (the
    single-segment analogue is ``_reorder_zigzag_chunks``).

    Args:
        shards: Per-rank local shards, indexed by CP rank.
        segment_lengths: Per-segment full lengths, in layout order.
        cp_size: CP world size.
        seq_dim: Sequence dimension.

    Returns:
        Full-sequence tensor of length ``sum(segment_lengths)``.
    """
    if cp_size == 1:
        return shards[0]
    segments = []
    local_offset = 0
    for seg_len in segment_lengths:
        local_len = seg_len // cp_size
        per_rank = [shard.narrow(seq_dim, local_offset, local_len) for shard in shards]
        segments.append(_reorder_zigzag_chunks(per_rank, cp_size, seq_dim))
        local_offset += local_len
    return torch.cat(segments, dim=seq_dim).contiguous()


def segmented_zigzag_index_table(
    segment_lengths, cp_rank: int, cp_size: int, device=None, dtype=torch.long
) -> torch.Tensor:
    """Materialize the local -> global index map as a 1-D lookup table.

    ``table[i]`` is the global position of local position ``i``. Built once per
    mask construction so that a consumer running INSIDE a compiled kernel -- a
    flex-attention ``mask_mod``, which is lowered into the attention Triton
    kernel and re-evaluated on every partial block, not just at build time --
    can remap an index with a single gather instead of re-deriving the segment
    arithmetic per element. A 1-D gather is the indexing pattern that path
    already relies on (``prompt_lengths[b]``).

    Costs 8 bytes per local position (512 KiB at a 128K doubled sequence with
    cp=4, against a 28 MiB BlockMask).

    Args:
        segment_lengths: Per-segment full lengths, in layout order.
        cp_rank: This rank's index within the CP group.
        cp_size: CP world size.
        device: Device to build the table on.
        dtype: Integer dtype of the table.

    Returns:
        ``[sum(segment_lengths) / cp_size]`` tensor of global positions.
    """
    local_len = sum(segment_lengths) // cp_size
    local_idx = torch.arange(local_len, device=device, dtype=dtype)
    return segmented_zigzag_local_to_global_idx(local_idx, segment_lengths, cp_rank, cp_size)
