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

"""Block-aware CP through the real attention layer (CPU, fp32, gloo).

``test_cp_block_aware`` validates the mask and the layout in isolation. This
file runs ``NemotronLabsDiffusionAttention._asymmetric_semi_ar_forward`` itself
with ``DIFFU_CP_BLOCK_AWARE`` on, over a real process group, and checks that the
per-rank outputs reassemble into the single-rank result.

It exists because three things only compose inside that method, and a mask-level
test cannot see any of them:

1. The CLEAN-ONLY gather. Only the clean K/V is all-gathered; the noisy K/V is
   concatenated in locally.
2. The RoPE position ids. Q rows are [noisy zigzag | clean zigzag] and K rows
   are [noisy zigzag local | clean FULL], so NEITHER is a slice of one global
   zigzag. The local-Q path rotates K with the full-length position ids; doing
   that here would rotate K's noisy half at the wrong positions -- silently
   wrong logprobs, not a crash. This test is aimed squarely at that.
3. The mask dispatch and the skipped output scatter.

``_asymmetric_semi_ar_forward`` is a pure function of (q, k, v) plus config and
metadata -- it carries no learned weights -- so the single-rank reference can be
computed in the parent without any weight sync.
"""

import os
import socket
import types
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

BLOCK_SIZE = 16
NOISY_LENGTH = 128
CLEAN_LENGTH = 128
NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 16
BATCH = 2

PROMPT_LENGTHS = [13, 21]
NOISY_VALID_LENGTHS = [96, 55]
CLEAN_LENGTHS = [109, 76]


def _make_config(cp_size: int):
    from megatron.core.transformer.transformer_config import TransformerConfig

    seq_len = NOISY_LENGTH + CLEAN_LENGTH
    hf_text_config = types.SimpleNamespace(
        max_position_embeddings=seq_len,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "llama_4_scaling_beta": 0.1,
            "original_max_position_embeddings": seq_len,
        },
        num_attention_heads=NUM_HEADS,
        hidden_size=NUM_HEADS * HEAD_DIM,
    )
    cfg = TransformerConfig(
        num_layers=1,
        hidden_size=NUM_HEADS * HEAD_DIM,
        num_attention_heads=NUM_HEADS,
        num_query_groups=NUM_KV_HEADS,
        kv_channels=HEAD_DIM,
        context_parallel_size=cp_size,
        tensor_model_parallel_size=1,
        use_cpu_initialization=True,
    )
    cfg.seq_length = seq_len
    cfg.block_size = BLOCK_SIZE
    cfg.apply_llama4_style_query_key_layer_scaling = True
    cfg.hf_config = types.SimpleNamespace(text_config=hf_text_config)
    # True takes the plain-dropout branch in the forward; the False branch calls
    # tensor_parallel.get_cuda_rng_tracker(), which needs CUDA.
    cfg.sequence_parallel = True
    cfg.apply_query_key_layer_scaling = False
    cfg.attention_dropout = 0.0
    return cfg


def _make_attention(cp_size: int):
    from megatron.core.transformer.enums import AttnMaskType

    from megatron.bridge.diffusion.models.common import (
        nemotron_labs_diffusion_attention as attn_mod,
    )

    pg_collection = MagicMock()
    pg_collection.tp = MagicMock()
    pg_collection.tp.size.return_value = 1
    with patch.object(attn_mod, "compute_block_bias", return_value=MagicMock()):
        attn = attn_mod.NemotronLabsDiffusionAttention(
            _make_config(cp_size), 1, AttnMaskType.causal, "self", pg_collection=pg_collection
        )
    attn.set_asymmetric_ar_metadata(
        noisy_length=NOISY_LENGTH,
        clean_length=CLEAN_LENGTH,
        noisy_response_offset=0,
        prompt_lengths=torch.tensor(PROMPT_LENGTHS),
        response_lengths=torch.tensor(NOISY_VALID_LENGTHS),
        noisy_valid_lengths=torch.tensor(NOISY_VALID_LENGTHS),
        clean_lengths=torch.tensor(CLEAN_LENGTHS),
    )
    return attn, attn_mod


# fp32, not fp64: the layer routes through the torch.compile'd
# fused_flex_attention, and inductor's CPU flex lowering supports only
# float/float16/bfloat16. (The mask-level suite can use fp64 because it calls
# eager flex_attention directly.) Tolerance is sized for fp32 summation-order
# differences between the two KV layouts; a wrong RoPE or a wrong mask moves the
# output by O(1), so this stays sharp for what it tests.
_DTYPE = torch.float32
_ATOL = 1e-5


def _qkv():
    """Deterministic q/k/v in [sq, b, np, hn] layout, identical on every rank."""
    gen = torch.Generator().manual_seed(1234)
    seq = NOISY_LENGTH + CLEAN_LENGTH
    shape = (seq, BATCH, NUM_HEADS, HEAD_DIM)
    kv_shape = (seq, BATCH, NUM_KV_HEADS, HEAD_DIM)
    q = torch.randn(*shape, dtype=_DTYPE, generator=gen)
    k = torch.randn(*kv_shape, dtype=_DTYPE, generator=gen)
    v = torch.randn(*kv_shape, dtype=_DTYPE, generator=gen)
    return q, k, v


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank: int, world_size: int, port: int, out_q) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        from megatron.bridge.diffusion.common.cp_utils import segmented_zigzag_slice

        attn, attn_mod = _make_attention(world_size)
        # Flip the flag on the module object: it is read from the environment at
        # import time, and the module is already imported here.
        attn_mod._CP_BLOCK_AWARE = True

        # The forward asks megatron for the CP group; point it at this gloo world
        # instead of initializing the whole model-parallel state for one method.
        group = dist.group.WORLD
        with patch.object(
            attn_mod.parallel_state, "get_context_parallel_group", return_value=group
        ), patch.object(
            attn_mod.parallel_state, "get_context_parallel_rank", return_value=rank
        ):
            q, k, v = _qkv()
            segments = (NOISY_LENGTH, CLEAN_LENGTH)
            q_local = segmented_zigzag_slice(q, segments, rank, world_size, seq_dim=0)
            k_local = segmented_zigzag_slice(k, segments, rank, world_size, seq_dim=0)
            v_local = segmented_zigzag_slice(v, segments, rank, world_size, seq_dim=0)
            out = attn._asymmetric_semi_ar_forward(q_local, k_local, v_local)
        out_q.put((rank, out.detach()))
    except Exception as exc:  # noqa: BLE001
        import traceback

        out_q.put((rank, f"EXC {exc!r}\n{traceback.format_exc()}"))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("cp_size", [2, 4])
def test_block_aware_layer_matches_no_cp(cp_size: int) -> None:
    from megatron.bridge.diffusion.common.cp_utils import reorder_segmented_zigzag_shards

    # Single-rank reference: cp_size == 1 never touches parallel_state.
    attn_ref, _ = _make_attention(1)
    q, k, v = _qkv()
    out_ref = attn_ref._asymmetric_semi_ar_forward(q, k, v).detach()
    assert torch.isfinite(out_ref).all(), "reference output is not finite"

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(target=_worker, args=(r, cp_size, port, queue)) for r in range(cp_size)
    ]
    for p in procs:
        p.start()
    results = [queue.get(timeout=300) for _ in range(cp_size)]
    for p in procs:
        p.join(timeout=300)

    shards = [None] * cp_size
    for rank, payload in results:
        assert not isinstance(payload, str), f"rank {rank} raised:\n{payload}"
        shards[rank] = payload

    expected_local = (NOISY_LENGTH + CLEAN_LENGTH) // cp_size
    for rank, shard in enumerate(shards):
        assert shard.shape[0] == expected_local, (
            f"rank {rank}: expected {expected_local} local rows, got {shard.shape[0]}"
        )

    regathered = reorder_segmented_zigzag_shards(
        shards, (NOISY_LENGTH, CLEAN_LENGTH), cp_size, seq_dim=0
    )
    assert torch.allclose(regathered, out_ref, rtol=_ATOL, atol=_ATOL), (
        f"cp={cp_size}: block-aware layer output differs from the single-rank "
        f"reference (max diff {float((regathered - out_ref).abs().max()):.3e})"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
