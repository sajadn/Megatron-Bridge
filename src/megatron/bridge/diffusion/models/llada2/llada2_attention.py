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

"""LLaDA2CoreAttention: SDPA-based core attention for LLaDA2 block-diffusion inference.

Replaces TEDotProductAttention in the Megatron GPTModel layer spec.

Key differences from standard Megatron attention:
- Partial RoPE: only first `rotary_dim` of `head_dim` dims are rotated
- Block-diagonal causal mask: stored as state, set once before the generation loop
- GQA: KV heads expanded to match Q heads
- Bidirectional (is_causal=False) always — causal structure comes from the block mask

The model provider sets position_embedding_type="none" so Megatron's SelfAttention
does NOT apply RoPE upstream; this module handles RoPE itself.

Megatron's SelfAttention DOES apply QK LayerNorm (when qk_layernorm=True) before
calling core_attention, so this module receives Q and K with layer-norm already applied.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import divide
from torch import Tensor


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------


def _rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_partial_rotary(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    rotary_dim: int,
) -> tuple[Tensor, Tensor]:
    """Apply RoPE to the first `rotary_dim` dimensions; leave the rest untouched."""
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_rot = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (_rotate_half(k_rot) * sin)
    return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)


def _repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """Expand KV heads for GQA: [B, nKV, S, D] → [B, nQ, S, D]."""
    if n_rep == 1:
        return x
    B, nKV, S, D = x.shape
    return x[:, :, None, :, :].expand(B, nKV, n_rep, S, D).reshape(B, nKV * n_rep, S, D)


class LLaDA2RotaryEmbedding(nn.Module):
    """Standard (non-scaled) RoPE for LLaDA2."""

    def __init__(self, rotary_dim: int, rope_theta: float, max_seq_len: int, device=None):
        super().__init__()
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len

    @torch.no_grad()
    def forward(self, x: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        # position_ids: [B, S] — use first batch to get cos/sin, then broadcast
        inv = self.inv_freq[None, :, None].float().to(x.device)  # [1, D/2, 1]
        pos = position_ids[:, None, :].float()  # [B, 1, S]
        freqs = (inv @ pos).transpose(1, 2)  # [B, S, D/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [B, S, D]
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


# ---------------------------------------------------------------------------
# LLaDA2CoreAttention
# ---------------------------------------------------------------------------


class LLaDA2CoreAttention(MegatronModule):
    """Core attention module for LLaDA2 block-diffusion inference.

    Replaces TEDotProductAttention in the layer spec.  Receives Q, K, V that
    have already gone through QK LayerNorm (applied by Megatron's SelfAttention
    when qk_layernorm=True).

    State for inference:
        _block_attn_mask: additive 4D mask [1, 1, S, S] (0.0 / -inf)
                          set once before the generation loop via set_block_mask().
                          None = fully bidirectional SDPA (no mask).
        _position_ids:    [1, S] position ids for the current forward pass,
                          set per-step via set_position_ids().
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float = None,
        softmax_scale: float = None,
        cp_comm_type: str = None,
        pg_collection: ProcessGroupCollection = None,
    ):
        super().__init__(config=config)
        self.config = config

        assert config.context_parallel_size == 1, "LLaDA2CoreAttention does not support context parallelism."

        self.layer_number = max(1, layer_number)

        projection_size = config.kv_channels * config.num_attention_heads
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp"])
        world_size = pg_collection.tp.size()

        self.hidden_size_per_partition = divide(projection_size, world_size)
        self.hidden_size_per_attention_head = divide(projection_size, config.num_attention_heads)
        self.num_attention_heads_per_partition = divide(config.num_attention_heads, world_size)
        self.num_query_groups_per_partition = divide(config.num_query_groups, world_size)
        self.n_rep = self.num_attention_heads_per_partition // self.num_query_groups_per_partition

        # Partial RoPE: rotary_dim from the HF config attached to TransformerConfig
        hf_config = getattr(config, "hf_config", None)
        if hf_config is not None:
            head_dim = getattr(hf_config, "head_dim", None) or (hf_config.hidden_size // hf_config.num_attention_heads)
            partial_rotary_factor = getattr(hf_config, "partial_rotary_factor", 1.0)
            self.rotary_dim = int(head_dim * partial_rotary_factor)
            rope_theta = getattr(hf_config, "rope_theta", 10000.0)
            max_seq_len = getattr(hf_config, "max_position_embeddings", 32768)
        else:
            # Fallback: full RoPE using kv_channels as head_dim
            self.rotary_dim = self.hidden_size_per_attention_head
            rope_theta = getattr(config, "rotary_base", 10000.0)
            max_seq_len = getattr(config, "seq_length", 32768)

        self.rope = LLaDA2RotaryEmbedding(self.rotary_dim, rope_theta, max_seq_len)

        if softmax_scale is None:
            self.softmax_scale = 1.0 / math.sqrt(self.hidden_size_per_attention_head)
        else:
            self.softmax_scale = softmax_scale

        self.attention_dropout = nn.Dropout(
            config.attention_dropout if attention_dropout is None else attention_dropout
        )

        # Inference state
        self._block_attn_mask: Optional[Tensor] = None
        self._position_ids: Optional[Tensor] = None

    # ------------------------------------------------------------------
    # State setters (called by inference_llada2.py before each forward)
    # ------------------------------------------------------------------

    def set_block_mask(self, mask: Optional[Tensor]):
        """Set the additive block-diagonal attention mask [1, 1, S, S] for inference."""
        self._block_attn_mask = mask

    def set_position_ids(self, position_ids: Tensor):
        """Set position ids [B, S] for the upcoming forward pass."""
        self._position_ids = position_ids

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        query: Tensor,  # [S, B, nQ, head_dim]  (Megatron: seq-first)
        key: Tensor,  # [S, B, nKV, head_dim]
        value: Tensor,  # [S, B, nKV, head_dim]
        attention_mask: Tensor = None,
        attn_mask_type: AttnMaskType = None,
        attention_bias: Tensor = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tensor:
        assert packed_seq_params is None, "LLaDA2CoreAttention does not support packed sequences."

        S, B = query.shape[:2]

        # [S, B, nH, D] → [B, nH, S, D]
        q = query.permute(1, 2, 0, 3)
        k = key.permute(1, 2, 0, 3)
        v = value.permute(1, 2, 0, 3)

        # Position ids: use stored or fall back to 0..S-1
        if self._position_ids is not None:
            pos_ids = self._position_ids.to(q.device)
        else:
            pos_ids = torch.arange(S, device=q.device).unsqueeze(0).expand(B, -1)

        cos, sin = self.rope(q, pos_ids)  # [B, S, rotary_dim]
        cos = cos.unsqueeze(1)  # [B, 1, S, rotary_dim]
        sin = sin.unsqueeze(1)

        q, k = _apply_partial_rotary(q, k, cos, sin, self.rotary_dim)

        # GQA: expand KV heads
        k = _repeat_kv(k, self.n_rep)
        v = _repeat_kv(v, self.n_rep)

        # Scale queries
        q = q * self.softmax_scale

        # SDPA — block mask is an additive bias [1, 1, S, S]
        attn_mask = self._block_attn_mask
        if attn_mask is not None:
            attn_mask = attn_mask[:, :, :S, :S].to(q.dtype)

        context = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attention_dropout.p if self.training else 0.0,
            is_causal=False,
            scale=1.0,  # already scaled q above
        )

        # [B, nQ, S, D] → [S, B, hidden_size_per_partition]
        context = context.permute(2, 0, 1, 3).contiguous()
        context = context.view(S, B, self.hidden_size_per_partition)

        return context
