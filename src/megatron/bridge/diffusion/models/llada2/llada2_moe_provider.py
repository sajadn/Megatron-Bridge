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

"""LLaDA2MoEModelProvider: GPTModel provider with LLaDA2CoreAttention injected."""

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Union

from megatron.bridge.diffusion.models.llada2.llada2_attention import LLaDA2CoreAttention
from megatron.bridge.models.gpt_provider import GPTModelProvider, ModuleSpec


def _llada2_block_spec(config: "LLaDA2MoEModelProvider", vp_stage=None):
    """Build a per-layer TransformerBlockSubmodules for LLaDA2, respecting moe_layer_freq.

    Uses get_gpt_decoder_block_spec so that layers where moe_layer_freq==0 get a
    standard dense MLP spec and layers where moe_layer_freq==1 get an MoE spec.
    LLaDA2CoreAttention is injected as core_attention into every layer spec.
    """
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec

    block_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=True, vp_stage=vp_stage)
    for layer_spec in block_spec.layer_specs:
        sa = getattr(getattr(layer_spec, "submodules", None), "self_attention", None)
        if sa is not None and hasattr(sa, "submodules"):
            sa.submodules.core_attention = LLaDA2CoreAttention
    return block_spec


@dataclass
class LLaDA2MoEModelProvider(GPTModelProvider):
    """GPTModel provider for LLaDA2 MoE block-diffusion models.

    Injects LLaDA2CoreAttention as the core_attention submodule so that:
    - Partial RoPE is handled inside the attention (not by Megatron's SelfAttention)
    - A per-layer block-diagonal mask can be set before each inference call
    - Standard Megatron infrastructure (QK LayerNorm, MoE FFN, etc.) is used unchanged

    position_embedding_type must be "none" so Megatron's SelfAttention does not
    apply its own positional embeddings before calling our core_attention.

    moe_layer_freq must be a list (e.g. [0]*first_k_dense + [1]*rest) so that
    get_gpt_decoder_block_spec produces the correct per-layer dense/MoE specs.
    """

    position_embedding_type: str = "none"

    # Use the per-layer block spec factory so moe_layer_freq is honoured
    transformer_layer_spec: Union[ModuleSpec, Callable] = _llada2_block_spec

    # HF config reference — read by LLaDA2CoreAttention for partial_rotary_factor / rope_theta
    hf_config: Optional[Any] = None

    # MoE fields not in GPTModelProvider base (mirroring GLM45VModelProvider / NemotronHModelProvider)
    moe_ffn_hidden_size: Optional[int] = None
    moe_router_topk: int = 8
    moe_router_num_groups: Optional[int] = None
    moe_router_group_topk: Optional[int] = None
    moe_router_score_function: str = "sigmoid"
    moe_router_enable_expert_bias: bool = True
    moe_router_topk_scaling_factor: float = 1.0
    moe_router_dtype: str = "fp32"
    moe_router_load_balancing_type: str = "aux_loss"
    moe_router_pre_softmax: bool = False
    moe_token_dispatcher_type: str = "alltoall"
    moe_permute_fusion: bool = True
    moe_aux_loss_coeff: float = 1e-3
    moe_layer_freq: Optional[Union[int, List[int]]] = None
    moe_shared_expert_intermediate_size: Optional[int] = None
