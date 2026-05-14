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

"""Megatron Bridge for LLaDA2 MoE block-diffusion language models.

Converts between HuggingFace LLaDA2MoeForCausalLM and Megatron-Core GPTModel.

Architecture notes:
- Fused QKV: HF uses a single `query_key_value` linear (not separate q/k/v proj).
  Megatron's `linear_qkv` uses the same packed format [(nQ+2*nKV)*head_dim, hidden],
  so the mapping is a direct AutoMapping (no QKVMapping split needed).
- QK LayerNorm: per-head, named `query_layernorm` / `key_layernorm` in HF.
- Output projection: named `dense` in HF (not `o_proj`).
- Attention module: `attention` (not `self_attn`) in HF decoder layers.
- Shared expert: `shared_experts` (GatedMLP) inside `LLaDA2MoeSparseMoeBlock`.
- Dense layers: layers 0..first_k_dense_replace-1 use standard MLP, rest are MoE.
  Handled via moe_layer_freq = [0]*first_k_dense_replace + [1]*rest.
- Routing: sigmoid scores with expert bias correction.
"""

import torch
import torch.nn.functional as F
from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.diffusion.models.llada2.llada2_moe_provider import LLaDA2MoEModelProvider
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, register_bridge_implementation
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
)
from megatron.bridge.models.conversion.transformers_compat import rope_theta_from_hf
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


class LLaDA2MoEBridge(MegatronModelBridge):
    """HF LLaDA2MoeModelLM ↔ Megatron GPTModel bridge.

    Registered under the string name "LLaDA2MoeModelLM" (the actual class name
    in modeling_llada2_moe.py).  The model uses trust_remote_code=True and is
    not a standard transformers class.
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> LLaDA2MoEModelProvider:
        hf_config = hf_pretrained.config

        head_dim = getattr(hf_config, "head_dim", None) or (hf_config.hidden_size // hf_config.num_attention_heads)
        partial_rotary_factor = getattr(hf_config, "partial_rotary_factor", 1.0)
        first_k_dense = getattr(hf_config, "first_k_dense_replace", 0)

        # moe_layer_freq: 0 = dense, 1 = MoE
        moe_layer_freq = [0] * first_k_dense + [1] * (hf_config.num_hidden_layers - first_k_dense)

        # Shared expert intermediate size (scaled by num_shared_experts)
        num_shared = getattr(hf_config, "num_shared_experts", 0) or 0
        shared_expert_ffn = hf_config.moe_intermediate_size * num_shared if num_shared else None

        provider = LLaDA2MoEModelProvider(
            hidden_size=hf_config.hidden_size,
            ffn_hidden_size=hf_config.intermediate_size,
            num_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            num_query_groups=hf_config.num_key_value_heads,
            kv_channels=head_dim,
            vocab_size=hf_config.vocab_size,
            seq_length=hf_config.max_position_embeddings,
            layernorm_epsilon=hf_config.rms_norm_eps,
            rotary_base=rope_theta_from_hf(hf_config),
            rotary_percent=partial_rotary_factor,
            share_embeddings_and_output_weights=getattr(hf_config, "tie_word_embeddings", False),
            add_bias_linear=getattr(hf_config, "use_bias", False),
            add_qkv_bias=getattr(hf_config, "use_qkv_bias", False),
            normalization="RMSNorm",
            gated_linear_unit=True,
            activation_func=F.silu,
            qk_layernorm=True,
            hidden_dropout=0.0,
            attention_dropout=getattr(hf_config, "attention_dropout", 0.0),
            autocast_dtype=torch.bfloat16,
            # MoE
            num_moe_experts=hf_config.num_experts,
            moe_ffn_hidden_size=hf_config.moe_intermediate_size,
            moe_router_topk=hf_config.num_experts_per_tok,
            moe_layer_freq=moe_layer_freq,
            moe_grouped_gemm=True,
            moe_router_pre_softmax=False,
            moe_router_score_function="sigmoid",
            moe_router_enable_expert_bias=True,
            moe_router_load_balancing_type="aux_loss",
            moe_aux_loss_coeff=1e-3,
            moe_token_dispatcher_type="alltoall",
            moe_permute_fusion=True,
            moe_shared_expert_intermediate_size=shared_expert_ffn,
            # n_group / topk_group for grouped top-k routing
            moe_router_num_groups=getattr(hf_config, "n_group", None),
            moe_router_group_topk=getattr(hf_config, "topk_group", None),
            moe_router_topk_scaling_factor=getattr(hf_config, "routed_scaling_factor", 1.0),
            # HF config reference for LLaDA2CoreAttention RoPE/partial RoPE setup
            hf_config=hf_config,
        )

        # Store for use in mapping_registry
        self._hf_config = hf_config
        return provider

    def build_conversion_tasks(self, hf_pretrained, megatron_model):
        self._hf_config = hf_pretrained.config
        return super().build_conversion_tasks(hf_pretrained, megatron_model)

    def mapping_registry(self) -> MegatronMappingRegistry:
        hf_config = getattr(self, "_hf_config", None)
        mapping_list = []

        # -----------------------------------------------------------------
        # Global (non-layer) mappings
        # -----------------------------------------------------------------
        global_mappings = {
            "embedding.word_embeddings.weight": "model.word_embeddings.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
        }
        for meg, hf in global_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=meg, hf_param=hf))

        # -----------------------------------------------------------------
        # Per-layer attention mappings (all layers share these)
        # -----------------------------------------------------------------
        # Fused QKV: direct copy, both sides use packed [(nQ+2*nKV)*head_dim, hidden]
        attn_mappings = {
            "decoder.layers.*.self_attention.linear_qkv.weight": "model.layers.*.attention.query_key_value.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.attention.query_layernorm.weight",
            "decoder.layers.*.self_attention.k_layernorm.weight": "model.layers.*.attention.key_layernorm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.attention.dense.weight",
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
        }
        for meg, hf in attn_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=meg, hf_param=hf))

        # QKV bias (only if use_qkv_bias=True; AutoMapping will silently skip missing keys)
        mapping_list.append(
            AutoMapping(
                megatron_param="decoder.layers.*.self_attention.linear_qkv.bias",
                hf_param="model.layers.*.attention.query_key_value.bias",
            )
        )

        # Output projection bias (only if use_bias=True)
        mapping_list.append(
            AutoMapping(
                megatron_param="decoder.layers.*.self_attention.linear_proj.bias",
                hf_param="model.layers.*.attention.dense.bias",
            )
        )

        # -----------------------------------------------------------------
        # Dense FFN layers (layers 0..first_k_dense-1)
        # -----------------------------------------------------------------
        # These layers use a standard GatedMLP (gate_proj / up_proj / down_proj).
        # The Megatron side uses linear_fc1 (gated) / linear_fc2.
        # We also map the layer-norm fused into linear_fc1 for dense layers.
        mapping_list.extend(
            [
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.layers.*.mlp.gate_proj.weight",
                    up="model.layers.*.mlp.up_proj.weight",
                ),
                AutoMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc2.weight",
                    hf_param="model.layers.*.mlp.down_proj.weight",
                ),
                AutoMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
                    hf_param="model.layers.*.post_attention_layernorm.weight",
                ),
            ]
        )

        # -----------------------------------------------------------------
        # MoE FFN layers (layers first_k_dense..num_layers-1)
        # -----------------------------------------------------------------
        # Router
        mapping_list.append(
            AutoMapping(
                megatron_param="decoder.layers.*.mlp.router.weight",
                hf_param="model.layers.*.mlp.gate.weight",
            )
        )
        # Expert bias (sigmoid routing correction)
        mapping_list.append(
            AutoMapping(
                megatron_param="decoder.layers.*.mlp.router.expert_bias",
                hf_param="model.layers.*.mlp.gate.expert_bias",
            )
        )

        # Routed experts — TEGroupedMLP (fused) format
        mapping_list.extend(
            [
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                    gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="model.layers.*.mlp.experts.*.up_proj.weight",
                ),
                AutoMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc2.weight*",
                    hf_param="model.layers.*.mlp.experts.*.down_proj.weight",
                ),
                # SequentialMLP format (used by quantization)
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight",
                    gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="model.layers.*.mlp.experts.*.up_proj.weight",
                ),
                AutoMapping(
                    megatron_param="decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight",
                    hf_param="model.layers.*.mlp.experts.*.down_proj.weight",
                ),
            ]
        )

        # Shared expert (GatedMLP)
        if getattr(hf_config, "num_shared_experts", 0):
            mapping_list.extend(
                [
                    GatedMLPMapping(
                        megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                        gate="model.layers.*.mlp.shared_experts.gate_proj.weight",
                        up="model.layers.*.mlp.shared_experts.up_proj.weight",
                    ),
                    AutoMapping(
                        megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc2.weight",
                        hf_param="model.layers.*.mlp.shared_experts.down_proj.weight",
                    ),
                ]
            )

        return MegatronMappingRegistry(*mapping_list)


# Register under the actual trust_remote_code class name
register_bridge_implementation(
    source="LLaDA2MoeModelLM",
    target=GPTModel,
    bridge_class=LLaDA2MoEBridge,
)
