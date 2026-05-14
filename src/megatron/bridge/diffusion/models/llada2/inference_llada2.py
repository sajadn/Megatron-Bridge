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

"""Block-diffusion generation for LLaDA2 MoE loaded into a Megatron GPTModel.

The generation algorithm mirrors modeling_llada2_moe.py::generate() but drives
a Megatron GPTModel rather than the HuggingFace model directly.

Key steps per block:
  1. Feed the growing prefix [prompt | unmasked_gen | MASK...MASK] through the
     Megatron model with a block-diagonal causal attention mask.
  2. Extract logits over the last block_length positions.
  3. Sample predicted tokens with temperature / top-k / top-p.
  4. Transfer the most confident tokens from MASK→real according to a uniform
     schedule (num_transfer_tokens_per_step = block_length // steps).
  5. Repeat until all positions in the block are filled or steps are exhausted.
  6. Advance to the next block.

Usage::

    from megatron.bridge.diffusion.models.llada2.inference_llada2 import generate_block_diffusion

    output_ids = generate_block_diffusion(
        model=megatron_gpt_model,
        input_ids=prompt_tensor,         # [1, prompt_len]
        gen_length=256,
        block_length=32,
        steps=32,
        mask_token_id=156895,
    )
"""

from typing import Optional

import torch
import torch.nn.functional as F

from megatron.bridge.diffusion.models.llada2.llada2_attention import LLaDA2CoreAttention


# ---------------------------------------------------------------------------
# Model unwrapping
# ---------------------------------------------------------------------------


def _unwrap(model):
    """Unwrap Float16Module, DDP, or similar wrappers to get the raw GPTModel."""
    if hasattr(model, "module"):
        return _unwrap(model.module)
    if hasattr(model, "language_model"):
        return _unwrap(model.language_model)
    return model


def _get_llada2_attentions(model) -> list[LLaDA2CoreAttention]:
    """Return all LLaDA2CoreAttention instances from the Megatron decoder."""
    m = _unwrap(model)
    attns = []
    for layer in m.decoder.layers:
        ca = layer.self_attention.core_attention
        assert isinstance(ca, LLaDA2CoreAttention), (
            f"Expected LLaDA2CoreAttention but found {type(ca).__name__}. "
            "Make sure the model was built with LLaDA2MoEModelProvider."
        )
        attns.append(ca)
    return attns


# ---------------------------------------------------------------------------
# Mask / position-id helpers
# ---------------------------------------------------------------------------


def _build_block_attention_mask(
    total_length: int,
    block_length: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Block-diagonal causal attention mask of shape [1, 1, total_length, total_length].

    Tokens within the same block attend to each other (bidirectional).
    Tokens attend causally (left-to-right) across blocks.
    Returns additive mask: 0.0 = attend, -inf = masked out.
    """
    num_blocks = (total_length + block_length - 1) // block_length
    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device))
    attn = (
        block_mask.repeat_interleave(block_length, dim=0)
        .repeat_interleave(block_length, dim=1)[:total_length, :total_length]
        .unsqueeze(0)
        .unsqueeze(0)
        .bool()
    )
    return torch.where(
        attn, torch.tensor(0.0, dtype=dtype, device=device), torch.tensor(float("-inf"), dtype=dtype, device=device)
    )


def _set_block_mask(model, mask: Optional[torch.Tensor]):
    """Set (or clear) the block-diagonal mask on every LLaDA2CoreAttention layer."""
    for attn in _get_llada2_attentions(model):
        attn.set_block_mask(mask)


def _set_position_ids(model, position_ids: Optional[torch.Tensor]):
    """Broadcast position_ids to every LLaDA2CoreAttention layer."""
    for attn in _get_llada2_attentions(model):
        attn.set_position_ids(position_ids)


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def _sample_with_temperature(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample from logits; return (token_ids, probabilities) both [B, L]."""
    B, L, V = logits.shape
    flat = logits.reshape(-1, V)

    # Greedy
    if temperature == 0.0 and (top_k is None or top_k <= 0) and (top_p is None or top_p >= 1.0):
        probs = F.softmax(flat, dim=-1)
        tokens = flat.argmax(dim=-1, keepdim=True)
        token_probs = probs.gather(-1, tokens)
        return tokens.squeeze(-1).view(B, L), token_probs.squeeze(-1).view(B, L)

    if temperature > 0 and temperature != 1.0:
        flat = flat / temperature
    if top_k is not None and top_k > 0:
        vals, _ = torch.topk(flat, top_k)
        flat = flat.masked_fill(flat < vals[..., -1:], float("-inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(flat, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum_probs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        mask = torch.zeros_like(flat, dtype=torch.bool).scatter(-1, sorted_idx, remove)
        flat = flat.masked_fill(mask, float("-inf"))

    probs = F.softmax(flat, dim=-1)
    tokens = torch.multinomial(probs, num_samples=1)
    token_probs = probs.gather(-1, tokens)
    return tokens.squeeze(-1).view(B, L), token_probs.squeeze(-1).view(B, L)


def _get_transfer_schedule(block_length: int, steps: int) -> torch.Tensor:
    """Uniform schedule: how many tokens to unmask at each step (sums to block_length)."""
    if steps == 0:
        return torch.tensor([], dtype=torch.int64)
    base = block_length // steps
    rem = block_length % steps
    sched = torch.full((steps,), base, dtype=torch.int64)
    sched[:rem] += 1
    return sched


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_block_diffusion(
    model,
    input_ids: torch.Tensor,
    gen_length: int = 512,
    block_length: int = 32,
    steps: int = 32,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    threshold: float = 0.95,
    eos_early_stop: bool = False,
    eos_token_id: Optional[int] = None,
    mask_token_id: int = 156895,
    minimal_topk: int = 1,
) -> torch.Tensor:
    """Generate tokens from a masked-diffusion LLaDA2 model loaded into Megatron.

    Args:
        model:           Megatron GPTModel built with LLaDA2MoEModelProvider.
        input_ids:       Prompt token ids [B, prompt_len].
        gen_length:      Maximum tokens to generate.
        block_length:    Number of tokens processed in parallel per block.
        steps:           Denoising steps per block.
        temperature:     Sampling temperature (0.0 = greedy).
        top_k / top_p:   Nucleus / top-k filtering.
        threshold:       Confidence threshold for accepting a token (0.95 default).
        eos_early_stop:  Stop when EOS is fully confirmed in the generated region.
        eos_token_id:    EOS token id for early stopping.
        mask_token_id:   Mask placeholder token id (LLaDA2 default: 156895).
        minimal_topk:    Clamp steps ≤ gen_length // minimal_topk.

    Returns:
        Token ids [B, prompt_len + generated_len].
    """
    steps = min(steps, gen_length // max(minimal_topk, 1))

    device = input_ids.device
    B, prompt_len = input_ids.shape

    num_blocks = (prompt_len + gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length

    # Full sequence: prompt tokens + MASK for gen region
    x = torch.full((B, total_length), mask_token_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids

    # Build and install block-diagonal attention mask (shared across all blocks)
    attn_mask_full = _build_block_attention_mask(total_length, block_length, device)
    _set_block_mask(model, attn_mask_full)

    # Full position ids [B, total_length]
    position_ids_full = torch.arange(total_length, device=device).unsqueeze(0).expand(B, -1)

    # Precompute transfer schedule (same for every block)
    transfer_schedule = _get_transfer_schedule(block_length, steps)

    prefill_blocks = prompt_len // block_length  # blocks fully covered by prompt

    for block_idx in range(prefill_blocks, num_blocks):
        window_end = (block_idx + 1) * block_length
        cur_x = x[:, :window_end]
        cur_pos = position_ids_full[:, :window_end]

        for step_idx in range(steps):
            active_block = cur_x[:, -block_length:]
            if (active_block != mask_token_id).all():
                break

            # Install position ids for this forward pass
            _set_position_ids(model, cur_pos)

            # Forward pass through Megatron model
            # attention_mask=None: our LLaDA2CoreAttention uses the stored block mask
            output = model(input_ids=cur_x, position_ids=cur_pos, attention_mask=None)
            logits = output if isinstance(output, torch.Tensor) else output[0]  # [B, window_end, vocab]

            block_logits = logits[:, -block_length:, :]  # [B, block_len, vocab]

            x0, x0_p = _sample_with_temperature(block_logits, temperature=temperature, top_k=top_k, top_p=top_p)

            # Confidence-threshold transfer
            num_to_transfer = int(transfer_schedule[step_idx].item())
            active_mask = active_block == mask_token_id
            confidence = torch.where(active_mask, x0_p, torch.full_like(x0_p, float("-inf")))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for b in range(B):
                high_conf = confidence[b] > threshold
                if high_conf.sum() >= num_to_transfer:
                    transfer_index[b] = high_conf
                else:
                    n = min(num_to_transfer, int(active_mask[b].sum().item()))
                    if n > 0:
                        _, idx = torch.topk(confidence[b], k=n)
                        transfer_index[b, idx] = True

            cur_x[:, -block_length:][transfer_index] = x0[transfer_index]

            # EOS early stop within the active block
            if eos_early_stop and eos_token_id is not None:
                if (cur_x[0] == eos_token_id).any():
                    eos_positions = (cur_x[0] == eos_token_id).nonzero(as_tuple=True)[0]
                    eos_pos = int(eos_positions[0].item())
                    gen_region = cur_x[0, prompt_len:eos_pos]
                    if (gen_region != mask_token_id).all():
                        x[:, :window_end] = cur_x
                        _set_block_mask(model, None)
                        _set_position_ids(model, None)
                        return x[:, : eos_pos + 1]

        x[:, :window_end] = cur_x

        # Block-level EOS stop
        if eos_token_id is not None and (x[0, prompt_len:window_end] == eos_token_id).any():
            break

    # Cleanup: clear stored masks/position_ids from attention layers
    _set_block_mask(model, None)
    _set_position_ids(model, None)

    # Trim to prompt + gen_length; find first EOS
    generated = x[:, : prompt_len + gen_length]
    if eos_token_id is not None:
        for b in range(B):
            eos_pos = (generated[b, prompt_len:] == eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                stop = int(eos_pos[0].item()) + 1
                return generated[:, : prompt_len + stop]

    return generated
