#!/usr/bin/env python3
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

"""
LLaDA2 MoE block-diffusion inference script.

Runs text generation over one or more prompts using a Megatron-format
LLaDA2 checkpoint.  Supports two loading paths:

  --megatron-path   Load from a converted Megatron checkpoint (requires prior
                    conversion via the LLaDA2MoEBridge).
  --hf-model-only   Load the HuggingFace model directly and convert weights
                    on-the-fly (useful for quick testing without a pre-converted
                    checkpoint).

Examples:

  # Convert HF weights on-the-fly and run inference (single GPU):
  python examples/diffusion/recipes/llada2/inference_llada2.py \\
      --hf-model inclusionAI/LLaDA2.1-mini \\
      --hf-model-only \\
      --prompts "What is the capital of France?"

  # Load from a pre-converted Megatron checkpoint (multi-GPU, TP=4):
  torchrun --nproc_per_node=4 examples/diffusion/recipes/llada2/inference_llada2.py \\
      --megatron-path /path/to/checkpoints/llada2_8b \\
      --hf-model inclusionAI/LLaDA2.1-mini \\
      --tp 4 \\
      --prompts "Explain quantum entanglement." \\
      --gen-length 256 --block-length 32 --steps-per-block 32

  # Multiple prompts with custom sampling:
  python examples/diffusion/recipes/llada2/inference_llada2.py \\
      --hf-model inclusionAI/LLaDA2.1-mini \\
      --hf-model-only \\
      --prompts "Prompt one" --prompts "Prompt two" \\
      --temperature 0.5 --top-p 0.9 --threshold 0.9
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from megatron.bridge.diffusion.conversion.llada2.llada2_moe_bridge import LLaDA2MoEBridge
from megatron.bridge.diffusion.models.llada2.inference_llada2 import generate_block_diffusion
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="LLaDA2 MoE block-diffusion inference")

    # ---- model source ----
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument(
        "--megatron-path",
        type=str,
        default=None,
        help="Path to a pre-converted Megatron-Bridge checkpoint directory.",
    )
    source.add_argument(
        "--hf-model-only",
        action="store_true",
        default=False,
        help="Convert HF weights on-the-fly (no Megatron checkpoint needed). Useful for quick testing.",
    )

    parser.add_argument(
        "--hf-model",
        type=str,
        required=True,
        help="HuggingFace model ID or local path (used for config, tokenizer, and "
        "weight source when --hf-model-only is set).",
    )

    # ---- prompts ----
    parser.add_argument(
        "--prompts",
        type=str,
        action="append",
        required=True,
        help="Input prompt(s). Repeat the flag for multiple prompts.",
    )

    # ---- generation ----
    parser.add_argument("--gen-length", type=int, default=256, help="Tokens to generate per prompt.")
    parser.add_argument("--block-length", type=int, default=32, help="Parallel denoising block size.")
    parser.add_argument("--steps-per-block", type=int, default=32, help="Denoising refinement steps per block.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 = greedy).")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k filtering (disabled by default).")
    parser.add_argument("--top-p", type=float, default=None, help="Nucleus sampling threshold (disabled by default).")
    parser.add_argument(
        "--threshold", type=float, default=0.95, help="Confidence threshold for unmasking (default: 0.95)."
    )
    parser.add_argument(
        "--mask-token-id", type=int, default=156895, help="Mask token ID for LLaDA2 (default: 156895)."
    )
    parser.add_argument(
        "--eos-early-stop", action="store_true", default=False, help="Stop generation early when EOS is confirmed."
    )

    # ---- parallelism ----
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism degree (must match saved checkpoint).")
    parser.add_argument("--seq-length", type=int, default=32768, help="Maximum sequence length for Megatron model.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_from_megatron(args, hf_pretrained):
    """Load model from a pre-converted Megatron checkpoint."""
    bridge = LLaDA2MoEBridge()
    model_provider = bridge.provider_bridge(hf_pretrained)
    model_provider.tensor_model_parallel_size = args.tp
    model_provider.pipeline_model_parallel_size = 1
    model_provider.pipeline_dtype = torch.bfloat16
    model_provider.params_dtype = torch.bfloat16
    model_provider.seq_length = args.seq_length
    model_provider.finalize()
    model_provider.initialize_model_parallel(seed=0)

    from megatron.bridge.training.model_load_save import build_and_load_model

    megatron_models = build_and_load_model(
        checkpoint_path=args.megatron_path,
        model_cfg=model_provider,
        skip_temp_dist_context=True,
    )
    model = megatron_models[0] if isinstance(megatron_models, list) else megatron_models
    return model.cuda().eval()


def load_from_hf(args, hf_pretrained):
    """Convert HF weights on-the-fly into a Megatron model."""
    bridge = LLaDA2MoEBridge()
    model_provider = bridge.provider_bridge(hf_pretrained)
    model_provider.tensor_model_parallel_size = 1  # on-the-fly conversion is single-GPU only
    model_provider.pipeline_model_parallel_size = 1
    model_provider.pipeline_dtype = torch.bfloat16
    model_provider.params_dtype = torch.bfloat16
    model_provider.seq_length = args.seq_length
    model_provider.finalize()
    model_provider.initialize_model_parallel(seed=0)

    megatron_model = model_provider.provide(pre_process=True, post_process=True)
    bridge.convert_hf_to_megatron(hf_pretrained, megatron_model)
    return megatron_model.cuda().eval()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Run LLaDA2 block-diffusion inference."""
    args = parse_args()

    if args.megatron_path is None and not args.hf_model_only:
        print("ERROR: Specify either --megatron-path or --hf-model-only.", file=sys.stderr)
        sys.exit(1)

    # Distributed setup
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # HF pretrained config (and weights, for on-the-fly conversion)
    hf_pretrained = PreTrainedCausalLM.from_pretrained(
        args.hf_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        load_weights=args.hf_model_only,  # skip loading weights if using Megatron checkpoint
    )

    # Load model
    if args.hf_model_only:
        model = load_from_hf(args, hf_pretrained)
    else:
        model = load_from_megatron(args, hf_pretrained)

    # Tokenize (left-pad for batch consistency)
    inputs = tokenizer(args.prompts, return_tensors="pt", padding=True, padding_side="left")
    prompt_ids = inputs.input_ids.cuda()
    prompt_len = prompt_ids.shape[1]

    # Generate
    with torch.no_grad():
        output = generate_block_diffusion(
            model=model,
            input_ids=prompt_ids,
            gen_length=args.gen_length,
            block_length=args.block_length,
            steps=args.steps_per_block,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            threshold=args.threshold,
            mask_token_id=args.mask_token_id,
            eos_early_stop=args.eos_early_stop,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode and print (rank 0 only, or one rank per TP group)
    if rank == 0 or (world_size > 1 and rank % args.tp == 0):
        generated_ids = output[:, prompt_len:]
        texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for i, (prompt, text) in enumerate(zip(args.prompts, texts)):
            print(f"\n--- Prompt {i + 1} ---")
            print(f"Input:  {prompt}")
            print(f"Output: {text}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
