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
LLaDA2 MoE checkpoint conversion: HuggingFace ↔ Megatron.

LLaDA2's HF architecture name is "LLaDA2MoeModelLM" (does not end in "ForCausalLM"),
so we use a thin AutoBridge subclass that bypasses architecture-name validation
and routes directly to LLaDA2MoEBridge.

Usage:

  # HuggingFace → Megatron  (single GPU, enough RAM to load the full model)
  python examples/diffusion/recipes/llada2/convert_checkpoints.py import \\
      --hf-model inclusionAI/LLaDA2.1-mini \\
      --megatron-path ./checkpoints/llada2_8b

  # Megatron → HuggingFace  (round-trip or export after fine-tuning)
  python examples/diffusion/recipes/llada2/convert_checkpoints.py export \\
      --hf-model inclusionAI/LLaDA2.1-mini \\
      --megatron-path ./checkpoints/llada2_8b \\
      --hf-path ./exports/llada2_8b_hf

Notes:
  - Conversion loads all weights on CPU then copies to checkpoint shards.
    For 8B+ models use a machine with ≥ 32 GB RAM.
  - For a local HF cache pass a local directory as --hf-model instead of
    a Hub ID (e.g. ~/.cache/huggingface/hub/inclusionAI/LLaDA2.1-mini).
  - Tensor-parallel conversion (TP > 1) is not supported in this script;
    all shards are written as TP=1.  Load the checkpoint with the matching
    TP degree at inference time.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import save_file

from megatron.bridge.diffusion.conversion.llada2.llada2_moe_bridge import LLaDA2MoEBridge
from megatron.bridge.models.conversion.auto_bridge import AutoBridge


class LLaDA2AutoBridge(AutoBridge):
    """AutoBridge subclass for LLaDA2 (architecture name: LLaDA2MoeModelLM).

    AutoBridge rejects architectures not ending in ForCausalLM/ForConditionalGeneration
    in _validate_config, _model_bridge, and save_hf_weights.  We override all three
    to route directly to LLaDA2MoEBridge.
    """

    def __init__(self, hf_pretrained):
        super().__init__(hf_pretrained)
        self._llada2_bridge = LLaDA2MoEBridge()

    @classmethod
    def _validate_config(cls, config, path=None):
        pass  # skip architecture-name check

    @property
    def _model_bridge(self):
        return self._llada2_bridge

    def save_hf_weights(
        self,
        model,
        path,
        show_progress=True,
        strict=True,
        merge_adapter_weights=True,
        distributed_save=False,
        **kwargs,
    ):
        """Override to avoid _causal_lm_architecture lookup in dispatch."""
        generator = self._llada2_bridge.stream_weights_megatron_to_hf(
            model,
            self.hf_pretrained,
            cpu=True,
            show_progress=show_progress,
            merge_adapter_weights=merge_adapter_weights,
        )
        state_dict = {name: tensor.contiguous().cpu() for name, tensor in generator}
        plan = split_torch_state_dict_into_shards(state_dict)
        safe_dir = Path(path)
        safe_dir.mkdir(parents=True, exist_ok=True)
        for filename, tensors in plan.filename_to_tensors.items():
            shard = {k: state_dict[k] for k in tensors}
            save_file(shard, safe_dir / filename)
        if plan.is_sharded:
            index = {"metadata": plan.metadata, "weight_map": plan.tensor_to_filename}
            with open(safe_dir / "model.safetensors.index.json", "w") as f:
                json.dump(index, f, indent=2)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert LLaDA2 checkpoints between HuggingFace and Megatron formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Conversion direction")

    # --- import: HF → Megatron ---
    imp = subparsers.add_parser("import", help="Import HuggingFace model to Megatron format")
    imp.add_argument(
        "--hf-model",
        required=True,
        help="HuggingFace model ID or local path (e.g. inclusionAI/LLaDA2.1-mini)",
    )
    imp.add_argument(
        "--megatron-path",
        required=True,
        help="Directory where the Megatron checkpoint will be written",
    )
    imp.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Weight dtype for the saved checkpoint (default: bfloat16)",
    )

    # --- export: Megatron → HF ---
    exp = subparsers.add_parser("export", help="Export Megatron checkpoint to HuggingFace format")
    exp.add_argument(
        "--hf-model",
        required=True,
        help="HuggingFace model ID or local path used as the config reference",
    )
    exp.add_argument(
        "--megatron-path",
        required=True,
        help="Directory of the Megatron checkpoint to export",
    )
    exp.add_argument(
        "--hf-path",
        required=True,
        help="Directory where the HuggingFace model will be written",
    )
    exp.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress the progress bar",
    )
    exp.add_argument(
        "--not-strict",
        action="store_true",
        help="Allow source and target to have different keys (use with caution)",
    )

    return parser.parse_args()


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def do_import(args):
    """Import HuggingFace model into Megatron checkpoint format."""
    dtype = DTYPE_MAP[args.dtype]
    print(f"Importing {args.hf_model} -> {args.megatron_path}  (dtype={args.dtype})")

    LLaDA2AutoBridge.import_ckpt(
        hf_model_id=args.hf_model,
        megatron_path=args.megatron_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    ckpt = Path(args.megatron_path)
    print(f"\nCheckpoint written to: {ckpt.resolve()}")
    if ckpt.exists():
        for item in sorted(ckpt.iterdir()):
            marker = "[dir]" if item.is_dir() else ""
            print(f"  {item.name} {marker}")


def do_export(args):
    """Export Megatron checkpoint to HuggingFace format."""
    print(f"Exporting {args.megatron_path} -> {args.hf_path}")

    bridge = LLaDA2AutoBridge.from_hf_pretrained(args.hf_model, trust_remote_code=True)
    bridge.export_ckpt(
        megatron_path=args.megatron_path,
        hf_path=args.hf_path,
        show_progress=not args.no_progress,
        strict=not args.not_strict,
    )

    out = Path(args.hf_path)
    print(f"\nHuggingFace model written to: {out.resolve()}")
    print("Load with:")
    print("  from transformers import AutoModelForCausalLM")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{args.hf_path}', trust_remote_code=True)")


def main():
    """Entry point for checkpoint conversion."""
    args = parse_args()
    if not args.command:
        print("Specify a sub-command: import | export", file=sys.stderr)
        return 1

    if args.command == "import":
        do_import(args)
    elif args.command == "export":
        do_export(args)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

    return 0


if __name__ == "__main__":
    sys.exit(main())
