#!/usr/bin/env python3
"""Diagnostic: print HF and Megatron parameter names/shapes side-by-side.

Usage:
  python examples/diffusion/recipes/llada2/diagnose_arch.py \
      --hf-model inclusionAI/LLaDA2.1-mini \
      --out arch_comparison.txt
"""

import argparse
import re
from pathlib import Path

import torch

from megatron.bridge.diffusion.conversion.llada2.llada2_moe_bridge import LLaDA2MoEBridge
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--hf-model", required=True, help="HF model ID or local path")
    p.add_argument("--out", default="arch_comparison.txt", help="Output file (default: arch_comparison.txt)")
    p.add_argument(
        "--no-hf-weights",
        action="store_true",
        help="Skip loading HF weights (only print param shapes from config estimate)",
    )
    return p.parse_args()


def build_megatron_model(hf_pretrained):
    """Build a Megatron GPTModel from the HF pretrained config."""
    bridge = LLaDA2MoEBridge()
    provider = bridge.provider_bridge(hf_pretrained)
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.pipeline_dtype = torch.bfloat16
    provider.params_dtype = torch.bfloat16
    provider.seq_length = 4096
    provider.finalize()
    provider.initialize_model_parallel(seed=0)
    return provider.provide(pre_process=True, post_process=True)


def pattern_to_regex(pat):
    """Convert a wildcard pattern (using * as single-segment wildcard) to a regex."""
    # Replace * with a capture group matching one path segment (no dots)
    return re.compile("^" + re.escape(pat).replace(r"\*", r"[^.]+") + "$")


def get_hf_patterns_from_mapping(mapping):
    """Return a list of HF key patterns from a mapping object (handles AutoMapping and GatedMLPMapping)."""
    hf_param = getattr(mapping, "hf_param", None)
    if hf_param is None:
        return []
    if isinstance(hf_param, str):
        return [hf_param]
    if isinstance(hf_param, dict):
        return list(hf_param.values())
    return []


def main():
    """Run parameter diagnostic between HF and Megatron models."""
    args = parse_args()

    print(f"Loading HF config/model from {args.hf_model} ...")
    hf_pretrained = PreTrainedCausalLM.from_pretrained(
        args.hf_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Access hf_pretrained.model to trigger weight loading (lazy load)
    if not args.no_hf_weights:
        hf_model = hf_pretrained.model
        hf_params = {n: tuple(p.shape) for n, p in hf_model.named_parameters()}
        print(f"HF model loaded: {len(hf_params)} parameters")
    else:
        hf_params = {}
        print("Skipping HF weight loading (--no-hf-weights)")

    print("Building Megatron model ...")
    meg_model = build_megatron_model(hf_pretrained)
    meg_params = {n: tuple(p.shape) for n, p in meg_model.named_parameters()}
    print(f"Megatron model built: {len(meg_params)} parameters")

    bridge = LLaDA2MoEBridge()
    bridge._hf_config = hf_pretrained.config
    registry = bridge.mapping_registry()

    out_path = Path(args.out)
    with open(out_path, "w") as f:
        # ---- HF parameters ----
        if hf_params:
            f.write("=" * 90 + "\n")
            f.write("HF MODEL PARAMETERS\n")
            f.write("=" * 90 + "\n")
            for name, shape in sorted(hf_params.items()):
                f.write(f"  {name:<85s}  {str(shape)}\n")
            f.write(f"\nTotal HF params: {len(hf_params)}\n\n")

        # ---- Megatron parameters ----
        f.write("=" * 90 + "\n")
        f.write("MEGATRON MODEL PARAMETERS\n")
        f.write("=" * 90 + "\n")
        for name, shape in sorted(meg_params.items()):
            f.write(f"  {name:<85s}  {str(shape)}\n")
        f.write(f"\nTotal Megatron params: {len(meg_params)}\n\n")

        # ---- Bridge mappings ----
        f.write("=" * 90 + "\n")
        f.write("BRIDGE MAPPINGS (megatron_pattern -> hf_pattern)\n")
        f.write("=" * 90 + "\n")
        for mapping in registry.mappings:
            meg_pat = getattr(mapping, "megatron_param", "???")
            hf_pats = get_hf_patterns_from_mapping(mapping)
            cls_name = type(mapping).__name__
            for hf_pat in hf_pats:
                f.write(f"  [{cls_name:<20s}]  {meg_pat:<60s}  ->  {hf_pat}\n")
        f.write("\n")

        # ---- Coverage analysis (only when HF weights loaded) ----
        if hf_params:
            f.write("=" * 90 + "\n")
            f.write("COVERAGE ANALYSIS\n")
            f.write("=" * 90 + "\n")

            meg_covered = set()
            hf_covered = set()
            unmatched_meg_patterns = []
            unmatched_hf_patterns = []

            for mapping in registry.mappings:
                meg_pat = getattr(mapping, "megatron_param", None)
                if meg_pat is None:
                    continue
                meg_re = pattern_to_regex(meg_pat)
                meg_matches = [k for k in meg_params if meg_re.match(k)]
                meg_covered.update(meg_matches)
                if not meg_matches:
                    unmatched_meg_patterns.append(meg_pat)

                for hf_pat in get_hf_patterns_from_mapping(mapping):
                    hf_re = pattern_to_regex(hf_pat)
                    hf_matches = [k for k in hf_params if hf_re.match(k)]
                    hf_covered.update(hf_matches)
                    if not hf_matches:
                        unmatched_hf_patterns.append(hf_pat)

            uncovered_meg = sorted(set(meg_params) - meg_covered)
            uncovered_hf = sorted(set(hf_params) - hf_covered)

            f.write("\nMegatron params NOT covered by any mapping:\n")
            for k in uncovered_meg:
                f.write(f"  UNCOV_MEG: {k:<80s}  {meg_params[k]}\n")
            if not uncovered_meg:
                f.write("  (none — all Megatron params are covered)\n")

            f.write("\nHF params NOT covered by any mapping:\n")
            for k in uncovered_hf:
                f.write(f"  UNCOV_HF:  {k:<80s}  {hf_params[k]}\n")
            if not uncovered_hf:
                f.write("  (none — all HF params are covered)\n")

            f.write("\nMapping patterns with no matching Megatron params:\n")
            for pat in unmatched_meg_patterns:
                f.write(f"  NO_MEG_MATCH: {pat}\n")
            if not unmatched_meg_patterns:
                f.write("  (none)\n")

            f.write("\nMapping patterns with no matching HF params:\n")
            for pat in unmatched_hf_patterns:
                f.write(f"  NO_HF_MATCH:  {pat}\n")
            if not unmatched_hf_patterns:
                f.write("  (none)\n")

            f.write("\nSummary:\n")
            f.write(f"  HF params:              {len(hf_params)}\n")
            f.write(f"  Megatron params:        {len(meg_params)}\n")
            f.write(f"  HF covered:             {len(hf_covered)} / {len(hf_params)}\n")
            f.write(f"  Megatron covered:       {len(meg_covered)} / {len(meg_params)}\n")
            f.write(f"  Uncovered HF:           {len(uncovered_hf)}\n")
            f.write(f"  Uncovered Megatron:     {len(uncovered_meg)}\n")

    print(f"\nDiagnostic written to: {out_path.resolve()}")
    print(f"Megatron params: {len(meg_params)}")
    if hf_params:
        print(f"HF params:       {len(hf_params)}")
        print(f"Uncovered MEG:   {len(uncovered_meg)}")
        print(f"Uncovered HF:    {len(uncovered_hf)}")


if __name__ == "__main__":
    main()
