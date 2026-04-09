#!/bin/bash
# Convert HF checkpoints to Megatron Bridge format for cascade evaluation.
#
# Usage:
#   bash convert_cascade_checkpoints.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MB_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
CONVERT_SCRIPT="${MB_ROOT}/examples/diffusion/recipes/nemotron_diffusion/convert_checkpoints.py"

export PYTHONPATH="${MB_ROOT}/src:${MB_ROOT}/examples:${MB_ROOT}:${PYTHONPATH:-}"

# --- 3B checkpoint ---
HF_3B="/lustre/fsw/portfolios/coreai/users/snorouzi/megatron_exp/ministral_3b/iter_0012500_hf"
MB_3B="/lustre/fsw/portfolios/coreai/users/snorouzi/megatron_exp/cascade_mb/ministral_3b"

if [ -d "${MB_3B}" ]; then
    echo "3B MB checkpoint already exists at ${MB_3B}, skipping."
else
    echo "Converting 3B HF → MB..."
    python "${CONVERT_SCRIPT}" import \
        --hf-model "${HF_3B}" \
        --megatron-path "${MB_3B}" \
        --torch-dtype bfloat16
    echo "3B conversion complete: ${MB_3B}"
fi

# --- 8B checkpoint ---
HF_8B="/lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/abhgarg/megatron_exp/ministral_8b_sbd64_llada_combine_before_weighting_1e_5_rerun_32/iter_0012500_hf"
MB_8B="/lustre/fsw/portfolios/coreai/users/snorouzi/megatron_exp/cascade_mb/ministral_8b"

if [ -d "${MB_8B}" ]; then
    echo "8B MB checkpoint already exists at ${MB_8B}, skipping."
else
    echo "Converting 8B HF → MB..."
    python "${CONVERT_SCRIPT}" import \
        --hf-model "${HF_8B}" \
        --megatron-path "${MB_8B}" \
        --torch-dtype bfloat16
    echo "8B conversion complete: ${MB_8B}"
fi

echo ""
echo "All checkpoint conversions complete!"
echo "3B MB: ${MB_3B}"
echo "8B MB: ${MB_8B}"
