#!/bin/bash
# Cascade evaluation launcher for 3B, 8B, and cascade (8B→3B) models.
#
# Runs all eval configs:
#   - steps_per_block: 32, 16, 1
#   - denoising_threshold=0.9 with steps_per_block=32
#   - All tasks: gsm8k_cot, humaneval, mbpp, humaneval_plus, mbpp_plus
#   - Latency logging enabled for all runs
#
# Usage:
#   bash eval_cascade.sh --direct                    # Run on current GPU node
#   bash eval_cascade.sh --direct --limit 8          # Quick validation
#   bash eval_cascade.sh --parallel-tasks            # Submit Slurm jobs per task
#   bash eval_cascade.sh --parallel-models           # Submit Slurm jobs per model

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/eval_megatron.sh"

# --- Checkpoint paths (Megatron Bridge format, converted by this repo) ---
CKPT_3B="${CKPT_3B:-/lustre/fsw/portfolios/coreai/users/snorouzi/megatron_exp/cascade_mb/ministral_3b}"
CKPT_8B="${CKPT_8B:-/lustre/fsw/portfolios/coreai/users/snorouzi/megatron_exp/cascade_mb/ministral_8b}"

# HF model IDs for config loading (AutoBridge) — must point to actual config.json dirs
HF_3B="${HF_3B:-/lustre/fsw/portfolios/coreai/users/snorouzi/megatron_exp/ministral_3b/iter_0012500_hf/Ministral-3-3B-Base-2512_converted}"
HF_8B="${HF_8B:-/lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/abhgarg/megatron_exp/ministral_8b_sbd64_llada_combine_before_weighting_1e_5_rerun_32/iter_0012500_hf/Ministral-3-8B-Base-2512_1t_ft}"

# --- Parse args ---
EXEC_MODE=""
LIMIT_ARG=""
EVAL_TASKS="gsm8k_cot,humaneval,mbpp,humaneval_plus,mbpp_plus"

while [[ $# -gt 0 ]]; do
  case $1 in
    --direct|--parallel-tasks|--parallel-models)
      EXEC_MODE="$1"; shift ;;
    --limit)
      LIMIT_ARG="--limit $2"; shift 2 ;;
    --eval-tasks)
      EVAL_TASKS="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [ -z "${EXEC_MODE}" ]; then
    echo "Usage: $0 [--direct|--parallel-tasks|--parallel-models] [--limit N] [--eval-tasks T1,T2]"
    exit 1
fi

COMMON_ARGS="--eval-tasks ${EVAL_TASKS} --seeds 42 --modes dllm ${LIMIT_ARG}"

echo "============================================"
echo " Cascade dLLM Evaluation Suite"
echo " Execution mode: ${EXEC_MODE}"
echo " Limit: ${LIMIT_ARG:-none}"
echo "============================================"

run_eval() {
    local name="$1"
    local ckpt="$2"
    local hf_id="$3"
    local spb="$4"
    local threshold="$5"
    local cascade="$6"

    local suffix="${name}_spb${spb}"
    if [ "${threshold}" != "None" ]; then
        suffix="${suffix}_thr${threshold}"
    fi

    local extra_args=""
    if [ -n "${cascade}" ]; then
        extra_args="${extra_args} --cascade-schedule ${cascade}"
    fi
    if [ "${threshold}" != "None" ]; then
        extra_args="${extra_args} --denoising-threshold ${threshold}"
    fi

    echo ""
    echo ">>> Running: ${suffix}"
    echo "    checkpoint: ${ckpt}"
    echo "    hf_model_id: ${hf_id}"
    echo "    steps_per_block: ${spb}, threshold: ${threshold}"
    if [ -n "${cascade}" ]; then
        echo "    cascade_schedule: ${cascade}"
    fi

    bash "${EVAL_SCRIPT}" "${EXEC_MODE}" \
        --checkpoint "${ckpt}" \
        --exp-name "${suffix}" \
        --hf-model-id "${hf_id}" \
        --steps-per-block "${spb}" \
        ${COMMON_ARGS} \
        ${extra_args}
}

# ================================================================
# 1. 3B standalone
# ================================================================
for spb in 32 16 1; do
    run_eval "3b" "${CKPT_3B}" "${HF_3B}" "${spb}" "None" ""
done
# 3B with denoising threshold
run_eval "3b" "${CKPT_3B}" "${HF_3B}" "32" "0.9" ""

# ================================================================
# 2. 8B standalone
# ================================================================
for spb in 32 16 1; do
    run_eval "8b" "${CKPT_8B}" "${HF_8B}" "${spb}" "None" ""
done
# 8B with denoising threshold
run_eval "8b" "${CKPT_8B}" "${HF_8B}" "32" "0.9" ""

# ================================================================
# 3. Cascade 8B→3B (50/50 split)
# For cascade, primary model is 3B (used for prefill + KV updates).
# cascade_schedule format: ckpt1|n_steps1|hf_id1|ckpt2|n_steps2|hf_id2
# ================================================================
CASCADE_32="${CKPT_8B}|16|${HF_8B}|${CKPT_3B}|16|${HF_3B}"
CASCADE_16="${CKPT_8B}|8|${HF_8B}|${CKPT_3B}|8|${HF_3B}"

run_eval "cascade_8b_3b" "${CKPT_3B}" "${HF_3B}" "32" "None" "${CASCADE_32}"
run_eval "cascade_8b_3b" "${CKPT_3B}" "${HF_3B}" "16" "None" "${CASCADE_16}"
# Cascade with denoising threshold
run_eval "cascade_8b_3b" "${CKPT_3B}" "${HF_3B}" "32" "0.9" "${CASCADE_32}"

echo ""
echo "============================================"
echo " All evaluation jobs submitted/completed!"
echo "============================================"
