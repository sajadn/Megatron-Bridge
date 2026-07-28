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

"""Block-aware context parallelism, stage (b).

``plans/cp_kv_sharding_blockaware_ring.md`` stage (b): give the noisy and clean
sections one zigzag each and all-gather ONLY the clean K/V, keeping the noisy
K/V on its owning rank. This suite pins the four claims the rest of (b) rests
on:

1. LOCALITY -- noisy keys are consumed block-diagonally only, so with a
   block-aligned noisy chunk grid no rank's queries ever reach another rank's
   noisy keys. Checked structurally on the dense global mask, plus a negative
   control showing the check fails once the chunk grid stops being
   block-aligned (the divisibility assert is load-bearing, not tidiness).
2. FORWARD PARITY -- per-rank ``[noisy local | clean full]`` attention
   reproduces the corresponding rows of the single-rank (no CP) reference, and
   so does the existing local-Q CP path (regression guard on the mask predicate
   they now share).
3. BACKWARD SPLIT -- dQ is slice-exact; the CLEAN K/V grads are partial and
   must be summed across ranks (all-reduce + slice semantics), while the NOISY
   K/V grads are already complete on their owning rank with no reduction at
   all.
4. COMPILED MASK BUILD == EAGER -- at 128K/cp4 the eager dense-grid build OOMs
   (see ``plans/mask_build_cost_findings.md``), so the compiled build is the
   ONLY way the block-aware mask can exist at its target geometry.
5. COMPILED ATTENTION == EAGER -- a ``mask_mod`` is not build-time-only: the
   compiled CUDA path lowers it into the flex Triton kernel and re-evaluates it
   on every partially-masked tile, and claims 2/3 run eager on CPU where no
   Triton kernel is involved. So the parity claim is re-run on the compiled
   CUDA path, which is what training actually executes. A mask that failed to
   lower would attend a large set of extra keys, redistributing the softmax and
   moving the output by O(0.1-1) -- far outside the kernel-drift tolerance
   below -- so a plain comparison is sufficient to catch it.

CP is simulated in a single process for the parity tests: the clean gather is an
exact reconstruction, so slicing the reference tensors is equivalent to running
the collective. ``test_cp_local_q`` covers the collective itself over gloo.
"""

import os
import sys
import warnings

import pytest
import torch
from torch.nn.attention.flex_attention import create_mask, flex_attention

from megatron.bridge.diffusion.common.cp_utils import (
    reorder_segmented_zigzag_shards,
    segmented_zigzag_index_table,
    segmented_zigzag_local_to_global_idx,
    segmented_zigzag_slice,
    zigzag_slice,
)
from megatron.bridge.diffusion.common.dllm import (
    asymmetric_semi_ar_mask_mod,
    compute_asymmetric_semi_ar_block_aware_mask,
    compute_asymmetric_semi_ar_mask,
)


# The conftest loads dllm from source; reach the module object for the
# compile-flag monkeypatching below.
_dllm = sys.modules["megatron.bridge.diffusion.common.dllm"]

# Production compiles flex with dynamic=False, which specializes on each
# (Q_LEN, KV_LEN) pair by value. This file exercises many distinct geometries in
# one process -- reference, local-Q and block-aware, at cp in {1,2,4} -- which
# blows dynamo's default limit of 8 recompiles. That is the exact failure mode
# documented in plans/mask_build_cost_findings.md, and it surfaces here as an
# unrelated-looking error in whichever test happens to cross the threshold, not
# as anything about masks. Name varies across torch versions; set whichever
# exists.
for _limit_name, _limit_value in (
    ("cache_size_limit", 64),
    ("recompile_limit", 64),
    ("accumulated_cache_size_limit", 512),
    ("accumulated_recompile_limit", 512),
):
    if hasattr(torch._dynamo.config, _limit_name):
        setattr(torch._dynamo.config, _limit_name, _limit_value)

# Realistic asymmetric case: clean = [prompt | response] with per-sample prompt
# lengths, response length == noisy valid length, and both sections padded past
# their valid extent (the mask, not the padding, excludes the tails).
_BLOCK_SIZE = 16
_NOISY_LENGTH = 128
_CLEAN_LENGTH = 128
_ATOL = 1e-9
_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

# Every index tensor a BlockMask carries; equality across all of them is the
# "bit-identical BlockMask" check from plans/mask_build_cost_findings.md.
_BLOCKMASK_TENSOR_ATTRS = (
    "kv_num_blocks",
    "kv_indices",
    "full_kv_num_blocks",
    "full_kv_indices",
    "q_num_blocks",
    "q_indices",
    "full_q_num_blocks",
    "full_q_indices",
)


@pytest.fixture(autouse=True)
def _eager_mask_build(monkeypatch):
    """Build masks eagerly by default.

    The compiled build is the production path but is asserted bit-identical
    below, so the numerical parity tests use the eager reference build: a
    mask-build regression should not surface as an attention-numerics failure.
    ``test_mask_build_compiles_identically`` opts back in explicitly.
    """
    monkeypatch.setattr(_dllm, "_MASK_COMPILE", False)


def _asym_case(device: str = "cpu") -> dict:
    return dict(
        block_size=_BLOCK_SIZE,
        noisy_length=_NOISY_LENGTH,
        clean_length=_CLEAN_LENGTH,
        noisy_response_offset=0,
        prompt_lengths=torch.tensor([13, 21], device=device),
        noisy_valid_lengths=torch.tensor([96, 55], device=device),
        clean_lengths=torch.tensor([109, 76], device=device),
    )


def _dense_global_mask(case: dict) -> torch.Tensor:
    """``[B, Q, KV]`` bool mask of the no-CP (global-index) predicate."""
    full_seq_len = case["noisy_length"] + case["clean_length"]
    mask_mod = asymmetric_semi_ar_mask_mod(
        case["block_size"],
        case["noisy_length"],
        case["noisy_response_offset"],
        case["prompt_lengths"],
        case["noisy_valid_lengths"],
        case["clean_lengths"],
    )
    dense = create_mask(
        mask_mod,
        B=case["prompt_lengths"].shape[0],
        H=1,
        Q_LEN=full_seq_len,
        KV_LEN=full_seq_len,
        device=str(case["prompt_lengths"].device),
    )
    return dense[:, 0]


# ---------------------------------------------------------------------------
# Layout primitives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("segment_lengths", [(128, 128), (64, 192), (64, 32, 96)])
def test_segmented_zigzag_slice_roundtrips(cp_size: int, segment_lengths: tuple) -> None:
    total = sum(segment_lengths)
    full = torch.arange(total).view(1, total, 1).expand(2, total, 3).contiguous()
    shards = [segmented_zigzag_slice(full, segment_lengths, r, cp_size, seq_dim=1) for r in range(cp_size)]
    for shard in shards:
        assert shard.shape[1] == total // cp_size
    restored = reorder_segmented_zigzag_shards(shards, segment_lengths, cp_size, seq_dim=1)
    assert torch.equal(restored, full)


@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("segment_lengths", [(128,), (128, 128), (64, 32, 96)])
def test_index_map_and_table_agree_with_slice(cp_size: int, segment_lengths: tuple) -> None:
    """The arithmetic map, the gather table, and the data-space slice agree.

    Three independent definitions of the same layout: the elementwise
    arithmetic (``segmented_zigzag_local_to_global_idx``), the precomputed
    lookup the compiled mask_mod actually indexes
    (``segmented_zigzag_index_table``), and the tensor slicing the data path
    performs (``segmented_zigzag_slice``). Any drift between them silently
    misaligns queries against keys.
    """
    total = sum(segment_lengths)
    positions = torch.arange(total).view(1, total)
    local_len = total // cp_size
    for rank in range(cp_size):
        owned = segmented_zigzag_slice(positions, segment_lengths, rank, cp_size, seq_dim=1).view(-1)
        mapped = segmented_zigzag_local_to_global_idx(
            torch.arange(local_len), segment_lengths, rank, cp_size
        )
        table = segmented_zigzag_index_table(segment_lengths, rank, cp_size)
        assert torch.equal(mapped, owned), f"arithmetic map != slice (rank {rank})"
        assert torch.equal(table, owned), f"gather table != slice (rank {rank})"


# ---------------------------------------------------------------------------
# Claim 1: noisy K/V never leaves its owning rank
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cp_size", [2, 4])
def test_noisy_kv_stays_on_owner_rank(cp_size: int) -> None:
    case = _asym_case()
    noisy_length = case["noisy_length"]
    chunk = noisy_length // (2 * cp_size)
    assert chunk % case["block_size"] == 0, "test case must use a block-aligned chunk grid"

    dense = _dense_global_mask(case)
    allowed_noisy = dense[:, :, :noisy_length]  # [B, full_seq_len, noisy_length]
    assert allowed_noisy.any(), "no noisy key is attended at all -- test case is degenerate"

    q_idx = torch.arange(dense.shape[1]).view(1, -1, 1)
    kv_idx = torch.arange(noisy_length).view(1, 1, -1)

    # Every allowed noisy key sits in the same noisy chunk as its query, so the
    # noisy K/V a rank's queries need is exactly the noisy K/V it already owns.
    same_chunk = (q_idx < noisy_length) & (
        torch.div(q_idx, chunk, rounding_mode="floor")
        == torch.div(kv_idx, chunk, rounding_mode="floor")
    )
    assert torch.equal(allowed_noisy & ~same_chunk, torch.zeros_like(allowed_noisy)), (
        f"cp={cp_size}: a query reads a noisy key outside its own chunk -- "
        "noisy K/V cannot stay local"
    )

    # No fully-masked local rows: the invalid-query self term keeps every row
    # non-empty, and its diagonal key is local in both sections.
    assert torch.all(dense.any(dim=-1)), "some query row has no visible key"


def test_unaligned_noisy_chunks_break_locality() -> None:
    """Negative control: the block-alignment requirement is load-bearing."""
    cp_size = 4
    case = _asym_case()
    case["block_size"] = 24  # chunk stays 16 -> blocks straddle chunk boundaries
    noisy_length = case["noisy_length"]
    chunk = noisy_length // (2 * cp_size)

    dense = _dense_global_mask(case)
    allowed_noisy = dense[:, :, :noisy_length]
    q_idx = torch.arange(dense.shape[1]).view(1, -1, 1)
    kv_idx = torch.arange(noisy_length).view(1, 1, -1)
    same_chunk = (q_idx < noisy_length) & (
        torch.div(q_idx, chunk, rounding_mode="floor")
        == torch.div(kv_idx, chunk, rounding_mode="floor")
    )
    assert (allowed_noisy & ~same_chunk).any(), (
        "expected cross-chunk noisy reads once the chunk grid is not block-aligned"
    )


def test_block_aware_mask_rejects_unaligned_shapes() -> None:
    case = _asym_case()
    # cp=8 -> noisy chunk 8 < block_size 16.
    with pytest.raises(AssertionError, match="block-aligned"):
        compute_asymmetric_semi_ar_block_aware_mask(**case, cp_rank=0, cp_size=8)
    with pytest.raises(AssertionError, match="noisy_response_offset"):
        compute_asymmetric_semi_ar_block_aware_mask(
            **{**case, "noisy_response_offset": 8}, cp_rank=0, cp_size=4
        )


# ---------------------------------------------------------------------------
# Claims 2 and 3: forward parity and the backward split, against no CP
# ---------------------------------------------------------------------------


# Prefer the PRODUCTION compiled wrapper. It is importable whenever megatron is
# installed: this directory's conftest only stubs ``megatron.*`` when those
# names are absent from sys.modules, and the repo-root conftest imports the real
# megatron.core first. The stubs (and this fallback) therefore only engage in a
# megatron-free container -- which is exactly what ``--confcutdir`` creates.
try:
    from megatron.bridge.diffusion.models.common.nemotron_labs_diffusion_attention import (
        fused_flex_attention as _fused_flex_attention,
    )

    FUSED_FLEX_SOURCE = "production"
except Exception:  # pragma: no cover - depends on the container, not the code
    _fused_flex_attention = None
    FUSED_FLEX_SOURCE = "reconstructed"

# Same knob and same construction as the production decorator, so the fallback
# differs from it only in identity. The test process defaults the mode to
# "default" rather than production's "max-autotune-no-cudagraphs" because
# autotune runs per shape and this file compiles several; set
# DIFFU_FLEX_COMPILE_MODE to override. Autotune selects among kernel configs
# and is not expected to change whether a mask_mod lowers, but that is an
# assumption, not something this test proves.
_FLEX_COMPILE_MODE = os.environ.get("DIFFU_FLEX_COMPILE_MODE", "default")
_COMPILED_FLEX = []


def _compiled_flex_attention(q, k, v, block_mask=None):
    if _fused_flex_attention is not None:
        return _fused_flex_attention(q, k, v, block_mask=block_mask)
    if not _COMPILED_FLEX:
        _COMPILED_FLEX.append(
            torch.compile(flex_attention, fullgraph=True, mode=_FLEX_COMPILE_MODE, dynamic=False)
        )
    return _COMPILED_FLEX[0](q, k, v, block_mask=block_mask)


def test_report_compiled_flex_source() -> None:
    """Surface which wrapper the compiled tests exercised.

    A silent fallback to a look-alike is the failure mode that makes a
    compile test worthless, so make the choice visible in the run rather than
    inferable only from the container.
    """
    assert FUSED_FLEX_SOURCE in ("production", "reconstructed")
    # warn, not print: pytest surfaces the warnings summary even under -q,
    # whereas stdout from a PASSING test is captured and never shown -- which
    # defeats the entire purpose of a test whose job is to report something.
    warnings.warn(
        f"compiled-flex tests exercised the {FUSED_FLEX_SOURCE} fused_flex_attention",
        stacklevel=1,
    )


# Two execution modes for the same parity claim. "eager" is the fp64 reference;
# "compiled" is what training runs.
#
# The compiled tolerance is 1e-4, set from drift MEASURED in the environment
# training actually runs (nemo-rl-nightly container, torch 2.10.0+cu129,
# production fused_flex_attention): max 1.1e-6 across out/dq/dk/dv at cp=1 and
# 7.0e-7 at cp=4 -- i.e. the compiled Triton kernel and the eager unfused path
# agree to about fp32 epsilon. 1e-4 keeps ~100x headroom and can still fail.
#
# An earlier 2e-2 was fitted to ~1e-3 drift seen on a DIFFERENT container
# (nvcr.io/nvidia/pytorch:25.06-py3, older torch). That was an artifact of that
# image, not a property of this kernel; do not re-loosen without re-measuring
# here.
#
# The eager arm is not merely "the same kernel uncompiled": torch warns that
# flex_attention outside torch.compile uses an unfused implementation that
# MATERIALIZES the full scores matrix. So this compares the BlockMask-driven
# Triton kernel against a dense-scores reference -- two different
# implementations -- which is what makes the comparison worth running at all.
_EXECUTIONS = {
    "eager": dict(device="cpu", dtype=torch.float64, attend=flex_attention, tol=1e-9),
    "compiled": dict(device="cuda", dtype=torch.float32, attend=_compiled_flex_attention, tol=1e-4),
}


def _attend(q, k, v, block_mask, grad_out, attend=flex_attention):
    """Run flex attention and return (out, dq, dk, dv) on fresh leaf tensors.

    Fresh leaves per call are load-bearing, not hygiene: the parity test runs
    the same q/k/v through the reference and then through every rank, and a
    shared leaf would let autograd accumulate ``k.grad`` across ranks -- which
    is exactly the cross-rank sum the noisy-K/V assertions are there to prove
    is NOT needed.
    """
    q = q.detach().clone().requires_grad_(True)
    k = k.detach().clone().requires_grad_(True)
    v = v.detach().clone().requires_grad_(True)
    out = attend(q, k, v, block_mask=block_mask)
    out.backward(grad_out)
    return out.detach(), q.grad, k.grad, v.grad


@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("execution", list(_EXECUTIONS))
def test_block_aware_cp_matches_no_cp(execution: str, cp_size: int) -> None:
    mode = _EXECUTIONS[execution]
    if mode["device"] == "cuda" and not torch.cuda.is_available():
        pytest.skip("compiled flex attention requires CUDA (Triton)")
    device, dtype, attend = mode["device"], mode["dtype"], mode["attend"]
    _ATOL = mode["tol"]
    # Preserve the eager path's original rtol; the compiled path needs a
    # relative term because TF32 drift scales with magnitude.
    _RTOL = 1e-5 if execution == "eager" else _ATOL

    torch.manual_seed(0)
    case = _asym_case(device)
    noisy_length, clean_length = case["noisy_length"], case["clean_length"]
    segments = (noisy_length, clean_length)
    full_seq_len = noisy_length + clean_length
    batch, heads, head_dim = case["prompt_lengths"].shape[0], 2, 16

    shape = (batch, heads, full_seq_len, head_dim)
    q = torch.randn(*shape, dtype=dtype, device=device)
    k = torch.randn(*shape, dtype=dtype, device=device)
    v = torch.randn(*shape, dtype=dtype, device=device)
    grad_out = torch.randn(*shape, dtype=dtype, device=device)

    # Reference: no context parallelism.
    out_ref, dq_ref, dk_ref, dv_ref = _attend(
        q, k, v, compute_asymmetric_semi_ar_mask(**case), grad_out, attend
    )
    # Anti-vacuity: the noisy-side grads must be non-trivial, or the
    # "complete without a reduction" assertions below would hold on zeros.
    assert torch.isfinite(out_ref).all(), "reference output has non-finite entries"
    assert dk_ref[:, :, :noisy_length].abs().max() > 0, "reference noisy dk is all zeros"
    assert dv_ref[:, :, :noisy_length].abs().max() > 0, "reference noisy dv is all zeros"

    noisy_local = noisy_length // cp_size
    dk_clean_sum = torch.zeros_like(dk_ref[:, :, noisy_length:])
    dv_clean_sum = torch.zeros_like(dv_ref[:, :, noisy_length:])
    out_shards = []

    for rank in range(cp_size):
        # Existing (naive) CP path: one global zigzag, Q local, FULL K/V gathered.
        out_lq, dq_lq, _, _ = _attend(
            zigzag_slice(q, rank, cp_size, seq_dim=2),
            k,
            v,
            compute_asymmetric_semi_ar_mask(**case, cp_rank=rank, cp_size=cp_size),
            zigzag_slice(grad_out, rank, cp_size, seq_dim=2),
            attend,
        )
        assert torch.allclose(out_lq, zigzag_slice(out_ref, rank, cp_size, 2), rtol=_RTOL, atol=_ATOL), (
            f"cp={cp_size} rank={rank}: local-Q CP forward differs from no-CP"
        )
        assert torch.allclose(dq_lq, zigzag_slice(dq_ref, rank, cp_size, 2), rtol=_RTOL, atol=_ATOL), (
            f"cp={cp_size} rank={rank}: local-Q CP dq differs from no-CP"
        )

        # Block-aware (b): two zigzags; K/V is [noisy local | clean full].
        q_local = segmented_zigzag_slice(q, segments, rank, cp_size, seq_dim=2)
        k_local = torch.cat(
            [zigzag_slice(k[:, :, :noisy_length], rank, cp_size, 2), k[:, :, noisy_length:]], dim=2
        )
        v_local = torch.cat(
            [zigzag_slice(v[:, :, :noisy_length], rank, cp_size, 2), v[:, :, noisy_length:]], dim=2
        )
        assert k_local.shape[2] == noisy_local + clean_length < full_seq_len

        out_local, dq_local, dk_local, dv_local = _attend(
            q_local,
            k_local,
            v_local,
            compute_asymmetric_semi_ar_block_aware_mask(**case, cp_rank=rank, cp_size=cp_size),
            segmented_zigzag_slice(grad_out, segments, rank, cp_size, seq_dim=2),
            attend,
        )

        assert torch.allclose(
            out_local, segmented_zigzag_slice(out_ref, segments, rank, cp_size, 2), rtol=_RTOL, atol=_ATOL
        ), f"cp={cp_size} rank={rank}: block-aware forward differs from no-CP"
        assert torch.allclose(
            dq_local, segmented_zigzag_slice(dq_ref, segments, rank, cp_size, 2), rtol=_RTOL, atol=_ATOL
        ), f"cp={cp_size} rank={rank}: block-aware dq differs from no-CP"

        # Noisy K/V grads are COMPLETE per rank: no other rank's queries touched
        # them, so plain local autograd is exact (no all-reduce).
        assert dk_local[:, :, :noisy_local].abs().max() > 0, "local noisy dk is all zeros"
        assert torch.allclose(
            dk_local[:, :, :noisy_local],
            zigzag_slice(dk_ref[:, :, :noisy_length], rank, cp_size, 2),
            rtol=_RTOL, atol=_ATOL,
        ), f"cp={cp_size} rank={rank}: noisy dk needs a cross-rank reduction"
        assert torch.allclose(
            dv_local[:, :, :noisy_local],
            zigzag_slice(dv_ref[:, :, :noisy_length], rank, cp_size, 2),
            rtol=_RTOL, atol=_ATOL,
        ), f"cp={cp_size} rank={rank}: noisy dv needs a cross-rank reduction"

        # Clean K/V grads are partial (this rank's query rows only) and sum.
        dk_clean_sum += dk_local[:, :, noisy_local:]
        dv_clean_sum += dv_local[:, :, noisy_local:]
        out_shards.append(out_local)

    assert torch.allclose(dk_clean_sum, dk_ref[:, :, noisy_length:], rtol=_RTOL, atol=_ATOL), (
        f"cp={cp_size}: summed clean dk differs from no-CP"
    )
    assert torch.allclose(dv_clean_sum, dv_ref[:, :, noisy_length:], rtol=_RTOL, atol=_ATOL), (
        f"cp={cp_size}: summed clean dv differs from no-CP"
    )
    # The re-gather every downstream consumer (logits, logprobs) has to perform.
    assert torch.allclose(
        reorder_segmented_zigzag_shards(out_shards, segments, cp_size, 2), out_ref, rtol=_RTOL, atol=_ATOL
    ), f"cp={cp_size}: re-gathered output differs from no-CP"


# ---------------------------------------------------------------------------
# Claim 4: the mask_mod survives torch.compile
# ---------------------------------------------------------------------------


def _assert_blockmask_identical(eager, compiled, label: str) -> None:
    assert eager.shape == compiled.shape, f"{label}: BlockMask shape differs"
    for attr in _BLOCKMASK_TENSOR_ATTRS:
        lhs, rhs = getattr(eager, attr, None), getattr(compiled, attr, None)
        assert (lhs is None) == (rhs is None), f"{label}: {attr} present in only one build"
        if lhs is not None:
            assert torch.equal(lhs, rhs), f"{label}: {attr} differs between eager and compiled"


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("cp_size", [1, 4])
@pytest.mark.parametrize("builder", ["local_q", "block_aware"])
def test_mask_build_compiles_identically(
    device: str, cp_size: int, builder: str, monkeypatch
) -> None:
    """Compiled and eager BlockMask builds must agree, for both CP layouts.

    The block-aware remap is a gather through a precomputed index table
    precisely so it lowers into the flex Triton kernel; this is the test that
    would catch it regressing to something dynamo cannot trace. At the target
    128K/cp4 geometry the eager build OOMs, so a mask that only builds eagerly
    is a mask that cannot be used.
    """
    case = _asym_case(device)
    build = (
        compute_asymmetric_semi_ar_mask
        if builder == "local_q"
        else compute_asymmetric_semi_ar_block_aware_mask
    )
    cp_rank = cp_size - 1

    monkeypatch.setattr(_dllm, "_MASK_COMPILE", False)
    eager = build(**case, cp_rank=cp_rank, cp_size=cp_size)

    monkeypatch.setattr(_dllm, "_MASK_COMPILE", True)
    compiled = build(**case, cp_rank=cp_rank, cp_size=cp_size)

    _assert_blockmask_identical(eager, compiled, f"{builder} cp={cp_size} {device}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compiled flex needs CUDA (Triton)")
@pytest.mark.parametrize("cp_size", [1, 4])
def test_block_aware_attention_compiled_matches_eager(cp_size: int) -> None:
    """Same BlockMask, eager vs compiled attention -- isolates the lowering.

    ``test_block_aware_cp_matches_no_cp[compiled]`` covers the same ground
    end-to-end; this narrows a failure down to the Triton lowering of the
    gather remap by holding the mask fixed and changing only how attention is
    executed. The mask build stays eager (autouse fixture) so both arms consume
    a byte-identical BlockMask.
    """
    torch.manual_seed(0)
    case = _asym_case("cuda")
    noisy_length, clean_length = case["noisy_length"], case["clean_length"]
    batch, heads, head_dim = case["prompt_lengths"].shape[0], 2, 64

    block_mask = compute_asymmetric_semi_ar_block_aware_mask(
        **case, cp_rank=cp_size - 1, cp_size=cp_size
    )
    q_len = (noisy_length + clean_length) // cp_size
    kv_len = noisy_length // cp_size + clean_length

    q = torch.randn(batch, heads, q_len, head_dim, dtype=torch.float32, device="cuda")
    k = torch.randn(batch, heads, kv_len, head_dim, dtype=torch.float32, device="cuda")
    v = torch.randn(batch, heads, kv_len, head_dim, dtype=torch.float32, device="cuda")
    grad_out = torch.randn(batch, heads, q_len, head_dim, dtype=torch.float32, device="cuda")

    eager = _attend(q, k, v, block_mask, grad_out, flex_attention)
    compiled = _attend(q, k, v, block_mask, grad_out, _compiled_flex_attention)

    drift = {}
    for name, lhs, rhs in zip(("out", "dq", "dk", "dv"), eager, compiled):
        drift[name] = float((lhs - rhs).abs().max())
        assert torch.allclose(lhs, rhs, rtol=1e-4, atol=1e-4), (
            f"cp={cp_size} [{FUSED_FLEX_SOURCE}]: compiled attention {name} differs "
            f"from eager by max {drift[name]:.3e}, well beyond measured kernel "
            f"drift -- suspect the mask_mod not lowering into the Triton kernel"
        )
    warnings.warn(f"cp={cp_size} compiled-vs-eager max drift: {drift}", stacklevel=1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
