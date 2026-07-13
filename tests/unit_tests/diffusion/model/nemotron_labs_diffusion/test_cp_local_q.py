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

"""Local-Q context parallelism for the asymmetric semi-AR flex path (CPU only).

Covers the DIFFU_CP_LOCAL_Q=1 building blocks:
  - ``zigzag_local_to_global_idx`` is the exact index inverse of ``zigzag_slice``.
  - ``all_gather_kv_seq_cp`` reconstructs the full sequence in forward and its
    backward is all-reduce + slice (sums every rank's contribution), validated
    over a CPU gloo process group with rank-dependent downstream losses.
  - End-to-end flex-attention parity: each rank's local Q rows against full K/V
    with the CP-remapped block mask reproduce the zigzag slice of the full
    attention output; dq is slice-exact and dk/dv summed over ranks match the
    full-attention grads.
"""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.attention.flex_attention import flex_attention

try:
    from megatron.bridge.diffusion.common.cp_utils import (
        all_gather_kv_seq_cp,
        zigzag_local_to_global_idx,
        zigzag_slice,
    )
    from megatron.bridge.diffusion.common.dllm import compute_asymmetric_semi_ar_mask
except ModuleNotFoundError:
    # mp.spawn children re-import this module in a fresh interpreter where the
    # conftest path-loader never ran; load the modules under test from source
    # the same way the conftest does.
    import importlib.util
    import sys
    import types
    from pathlib import Path

    _SRC = Path(__file__).parents[5] / "src"

    for _ns in [
        "megatron",
        "megatron.bridge",
        "megatron.bridge.diffusion",
        "megatron.bridge.diffusion.common",
    ]:
        if _ns not in sys.modules:
            _m = types.ModuleType(_ns)
            _m.__path__ = []
            _m.__package__ = _ns
            sys.modules[_ns] = _m

    def _load(module_name: str, rel_path: str) -> None:
        spec = importlib.util.spec_from_file_location(module_name, _SRC / rel_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)

    _load(
        "megatron.bridge.diffusion.common.cp_utils",
        "megatron/bridge/diffusion/common/cp_utils.py",
    )
    _load(
        "megatron.bridge.diffusion.common.dllm",
        "megatron/bridge/diffusion/common/dllm.py",
    )
    from megatron.bridge.diffusion.common.cp_utils import (
        all_gather_kv_seq_cp,
        zigzag_local_to_global_idx,
        zigzag_slice,
    )
    from megatron.bridge.diffusion.common.dllm import compute_asymmetric_semi_ar_mask


# ---------------------------------------------------------------------------
# Index-map inverse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_zigzag_local_to_global_idx_inverts_slice(cp_size: int) -> None:
    seq = 2 * cp_size * 8
    positions = torch.arange(seq)
    local_len = seq // cp_size
    for rank in range(cp_size):
        owned = zigzag_slice(positions.view(1, -1), rank, cp_size, seq_dim=1).view(-1)
        mapped = zigzag_local_to_global_idx(torch.arange(local_len), rank, cp_size, local_len)
        assert torch.equal(mapped, owned)


# ---------------------------------------------------------------------------
# Distributed (gloo, CPU) check for the K/V all-gather autograd
# ---------------------------------------------------------------------------


def _kv_gather_worker(rank: int, world_size: int, port: int, seq_len: int, hidden: int, out_q) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        group = dist.group.WORLD
        full = torch.arange(seq_len * hidden).float().view(1, seq_len, hidden)  # identical on all ranks

        local = zigzag_slice(full, rank, world_size, seq_dim=1).clone().requires_grad_(True)
        gathered = all_gather_kv_seq_cp(local, group, seq_dim=1)
        fwd_ok = torch.equal(gathered.detach(), full)

        # Rank-dependent downstream weight (as with local-Q attention, where each
        # rank's loss only covers its own query rows). The true grad of the local
        # K/V shard is the zigzag slice of the SUM of all ranks' weights -- a
        # slice-only backward would keep just this rank's own term.
        base = torch.arange(seq_len * hidden).float().view(1, seq_len, hidden) + 1.0
        (gathered * (base * (rank + 1))).sum().backward()
        w_sum = base * sum(r + 1 for r in range(world_size))
        bwd_ok = torch.allclose(local.grad, zigzag_slice(w_sum, rank, world_size, seq_dim=1))

        out_q.put((rank, bool(fwd_ok), bool(bwd_ok)))
    finally:
        dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("world_size", [2, 4])
def test_all_gather_kv_seq_cp_roundtrip_and_reduced_grad(world_size: int) -> None:
    seq_len, hidden = 2 * world_size * 5, 3
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(target=_kv_gather_worker, args=(r, world_size, port, seq_len, hidden, q))
        for r in range(world_size)
    ]
    for p in procs:
        p.start()
    results = [q.get(timeout=120) for _ in range(world_size)]
    for p in procs:
        p.join(timeout=120)
    for rank, fwd_ok, bwd_ok in results:
        assert fwd_ok, f"cp={world_size} rank={rank}: forward reconstruction wrong"
        assert bwd_ok, f"cp={world_size} rank={rank}: backward grad not all-reduced"


# ---------------------------------------------------------------------------
# End-to-end local-Q flex attention parity (single process; the gather is an
# exact reconstruction, so per-rank local-Q attention can be simulated by
# slicing Q and reusing the full K/V directly)
# ---------------------------------------------------------------------------


def _asym_mask_inputs(device: str = "cpu") -> dict:
    return dict(
        block_size=16,
        noisy_length=64,
        clean_length=64,
        noisy_response_offset=8,
        prompt_lengths=torch.tensor([5, 9], device=device),
        noisy_valid_lengths=torch.tensor([40, 24], device=device),
        clean_lengths=torch.tensor([50, 30], device=device),
    )


@pytest.mark.parametrize("cp_size", [2, 4])
def test_local_q_flex_attention_parity(cp_size: int) -> None:
    torch.manual_seed(0)
    m = _asym_mask_inputs()
    full_len = m["noisy_length"] + m["clean_length"]
    b, h, d = m["prompt_lengths"].shape[0], 2, 16

    q = torch.randn(b, h, full_len, d, dtype=torch.float64, requires_grad=True)
    k = torch.randn(b, h, full_len, d, dtype=torch.float64, requires_grad=True)
    v = torch.randn(b, h, full_len, d, dtype=torch.float64, requires_grad=True)
    grad_out = torch.randn(b, h, full_len, d, dtype=torch.float64)

    full_mask = compute_asymmetric_semi_ar_mask(**m)
    out_full = flex_attention(q, k, v, block_mask=full_mask)
    out_full.backward(grad_out)
    dq_full, dk_full, dv_full = q.grad.clone(), k.grad.clone(), v.grad.clone()

    dk_sum = torch.zeros_like(dk_full)
    dv_sum = torch.zeros_like(dv_full)
    for rank in range(cp_size):
        q_local = zigzag_slice(q.detach(), rank, cp_size, seq_dim=2).clone().requires_grad_(True)
        k_full = k.detach().clone().requires_grad_(True)
        v_full = v.detach().clone().requires_grad_(True)
        local_mask = compute_asymmetric_semi_ar_mask(**m, cp_rank=rank, cp_size=cp_size)
        out_local = flex_attention(q_local, k_full, v_full, block_mask=local_mask)
        assert torch.allclose(
            out_local, zigzag_slice(out_full.detach(), rank, cp_size, seq_dim=2), atol=1e-9
        ), f"cp={cp_size} rank={rank}: local-Q forward output mismatch"

        out_local.backward(zigzag_slice(grad_out, rank, cp_size, seq_dim=2))
        assert torch.allclose(
            q_local.grad, zigzag_slice(dq_full, rank, cp_size, seq_dim=2), atol=1e-9
        ), f"cp={cp_size} rank={rank}: dq mismatch"
        dk_sum += k_full.grad
        dv_sum += v_full.grad

    assert torch.allclose(dk_sum, dk_full, atol=1e-9), f"cp={cp_size}: summed dk mismatch"
    assert torch.allclose(dv_sum, dv_full, atol=1e-9), f"cp={cp_size}: summed dv mismatch"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
