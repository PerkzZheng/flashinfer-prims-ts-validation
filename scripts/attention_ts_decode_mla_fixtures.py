# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared deterministic fixtures and references for Attention-TS MLA decode."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Union

import torch

MLA_LATENT_DIM = 512
MLA_ROPE_DIM = 64
MLA_QK_DIM = MLA_LATENT_DIM + MLA_ROPE_DIM
MLA_PAGE_SIZE = 32
MLA_FP8_PROBABILITY_SCALE = 448.0


@dataclass(frozen=True)
class AttentionTSMLADecodeCase:
    """One canonical public-ABI MLA decode problem."""

    query: torch.Tensor
    kv_cache: torch.Tensor
    block_tables: torch.Tensor
    seq_lens: torch.Tensor
    max_seq_len: int
    output_dtype: torch.dtype
    bmm1_scale: float
    bmm2_scale: float
    q_scale: float
    kv_scale: float


def deterministic_variable_seq_lens(
    batch_size: int, max_seq_len: int
) -> tuple[int, ...]:
    """Return stable runtime lengths spanning roughly half to all of ``max_seq_len``.

    Request zero keeps the advertised maximum. Additional requests deliberately
    use non-page-aligned, distinct tails so the matrix exercises runtime length
    masking rather than only rectangular full-page rows.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_seq_len < MLA_PAGE_SIZE:
        raise ValueError(f"max_seq_len must be at least {MLA_PAGE_SIZE}")
    if batch_size == 1:
        return (max_seq_len,)

    lower = max(MLA_PAGE_SIZE, max_seq_len // 2)
    span = max_seq_len - lower
    lengths = [max_seq_len]
    used = {max_seq_len}
    for batch_idx in range(1, batch_size):
        # The prime step keeps B4/B8 rows reproducible without coupling the
        # fixture to Torch's RNG implementation.
        candidate = lower + ((batch_idx * 104729 + batch_size * 37) % span)
        candidate = min(candidate, max_seq_len - 1)
        if candidate % MLA_PAGE_SIZE == 0:
            candidate -= 1
        while candidate in used and candidate > lower:
            candidate -= 1
            if candidate % MLA_PAGE_SIZE == 0:
                candidate -= 1
        used.add(candidate)
        lengths.append(candidate)
    return tuple(lengths)


def _stored_tensor(
    real: torch.Tensor, dtype: torch.dtype, scale: float
) -> torch.Tensor:
    if dtype == torch.float8_e4m3fn:
        return (real / scale).to(dtype)
    return real.to(dtype)


def make_attention_ts_mla_decode_case(
    *,
    batch_size: int,
    num_qo_heads: int,
    max_seq_len: int,
    qkv_dtype: torch.dtype,
    seq_len_q: int = 1,
    device: Union[str, torch.device] = "cuda",
    seed: int = 0,
    seq_lens: Sequence[int] | None = None,
) -> AttentionTSMLADecodeCase:
    """Create one same-input TS/TRT MLA case with nonidentity page IDs.

    The cache is stored in FlashInfer's canonical dense MLA page form
    ``[num_pages, 1, page_size, 576]``. Q and cache use one shared storage
    scale each. Since MLA's compressed latent cache is both K and V, its scale
    is folded into both public BMM scales.
    """

    if batch_size <= 0 or num_qo_heads <= 0 or max_seq_len <= 0:
        raise ValueError("batch_size, num_qo_heads, and max_seq_len must be positive")
    if seq_len_q <= 0:
        raise ValueError("seq_len_q must be positive")
    if qkv_dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise ValueError(f"unsupported MLA test dtype {qkv_dtype}")

    if seq_lens is None:
        seq_lens = deterministic_variable_seq_lens(batch_size, max_seq_len)
    seq_lens = tuple(int(length) for length in seq_lens)
    if len(seq_lens) != batch_size:
        raise ValueError("seq_lens must contain one value per request")
    if any(length <= 0 or length > max_seq_len for length in seq_lens):
        raise ValueError("seq_lens must be in [1, max_seq_len]")
    if any(length < seq_len_q for length in seq_lens):
        raise ValueError(
            "bottom-right causal attention requires every sequence length to be "
            "at least seq_len_q"
        )

    pages_per_request = tuple(
        (length + MLA_PAGE_SIZE - 1) // MLA_PAGE_SIZE for length in seq_lens
    )
    max_pages_per_request = max(pages_per_request)
    total_referenced_pages = sum(pages_per_request)
    num_physical_pages = total_referenced_pages + 7

    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(seed)
    page_ids = torch.randperm(num_physical_pages, generator=cpu_generator)[
        :total_referenced_pages
    ]
    if torch.equal(page_ids, torch.arange(total_referenced_pages)):
        page_ids = torch.roll(page_ids, 1)

    block_tables_cpu = torch.zeros(
        (batch_size, max_pages_per_request), dtype=torch.int32
    )
    page_offset = 0
    for batch_idx, page_count in enumerate(pages_per_request):
        block_tables_cpu[batch_idx, :page_count] = page_ids[
            page_offset : page_offset + page_count
        ]
        page_offset += page_count

    device = torch.device(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 1)
    q_real = 0.2 * torch.randn(
        batch_size,
        seq_len_q,
        num_qo_heads,
        MLA_QK_DIM,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    kv_real = 0.2 * torch.randn(
        num_physical_pages,
        1,
        MLA_PAGE_SIZE,
        MLA_QK_DIM,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )

    if qkv_dtype == torch.float8_e4m3fn:
        q_scale = 0.0625
        kv_scale = 0.125
    else:
        q_scale = 1.0
        kv_scale = 1.0

    query = _stored_tensor(q_real, qkv_dtype, q_scale)
    kv_cache = _stored_tensor(kv_real, qkv_dtype, kv_scale)
    block_tables = block_tables_cpu.to(device=device)
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    sm_scale = 1.0 / math.sqrt(128 + MLA_ROPE_DIM)

    return AttentionTSMLADecodeCase(
        query=query,
        kv_cache=kv_cache,
        block_tables=block_tables,
        seq_lens=seq_lens_tensor,
        max_seq_len=max_seq_len,
        output_dtype=torch.bfloat16,
        bmm1_scale=sm_scale * q_scale * kv_scale,
        bmm2_scale=kv_scale,
        q_scale=q_scale,
        kv_scale=kv_scale,
    )


def _gather_request_cache(
    case: AttentionTSMLADecodeCase, batch_idx: int
) -> torch.Tensor:
    seq_len = int(case.seq_lens[batch_idx].item())
    page_count = (seq_len + MLA_PAGE_SIZE - 1) // MLA_PAGE_SIZE
    page_ids = case.block_tables[batch_idx, :page_count].to(torch.long)
    return (
        case.kv_cache[page_ids, 0].reshape(-1, MLA_QK_DIM)[:seq_len].to(torch.float32)
    )


def _fp8_mla_request_reference(
    q_stored: torch.Tensor,
    cache_stored: torch.Tensor,
    *,
    bmm1_scale: float,
    bmm2_scale: float,
    num_insts_kv: int,
    tile_size_kv: int,
) -> torch.Tensor:
    """Match TS MLA's TRTLLM-gen-compatible 448-scaled FP8 P path."""

    if num_insts_kv <= 0 or tile_size_kv <= 0:
        raise ValueError("num_insts_kv and tile_size_kv must be positive")

    q_latent = q_stored[:, :MLA_LATENT_DIM]
    q_rope = q_stored[:, MLA_LATENT_DIM:]
    c_latent = cache_stored[:, :MLA_LATENT_DIM]
    c_rope = cache_stored[:, MLA_LATENT_DIM:]
    scores = q_latent @ c_latent.T + q_rope @ c_rope.T
    num_heads, seq_len = scores.shape
    num_tiles = (seq_len + tile_size_kv - 1) // tile_size_kv

    stream_max = [
        torch.full((num_heads,), -torch.inf, dtype=torch.float32, device=scores.device)
        for _ in range(num_insts_kv)
    ]
    stream_sum = [
        torch.zeros(num_heads, dtype=torch.float32, device=scores.device)
        for _ in range(num_insts_kv)
    ]
    stream_acc = [
        torch.zeros(
            (num_heads, MLA_LATENT_DIM),
            dtype=torch.float32,
            device=scores.device,
        )
        for _ in range(num_insts_kv)
    ]
    stream_valid = [False] * num_insts_kv

    for stream_idx in range(num_insts_kv):
        for tile_idx in range(stream_idx, num_tiles, num_insts_kv):
            tile_begin = tile_idx * tile_size_kv
            tile_end = min(tile_begin + tile_size_kv, seq_len)
            tile_scores = scores[:, tile_begin:tile_end]
            local_max = tile_scores.max(dim=-1).values
            new_max = (
                torch.maximum(stream_max[stream_idx], local_max)
                if stream_valid[stream_idx]
                else local_max
            )
            probabilities_scaled = (
                torch.exp((tile_scores - new_max.unsqueeze(-1)) * float(bmm1_scale))
                * MLA_FP8_PROBABILITY_SCALE
            )
            probabilities_quantized = probabilities_scaled.to(torch.float8_e4m3fn).to(
                torch.float32
            )
            local_sum = probabilities_scaled.sum(dim=-1)
            tile_acc = probabilities_quantized @ c_latent[tile_begin:tile_end]

            if stream_valid[stream_idx]:
                correction = torch.exp(
                    (stream_max[stream_idx] - new_max) * float(bmm1_scale)
                )
                stream_sum[stream_idx] = stream_sum[stream_idx] * correction + local_sum
                stream_acc[stream_idx] = (
                    stream_acc[stream_idx] * correction.unsqueeze(-1) + tile_acc
                )
            else:
                stream_sum[stream_idx] = local_sum
                stream_acc[stream_idx] = tile_acc
                stream_valid[stream_idx] = True
            stream_max[stream_idx] = new_max

    valid_maxima = [stream_max[idx] for idx in range(num_insts_kv) if stream_valid[idx]]
    final_max = torch.stack(valid_maxima, dim=0).max(dim=0).values
    final_sum = torch.zeros_like(final_max)
    final_acc = torch.zeros_like(stream_acc[0])
    for stream_idx in range(num_insts_kv):
        if not stream_valid[stream_idx]:
            continue
        correction = torch.exp((stream_max[stream_idx] - final_max) * float(bmm1_scale))
        final_sum += stream_sum[stream_idx] * correction
        final_acc += stream_acc[stream_idx] * correction.unsqueeze(-1)
    return final_acc / final_sum.unsqueeze(-1) * float(bmm2_scale)


@torch.no_grad()
def attention_ts_mla_decode_ideal_reference(
    case: AttentionTSMLADecodeCase,
    *,
    batch_indices: Sequence[int] | None = None,
) -> torch.Tensor:
    """Return ideal IEEE-FP32 MLA outputs for selected requests."""

    if batch_indices is None:
        batch_indices = range(case.query.shape[0])
    batch_indices = tuple(int(index) for index in batch_indices)
    if not batch_indices:
        raise ValueError("batch_indices must not be empty")
    if any(index < 0 or index >= case.query.shape[0] for index in batch_indices):
        raise IndexError("batch index is outside the MLA case")

    previous_precision = torch.backends.cuda.matmul.fp32_precision
    try:
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        outputs = []
        seq_len_q = case.query.shape[1]
        for batch_idx in batch_indices:
            request_outputs = []
            full_cache_stored = _gather_request_cache(case, batch_idx)
            for query_idx in range(seq_len_q):
                visible_kv_len = full_cache_stored.shape[0] - (
                    seq_len_q - 1 - query_idx
                )
                if visible_kv_len <= 0:
                    raise ValueError(
                        "bottom-right causal attention requires every sequence "
                        "length to be at least seq_len_q"
                    )
                q_stored = case.query[batch_idx, query_idx].to(torch.float32)
                cache_stored = full_cache_stored[:visible_kv_len]
                q_latent = q_stored[:, :MLA_LATENT_DIM]
                q_rope = q_stored[:, MLA_LATENT_DIM:]
                c_latent = cache_stored[:, :MLA_LATENT_DIM]
                c_rope = cache_stored[:, MLA_LATENT_DIM:]
                scores = q_latent @ c_latent.T + q_rope @ c_rope.T
                probabilities = torch.softmax(scores * case.bmm1_scale, dim=-1)
                request_outputs.append(probabilities @ c_latent * case.bmm2_scale)
            outputs.append(torch.stack(request_outputs, dim=0))
        return torch.stack(outputs, dim=0)
    finally:
        torch.backends.cuda.matmul.fp32_precision = previous_precision


@torch.no_grad()
def attention_ts_mla_decode_reference(
    case: AttentionTSMLADecodeCase,
    *,
    num_insts_kv: int = 1,
    tile_size_kv: int = 128,
    batch_indices: Sequence[int] | None = None,
) -> torch.Tensor:
    """Return the selected TS MLA policy's FP32 output oracle."""

    if batch_indices is None:
        batch_indices = range(case.query.shape[0])
    batch_indices = tuple(int(index) for index in batch_indices)
    if not batch_indices:
        raise ValueError("batch_indices must not be empty")
    if any(index < 0 or index >= case.query.shape[0] for index in batch_indices):
        raise IndexError("batch index is outside the MLA case")

    outputs = []
    is_fp8 = case.query.dtype == torch.float8_e4m3fn
    seq_len_q = case.query.shape[1]
    for batch_idx in batch_indices:
        request_outputs = []
        full_cache_stored = _gather_request_cache(case, batch_idx)
        for query_idx in range(seq_len_q):
            visible_kv_len = full_cache_stored.shape[0] - (seq_len_q - 1 - query_idx)
            if visible_kv_len <= 0:
                raise ValueError(
                    "bottom-right causal attention requires every sequence length "
                    "to be at least seq_len_q"
                )
            q_stored = case.query[batch_idx, query_idx].to(torch.float32)
            cache_stored = full_cache_stored[:visible_kv_len]
            if is_fp8:
                output = _fp8_mla_request_reference(
                    q_stored,
                    cache_stored,
                    bmm1_scale=case.bmm1_scale,
                    bmm2_scale=case.bmm2_scale,
                    num_insts_kv=num_insts_kv,
                    tile_size_kv=tile_size_kv,
                )
            else:
                q_latent = q_stored[:, :MLA_LATENT_DIM]
                q_rope = q_stored[:, MLA_LATENT_DIM:]
                c_latent = cache_stored[:, :MLA_LATENT_DIM]
                c_rope = cache_stored[:, MLA_LATENT_DIM:]
                scores = q_latent @ c_latent.T + q_rope @ c_rope.T
                probabilities = torch.softmax(scores * case.bmm1_scale, dim=-1)
                output = probabilities @ c_latent * case.bmm2_scale
            request_outputs.append(output)
        outputs.append(torch.stack(request_outputs, dim=0))
    return torch.stack(outputs, dim=0)


def attention_ts_mla_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    """Return established FlashInfer MLA correctness tolerances."""

    if dtype == torch.float8_e4m3fn:
        # This reference already models E4M3 Q/KV storage, the TS kernel's
        # 448-scaled E4M3 probability path, per-instance online-softmax order,
        # split merging, and BF16 output scaling. Representative 1CTA/2CTA
        # residuals are below 7e-4, so retain margin for accumulation order
        # without allowing an all-zero output (~1e-2 error) to pass.
        return 5e-2, 1.5e-3
    # Eight public-wrapper probes spanning both kernel families, B1/B8, and
    # K2K/K8K measured at most 1.04e-4 absolute error. Retain nearly 5x
    # accumulation-order margin while ensuring a zero result cannot pass.
    return 1e-2, 5e-4


__all__ = [
    "AttentionTSMLADecodeCase",
    "MLA_LATENT_DIM",
    "MLA_FP8_PROBABILITY_SCALE",
    "MLA_PAGE_SIZE",
    "MLA_QK_DIM",
    "MLA_ROPE_DIM",
    "attention_ts_mla_decode_ideal_reference",
    "attention_ts_mla_decode_reference",
    "attention_ts_mla_tolerances",
    "deterministic_variable_seq_lens",
    "make_attention_ts_mla_decode_case",
]
