# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared native-HND fixtures and semantic references for Attention-TS."""

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import torch

PagedKVCache = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


# The selected current FP8 policy fixes these schedule values. They are
# deliberately benchmark-local oracle constants rather than tuning knobs.
_FP8_PROBABILITY_SCALE = 448.0
_FP8_KV_TILE_SIZE = 128
_FP8_NUM_KV_INSTANCES = 2


@dataclass(frozen=True)
class AttentionTSDecodeCase:
    q: torch.Tensor
    paged_kv_cache: PagedKVCache
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    paged_kv_indptr: torch.Tensor
    paged_kv_indices: torch.Tensor
    paged_kv_last_page_len: torch.Tensor
    reference_real: torch.Tensor
    output_dtype: torch.dtype
    mask_type: str
    bmm1_scale: float
    bmm2_scale: float
    q_scale: float
    k_scale: float
    v_scale: float
    o_scale: float


@dataclass(frozen=True)
class AttentionTSDecodeCaseSpec:
    """One curated row shared by TS and TRTLLM-gen decode tests."""

    case_id: str
    kv_lens: Tuple[int, ...]
    num_qo_heads: int
    num_kv_heads: int
    page_size: int
    mask_type: str
    qkv_dtype: torch.dtype
    output_dtype: torch.dtype
    cache_form: str
    provide_out: bool
    expected_bucket: int
    seed: int
    head_dim: int = 128
    seq_len_q: int = 1


ATTENTION_TS_SHARED_CASE_SPECS = (
    AttentionTSDecodeCaseSpec(
        "smoke",
        (128,),
        8,
        1,
        16,
        "dense",
        torch.float16,
        torch.float16,
        "combined",
        False,
        128,
        17,
    ),
    AttentionTSDecodeCaseSpec(
        "tail129",
        (129,),
        8,
        1,
        16,
        "dense",
        torch.float16,
        torch.float16,
        "tuple",
        False,
        256,
        23,
    ),
    AttentionTSDecodeCaseSpec(
        "ragged16",
        (15, 16, 129, 255),
        8,
        1,
        16,
        "dense",
        torch.float16,
        torch.float16,
        "combined",
        True,
        256,
        29,
    ),
    AttentionTSDecodeCaseSpec(
        "mha32",
        (31, 32, 127, 128),
        2,
        2,
        32,
        "causal",
        torch.float16,
        torch.float16,
        "tuple",
        False,
        128,
        31,
    ),
    AttentionTSDecodeCaseSpec(
        "bf16-r5",
        (257,),
        10,
        2,
        32,
        "dense",
        torch.bfloat16,
        torch.bfloat16,
        "combined",
        True,
        512,
        37,
    ),
    AttentionTSDecodeCaseSpec(
        "bf16-tail",
        (17, 127, 128, 129),
        8,
        1,
        16,
        "causal",
        torch.bfloat16,
        torch.bfloat16,
        "tuple",
        False,
        256,
        41,
    ),
    AttentionTSDecodeCaseSpec(
        "fp8-f16",
        (129,),
        64,
        8,
        16,
        "dense",
        torch.float8_e4m3fn,
        torch.float16,
        "combined",
        False,
        256,
        43,
    ),
    AttentionTSDecodeCaseSpec(
        "fp8-fp8",
        (31, 32, 33, 257),
        10,
        2,
        32,
        "causal",
        torch.float8_e4m3fn,
        torch.float8_e4m3fn,
        "tuple",
        True,
        512,
        47,
    ),
)


def _pr2265_case_specs() -> tuple[AttentionTSDecodeCaseSpec, ...]:
    """Return the pinned 30-row speculative-decode catalog from PR 2265."""

    model_geometries = (
        ("qwen", 96, 8, 128, 226500),
        ("gpt-oss", 64, 8, 64, 226600),
    )
    specs = []
    for model_name, num_qo_heads, num_kv_heads, head_dim, seed_base in model_geometries:
        for seq_len_q in (2, 4, 8):
            for batch_size in (8, 16, 32, 40, 64):
                specs.append(
                    AttentionTSDecodeCaseSpec(
                        case_id=(f"pr2265-{model_name}-b{batch_size}-sq{seq_len_q}"),
                        kv_lens=(16384,) * batch_size,
                        num_qo_heads=num_qo_heads,
                        num_kv_heads=num_kv_heads,
                        page_size=32,
                        mask_type="causal",
                        qkv_dtype=torch.float8_e4m3fn,
                        output_dtype=torch.float8_e4m3fn,
                        cache_form="combined",
                        provide_out=True,
                        expected_bucket=16384,
                        seed=seed_base + seq_len_q * 100 + batch_size,
                        head_dim=head_dim,
                        seq_len_q=seq_len_q,
                    )
                )
    if len(specs) != 30:
        raise AssertionError("the pinned PR2265 catalog must contain 30 rows")
    return tuple(specs)


ATTENTION_TS_PR2265_CASE_SPECS = _pr2265_case_specs()


def attention_ts_seq_lens_from_csr(
    paged_kv_indptr: torch.Tensor,
    paged_kv_last_page_len: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """Derive TRT/TS sequence lengths from the native CSR paging metadata."""

    page_counts = paged_kv_indptr[1:] - paged_kv_indptr[:-1]
    if page_counts.shape != paged_kv_last_page_len.shape:
        raise ValueError("CSR row count and last-page-length shapes must match")
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    return (page_counts - 1) * page_size + paged_kv_last_page_len


def attention_ts_rectangular_block_tables(
    paged_kv_indptr: torch.Tensor,
    paged_kv_indices: torch.Tensor,
) -> torch.Tensor:
    """Present the same native CSR page IDs through TRTLLM-gen's table ABI."""

    row_counts = paged_kv_indptr[1:] - paged_kv_indptr[:-1]
    if row_counts.numel() == 0:
        raise ValueError("paged_kv_indptr must describe at least one request")
    max_pages = int(row_counts.max().item())
    block_tables = torch.zeros(
        (row_counts.numel(), max_pages),
        dtype=torch.int32,
        device=paged_kv_indices.device,
    )
    for batch_idx in range(row_counts.numel()):
        row_begin = int(paged_kv_indptr[batch_idx].item())
        row_end = int(paged_kv_indptr[batch_idx + 1].item())
        block_tables[batch_idx, : row_end - row_begin] = paged_kv_indices[
            row_begin:row_end
        ]
    return block_tables


def fold_attention_ts_scales(
    *,
    sm_scale: float,
    q_scale: float,
    k_scale: float,
    v_scale: float,
    o_scale: float = 1.0,
) -> Tuple[float, float]:
    """Fold semantic Q/K/V/O calibration into the two public TS scales."""

    return sm_scale * q_scale * k_scale, v_scale / o_scale


@torch.no_grad()
def attention_ts_decode_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    paged_kv_indptr: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    paged_kv_last_page_len: torch.Tensor,
    *,
    sm_scale: float,
    q_scale: float = 1.0,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    mask_type: str = "dense",
) -> torch.Tensor:
    """Pure Torch FP32 paged-GQA semantic oracle.

    The returned tensor is in real (dequantized) output units. A stored FP8
    kernel output must be multiplied by ``o_scale`` before comparison.
    """

    if q.ndim == 3:
        batch_size, num_qo_heads, head_dim = q.shape
        q_tokens = q.unsqueeze(1)
        preserve_sq1_shape = True
    elif q.ndim == 4:
        batch_size, seq_len_q, num_qo_heads, head_dim = q.shape
        q_tokens = q
        preserve_sq1_shape = False
    else:
        raise ValueError("Q must have shape [B, H, D] or [B, SQ, H, D]")
    seq_len_q = q_tokens.shape[1]
    if mask_type not in ("dense", "causal"):
        raise ValueError("mask_type must be 'dense' or 'causal'")
    _, num_kv_heads, page_size, cache_head_dim = k_cache.shape
    if head_dim != cache_head_dim:
        raise ValueError("Q and KV head dimensions must match")
    if num_qo_heads % num_kv_heads != 0:
        raise ValueError("num_qo_heads must be divisible by num_kv_heads")

    group_size = num_qo_heads // num_kv_heads
    q_real = q_tokens.to(torch.float32) * q_scale
    k_real = k_cache.to(torch.float32) * k_scale
    v_real = v_cache.to(torch.float32) * v_scale
    output = torch.empty(
        (batch_size, seq_len_q, num_qo_heads, head_dim),
        dtype=torch.float32,
        device=q.device,
    )

    for batch_idx in range(batch_size):
        row_begin = int(paged_kv_indptr[batch_idx].item())
        row_end = int(paged_kv_indptr[batch_idx + 1].item())
        page_ids = paged_kv_indices[row_begin:row_end].to(torch.long)
        num_pages = row_end - row_begin
        last_page_len = int(paged_kv_last_page_len[batch_idx].item())
        kv_len = (num_pages - 1) * page_size + last_page_len

        # HND pages become token-major [tokens, Hkv, D] before GQA expansion.
        keys = (
            k_real[page_ids]
            .permute(0, 2, 1, 3)
            .reshape(num_pages * page_size, num_kv_heads, head_dim)[:kv_len]
        )
        values = (
            v_real[page_ids]
            .permute(0, 2, 1, 3)
            .reshape(num_pages * page_size, num_kv_heads, head_dim)[:kv_len]
        )
        keys = keys.repeat_interleave(group_size, dim=1)
        values = values.repeat_interleave(group_size, dim=1)
        for query_idx in range(seq_len_q):
            visible_kv_len = (
                kv_len if mask_type == "dense" else kv_len - (seq_len_q - 1 - query_idx)
            )
            if visible_kv_len <= 0:
                raise ValueError(
                    "bottom-right causal attention requires every KV length to be "
                    "at least seq_len_q"
                )
            logits = (
                torch.einsum(
                    "hd,thd->ht",
                    q_real[batch_idx, query_idx],
                    keys[:visible_kv_len],
                )
                * sm_scale
            )
            probabilities = torch.softmax(logits, dim=-1)
            output[batch_idx, query_idx] = torch.einsum(
                "ht,thd->hd", probabilities, values[:visible_kv_len]
            )

    return output[:, 0] if preserve_sq1_shape else output


@torch.no_grad()
def attention_ts_decode_fp8_kernel_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    paged_kv_indptr: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    paged_kv_last_page_len: torch.Tensor,
    *,
    bmm1_scale: float,
    bmm2_scale: float,
    output_dtype: torch.dtype,
    o_scale: float = 1.0,
    mask_type: str = "dense",
) -> torch.Tensor:
    """Match the selected TS FP8 decode policy in real output units.

    The FP8 kernel does not feed full-precision softmax probabilities to
    BMM2. Each of its two KV instruction streams visits alternating 128-token
    tiles, rescales an independent running max/sum, multiplies the exponentials
    by 448, and rounds only that BMM2 operand to E4M3. The denominator retains
    the unrounded scaled exponentials. The two streams are then merged using a
    common maximum.

    ``bmm1_scale`` and ``bmm2_scale`` are the already-folded public API scales.
    The returned values include output-storage rounding and multiplication by
    ``o_scale``, so they can be compared directly with
    :func:`dequantize_attention_ts_output`.
    """

    fp8_dtype = torch.float8_e4m3fn
    if q.dtype != fp8_dtype or k_cache.dtype != fp8_dtype or v_cache.dtype != fp8_dtype:
        raise ValueError("the FP8 kernel reference requires E4M3 Q, K, and V")
    if output_dtype not in (torch.float16, fp8_dtype):
        raise ValueError("FP8 TS decode supports float16 or E4M3 output")

    if q.ndim == 3:
        batch_size, num_qo_heads, head_dim = q.shape
        q_tokens = q.unsqueeze(1)
        preserve_sq1_shape = True
    elif q.ndim == 4:
        batch_size, seq_len_q, num_qo_heads, head_dim = q.shape
        q_tokens = q
        preserve_sq1_shape = False
    else:
        raise ValueError("Q must have shape [B, H, D] or [B, SQ, H, D]")
    seq_len_q = q_tokens.shape[1]
    if mask_type not in ("dense", "causal"):
        raise ValueError("mask_type must be 'dense' or 'causal'")
    _, num_kv_heads, page_size, cache_head_dim = k_cache.shape
    if head_dim != cache_head_dim:
        raise ValueError("Q and KV head dimensions must match")
    if num_qo_heads % num_kv_heads != 0:
        raise ValueError("num_qo_heads must be divisible by num_kv_heads")
    if paged_kv_indptr.numel() != batch_size + 1:
        raise ValueError("paged_kv_indptr must contain batch_size + 1 entries")

    group_size = num_qo_heads // num_kv_heads
    q_stored = q_tokens.to(torch.float32)
    k_stored = k_cache.to(torch.float32)
    v_stored = v_cache.to(torch.float32)
    output = torch.empty(
        (batch_size, seq_len_q, num_qo_heads, head_dim),
        dtype=torch.float32,
        device=q.device,
    )

    for batch_idx in range(batch_size):
        row_begin = int(paged_kv_indptr[batch_idx].item())
        row_end = int(paged_kv_indptr[batch_idx + 1].item())
        if row_end <= row_begin:
            raise ValueError("every request must reference at least one KV page")
        page_ids = paged_kv_indices[row_begin:row_end].to(torch.long)
        num_pages = row_end - row_begin
        last_page_len = int(paged_kv_last_page_len[batch_idx].item())
        if not 1 <= last_page_len <= page_size:
            raise ValueError("last-page lengths must be in [1, page_size]")
        kv_len = (num_pages - 1) * page_size + last_page_len

        keys = (
            k_stored[page_ids]
            .permute(0, 2, 1, 3)
            .reshape(num_pages * page_size, num_kv_heads, head_dim)[:kv_len]
            .repeat_interleave(group_size, dim=1)
        )
        values = (
            v_stored[page_ids]
            .permute(0, 2, 1, 3)
            .reshape(num_pages * page_size, num_kv_heads, head_dim)[:kv_len]
            .repeat_interleave(group_size, dim=1)
        )
        for query_idx in range(seq_len_q):
            visible_kv_len = (
                kv_len if mask_type == "dense" else kv_len - (seq_len_q - 1 - query_idx)
            )
            if visible_kv_len <= 0:
                raise ValueError(
                    "bottom-right causal attention requires every KV length to be "
                    "at least seq_len_q"
                )
            scores = torch.einsum(
                "hd,thd->ht",
                q_stored[batch_idx, query_idx],
                keys[:visible_kv_len],
            )

            stream_max = [
                torch.full(
                    (num_qo_heads,),
                    -torch.inf,
                    dtype=torch.float32,
                    device=q.device,
                )
                for _ in range(_FP8_NUM_KV_INSTANCES)
            ]
            stream_sum = [
                torch.zeros(num_qo_heads, dtype=torch.float32, device=q.device)
                for _ in range(_FP8_NUM_KV_INSTANCES)
            ]
            stream_acc = [
                torch.zeros(
                    (num_qo_heads, head_dim),
                    dtype=torch.float32,
                    device=q.device,
                )
                for _ in range(_FP8_NUM_KV_INSTANCES)
            ]
            stream_valid = [False] * _FP8_NUM_KV_INSTANCES
            num_tiles = (visible_kv_len + _FP8_KV_TILE_SIZE - 1) // _FP8_KV_TILE_SIZE

            for stream_idx in range(_FP8_NUM_KV_INSTANCES):
                for tile_idx in range(stream_idx, num_tiles, _FP8_NUM_KV_INSTANCES):
                    tile_begin = tile_idx * _FP8_KV_TILE_SIZE
                    tile_end = min(tile_begin + _FP8_KV_TILE_SIZE, visible_kv_len)
                    tile_scores = scores[:, tile_begin:tile_end]
                    local_max = tile_scores.max(dim=-1).values
                    new_max = (
                        torch.maximum(stream_max[stream_idx], local_max)
                        if stream_valid[stream_idx]
                        else local_max
                    )
                    probabilities_scaled = (
                        torch.exp((tile_scores - new_max.unsqueeze(-1)) * bmm1_scale)
                        * _FP8_PROBABILITY_SCALE
                    )
                    probabilities_quantized = probabilities_scaled.to(fp8_dtype).to(
                        torch.float32
                    )
                    local_sum = probabilities_scaled.sum(dim=-1)
                    tile_acc = torch.einsum(
                        "ht,thd->hd",
                        probabilities_quantized,
                        values[tile_begin:tile_end],
                    )

                    if stream_valid[stream_idx]:
                        previous_scale = torch.exp(
                            (stream_max[stream_idx] - new_max) * bmm1_scale
                        )
                        stream_sum[stream_idx] = (
                            stream_sum[stream_idx] * previous_scale + local_sum
                        )
                        stream_acc[stream_idx] = (
                            stream_acc[stream_idx] * previous_scale.unsqueeze(-1)
                            + tile_acc
                        )
                    else:
                        stream_sum[stream_idx] = local_sum
                        stream_acc[stream_idx] = tile_acc
                        stream_valid[stream_idx] = True
                    stream_max[stream_idx] = new_max

            valid_stream_maxima = [
                stream_max[idx]
                for idx in range(_FP8_NUM_KV_INSTANCES)
                if stream_valid[idx]
            ]
            final_max = torch.stack(valid_stream_maxima, dim=0).max(dim=0).values
            final_sum = torch.zeros_like(final_max)
            final_acc = torch.zeros_like(stream_acc[0])
            for stream_idx in range(_FP8_NUM_KV_INSTANCES):
                if not stream_valid[stream_idx]:
                    continue
                merge_scale = torch.exp(
                    (stream_max[stream_idx] - final_max) * bmm1_scale
                )
                final_sum += stream_sum[stream_idx] * merge_scale
                final_acc += stream_acc[stream_idx] * merge_scale.unsqueeze(-1)

            output_stored = final_acc / final_sum.unsqueeze(-1) * bmm2_scale
            output[batch_idx, query_idx] = (
                output_stored.to(output_dtype).to(torch.float32) * o_scale
            )

    return output[:, 0] if preserve_sq1_shape else output


def dequantize_attention_ts_output(
    output: torch.Tensor, *, o_scale: float
) -> torch.Tensor:
    """Convert a stored TS output to FP32 real-output units."""

    return output.to(torch.float32) * o_scale


def _stored_tensor(
    real: torch.Tensor, dtype: torch.dtype, scale: float
) -> torch.Tensor:
    if dtype == torch.float8_e4m3fn:
        return (real / scale).to(dtype)
    return real.to(dtype)


def make_attention_ts_decode_case(
    *,
    kv_lens: Sequence[int] = (128,),
    num_qo_heads: int = 8,
    num_kv_heads: int = 1,
    head_dim: int = 128,
    seq_len_q: int = 1,
    page_size: int = 16,
    qkv_dtype: torch.dtype = torch.float16,
    output_dtype: Optional[torch.dtype] = None,
    cache_form: str = "combined",
    mask_type: str = "dense",
    device: Union[str, torch.device] = "cuda",
    seed: int = 0,
    q_scale: float = 0.5,
    k_scale: float = 0.25,
    v_scale: float = 0.75,
    o_scale: Optional[float] = None,
) -> AttentionTSDecodeCase:
    """Create a deterministic native-CSR case with non-identity page IDs."""

    if seq_len_q <= 0:
        raise ValueError("seq_len_q must be positive")
    if not kv_lens or any(kv_len <= 0 for kv_len in kv_lens):
        raise ValueError("kv_lens must contain positive lengths")
    if cache_form not in ("combined", "tuple"):
        raise ValueError("cache_form must be 'combined' or 'tuple'")
    if mask_type not in ("dense", "causal"):
        raise ValueError("mask_type must be 'dense' or 'causal'")
    if mask_type == "causal" and any(kv_len < seq_len_q for kv_len in kv_lens):
        raise ValueError(
            "bottom-right causal attention requires every KV length to be at "
            "least seq_len_q"
        )
    if output_dtype is None:
        output_dtype = qkv_dtype

    if qkv_dtype != torch.float8_e4m3fn:
        q_scale = k_scale = v_scale = 1.0
    if o_scale is None:
        o_scale = 0.625 if output_dtype == torch.float8_e4m3fn else 1.0

    batch_size = len(kv_lens)
    pages_per_request = [(kv_len + page_size - 1) // page_size for kv_len in kv_lens]
    total_referenced_pages = sum(pages_per_request)
    num_physical_pages = total_referenced_pages + 3

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    page_ids = torch.randperm(num_physical_pages, generator=generator)[
        :total_referenced_pages
    ]
    if torch.equal(page_ids, torch.arange(total_referenced_pages)):
        page_ids = torch.roll(page_ids, 1)

    indptr = [0]
    for num_pages in pages_per_request:
        indptr.append(indptr[-1] + num_pages)
    last_page_lens = [(kv_len - 1) % page_size + 1 for kv_len in kv_lens]

    q_shape = (
        (batch_size, num_qo_heads, head_dim)
        if seq_len_q == 1
        else (batch_size, seq_len_q, num_qo_heads, head_dim)
    )
    q_real = 0.25 * torch.randn(q_shape, generator=generator, dtype=torch.float32)
    k_real = 0.25 * torch.randn(
        num_physical_pages,
        num_kv_heads,
        page_size,
        head_dim,
        generator=generator,
        dtype=torch.float32,
    )
    v_real = 0.25 * torch.randn(
        num_physical_pages,
        num_kv_heads,
        page_size,
        head_dim,
        generator=generator,
        dtype=torch.float32,
    )
    q = _stored_tensor(q_real, qkv_dtype, q_scale).to(device)
    k = _stored_tensor(k_real, qkv_dtype, k_scale).to(device)
    v = _stored_tensor(v_real, qkv_dtype, v_scale).to(device)
    paged_kv_indptr = torch.tensor(indptr, dtype=torch.int32, device=device)
    paged_kv_indices = page_ids.to(dtype=torch.int32, device=device)
    paged_kv_last_page_len = torch.tensor(
        last_page_lens, dtype=torch.int32, device=device
    )

    if cache_form == "combined":
        combined_cache = torch.stack((k, v), dim=1)
        paged_kv_cache: PagedKVCache = combined_cache
        k_cache = combined_cache[:, 0]
        v_cache = combined_cache[:, 1]
    else:
        paged_kv_cache = (k, v)
        k_cache, v_cache = k, v

    sm_scale = 1.0 / math.sqrt(head_dim)
    bmm1_scale, bmm2_scale = fold_attention_ts_scales(
        sm_scale=sm_scale,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        o_scale=o_scale,
    )
    reference_real = attention_ts_decode_reference(
        q,
        k_cache,
        v_cache,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        sm_scale=sm_scale,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        mask_type=mask_type,
    )
    return AttentionTSDecodeCase(
        q=q,
        paged_kv_cache=paged_kv_cache,
        k_cache=k_cache,
        v_cache=v_cache,
        paged_kv_indptr=paged_kv_indptr,
        paged_kv_indices=paged_kv_indices,
        paged_kv_last_page_len=paged_kv_last_page_len,
        reference_real=reference_real,
        output_dtype=output_dtype,
        mask_type=mask_type,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        o_scale=o_scale,
    )


__all__ = [
    "ATTENTION_TS_PR2265_CASE_SPECS",
    "ATTENTION_TS_SHARED_CASE_SPECS",
    "AttentionTSDecodeCase",
    "AttentionTSDecodeCaseSpec",
    "attention_ts_decode_fp8_kernel_reference",
    "attention_ts_decode_reference",
    "attention_ts_rectangular_block_tables",
    "attention_ts_seq_lens_from_csr",
    "dequantize_attention_ts_output",
    "fold_attention_ts_scales",
    "make_attention_ts_decode_case",
]
