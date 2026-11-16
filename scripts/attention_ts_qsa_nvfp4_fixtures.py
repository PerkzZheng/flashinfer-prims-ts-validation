# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic QSA NVFP4/FP8 workloads, byte accounting and an FP32 oracle.

Index generation and accounting are NumPy-only so CPU checks can run without
CUDA. Tensor helpers import Torch lazily.
"""

import numpy as np

HEAD_DIM = 256
SPARSE_BLOCK = 4
# Tensor parallelism splits both the model's Q and KV heads.
TP_DEGREES = (1, 2, 4)
NVFP4_ROW_BYTES = HEAD_DIM // 2 + HEAD_DIM // 16
E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def tail_tokens(position, count_current):
    """Visible tokens of the causal current block that the kernel appends."""
    if count_current and (position + 1) % SPARSE_BLOCK == 0:
        return 0
    return position % SPARSE_BLOCK + 1


def _sample_excluding(rng, bound, count, exclude):
    """Draw ``count`` distinct block IDs in ``[0, bound)`` outside ``exclude``."""
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    taken = np.zeros(bound, dtype=bool)
    taken[exclude] = True
    out, need = [], count
    while need > 0:
        cand = rng.integers(0, bound, size=need * 2 + 16)
        _, first = np.unique(cand, return_index=True)
        cand = cand[np.sort(first)]
        cand = cand[~taken[cand]][:need]
        taken[cand] = True
        out.append(cand)
        need -= cand.size
    return np.concatenate(out)


def request_chain(positions, topk, rng, *, overlap, count_current):
    """Top-k block IDs for consecutive query tokens of one request.

    The first token draws distinct uniform blocks from its causal range, as
    FlashMLA's random top-k tests do. Each next token keeps a random
    ``overlap`` fraction of its predecessor's blocks and draws the rest from
    its own causal range, so overlap(t, t+k) is about ``overlap**k``. Rows are
    shuffled; entries beyond the live prefix are -1.
    """
    out = np.full((positions.size, topk), -1, dtype=np.int32)
    prev = None
    for t, position in enumerate(positions):
        bound = (int(position) + int(count_current)) // SPARSE_BLOCK
        count = min(topk, bound)
        if prev is None or prev.size == 0:
            sel = _sample_excluding(rng, bound, count, np.empty(0, dtype=np.int64))
        else:
            keep = rng.choice(
                prev, size=min(count, round(overlap * prev.size)), replace=False
            )
            sel = np.concatenate(
                [keep, _sample_excluding(rng, bound, count - keep.size, keep)]
            )
        sel = rng.permutation(sel)
        out[t, : sel.size] = sel
        prev = sel
    return out


def build_indices(case):
    """Deterministic indices and accounting; each request has its own RNG."""
    q_per_req = case["query_tokens"]
    ids, positions, request_of, unique_blocks, attended = [], [], [], [], []
    overlaps = []
    tag = 0 if case["phase"] == "decode" else 1
    for r in range(case["batch"]):
        rng = np.random.default_rng([case["seed"], tag, r])
        pos = np.arange(
            case["context_tokens"] - q_per_req, case["context_tokens"], dtype=np.int64
        )
        chain = request_chain(
            pos,
            case["topk_blocks"],
            rng,
            overlap=case["overlap"],
            count_current=case["count_current_block"],
        )
        live = [row[row >= 0].astype(np.int64) for row in chain]
        tails = [
            p // SPARSE_BLOCK
            for p in pos
            if tail_tokens(int(p), case["count_current_block"])
        ]
        unique_blocks.append(np.unique(np.concatenate([*live, tails])).size)
        attended.extend(
            live[i].size * SPARSE_BLOCK
            + tail_tokens(int(pos[i]), case["count_current_block"])
            for i in range(q_per_req)
        )
        overlaps.extend(
            np.intersect1d(live[i - 1], live[i]).size / max(1, live[i].size)
            for i in range(1, q_per_req)
        )
        ids.append(chain)
        positions.append(pos)
        request_of.append(np.full(q_per_req, r, dtype=np.int32))
    return dict(
        ids=np.concatenate(ids),
        positions=np.concatenate(positions),
        request_of=np.concatenate(request_of),
        unique_blocks=np.asarray(unique_blocks, dtype=np.int64),
        attended_tokens=np.asarray(attended, dtype=np.int64),
        mean_adjacent_overlap=float(np.mean(overlaps)) if overlaps else None,
    )


def accounting(case, indices):
    """Bytes and FLOPs for one call: unique needed KV once, plus Q and O.

    Grouping overfetch is excluded. NVFP4 counts payload plus E4M3 scales (in
    either layout); FP8 counts E4M3 K/V and BF16 counts BF16 K/V.
    FLOPs = 4 * D * Hq * attended tokens.
    """
    hq, hkv = case["heads_q"], case["heads_kv"]
    tokens = case["batch"] * case["query_tokens"]
    unique_tokens = int(indices["unique_blocks"].sum()) * SPARSE_BLOCK
    kv_row = {"nvfp4": 2 * NVFP4_ROW_BYTES, "fp8": 2 * HEAD_DIM, "bf16": 4 * HEAD_DIM}
    q_elem = {"fp8": 1, "bf16": 2}
    out = {}
    for storage, row_bytes in kv_row.items():
        for q_dtype, q_bytes in q_elem.items():
            out[f"{storage}-q{q_dtype}"] = (
                unique_tokens * hkv * row_bytes + tokens * hq * HEAD_DIM * (q_bytes + 2)
            )
    return dict(
        bytes=out,
        flops=4.0 * HEAD_DIM * hq * float(indices["attended_tokens"].sum()),
        unique_blocks_per_request=float(indices["unique_blocks"].mean()),
        mean_adjacent_overlap=indices["mean_adjacent_overlap"],
    )


def quantize_nvfp4(x):
    """Return (payload uint8 [..., D/2], E4M3 scales [..., D/16]).

    Scale = amax/6 rounded to E4M3; values round to the nearest E2M1 magnitude
    with the sign in bit 3, and the even element sits in the low nibble.
    """
    import torch

    shape = x.shape
    blocks = x.float().reshape(*shape[:-1], shape[-1] // 16, 16)
    scales = (blocks.abs().amax(-1) / 6.0).clamp_min(2.0**-9).to(torch.float8_e4m3fn)
    y = blocks / scales.float().unsqueeze(-1)
    mids = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device)
    code = torch.bucketize(y.abs(), mids).to(torch.uint8)
    code = (code | ((y < 0).to(torch.uint8) << 3)).reshape(*shape[:-1], -1, 2)
    return (code[..., 0] | (code[..., 1] << 4)).contiguous(), scales.contiguous()


def dequantize_nvfp4_rows(rows):
    """Interleaved uint8 rows [..., D/2 + D/16] -> float32 [..., D]."""
    import torch

    payload = rows[..., : HEAD_DIM // 2]
    scales = rows[..., HEAD_DIM // 2 :].contiguous().view(torch.float8_e4m3fn)
    lut = torch.tensor(E2M1_GRID + tuple(-v for v in E2M1_GRID), device=rows.device)
    codes = torch.stack((payload & 0x0F, payload >> 4), -1).reshape(
        *payload.shape[:-1], HEAD_DIM
    )
    return lut[codes.long()] * scales.float().repeat_interleave(16, -1)


def make_pools(heads_kv, pages, page_size, seed, device):
    """Matched caches [pages, Hkv, page, row] built from one quantized tile.

    One random BF16 tile is quantized to interleaved NVFP4 rows (payload, then
    the token's scales) and repeated over the pool; the FP8 pool holds the
    E4M3 rounding of exactly those dequantized values. Repetition keeps the
    allocation fast; the shuffled block table still scatters every request.
    """
    import torch

    gen = torch.Generator(device=device).manual_seed(seed)
    tile = max(1, min(pages, (64 << 20) // (heads_kv * page_size * HEAD_DIM)))
    shape = (pages, heads_kv, page_size)
    pools = {}
    for kind in ("k", "v"):
        x = torch.randn(
            tile, heads_kv, page_size, HEAD_DIM, device=device, generator=gen
        ).to(torch.bfloat16)
        payload, scales = quantize_nvfp4(x)
        rows = torch.cat([payload, scales.view(torch.uint8)], -1)
        fp8 = dequantize_nvfp4_rows(rows).to(torch.float8_e4m3fn)
        nv = torch.empty(*shape, NVFP4_ROW_BYTES, dtype=torch.uint8, device=device)
        f8 = torch.empty(*shape, HEAD_DIM, dtype=torch.float8_e4m3fn, device=device)
        for start in range(0, pages, tile):
            stop = min(pages, start + tile)
            nv[start:stop].copy_(rows[: stop - start])
            f8[start:stop].copy_(fp8[: stop - start])
        pools[kind] = (nv, f8)
    order = torch.randperm(pages, generator=torch.Generator().manual_seed(seed + 1))
    return dict(
        nvfp4=(pools["k"][0], pools["v"][0]),
        fp8=(pools["k"][1], pools["v"][1]),
        page_order=order.to(device=device, dtype=torch.int32),
    )


def bf16_pool(rows, chunk_pages=4096):
    """Plain BF16 K or V holding exactly the dequantized NVFP4 values.

    ``rows`` is an interleaved NVFP4 pool [pages, Hkv, page, D/2 + D/16]; every
    E2M1 * E4M3 product is representable in BF16, so the conversion is exact.
    Pages are converted in chunks to bound the FP32 temporary.
    """
    import torch

    out = torch.empty(
        *rows.shape[:-1], HEAD_DIM, dtype=torch.bfloat16, device=rows.device
    )
    for start in range(0, rows.shape[0], chunk_pages):
        stop = min(rows.shape[0], start + chunk_pages)
        out[start:stop] = dequantize_nvfp4_rows(rows[start:stop]).to(torch.bfloat16)
    return out


def interleave_v_scales(scales):
    """Reorder V block scales [pages, Hkv, page, D/16] into TRT-LLM's layout.

    Separate NVFP4 V scale factors interleave four consecutive tokens. With
    ``Q = D/64`` scales per quarter row, the scale of (token ``4g + t``, block
    ``b``) moves to token ``4g + b // Q``, slot ``(b % Q) * 4 + t``, as
    FlashInfer's tests build them. K scales stay in natural token order.
    """
    pages, heads, page, blocks = scales.shape
    return (
        scales.reshape(pages, heads, page // 4, 4, 4, blocks // 4)
        .permute(0, 1, 2, 4, 5, 3)
        .reshape(scales.shape)
        .contiguous()
    )


def split_nvfp4_pools(k_rows, v_rows):
    """Split interleaved NVFP4 pools into payloads and ``kv_scale_factors``.

    Returns ((K payload, V payload), (K scales, V scales)): payloads are
    uint8 [..., D/2] and scales E4M3 [..., D/16], with V scales interleaved as
    in :func:`interleave_v_scales`. The values are identical to the rows.
    """
    import torch

    payloads = tuple(r[..., : HEAD_DIM // 2].contiguous() for r in (k_rows, v_rows))
    k_scales, v_scales = (
        r[..., HEAD_DIM // 2 :].contiguous().view(torch.float8_e4m3fn)
        for r in (k_rows, v_rows)
    )
    return payloads, (k_scales, interleave_v_scales(v_scales))


def reference(case, indices, q, k_rows, v_rows, block_table, tokens):
    """FP32 attention for selected query tokens over dequantized NVFP4 rows.

    ``q`` is [T, Hq, D] in real units. The oracle attends every selected
    block plus the causal tail; returns [len(tokens), Hq, D].
    """
    import torch

    hq, hkv = case["heads_q"], case["heads_kv"]
    page = case["page_size"]
    out = []
    for t in tokens:
        position = int(indices["positions"][t])
        request = int(indices["request_of"][t])
        blocks = indices["ids"][t]
        blocks = blocks[blocks >= 0].astype(np.int64)
        toks = (blocks[:, None] * SPARSE_BLOCK + np.arange(SPARSE_BLOCK)).reshape(-1)
        tail = tail_tokens(position, case["count_current_block"])
        if tail:
            toks = np.concatenate([toks, np.arange(position - tail + 1, position + 1)])
        toks = torch.from_numpy(toks).to(q.device)
        phys = block_table[request, toks // page].long()
        slot = toks % page
        kk = dequantize_nvfp4_rows(k_rows[phys, :, slot])
        vv = dequantize_nvfp4_rows(v_rows[phys, :, slot])
        qq = q[t].float().view(hkv, hq // hkv, HEAD_DIM)
        s = torch.einsum("hgd,nhd->hgn", qq, kk) * HEAD_DIM**-0.5
        out.append(torch.einsum("hgn,nhd->hgd", s.softmax(-1), vv).reshape(hq, -1))
    return torch.stack(out)


def check_output(out, expected, *, rel_l2_limit):
    """Finite output and relative L2 / cosine against the FP32 oracle."""
    import torch

    out = out.float()
    finite = bool(torch.isfinite(out).all())
    diff = out - expected
    rel_l2 = float(diff.norm() / expected.norm())
    cosine = float(
        torch.nn.functional.cosine_similarity(out.flatten(), expected.flatten(), 0)
    )
    result = dict(
        finite=finite,
        rel_l2=rel_l2,
        cosine=cosine,
        max_abs=float(diff.abs().max()),
        rel_l2_limit=rel_l2_limit,
    )
    if not finite or not rel_l2 <= rel_l2_limit:
        raise AssertionError(f"Output check failed: {result}")
    return result
