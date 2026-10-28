# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Token-interleaved NVFP4 fixtures and independent FP64 sparse MLA references.

The 352-byte D576 layout is [256 E2M1 bytes, 64 E4M3 RoPE bytes,
32 permuted E4M3 scale bytes]. The 288-byte D512 layout is
[256 E2M1 bytes, 32 linear E4M3 scale bytes], including rotary channels.
Each scale covers 16 values; neither layout uses a global scale.
"""

import hashlib
from types import SimpleNamespace

import torch


def quantize_nvfp4(values):
    """Pack 512/576 channels with nearest-even E2M1 and E4M3 block16 scales."""
    dim = values.shape[-1]
    if dim not in (512, 576):
        raise ValueError("NVFP4 fixture expects 512 or 576 channels")
    groups = values[..., :512].float().unflatten(-1, (32, 16))
    scales = (groups.abs().amax(-1) / 6).clamp(2**-9, 448).to(torch.float8_e4m3fn)
    scaled = groups / scales.float().unsqueeze(-1)
    # E2M1 magnitudes: 0, .5, 1, 1.5, 2, 3, 4, 6. At a midpoint,
    # choose the even code (the low significand bit), including ties to zero.
    midpoints = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=values.device
    )
    low = torch.bucketize(scaled.abs().contiguous(), midpoints, right=False)
    high = torch.bucketize(scaled.abs().contiguous(), midpoints, right=True)
    codes = (low + ((high != low) & (low % 2 == 1))).to(torch.uint8)
    codes |= (scaled < 0).to(torch.uint8) << 3
    codes = codes.flatten(-2)
    payload = codes[..., 0::2] | (codes[..., 1::2] << 4)
    scale_bytes = scales.view(torch.uint8)
    if dim == 576:
        # FlashMLA V32 scale order is [s0,s4,...,s28,s1,s5,...,s31].
        scale_bytes = scale_bytes.unflatten(-1, (8, 4)).transpose(-1, -2).flatten(-2)
        rope = values[..., 512:].to(torch.float8_e4m3fn).view(torch.uint8)
        return torch.cat((payload, rope, scale_bytes), -1).contiguous()
    return torch.cat((payload, scale_bytes), -1).contiguous()


def decode_cache(cache, *, fp8_mma, tensor_scale=1.0):
    """Decode actual storage bytes; round NVFP4 products once for FP8 MMA."""
    if cache.dtype == torch.float8_e4m3fn:
        return cache.float() * tensor_scale
    width = cache.shape[-1]
    if width not in (288, 352):
        raise ValueError(f"Unknown packed cache width {width}")
    payload = cache[..., :256]
    codes = torch.stack((payload & 15, payload >> 4), -1).flatten(-2)
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=cache.device)
    values = lut[(codes & 7).long()] * torch.where(codes & 8 != 0, -1.0, 1.0)
    scale_bytes = cache[..., 320:] if width == 352 else cache[..., 256:]
    if width == 352:
        scale_bytes = scale_bytes.unflatten(-1, (4, 8)).transpose(-1, -2).flatten(-2)
    scales = scale_bytes.contiguous().view(torch.float8_e4m3fn).float()
    values = (values.unflatten(-1, (32, 16)) * scales.unsqueeze(-1)).flatten(-2)
    if fp8_mma:
        values = values.clamp(-448, 448).to(torch.float8_e4m3fn).float()
    if width == 352:
        rope = cache[..., 256:320].contiguous().view(torch.float8_e4m3fn).float()
        values = torch.cat((values, rope), -1)
    return values


def expand_fp8(cache, dim, tensor_scale=1.0):
    """Build the equivalent FP8-MMA operands outside graph capture, in chunks."""
    flat = cache.reshape(-1, cache.shape[-1])
    result = torch.empty(
        (flat.shape[0], dim), device=cache.device, dtype=torch.float8_e4m3fn
    )
    for start in range(0, len(flat), 16384):
        result[start : start + 16384] = decode_cache(
            flat[start : start + 16384], fp8_mma=True, tensor_scale=tensor_scale
        ).to(result.dtype)
    return result.view(*cache.shape[:-1], dim)


def tensor_hash(tensor):
    """Hash all tensor bytes in bounded host copies, outside measurement."""
    digest = hashlib.sha256(str((tuple(tensor.shape), str(tensor.dtype))).encode())
    raw = tensor.contiguous().view(torch.uint8).flatten()
    for start in range(0, raw.numel(), 1 << 26):
        digest.update(raw[start : start + (1 << 26)].cpu().numpy().tobytes())
    return digest.hexdigest()


def make_fixture(case, *, device):
    """Generate independent 64K request contexts, shuffled pages and distinct top-k.

    Explicit sparse lists define visibility; this is decode without an extra
    causal mask. D512 SWA uses overlapping per-query 128-token windows.
    All fixtures use FP8 Q, and storage is NHD with one KV head per row.
    """
    b, sq, h = case["batch"], case["queries"], case["heads"]
    context, topk, page = case["context_tokens"], case["topk"], case["page_size"]
    dim = 576 if case["variant"] == "d576" else 512
    g = torch.Generator(device=device).manual_seed(case["seed"])
    pages = torch.randperm(b * context // page, generator=g, device=device).view(b, -1)
    scores = torch.rand((b, sq, context), generator=g, device=device)
    logical = scores.topk(topk, largest=False, dim=-1).indices
    physical = torch.gather(pages[:, None, :].expand(-1, sq, -1), 2, logical // page)
    indices = (physical * page + logical % page).to(torch.int32)
    q = torch.randn((b, sq, h, dim), generator=g, device=device).clamp_(-1, 1)
    q = q.bfloat16().to(torch.float8_e4m3fn)
    del scores, logical, physical
    kv_generator = torch.Generator(device=device).manual_seed(case["seed"] + 1)

    def packed_pool(rows):
        """Bound temporary FP32 quantizer memory independently of context size."""
        result = torch.empty((rows, 288 + dim - 512), dtype=torch.uint8, device=device)
        for start in range(0, rows, 65536):
            n = min(65536, rows - start)
            values = (
                (torch.randn((n, dim), generator=kv_generator, device=device) / 10)
                .clamp_(-1, 1)
                .bfloat16()
            )
            result[start : start + n] = quantize_nvfp4(values)
        return result.view(-1, page, 1, result.shape[-1])

    primary = packed_pool(b * context)
    extra, extra_indices, primary_scale = None, None, 1.0
    if dim == 512:
        extra, extra_indices = primary, indices
        swa = case["swa_tokens"]
        swa_pages = (swa + sq - 1 + page - 1) // page
        g = torch.Generator(device=device).manual_seed(case["seed"] + 2)
        table = torch.randperm(b * swa_pages, generator=g, device=device).view(b, -1)
        positions = (
            torch.arange(swa, device=device)[None, None]
            + torch.arange(sq, device=device)[None, :, None]
        )
        physical = torch.gather(
            table[:, None].expand(-1, sq, -1), 2, (positions // page).expand(b, -1, -1)
        )
        indices = (physical * page + positions % page).to(torch.int32)
        if case["variant"] == "d512-nvfp4-extra":
            primary = packed_pool(b * swa_pages * page)
        else:
            effective = (
                (
                    torch.randn(
                        (b * swa_pages, page, 1, dim), generator=g, device=device
                    )
                    / 10
                )
                .clamp_(-1, 1)
                .to(torch.float8_e4m3fn)
            )
            primary_scale = 0.5
            primary = (effective.float() / primary_scale).to(torch.float8_e4m3fn)
            if not torch.equal(primary.float() * primary_scale, effective.float()):
                raise AssertionError(
                    "Per-tensor scale must preserve the FP8 baseline exactly"
                )
    return SimpleNamespace(
        q=q,
        primary=primary,
        extra=extra,
        indices=indices,
        extra_indices=extra_indices,
        primary_scale=primary_scale,
        dim=dim,
        case=case,
    )


def reference(fixture, *, fp8_mma):
    """FP64 oracle over the actual bytes, checking every query and output channel.

    The FP8 bound accounts for E4M3 probabilities, BF16 output rounding and
    probability underflow at P scale 448. Q/KV operand rounding is modeled
    directly, rather than absorbed by a larger tolerance.
    """
    q = fixture.q.flatten(0, 1).double()
    outputs, lses, bounds = [], [], []
    for start in range(0, len(q), 8):
        parts = []
        for cache, indices, scale in (
            (fixture.primary, fixture.indices, fixture.primary_scale),
            (fixture.extra, fixture.extra_indices, 1.0),
        ):
            if cache is not None:
                ids = indices.flatten(0, 1)[start : start + 8].long()
                selected = cache.reshape(-1, cache.shape[-1])[ids]
                parts.append(
                    decode_cache(selected, fp8_mma=fp8_mma, tensor_scale=scale).double()
                )
        kv = torch.cat(parts, 1)
        scores = torch.bmm(q[start : start + 8], kv.transpose(1, 2)) * fixture.dim**-0.5
        lse = scores.logsumexp(-1)
        probability = (scores - lse.unsqueeze(-1)).exp()
        v = kv[..., :512]
        out = torch.bmm(probability, v)
        if fp8_mma:
            bound = (2**-4 + 3 * 2**-8 + 1e-5) * torch.bmm(probability, v.abs())
            bound += (2**-10 / 448) * (scores.amax(-1) - lse).exp().unsqueeze(
                -1
            ) * v.abs().sum(1).unsqueeze(1) + 1e-6
        else:
            bound = 1e-3 + (2.01 / 128) * out.abs()
        outputs.append(out)
        lses.append(lse)
        bounds.append(bound)
    return torch.cat(outputs), torch.cat(lses), torch.cat(bounds)


def check_output(output, lse, expected, *, fp8_mma):
    """Apply output, LSE and (for BF16 MMA) cosine gates; never waive NaNs."""
    ref, ref_lse, bound = expected
    actual = output.reshape(ref.shape).double()
    delta = (actual - ref).abs()
    lse_delta = (lse.reshape(ref_lse.shape).double() - ref_lse).abs()
    lse_bound = (
        2e-4 + 1e-4 * ref_lse.abs()
        if fp8_mma
        else 1e-6 + (8.01 / 65536) * ref_lse.abs()
    )
    cosine_diff = float(
        1
        - 2
        * (actual * ref).sum()
        / (actual.square().sum() + ref.square().sum()).clamp_min(1e-30)
    )
    passed = bool((delta <= bound).all() and (lse_delta <= lse_bound).all())
    passed &= fp8_mma or cosine_diff <= 5e-6
    result = dict(
        passed=passed,
        max_abs_error=float(delta.max()),
        max_error_over_bound=float((delta / bound).max()),
        max_lse_error=float(lse_delta.max()),
        cosine_diff=cosine_diff,
    )
    if not passed:
        raise AssertionError(result)
    return result
