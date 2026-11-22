#!/usr/bin/env python3
# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Run the mask-tile workloads against one FlashInfer checkout.

Each case calls a public Prims-TS functional API (context, paged context,
paged decode or paged MLA decode) on deterministic inputs, checks the output
against an independent FP32 oracle on sampled requests, records a SHA-256 of
the output bytes for bitwise base/candidate comparison, and times the
FlashInfer kernels with ``torch.profiler``. Use ``run_mask_tiles_ab.py`` to
interleave a base and a candidate checkout on one GPU.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import sys
from pathlib import Path

from attention_ts_decode_artifacts import atomic_write_json, git_value, sha256_file
from mask_tiles_suite import (
    case_seed,
    case_tolerance,
    expand_lengths,
    load_suite,
    local_kv_length,
    owned_positions,
    result_summary,
    timing_summary,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = ROOT / "suites/mask_tiles/mask_tiles.json"
REFERENCE_REQUESTS = 2
QUERY_CHUNK = 512
# PyTorch-native kernels and copies belong to input handling, not attention.
EXCLUDED_KERNELS = ("at::", "memcpy", "memset", "Memcpy", "Memset")


def load_runtime(source_root, workdir):
    """Select one clean source checkout before importing FlashInfer."""
    source_root = source_root.resolve()
    head = git_value(source_root, "rev-parse", "HEAD")
    if not head or not (source_root / "flashinfer").is_dir():
        raise ValueError("--source-root must be a FlashInfer Git checkout")
    if git_value(source_root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError(
            "Commit tracked source changes before collecting reproducible results"
        )
    # One JIT workspace per source commit lets interleaved rounds reuse kernels.
    os.environ.setdefault(
        "FLASHINFER_WORKSPACE_BASE", str(workdir.resolve() / "jit" / head)
    )
    sys.path.insert(0, str(source_root))
    import flashinfer
    import torch
    from flashinfer.attention import prims_ts

    if Path(flashinfer.__file__).resolve().parent.parent != source_root:
        raise ValueError("FlashInfer did not import from --source-root")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    metadata = {
        "source_commit": head,
        "gpu": props.name,
        "capability": list(torch.cuda.get_device_capability()),
        "sms": props.multi_processor_count,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
        "harness_sha256": {
            p.name: sha256_file(p)
            for p in (
                Path(__file__),
                ROOT / "scripts/mask_tiles_suite.py",
                ROOT / "scripts/attention_ts_decode_artifacts.py",
            )
        },
    }
    return torch, prims_ts, metadata


# ------------------------------------------------------------------ inputs
def _cumsum(torch, xs):
    out = [0]
    for x in xs:
        out.append(out[-1] + x)
    return torch.tensor(out, dtype=torch.int32, device="cuda")


def _randn(torch, shape, g, dtype):
    return torch.randn(shape, generator=g, device="cuda", dtype=torch.float32).to(dtype)


def _page_table(torch, lengths, page, g):
    """Shuffle complete pages across requests; return (table, pages per row)."""
    pages = [max(math.ceil(x / page), 1) for x in lengths]
    total = sum(pages)
    perm = torch.randperm(total, generator=g, device="cuda").to(torch.int32)
    table = torch.zeros((len(lengths), max(pages)), dtype=torch.int32, device="cuda")
    offset = 0
    for i, n in enumerate(pages):
        table[i, :n] = perm[offset : offset + n]
        offset += n
    return table, total


def _gather_hnd(cache, table_row, length):
    """Return ``[length, H, D]`` keys of one request from an HND page pool."""
    pages = cache[table_row[: math.ceil(length / cache.shape[2])].long()]
    return pages.permute(0, 2, 1, 3).reshape(-1, cache.shape[1], cache.shape[3])[
        :length
    ]


def _visible(q_len, k_len, query_idx, mask, window_left):
    """FlashInfer's bottom-right visible interval ``[begin, end)``."""
    end = k_len if mask == "dense" else k_len - q_len + query_idx + 1
    begin = 0 if window_left < 0 else max(0, end - window_left - 1)
    return begin, end


def _attend(torch, q, k, v, valid, scale):
    """FP32 attention of ``q[T, Hq, D]`` over ``k/v[L, Hkv, D]`` with ``valid[T, L]``."""
    ratio = q.shape[1] // k.shape[1]
    k = k.float().repeat_interleave(ratio, dim=1)
    v = v.float().repeat_interleave(ratio, dim=1)
    scores = torch.einsum("thd,lhd->htl", q.float(), k) * scale
    scores = scores.masked_fill(~valid.unsqueeze(0), -torch.inf)
    probs = torch.softmax(scores, dim=-1).nan_to_num(0.0)
    return torch.einsum("htl,lhd->thd", probs, v)


def _interval_mask(torch, q_len, k_len, mask, window_left, rows=None):
    rows = range(q_len) if rows is None else rows
    keys = torch.arange(k_len, device="cuda")
    out = []
    for t in rows:
        begin, end = _visible(q_len, k_len, t, mask, window_left)
        out.append((keys >= begin) & (keys < end))
    return torch.stack(out)


def build_context(torch, prims_ts, c, g):
    b, hq, hkv, d = c["batch"], c["heads_q"], c["heads_kv"], c["head_dim"]
    q_lens = expand_lengths(c["q_lens"], b, "q_lens")
    kv_lens = [x + c.get("kv_extra", 0) for x in q_lens]
    dtype = torch.bfloat16
    scale = d**-0.5
    kw = dict(mask_type=c["mask"], window_left=c["window_left"], softmax_scale=scale)
    layout = c["layout"]
    starts = ends = None
    if layout == "fixed":
        q = _randn(torch, (b, q_lens[0], hq, d), g, dtype)
        k = _randn(torch, (b, kv_lens[0], hkv, d), g, dtype)
        v = _randn(torch, (b, kv_lens[0], hkv, d), g, dtype)
        if c.get("variable_window"):
            s = q_lens[0]
            qi = torch.arange(s, device="cuda", dtype=torch.int32)
            ends = qi.unsqueeze(0).repeat(b, 1).contiguous()
            starts = (ends - c["variable_window"] + 1).clamp_min(0)
            starts = (starts + (qi % 37).unsqueeze(0)).clamp_max(ends).contiguous()
            kw.update(
                mask_type="variable_window",
                variable_window_token_starts=starts,
                variable_window_token_ends=ends,
            )
            kw.pop("window_left")

        def call():
            return prims_ts.batch_prefill(q, k, v, **kw)

        def request(i):
            return q[i], k[i], v[i]

        inputs = (q, k, v)
    elif layout == "ragged":
        qo, kvp = _cumsum(torch, q_lens), _cumsum(torch, kv_lens)
        q = _randn(torch, (sum(q_lens), hq, d), g, dtype)
        k = _randn(torch, (sum(kv_lens), hkv, d), g, dtype)
        v = _randn(torch, (sum(kv_lens), hkv, d), g, dtype)

        def call():
            return prims_ts.batch_prefill(q, k, v, qo_indptr=qo, kv_indptr=kvp, **kw)

        def request(i):
            return (
                q[sum(q_lens[:i]) : sum(q_lens[: i + 1])],
                k[sum(kv_lens[:i]) : sum(kv_lens[: i + 1])],
                v[sum(kv_lens[:i]) : sum(kv_lens[: i + 1])],
            )

        inputs = (q, k, v)
    else:
        page = c["page_size"]
        qo = _cumsum(torch, q_lens)
        lens = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
        q = _randn(torch, (sum(q_lens), hq, d), g, dtype)
        table, total = _page_table(torch, kv_lens, page, g)
        kc = _randn(torch, (total, hkv, page, d), g, dtype)
        vc = _randn(torch, (total, hkv, page, d), g, dtype)

        def call():
            return prims_ts.batch_prefill_with_paged_kv_cache(
                q, kc, vc, qo, table, lens, page_size=page, **kw
            )

        def request(i):
            return (
                q[sum(q_lens[:i]) : sum(q_lens[: i + 1])],
                _gather_hnd(kc, table[i], kv_lens[i]),
                _gather_hnd(vc, table[i], kv_lens[i]),
            )

        inputs = (q, kc, vc, table)

    def reference(out):
        """Yield (expected, actual) for sampled requests, in query chunks."""
        for i in range(min(b, REFERENCE_REQUESTS)):
            qi, ki, vi = request(i)
            if layout == "fixed":
                oi = out[i]
            else:
                oi = out[sum(q_lens[:i]) : sum(q_lens[: i + 1])]
            for lo in range(0, q_lens[i], QUERY_CHUNK):
                rows = range(lo, min(lo + QUERY_CHUNK, q_lens[i]))
                if starts is not None:
                    keys = torch.arange(kv_lens[i], device="cuda")
                    valid = (keys >= starts[i, lo : rows.stop, None]) & (
                        keys <= ends[i, lo : rows.stop, None]
                    )
                else:
                    valid = _interval_mask(
                        torch, q_lens[i], kv_lens[i], c["mask"], c["window_left"], rows
                    )
                yield (
                    _attend(torch, qi[lo : rows.stop], ki, vi, valid, scale),
                    oi[lo : rows.stop],
                )

    return call, reference, inputs


def build_decode(torch, prims_ts, c, g):
    b, sq, hq, hkv, d = (
        c["batch"],
        c["queries"],
        c["heads_q"],
        c["heads_kv"],
        c["head_dim"],
    )
    page = c["page_size"]
    dtype = torch.float8_e4m3fn if c["dtype"] == "fp8" else torch.bfloat16
    kv_lens = expand_lengths(c["kv_lens"], b, "kv_lens")
    lens = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    table, total = _page_table(torch, kv_lens, page, g)
    kc = _randn(torch, (total, hkv, page, d), g, dtype)
    vc = _randn(torch, (total, hkv, page, d), g, dtype)
    q = _randn(torch, (b, hq, d) if sq == 1 else (b, sq, hq, d), g, dtype)
    scale = d**-0.5

    def call():
        return prims_ts.batch_decode_with_paged_kv_cache(
            q,
            (kc, vc),
            table,
            lens,
            seq_len_q=sq,
            mask_type=c["mask"],
            window_left=c["window_left"],
            softmax_scale=scale,
            out_dtype=torch.bfloat16,
        )

    def reference(out):
        for i in range(min(b, REFERENCE_REQUESTS)):
            qi = q[i].reshape(sq, hq, d)
            ki = _gather_hnd(kc, table[i], kv_lens[i])
            vi = _gather_hnd(vc, table[i], kv_lens[i])
            valid = _interval_mask(torch, sq, kv_lens[i], c["mask"], c["window_left"])
            yield _attend(torch, qi, ki, vi, valid, scale), out[i].reshape(sq, hq, d)

    return call, reference, (q, kc, vc, table)


def build_mla(torch, prims_ts, c, g):
    b, hq, page = c["batch"], c["heads_q"], c["page_size"]
    q_lens = expand_lengths(c["q_lens"], b, "q_lens")
    kv_lens = expand_lengths(c["kv_lens"], b, "kv_lens")
    cp = c.get("cp")
    local = [local_kv_length(x, cp) if cp else x for x in kv_lens]
    lens = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    table, total = _page_table(torch, local, page, g)
    kv = _randn(torch, (total, page, 576), g, torch.bfloat16)
    scale = 576**-0.5
    kw = dict(mask_type=c["mask"], softmax_scale=scale)
    if cp:
        kw.update(
            cp_world=cp[0],
            cp_interleave_size=cp[1],
            cp_rank=cp[2],
            max_kv_len=table.shape[1] * page,
        )
    if c.get("packed"):
        q = _randn(torch, (sum(q_lens), hq, 576), g, torch.bfloat16)
        kw.update(qo_indptr=_cumsum(torch, q_lens), max_q_len=max(q_lens))
    else:
        q = _randn(torch, (b, q_lens[0], hq, 576), g, torch.bfloat16)

    def call():
        return prims_ts.batch_mla_decode_with_paged_kv_cache(q, kv, table, lens, **kw)

    def reference(out):
        for i in range(min(b, REFERENCE_REQUESTS)):
            if c.get("packed"):
                rows = slice(sum(q_lens[:i]), sum(q_lens[: i + 1]))
                qi, oi = q[rows], out[rows]
            else:
                qi, oi = q[i], out[i]
            pages = kv[table[i, : math.ceil(local[i] / page)].long()]
            ki = pages.reshape(-1, 576)[: local[i]].unsqueeze(1)
            pos = torch.tensor(
                owned_positions(local[i], cp) if cp else list(range(local[i])),
                device="cuda",
            )
            ends = torch.tensor(
                [
                    _visible(q_lens[i], kv_lens[i], t, c["mask"], -1)[1]
                    for t in range(q_lens[i])
                ],
                device="cuda",
            )
            valid = pos.unsqueeze(0) < ends.unsqueeze(1)
            yield _attend(torch, qi, ki, ki[..., :512], valid, scale), oi

    return call, reference, (q, kv, table, lens)


BUILDERS = {"context": build_context, "decode": build_decode, "mla": build_mla}


# ----------------------------------------------------------------- measure
def output_tensor(result):
    return result[0] if isinstance(result, tuple) else result


def tensor_sha256(torch, tensors):
    digest = hashlib.sha256()
    for t in tensors:
        t = t.detach().contiguous()
        digest.update(str((t.dtype, tuple(t.shape))).encode())
        digest.update(t.view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def check_reference(torch, reference, out, tol):
    """Compare sampled requests with the FP32 oracle."""
    err_sq = ref_sq = 0.0
    max_abs = 0.0
    for expected, actual in reference(out):
        diff = actual.float() - expected
        err_sq += float(diff.square().sum())
        ref_sq += float(expected.square().sum())
        max_abs = max(max_abs, float(diff.abs().max()))
    rel_l2 = math.sqrt(err_sq / max(ref_sq, 1e-30))
    return {
        "oracle_pass": rel_l2 <= tol["rel_l2"] and max_abs <= tol["max_abs"],
        "rel_l2": rel_l2,
        "max_abs": max_abs,
        "tolerance": tol,
    }


def kernel_time_us(torch, call, calls):
    """Return per-call FlashInfer kernel time (µs) and the kernels counted."""
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(calls):
            call()
        torch.cuda.synchronize()
    total, names = 0.0, {}
    for event in prof.events():
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        if any(tag in event.name for tag in EXCLUDED_KERNELS):
            continue
        total += event.device_time
        names[event.name[:96]] = names.get(event.name[:96], 0.0) + event.device_time
    kernels = sorted(names.items(), key=lambda kv: -kv[1])
    return total / calls, [(n, t / calls) for n, t in kernels[:4]]


def run_case(torch, prims_ts, spec, index, c, timing):
    g = torch.Generator(device="cuda")
    g.manual_seed(case_seed(spec, index))
    call, reference, inputs = BUILDERS[c["family"]](torch, prims_ts, c, g)
    row = {"case_id": c["id"], "family": c["family"], "status": "failed"}
    row["input_sha256"] = tensor_sha256(torch, inputs)
    out = output_tensor(call())
    torch.cuda.synchronize()
    row["output_sha256"] = tensor_sha256(torch, [out])
    row["all_outputs_finite"] = bool(torch.isfinite(out.float()).all())
    row.update(check_reference(torch, reference, out, case_tolerance(spec, c)))
    for _ in range(timing["warmups"]):
        call()
    samples, kernels = [], None
    for _ in range(timing["samples"]):
        us, top = kernel_time_us(torch, call, timing["calls_per_sample"])
        samples.append(us)
        kernels = kernels or top
    row["timing"] = timing_summary(samples)
    row["kernels"] = kernels
    replay = output_tensor(call())
    torch.cuda.synchronize()
    row["deterministic"] = tensor_sha256(torch, [replay]) == row["output_sha256"]
    if row["all_outputs_finite"] and row["oracle_pass"] and row["deterministic"]:
        row["status"] = "passed"
    return row


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    p.add_argument("--source-root", type=Path)
    p.add_argument("--output", type=Path, help="New result JSON path")
    p.add_argument(
        "--workdir",
        type=Path,
        help="Directory for per-commit JIT workspaces (default: output's directory)",
    )
    p.add_argument("--case", action="append", help="Select a case ID (repeatable)")
    p.add_argument("--list", action="store_true", help="Print the selected cases")
    p.add_argument("--samples", type=int, help="Override timing samples")
    return p


def select_cases(spec, wanted):
    cases = list(enumerate(spec["cases"]))
    if not wanted:
        return cases
    ids = {c["id"] for _, c in cases}
    if len(set(wanted)) != len(wanted) or not set(wanted) <= ids:
        raise ValueError("--case must select unique existing case IDs")
    return [(i, c) for i, c in cases if c["id"] in wanted]


def main(argv=None):
    args = parser().parse_args(argv)
    spec = load_suite(args.suite)
    cases = select_cases(spec, args.case)
    if args.list:
        print(json.dumps([c for _, c in cases], indent=2))
        return 0
    if not args.source_root or not args.output:
        raise ValueError("--source-root and --output are required for execution")
    if args.output.exists():
        raise ValueError("Choose a new output path; results are never overwritten")
    timing = dict(spec["timing"])
    if args.samples is not None:
        if args.samples <= 0:
            raise ValueError("--samples must be positive")
        timing["samples"] = args.samples
    workdir = args.workdir or args.output.resolve().parent
    torch, prims_ts, metadata = load_runtime(args.source_root, workdir)
    payload = {
        "schema_version": 1,
        "metadata": metadata,
        "suite": spec,
        "timing": timing,
        "selected_cases": [c["id"] for _, c in cases],
        "rows": [],
    }
    for index, c in cases:
        try:
            row = run_case(torch, prims_ts, spec, index, c, timing)
        except Exception as exc:
            print(
                f"{c['id']}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True
            )
            row = {
                "case_id": c["id"],
                "family": c["family"],
                "status": "failed",
                "error_type": type(exc).__name__,
            }
        payload["rows"].append(row)
        if row["status"] == "passed":
            print(
                f"{c['id']:44s} {row['timing']['median_us']:10.2f} us  "
                f"rel_l2={row['rel_l2']:.2e}",
                flush=True,
            )
        else:
            print(f"{c['id']:44s} FAILED", flush=True)
        payload["summary"] = result_summary(payload["rows"], [c for _, c in cases])
        atomic_write_json(args.output, payload)
        gc.collect()
        torch.cuda.empty_cache()
    if git_value(args.source_root, "rev-parse", "HEAD") != metadata[
        "source_commit"
    ] or git_value(args.source_root, "status", "--porcelain", "--untracked-files=no"):
        payload["summary"]["passed"] = False
        payload["summary"]["source_changed_during_run"] = True
        atomic_write_json(args.output, payload)
    return 0 if payload["summary"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
