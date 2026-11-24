#!/usr/bin/env python3
# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Record and compare decode dispatch choices; this is not a golden-policy test."""

import argparse
import json
from pathlib import Path

from attention_ts_decode_artifacts import atomic_write_json, git_value, sha256_file
from bench_attention_ts_nvfp4_decode import load_runtime
from nvfp4_decode_suite import POLICY_FIELDS, policy_cases


def compare_policy(before, after):
    """Compare every requested case, including rejection reasons, on equal capacity."""
    for key in (
        "gpu",
        "capability",
        "sms",
        "dsl",
        "torch",
        "cuda",
        "audit_sha256",
        "harness_sha256",
    ):
        if before["metadata"][key] != after["metadata"][key]:
            raise ValueError(f"Incompatible policy environments: {key}")
    if [r["case"] for r in before["rows"]] != [r["case"] for r in after["rows"]]:
        raise ValueError("Policy comparisons require identical case lists")
    return [
        {"case": a["case"], "before": b, "after": a}
        for b, a in zip(before["rows"], after["rows"])
        if b != a
    ]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--kind", choices=["all", "fp8", "nvfp4"], default="all")
    p.add_argument(
        "--baseline", type=Path, help="fail if policy/rejection records differ"
    )
    args = p.parse_args(argv)
    if args.output.exists():
        raise ValueError("Choose a new output filename")
    _, _, _, metadata = load_runtime(args.source_root, args.output.resolve().parent)
    from cutlass import BFloat16, Float4E2M1FN, Float8E4M3FN, Float16
    try:  # feature packages under prims_ts/variants/
        from flashinfer.attention.prims_ts.variants.decode.kernel.selection import (
            make_decode_config,
        )
    except ImportError:
        try:  # one folder per feature (prims_ts/decode/kernel/selection.py)
            from flashinfer.attention.prims_ts.decode.kernel.selection import (
                make_decode_config,
            )
        except ImportError:  # kernels/ layout
            from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_config import (
                make_decode_config,
            )

    metadata["audit_sha256"] = sha256_file(__file__)
    rows = []
    for case in policy_cases():
        kind, dim, out, b, q, hkv, ratio, kv, page, scope, force = case
        if args.kind != "all" and kind != args.kind:
            continue
        options = {"use_variable_seqlens_q": "packed" in scope}
        if force:
            options.update(
                use_keeps_mma_ab=True, tile_size_q=64, groups_tokens_heads_q=True
            )
        row = {"case": list(case)}
        try:
            cfg = make_decode_config(
                headdim=dim,
                args=options,
                seq_len_q=q,
                seq_len_kv=kv,
                batch_size=b,
                num_heads_q=hkv * ratio,
                num_heads_kv=hkv,
                q_dtype=Float8E4M3FN,
                k_dtype=Float4E2M1FN if kind == "nvfp4" else Float8E4M3FN,
                v_dtype=Float4E2M1FN if kind == "nvfp4" else Float8E4M3FN,
                o_dtype=Float16 if out == "fp16" else BFloat16,
                qkv_layout="pagedKv",
                num_tokens_per_page=page,
                mask_type="causal",
                use_attention_sinks="sinks" in scope,
                auto_tuner=not force,
            )
            row["policy"] = {k: getattr(cfg, k) for k in POLICY_FIELDS}
        except (ValueError, TypeError) as exc:
            row["rejection"] = {"type": type(exc).__name__, "reason": str(exc)}
        rows.append(row)
    payload = {
        "schema_version": 1,
        "metadata": metadata,
        "kind": args.kind,
        "rows": rows,
    }
    if args.baseline:
        payload["differences"] = compare_policy(
            json.loads(args.baseline.read_text()), payload
        )
    if git_value(args.source_root, "rev-parse", "HEAD") != metadata[
        "source_commit"
    ] or git_value(args.source_root, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Source changed during policy audit")
    atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "cases": len(rows),
                "accepted": sum("policy" in r for r in rows),
                "differences": len(payload.get("differences", [])),
            }
        )
    )
    return 1 if payload.get("differences") else 0


if __name__ == "__main__":
    raise SystemExit(main())
