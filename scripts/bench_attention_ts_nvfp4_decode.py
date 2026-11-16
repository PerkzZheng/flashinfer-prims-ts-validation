#!/usr/bin/env python3
# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Run dense NVFP4/FP8 decode workloads through public automatic dispatch."""

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from attention_ts_decode_artifacts import atomic_write_json, git_value, sha256_file
from nvfp4_decode_suite import (
    BACKENDS,
    POLICY_FIELDS,
    load_suite,
    result_summary,
    timing_summary,
)

ROOT = Path(__file__).resolve().parents[1]


def load_runtime(source_root, output_dir):
    """Select one clean source checkout before importing FlashInfer or test oracles."""
    source_root = source_root.resolve()
    head = git_value(source_root, "rev-parse", "HEAD")
    if not head or not (source_root / "flashinfer").is_dir():
        raise ValueError("--source-root must be a FlashInfer Git checkout")
    if git_value(source_root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError(
            "Commit tracked source changes before collecting reproducible results"
        )
    os.environ.setdefault("CUTE_DSL_DISABLE_FILE_CACHING", "1")
    os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", str(output_dir / "jit" / head))
    sys.path.insert(0, str(source_root))
    import flashinfer
    import torch

    if Path(flashinfer.__file__).resolve().parent.parent != source_root:
        raise ValueError("FlashInfer did not import from --source-root")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    # The source revision owns its precision-aware reference and unchanged tolerances.
    tests = source_root / "tests/attention"
    sys.path.insert(0, str(tests))
    oracle = importlib.import_module("test_attention_ts_decode")
    if Path(oracle.__file__).resolve() != tests / "test_attention_ts_decode.py":
        raise ValueError("Reference module did not import from selected checkout")
    metadata = {
        "source_commit": head,
        "gpu": props.name,
        "capability": list(torch.cuda.get_device_capability()),
        "sms": props.multi_processor_count,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
        "oracle_sha256": sha256_file(oracle.__file__),
        "harness_sha256": {
            p.name: sha256_file(p)
            for p in (
                Path(__file__),
                ROOT / "scripts/nvfp4_decode_suite.py",
                ROOT / "scripts/attention_ts_decode_artifacts.py",
            )
        },
    }
    return torch, flashinfer, oracle, metadata


def logical_v_scales(scales, c):
    """Undo the stored four-token V-scale ordering for logical dequantization."""
    h, p, d = c["heads_kv"], c["page_size"], c["head_dim"]
    return (
        scales.reshape(-1, h, p // 4, 4, d // 64, 4)
        .permute(0, 1, 2, 5, 3, 4)
        .reshape(-1, h, p, d // 16)
        .contiguous()
    )


def make_inputs(torch, oracle, c, spec):
    """Materialize the historical fixture with deterministic RNG draw ordering."""
    b, ql, hq, hkv, d, p, kv = (
        c[k]
        for k in (
            "batch",
            "queries",
            "heads_q",
            "heads_kv",
            "head_dim",
            "page_size",
            "kv_len",
        )
    )
    normal = spec["fixture"] == "normal-fp8"
    seed = spec["seed"] if normal else spec["seed"] + b * 31 + ql
    torch.manual_seed(seed)
    pages, total = kv // p, b * (kv // p)
    fp8 = torch.float8_e4m3fn
    if not normal:
        table = torch.randperm(total, dtype=torch.int32).view(b, pages).cuda()
    q = torch.randn((b * ql, hq, d), device="cuda").to(fp8)
    if normal:
        k8, v8 = [
            (torch.randn(total, hkv, p, d, device="cuda") * 0.5).to(fp8)
            for _ in range(2)
        ]
        table = torch.randperm(total, device="cuda", dtype=torch.int32).view(b, pages)
        kv4 = sf = None
    else:
        k8, v8 = [
            (
                torch.randint(
                    0, 255, (total, hkv, p, d), dtype=torch.uint8, device="cuda"
                )
                & 0x3F
            ).view(fp8)
            for _ in range(2)
        ]
        k4, v4 = [
            torch.randint(
                0, 255, (total, hkv, p, d // 2), dtype=torch.uint8, device="cuda"
            )
            for _ in range(2)
        ]
        scale_shape = (total, hkv, p, d // 16)
        if spec["fixture"] == "independent-issue":
            ks, vs = [
                torch.randint(
                    0x30, 0x38, scale_shape, dtype=torch.uint8, device="cuda"
                ).view(fp8)
                for _ in range(2)
            ]
        else:
            ks, vs = [
                (torch.randn(scale_shape, device="cuda") * 0.5).to(fp8)
                for _ in range(2)
            ]
            k8 = oracle._dequantize_nvfp4_cache(k4, ks, fp8)
            v8 = oracle._dequantize_nvfp4_cache(v4, logical_v_scales(vs, c), fp8)
        kv4, sf = (k4, v4), (ks, vs)
    tensors = (q, *(kv4 if kv4 else (k8, v8)), *(sf or ()), table)
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(
            tensor.contiguous()
            .view(torch.uint8)
            .flatten()[:2048]
            .cpu()
            .numpy()
            .tobytes()
        )
    return SimpleNamespace(
        q=q,
        q_ts=q if ql == 1 else q.view(b, ql, hq, d),
        kv8=(k8, v8),
        kv4=kv4,
        sf=sf,
        table=table,
        lens=torch.full((b,), kv, dtype=torch.int32, device="cuda"),
        seed=seed,
        input_sample_sha256=digest.hexdigest(),
    )


def make_call(torch, flashinfer, f, c, spec, backend):
    """Plan and allocate outside timing, retaining the complete backend call."""
    fp8 = torch.float8_e4m3fn
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp8": fp8}[c["out_dtype"]]
    nvfp4 = backend.endswith("nvfp4")
    cache, scales = (f.kv4, f.sf) if nvfp4 else (f.kv8, None)
    sm_scale = c["head_dim"] ** -0.5
    mask = (
        "dense"
        if spec["mask"] == "dense-q1-causal-multi" and c["queries"] == 1
        else "causal"
    )
    if backend.startswith("prims_"):
        wrapper = flashinfer.attention.prims_ts.BatchDecodePagedTSWrapper("HND")
        wrapper.plan(
            "cuda",
            c["batch"],
            c["heads_q"],
            c["heads_kv"],
            c["head_dim"],
            c["page_size"],
            c["kv_len"],
            max_q_len=c["queries"],
            q_dtype=fp8,
            kv_dtype=cache[0].dtype,
            out_dtype=dtype,
            mask_type=mask,
            **({"seq_lens_kv": f.lens.cpu()} if spec["planned_lengths"] else {}),
        )
        state = wrapper._plan_state
        cfg = state.config
        policy = {k: getattr(cfg, k) for k in POLICY_FIELDS}
        policy["requested_splits"] = int(state.requested_splits or cfg.splits_kv)
        policy["q_ctas_per_kv_head"] = math.ceil(c["queries"] / cfg.q_tokens_per_cta)
        out = torch.empty_like(f.q_ts, dtype=dtype)

        def call():
            return wrapper.run(
                f.q_ts,
                cache,
                None if spec["planned_lengths"] else f.lens,
                f.table,
                kv_scale_factors=scales,
                softmax_scale=sm_scale,
                out=out,
                validate=False,
            )
    else:
        policy = None
        # TRT NVFP4's historical comparison uses FP8 output; record this explicitly.
        out = torch.empty_like(f.q, dtype=fp8 if nvfp4 else dtype)
        workspace = torch.zeros(512 << 20, dtype=torch.uint8, device="cuda")

        def call():
            return flashinfer.decode.trtllm_batch_decode_with_kv_cache(
                f.q,
                cache,
                workspace,
                f.table,
                f.lens,
                c["kv_len"],
                bmm1_scale=sm_scale,
                bmm2_scale=1.0,
                out=out,
                backend="trtllm-gen",
                q_len_per_req=c["queries"],
                **({"kv_cache_sf": scales} if nvfp4 else {}),
            )

    return call, out, cache, scales, policy, mask


def check_output(torch, oracle, f, c, out, cache, scales, policy, mask):
    """Check all outputs finite and Prims samples with the revision's strict oracle."""
    if not torch.isfinite(out.float()).all():
        raise AssertionError("Nonfinite output")
    if policy is None:
        return {
            "scope": "finiteness-only",
            "all_outputs_finite": True,
            "oracle_pass": None,
        }
    n, pages, p = min(c["batch"], 2), f.table.shape[1], c["page_size"]
    indices = f.table[:n].reshape(-1).long()
    k, v = (x.index_select(0, indices) for x in cache)
    if scales is not None:
        ks, vs = (x.index_select(0, indices) for x in scales)
        k = oracle._dequantize_nvfp4_cache(k, ks, torch.float8_e4m3fn)
        v = oracle._dequantize_nvfp4_cache(
            v, logical_v_scales(vs, c), torch.float8_e4m3fn
        )
    expected = (
        oracle._fp8_decode_reference(
            f.q_ts[:n],
            k,
            v,
            torch.arange(0, (n + 1) * pages, pages, dtype=torch.int32, device="cuda"),
            torch.arange(n * pages, dtype=torch.int32, device="cuda"),
            torch.full((n,), p, dtype=torch.int32, device="cuda"),
            softmax_scale=c["head_dim"] ** -0.5,
            output_scale=1.0,
            out_dtype=out.dtype,
            o_scale=1.0,
            mask_type=mask,
            splits_kv=policy["requested_splits"],
            num_insts_kv=policy["num_insts_kv"],
            q_tokens_per_cta=policy["q_tokens_per_cta"],
        )
        .to(out.dtype)
        .float()
    )
    actual = out[:n]
    oracle._assert_case_correct(
        actual,
        SimpleNamespace(
            q=f.q_ts[:n],
            k_cache=cache[0],
            out_dtype=out.dtype,
            o_scale=1.0,
            reference_real=expected,
        ),
    )
    error = actual.float() - expected
    return {
        "scope": "precision-aware-sampled-reference",
        "oracle_pass": True,
        "reference_requests": n,
        "all_outputs_finite": True,
        "max_abs": float(error.abs().max()),
        "relative_l2": float(error.norm() / expected.norm().clamp_min(1e-30)),
    }


def measure(torch, call, protocol, repeats, calls_per_replay, warmups):
    """Retain each GPU sample; host preparation and compilation are excluded."""
    if protocol == "cold-cupti":
        from flashinfer.testing import bench_gpu_time

        samples = [
            float(t) * 1000
            for t in bench_gpu_time(
                call,
                dry_run_iters=warmups,
                repeat_iters=repeats,
                use_cuda_graph=True,
                enable_cupti=True,
                cold_l2_cache=True,
            )
        ]
    else:
        call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(calls_per_replay):
                call()
        samples = []
        for _ in range(repeats):
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000 / calls_per_replay)
    return timing_summary(samples)


def run_case(torch, flashinfer, oracle, c, spec, backends, controls):
    """Keep one fixture alive across backends and retain failures without timing them."""
    f = make_inputs(torch, oracle, c, spec)
    rows = []
    for backend in backends:
        row = {
            "case": c,
            "backend": backend,
            "seed": f.seed,
            "input_sample_sha256": f.input_sample_sha256,
            "status": "failed",
        }
        call = out = cache = scales = graph = None
        try:
            call, out, cache, scales, policy, mask = make_call(
                torch, flashinfer, f, c, spec, backend
            )
            row.update(policy=policy, out_dtype=str(out.dtype), mask=mask)
            call()
            torch.cuda.synchronize()
            row["before"] = check_output(
                torch, oracle, f, c, out, cache, scales, policy, mask
            )
            row["timing"] = measure(torch, call, **controls)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                call()
            out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            row["after"] = check_output(
                torch, oracle, f, c, out, cache, scales, policy, mask
            )
            row["status"] = "ok"
        except Exception as exc:
            # Full diagnostic text stays on stderr; public JSON avoids local paths.
            print(
                f"{c['id']} {backend}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            row["error_type"] = type(exc).__name__
        rows.append(row)
        print(
            json.dumps(
                {
                    "case": c["id"],
                    "backend": backend,
                    "status": row["status"],
                    "median_us": row.get("timing", {}).get("median_us"),
                }
            ),
            flush=True,
        )
        del call, out, cache, scales, graph
    return rows


def parser():
    """Expose workload selection independently of GPU/source initialization."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--suite", type=Path, default=ROOT / "suites/nvfp4_decode/issue9311.json"
    )
    p.add_argument("--source-root", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument(
        "--case", action="append", help="exact case ID; repeat to select multiple rows"
    )
    p.add_argument("--backend", choices=BACKENDS, action="append")
    p.add_argument(
        "--repeats", type=int, help="default: 10 hot graph or 50 cold CUPTI samples"
    )
    p.add_argument(
        "--calls-per-replay", type=int, default=20, help="hot-graph protocol only"
    )
    p.add_argument("--warmups", type=int, default=10, help="cold-CUPTI protocol only")
    p.add_argument(
        "--list", action="store_true", help="print cases without importing CUDA"
    )
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    spec = load_suite(args.suite)
    cases = spec["cases"]
    if args.case:
        if len(set(args.case)) != len(args.case) or not set(args.case) <= {
            c["id"] for c in cases
        }:
            raise ValueError("--case must select unique existing case IDs")
        cases = [c for c in cases if c["id"] in args.case]
    backends = args.backend or spec["backends"]
    if len(set(backends)) != len(backends) or not set(backends) <= set(
        spec["backends"]
    ):
        raise ValueError("--backend must select unique backends from the suite")
    if args.list:
        print(json.dumps({"cases": cases, "backends": backends}, indent=2))
        return 0
    if not args.source_root or not args.output:
        raise ValueError("--source-root and --output are required for execution")
    if args.output.exists():
        raise ValueError(
            "Choose a new output filename; existing results are never overwritten"
        )
    controls = dict(
        protocol=spec["timing"],
        repeats=args.repeats or (10 if spec["timing"] == "hot-graph" else 50),
        calls_per_replay=args.calls_per_replay,
        warmups=args.warmups,
    )
    if (
        any(controls[k] <= 0 for k in ("repeats", "calls_per_replay", "warmups"))
        or args.repeats == 0
    ):
        raise ValueError("Timing counts must be positive")
    torch, flashinfer, oracle, metadata = load_runtime(
        args.source_root, args.output.resolve().parent
    )
    payload = {
        "schema_version": 1,
        "metadata": metadata,
        "suite": spec,
        "controls": controls,
        "selected_cases": [c["id"] for c in cases],
        "backends": backends,
        "rows": [],
    }
    for c in cases:
        try:
            payload["rows"].extend(
                run_case(torch, flashinfer, oracle, c, spec, backends, controls)
            )
        except Exception as exc:
            print(
                f"Fixture failure: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            payload["rows"].extend(
                {
                    "case": c,
                    "backend": b,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                }
                for b in backends
            )
        payload["summary"] = result_summary(payload["rows"], cases, backends)
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
